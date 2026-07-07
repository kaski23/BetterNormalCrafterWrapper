"""Memory-conscious NormalCrafter inference engine.

The engine is intentionally organized as four explicit stages:

1. CLIP image encoding: RGB frames -> semantic frame embeddings.
2. VAE encoding: RGB frames -> spatial RGB latents.
3. UNet inference: RGB latents + CLIP embeddings -> normal-map latents.
4. VAE decoding: normal-map latents -> three-channel normal maps.

The central ownership rule is simple: large, long-lived intermediate videos stay
on CPU. Only the currently active chunk or temporal window is moved to the
inference device. This avoids retaining an entire video on CUDA and makes peak
VRAM mostly independent of total clip length.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import inspect
import logging
import threading
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .preprocess import FrameSource
from .unet import NormalCrafterUNet
from .windows import Window, build_windows, overlap_length

logger = logging.getLogger(__name__)

# Called after every completed chunk/window. The strings make progress reporting
# descriptive without tying the engine to ComfyUI itself.
ProgressCallback = Callable[[str, int, int], None]

# ``staged`` minimizes VRAM by moving only one model component to the device.
# ``resident`` avoids repeated device transfers but keeps every component loaded.
OffloadMode = Literal["staged", "resident"]

# ``original`` restores the source dimensions after decoding. ``processed`` keeps
# the aspect-preserving, max-resolution-limited dimensions used by the model.
OutputSize = Literal["original", "processed"]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Immutable runtime settings for one inference call.

    Keeping these values in a frozen dataclass prevents an inference run from
    observing settings that change halfway through execution.
    """

    max_resolution: int = 1024
    window_size: int = 14
    step_size: int = 10
    clip_chunk_size: int = 16
    vae_encode_chunk_size: int = 7
    vae_decode_chunk_size: int = 4
    offload_mode: OffloadMode = "staged"
    offload_after: bool = True
    output_size: OutputSize = "original"
    renormalize_normals: bool = True

    def validate(self) -> None:
        """Reject combinations that would produce gaps or invalid tensors."""

        if self.max_resolution < 64:
            raise ValueError("max_resolution must be at least 64")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")

        # A step larger than the window would leave uncovered frames between
        # windows. A step of zero would never advance.
        if not 1 <= self.step_size <= self.window_size:
            raise ValueError("step_size must be between 1 and window_size")

        for name, value in (
            ("clip_chunk_size", self.clip_chunk_size),
            ("vae_encode_chunk_size", self.vae_encode_chunk_size),
            ("vae_decode_chunk_size", self.vae_decode_chunk_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


class NormalCrafterModel:
    """Own and execute every model component required for inference.

    Ownership guarantees:

    * No module-level model globals.
    * No duplicate pipeline object pointing at the same modules.
    * No CUDA tensor is returned to ComfyUI.
    * Intermediate full-video tensors live on CPU.
    * A re-entrant lock serializes device moves and inference on this instance.

    The object may stay alive in a ComfyUI graph cache, but its resource state is
    explicit and controllable through ``offload_to_cpu``.
    """

    # These values reproduce the conditioning used by the released pipeline.
    # They are inherited from the Stable Video Diffusion interface, even though
    # NormalCrafter performs a deterministic one-step prediction.
    FPS_CONDITIONING = 7
    MOTION_BUCKET_ID = 127
    NOISE_AUG_STRENGTH = 0.0

    def __init__(
        self,
        *,
        unet: NormalCrafterUNet,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        scheduler: EulerDiscreteScheduler,
        dtype: torch.dtype,
        attention_mode: str,
    ) -> None:
        # ``eval`` disables training behavior such as dropout and documents that
        # these modules are inference-only in this implementation.
        self.unet = unet.eval()
        self.vae = vae.eval()
        self.image_encoder = image_encoder.eval()

        # The feature extractor only provides normalization metadata here; image
        # resizing is done directly in Torch to avoid PIL/NumPy round-trips.
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.dtype = dtype
        self.attention_mode = attention_mode

        # Device moves are not thread-safe when two graph executions share one
        # model instance. The lock also protects scheduler mutation.
        self._lock = threading.RLock()

        self._configure_attention(attention_mode)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_repo: str = "Yanrui95/NormalCrafter",
        base_repo: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        dtype: torch.dtype = torch.float16,
        attention_mode: str = "auto",
        local_files_only: bool = False,
    ) -> "NormalCrafterModel":
        """Load the released weights and assemble one clean model owner.

        NormalCrafter supplies its fine-tuned UNet and VAE. The CLIP vision
        encoder, preprocessing statistics, and Euler scheduler come from the SVD
        base repository used during training.
        """

        common = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }

        logger.info("Loading NormalCrafter UNet from %s", model_repo)
        unet = NormalCrafterUNet.from_pretrained(model_repo, subfolder="unet", **common)

        logger.info("Loading NormalCrafter VAE from %s", model_repo)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_repo,
            subfolder="vae",
            **common,
        )

        # Some Hugging Face repos expose an explicit fp16 variant. Try it first
        # for half-precision execution, then fall back to the default files.
        encoder_kwargs = dict(common)
        if dtype in (torch.float16, torch.bfloat16):
            encoder_kwargs["variant"] = "fp16"

        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                base_repo,
                subfolder="image_encoder",
                **encoder_kwargs,
            )
        except (OSError, ValueError, TypeError):
            encoder_kwargs.pop("variant", None)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                base_repo,
                subfolder="image_encoder",
                **encoder_kwargs,
            )

        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_repo,
            subfolder="feature_extractor",
            local_files_only=local_files_only,
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            base_repo,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        model = cls(
            unet=unet,
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            scheduler=scheduler,
            dtype=dtype,
            attention_mode=attention_mode,
        )

        # Loading libraries may choose a device based on environment/config. The
        # public lifecycle contract is that a newly loaded model starts on CPU.
        model.offload_to_cpu(empty_cache=False)
        return model

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        """Return the heavy components participating in device management."""

        return {
            "image_encoder": self.image_encoder,
            "vae": self.vae,
            "unet": self.unet,
        }

    def _configure_attention(self, mode: str) -> None:
        """Enable xFormers when available or explicitly requested."""

        if mode not in {"auto", "default", "xformers"}:
            raise ValueError(f"unsupported attention mode: {mode}")

        if mode == "default":
            return

        try:
            self.unet.enable_xformers_memory_efficient_attention()
            logger.info("NormalCrafter xFormers attention enabled")
        except Exception as exc:
            # Explicit requests should fail loudly. ``auto`` is allowed to fall
            # back because current PyTorch/Diffusers attention may be sufficient.
            if mode == "xformers":
                raise RuntimeError("xFormers was requested but could not be enabled") from exc
            logger.info("xFormers unavailable; using Diffusers default attention: %s", exc)

    @staticmethod
    def _device_of(module: torch.nn.Module) -> torch.device:
        """Read a module's current device from its first parameter."""

        try:
            return next(module.parameters()).device
        except StopIteration:
            # Defensive fallback for parameterless modules.
            return torch.device("cpu")

    def _activate(self, name: str, device: torch.device, offload_mode: OffloadMode) -> None:
        """Place model components according to the selected residency policy.

        ``resident``:
            Move all heavy modules to the inference device and leave them there.

        ``staged``:
            Move only the named component to the inference device and force the
            others to CPU. After those moves, release now-unused CUDA cache blocks.
        """

        if offload_mode == "resident":
            for module in self.modules.values():
                if self._device_of(module) != device:
                    module.to(device)
            return

        for module_name, module in self.modules.items():
            target = device if module_name == name else torch.device("cpu")
            if self._device_of(module) != target:
                module.to(target)

        self._release_cuda_cache(device)

    @staticmethod
    def _release_cuda_cache(device: torch.device) -> None:
        """Release unreachable Python objects and unused CUDA allocator blocks.

        ``empty_cache`` does not free live tensors. Its usefulness depends on the
        preceding ownership discipline: temporary tensors must first go out of
        scope or be deleted, and inactive modules must already have moved to CPU.
        """

        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_to_cpu(self, *, empty_cache: bool = True) -> None:
        """Move all owned model weights to CPU and optionally empty CUDA cache."""

        for module in self.modules.values():
            module.to("cpu")

        if empty_cache:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def infer(
        self,
        images_bhwc: torch.Tensor,
        *,
        device: torch.device,
        config: InferenceConfig,
        progress: ProgressCallback | None = None,
    ) -> torch.Tensor:
        """Run complete inference and return CPU normal maps in ComfyUI format.

        Input shape:
            ``[frames, height, width, channels]``, values expected in ``[0, 1]``.

        Output shape:
            ``[real_frames, output_height, output_width, 3]`` on CPU, values in
            ``[0, 1]`` where encoded RGB corresponds to XYZ normal direction.
        """

        config.validate()

        # FrameSource owns the canonical CPU input and lazily prepares only the
        # chunk needed by the current stage.
        source = FrameSource(images_bhwc, max_resolution=config.max_resolution)

        # The temporal UNet requires at least one full window. Short clips repeat
        # their final frame internally; decoding is still limited to real frames.
        effective_frames = source.effective_frame_count(config.window_size)
        completed = False

        # ``torch.inference_mode`` removes autograd bookkeeping more aggressively
        # than ``no_grad`` and prevents accidental graph retention.
        with self._lock, torch.inference_mode():
            try:
                # Stage 1: semantic embedding per frame, stored on CPU.
                clip_embeddings = self._encode_clip(
                    source,
                    effective_frames,
                    device,
                    config,
                    progress,
                )

                # Stage 2: spatial RGB latent per frame, stored on CPU.
                rgb_latents = self._encode_vae(
                    source,
                    effective_frames,
                    device,
                    config,
                    progress,
                )

                # Stage 3: sliding-window temporal prediction, stored on CPU.
                normal_latents = self._predict_latents(
                    rgb_latents,
                    clip_embeddings,
                    device,
                    config,
                    progress,
                )

                # These full-video intermediates are no longer needed before the
                # decoder stage. Delete references early to reduce host memory.
                del rgb_latents, clip_embeddings

                # Stage 4: decode only real input frames into final normals.
                output = self._decode_normals(
                    normal_latents,
                    source,
                    device,
                    config,
                    progress,
                )

                completed = True
                return output
            finally:
                # On failure we always offload. On success this follows the user's
                # setting, allowing ``resident`` mode to remain fast across runs.
                if config.offload_after or not completed:
                    self.offload_to_cpu(empty_cache=True)

    def _encode_clip(
        self,
        source: FrameSource,
        frame_count: int,
        device: torch.device,
        config: InferenceConfig,
        progress: ProgressCallback | None,
    ) -> torch.Tensor:
        """Encode every frame into one CLIP vision embedding on CPU.

        Shape flow, typically:

        ``[chunk, 3, H, W]`` -> ``[chunk, 3, 224, 224]`` ->
        ``[chunk, 1, embedding_dim]``.
        """

        self._activate("image_encoder", device, config.offload_mode)

        # Reproduce the CLIP normalization configured by the SVD repository.
        # Mean/std remain on CPU because subtraction occurs before device transfer.
        mean = torch.tensor(
            self.feature_extractor.image_mean,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.feature_extractor.image_std,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        output: torch.Tensor | None = None
        total = (frame_count + config.clip_chunk_size - 1) // config.clip_chunk_size

        for chunk_index, start in enumerate(
            range(0, frame_count, config.clip_chunk_size),
            start=1,
        ):
            end = min(start + config.clip_chunk_size, frame_count)

            # FrameSource resizes/pads lazily and repeats the final real frame for
            # synthetic short-clip indices.
            frames = source.get_bchw(start, end)

            # CLIP vision expects its own fixed spatial resolution.
            frames = F.interpolate(
                frames,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)

            # Only the current normalized chunk crosses to the inference device.
            pixels = ((frames - mean) / std).to(device=device, dtype=self.dtype)
            embeddings = self.image_encoder(pixel_values=pixels).image_embeds

            # Diffusers cross-attention expects a sequence dimension. Depending on
            # Transformers version, ``image_embeds`` may arrive as rank two.
            if embeddings.ndim == 2:
                embeddings = embeddings.unsqueeze(1)

            # Copy the result back immediately so total video length does not grow
            # CUDA residency. Half precision is retained to limit host memory.
            embeddings_cpu = embeddings.detach().to(device="cpu", dtype=self.dtype)

            # Allocate the final CPU tensor once the runtime embedding shape is
            # known, then fill slices instead of accumulating and concatenating.
            if output is None:
                output = torch.empty(
                    (frame_count, *embeddings_cpu.shape[1:]),
                    dtype=self.dtype,
                    device="cpu",
                )
            output[start:end].copy_(embeddings_cpu)

            # Drop all references to current chunk/device tensors before moving on.
            del frames, pixels, embeddings, embeddings_cpu

            if progress:
                progress("CLIP", chunk_index, total)

        assert output is not None
        return output

    def _encode_vae(
        self,
        source: FrameSource,
        frame_count: int,
        device: torch.device,
        config: InferenceConfig,
        progress: ProgressCallback | None,
    ) -> torch.Tensor:
        """Encode RGB frames into VAE latents and store them on CPU.

        Unlike generative sampling, this path uses the latent distribution mode,
        making encoding deterministic and avoiding random sampling noise.
        """

        self._activate("vae", device, config.offload_mode)

        original_dtype = next(self.vae.parameters()).dtype

        # Some Diffusers VAEs declare ``force_upcast`` because selected operations
        # are unstable in FP16. The dtype is restored in ``finally`` even on error.
        needs_upcast = (
            original_dtype == torch.float16
            and bool(getattr(self.vae.config, "force_upcast", False))
        )
        if needs_upcast:
            self.vae.to(dtype=torch.float32)
        compute_dtype = next(self.vae.parameters()).dtype

        output: torch.Tensor | None = None
        total = (
            frame_count + config.vae_encode_chunk_size - 1
        ) // config.vae_encode_chunk_size

        try:
            for chunk_index, start in enumerate(
                range(0, frame_count, config.vae_encode_chunk_size),
                start=1,
            ):
                end = min(start + config.vae_encode_chunk_size, frame_count)

                frames = source.get_bchw(start, end)

                # VAE input convention is [-1, 1], whereas ComfyUI uses [0, 1].
                frames = (frames * 2.0 - 1.0).to(
                    device=device,
                    dtype=compute_dtype,
                )

                latents = self.vae.encode(frames).latent_dist.mode()
                latents_cpu = latents.detach().to(device="cpu", dtype=self.dtype)

                # Runtime allocation avoids assumptions about latent channel count
                # or spatial scale while still eliminating list + torch.cat churn.
                if output is None:
                    output = torch.empty(
                        (frame_count, *latents_cpu.shape[1:]),
                        dtype=self.dtype,
                        device="cpu",
                    )
                output[start:end].copy_(latents_cpu)

                del frames, latents, latents_cpu

                if progress:
                    progress("VAE encode", chunk_index, total)
        finally:
            if needs_upcast:
                self.vae.to(dtype=original_dtype)

        assert output is not None
        return output

    def _added_time_ids(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Construct the three Stable Video Diffusion conditioning values."""

        values = [
            self.FPS_CONDITIONING,
            self.MOTION_BUCKET_ID,
            self.NOISE_AUG_STRENGTH,
        ]

        # Fail explicitly if a different UNet config expects a different added
        # conditioning width. Silent shape coercion here would produce bad output.
        expected = self.unet.add_embedding.linear_1.in_features
        actual = int(self.unet.config.addition_time_embed_dim) * len(values)
        if expected != actual:
            raise ValueError(
                f"UNet added-time embedding mismatch: expected {expected}, constructed {actual}"
            )

        return torch.tensor([values], dtype=dtype, device=device)

    def _predict_latents(
        self,
        rgb_latents_cpu: torch.Tensor,
        clip_embeddings_cpu: torch.Tensor,
        device: torch.device,
        config: InferenceConfig,
        progress: ProgressCallback | None,
    ) -> torch.Tensor:
        """Predict normal latents over overlapping temporal windows.

        Full-video input/output tensors remain on CPU. Each iteration moves only
        one window of RGB latents and CLIP embeddings to the inference device.

        The released inference path is a deterministic single scheduler step:
        normal latents start at zero, RGB latents are concatenated as conditioning,
        and the UNet predicts the one-step update.
        """

        self._activate("unet", device, config.offload_mode)

        frame_count = int(rgb_latents_cpu.shape[0])
        windows = build_windows(frame_count, config.window_size, config.step_size)

        # The output shape matches the VAE latent shape. It is allocated once and
        # overwritten by each anchored window instead of repeatedly concatenated.
        output_cpu = torch.empty_like(rgb_latents_cpu, device="cpu")

        previous_window: Window | None = None
        previous_latents: torch.Tensor | None = None

        for index, window in enumerate(windows, start=1):
            # RGB latent window: [frames, channels, h, w] ->
            # [batch=1, frames, channels, h, w].
            image_latents = rgb_latents_cpu[window.start : window.end].to(
                device=device,
                dtype=self.dtype,
            ).unsqueeze(0)

            # One CLIP embedding is supplied for every video frame.
            embeddings = clip_embeddings_cpu[window.start : window.end].to(
                device=device,
                dtype=self.dtype,
            )

            # NormalCrafter's prediction state starts from zero rather than random
            # diffusion noise. This is why the node intentionally has no seed.
            latents = torch.zeros(
                (
                    1,
                    window.length,
                    int(self.unet.config.out_channels),
                    *image_latents.shape[-2:],
                ),
                device=device,
                dtype=self.dtype,
            )

            overlap = overlap_length(previous_window, window)
            old_overlap: torch.Tensor | None = None

            if overlap and previous_window is not None and previous_latents is not None:
                # Determine where the current window begins inside the previous
                # window, then seed the overlapping prefix with prior predictions.
                previous_offset = window.start - previous_window.start
                old_overlap = previous_latents[
                    :,
                    previous_offset : previous_offset + overlap,
                ]
                latents[:, :overlap].copy_(old_overlap)

            # NormalCrafter uses exactly one Euler scheduler timestep.
            self.scheduler.set_timesteps(1, device=device)
            timestep = self.scheduler.timesteps[0]

            # Scheduler scaling is kept even for one step because it is part of
            # the trained SVD inference convention.
            model_input = self.scheduler.scale_model_input(latents, timestep)

            # Channel concatenation: four predicted-normal channels plus four RGB
            # latent conditioning channels -> eight UNet input channels.
            model_input = torch.cat((model_input, image_latents), dim=2)
            added_time_ids = self._added_time_ids(dtype=self.dtype, device=device)

            noise_prediction = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            predicted = self.scheduler.step(
                noise_prediction,
                timestep,
                latents,
            ).prev_sample

            if old_overlap is not None:
                # Crossfade old -> new prediction over the overlap. Endpoint values
                # are excluded so neither side receives a fully duplicated weight.
                weights = torch.linspace(
                    1.0,
                    0.0,
                    overlap + 2,
                    device=device,
                    dtype=self.dtype,
                )[1:-1].view(1, overlap, 1, 1, 1)

                predicted[:, :overlap] = (
                    old_overlap * weights
                    + predicted[:, :overlap] * (1.0 - weights)
                )

            # Later windows intentionally overwrite their entire covered region.
            # The overlap has already been blended, and the final anchored window
            # guarantees coverage through the last frame.
            output_cpu[window.start : window.end].copy_(
                predicted[0].detach().to("cpu")
            )

            # Retain only the immediately previous GPU window for overlap seeding.
            # This bounds temporal state by ``window_size``, not total clip length.
            previous_window = window
            previous_latents = predicted.detach()

            del image_latents, embeddings, latents, model_input, added_time_ids, noise_prediction

            if progress:
                progress("UNet", index, len(windows))

        return output_cpu

    def _decode_normals(
        self,
        normal_latents_cpu: torch.Tensor,
        source: FrameSource,
        device: torch.device,
        config: InferenceConfig,
        progress: ProgressCallback | None,
    ) -> torch.Tensor:
        """Decode normal latents into final CPU ComfyUI IMAGE tensors."""

        self._activate("vae", device, config.offload_mode)

        original_dtype = next(self.vae.parameters()).dtype
        needs_upcast = (
            original_dtype == torch.float16
            and bool(getattr(self.vae.config, "force_upcast", False))
        )
        if needs_upcast:
            self.vae.to(dtype=torch.float32)
        compute_dtype = next(self.vae.parameters()).dtype

        if config.output_size == "original":
            out_height, out_width = source.original_height, source.original_width
        else:
            out_height, out_width = source.resized_height, source.resized_width

        # Allocate only the real frame count. Any synthetic padding frames used to
        # satisfy a short temporal window are never decoded or returned.
        output = torch.empty(
            (source.frame_count, out_height, out_width, 3),
            dtype=torch.float32,
            device="cpu",
        )

        scaling_factor = float(self.vae.config.scaling_factor)

        # Diffusers versions differ in whether temporal decoders accept an
        # explicit ``num_frames`` argument. Inspect once and adapt safely.
        forward_vae = (
            self.vae._orig_mod.forward
            if hasattr(self.vae, "_orig_mod")
            else self.vae.forward
        )
        accepts_num_frames = "num_frames" in inspect.signature(forward_vae).parameters

        total = (
            source.frame_count + config.vae_decode_chunk_size - 1
        ) // config.vae_decode_chunk_size

        try:
            for chunk_index, start in enumerate(
                range(0, source.frame_count, config.vae_decode_chunk_size),
                start=1,
            ):
                end = min(start + config.vae_decode_chunk_size, source.frame_count)

                latents = normal_latents_cpu[start:end].to(
                    device=device,
                    dtype=compute_dtype,
                )

                # Undo the latent scaling expected by the VAE decoder.
                latents = latents / scaling_factor

                decode_kwargs = (
                    {"num_frames": end - start}
                    if accepts_num_frames
                    else {}
                )
                # Decode, promote to FP32, and transfer to CPU in one expression.
                # The chained calls are kept identical to the validated version.
                normals = self.vae.decode(latents, **decode_kwargs).sample.float().to("cpu")

                # Remove the white spatial padding introduced before encoding.
                normals = source.crop_padding(normals)

                if (out_height, out_width) != (
                    source.resized_height,
                    source.resized_width,
                ):
                    normals = F.interpolate(
                        normals,
                        size=(out_height, out_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Decoder output represents XYZ components in [-1, 1].
                normals = normals.clamp(-1.0, 1.0)

                # Bilinear resizing blends vector components and can reduce vector
                # magnitude. Renormalization restores unit-length surface normals.
                if config.renormalize_normals:
                    normals = F.normalize(normals, p=2.0, dim=1, eps=1e-6)

                # Encode signed XYZ into display/image range [0, 1].
                normals = (normals * 0.5 + 0.5).clamp_(0.0, 1.0)

                # Convert BCHW -> BHWC, the layout required by ComfyUI IMAGE.
                output[start:end].copy_(
                    normals.permute(0, 2, 3, 1).contiguous()
                )

                del latents, normals

                if progress:
                    progress("VAE decode", chunk_index, total)
        finally:
            if needs_upcast:
                self.vae.to(dtype=original_dtype)

        return output
