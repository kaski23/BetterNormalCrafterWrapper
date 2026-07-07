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
ProgressCallback = Callable[[str, int, int], None]
OffloadMode = Literal["staged", "resident"]
OutputSize = Literal["original", "processed"]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
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
        if self.max_resolution < 64:
            raise ValueError("max_resolution must be at least 64")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
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
    """Single-owner NormalCrafter inference model.

    No module globals, duplicate pipeline references, or hidden CUDA output tensors.
    All long-lived intermediate video data is stored on CPU. GPU residency is
    explicitly controlled per inference stage.
    """

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
        self.unet = unet.eval()
        self.vae = vae.eval()
        self.image_encoder = image_encoder.eval()
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.dtype = dtype
        self.attention_mode = attention_mode
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
        model.offload_to_cpu(empty_cache=False)
        return model

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return {
            "image_encoder": self.image_encoder,
            "vae": self.vae,
            "unet": self.unet,
        }

    def _configure_attention(self, mode: str) -> None:
        if mode not in {"auto", "default", "xformers"}:
            raise ValueError(f"unsupported attention mode: {mode}")
        if mode == "default":
            return
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            logger.info("NormalCrafter xFormers attention enabled")
        except Exception as exc:
            if mode == "xformers":
                raise RuntimeError("xFormers was requested but could not be enabled") from exc
            logger.info("xFormers unavailable; using Diffusers default attention: %s", exc)

    @staticmethod
    def _device_of(module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _activate(self, name: str, device: torch.device, offload_mode: OffloadMode) -> None:
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
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_to_cpu(self, *, empty_cache: bool = True) -> None:
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
        config.validate()
        source = FrameSource(images_bhwc, max_resolution=config.max_resolution)
        effective_frames = source.effective_frame_count(config.window_size)
        completed = False

        with self._lock, torch.inference_mode():
            try:
                clip_embeddings = self._encode_clip(
                    source,
                    effective_frames,
                    device,
                    config,
                    progress,
                )
                rgb_latents = self._encode_vae(
                    source,
                    effective_frames,
                    device,
                    config,
                    progress,
                )
                normal_latents = self._predict_latents(
                    rgb_latents,
                    clip_embeddings,
                    device,
                    config,
                    progress,
                )
                del rgb_latents, clip_embeddings

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
        self._activate("image_encoder", device, config.offload_mode)
        mean = torch.tensor(self.feature_extractor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.feature_extractor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        output: torch.Tensor | None = None
        total = (frame_count + config.clip_chunk_size - 1) // config.clip_chunk_size

        for chunk_index, start in enumerate(range(0, frame_count, config.clip_chunk_size), start=1):
            end = min(start + config.clip_chunk_size, frame_count)
            frames = source.get_bchw(start, end)
            frames = F.interpolate(
                frames,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)
            pixels = ((frames - mean) / std).to(device=device, dtype=self.dtype)
            embeddings = self.image_encoder(pixel_values=pixels).image_embeds
            if embeddings.ndim == 2:
                embeddings = embeddings.unsqueeze(1)
            embeddings_cpu = embeddings.detach().to(device="cpu", dtype=self.dtype)

            if output is None:
                output = torch.empty(
                    (frame_count, *embeddings_cpu.shape[1:]),
                    dtype=self.dtype,
                    device="cpu",
                )
            output[start:end].copy_(embeddings_cpu)
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
        self._activate("vae", device, config.offload_mode)
        original_dtype = next(self.vae.parameters()).dtype
        needs_upcast = original_dtype == torch.float16 and bool(getattr(self.vae.config, "force_upcast", False))
        if needs_upcast:
            self.vae.to(dtype=torch.float32)
        compute_dtype = next(self.vae.parameters()).dtype

        output: torch.Tensor | None = None
        total = (frame_count + config.vae_encode_chunk_size - 1) // config.vae_encode_chunk_size
        try:
            for chunk_index, start in enumerate(
                range(0, frame_count, config.vae_encode_chunk_size),
                start=1,
            ):
                end = min(start + config.vae_encode_chunk_size, frame_count)
                frames = source.get_bchw(start, end)
                frames = (frames * 2.0 - 1.0).to(device=device, dtype=compute_dtype)
                latents = self.vae.encode(frames).latent_dist.mode()
                latents_cpu = latents.detach().to(device="cpu", dtype=self.dtype)

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
        values = [self.FPS_CONDITIONING, self.MOTION_BUCKET_ID, self.NOISE_AUG_STRENGTH]
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
        self._activate("unet", device, config.offload_mode)
        frame_count = int(rgb_latents_cpu.shape[0])
        windows = build_windows(frame_count, config.window_size, config.step_size)
        output_cpu = torch.empty_like(rgb_latents_cpu, device="cpu")

        previous_window: Window | None = None
        previous_latents: torch.Tensor | None = None

        for index, window in enumerate(windows, start=1):
            image_latents = rgb_latents_cpu[window.start : window.end].to(
                device=device,
                dtype=self.dtype,
            ).unsqueeze(0)
            embeddings = clip_embeddings_cpu[window.start : window.end].to(
                device=device,
                dtype=self.dtype,
            )

            latents = torch.zeros(
                (1, window.length, int(self.unet.config.out_channels), *image_latents.shape[-2:]),
                device=device,
                dtype=self.dtype,
            )

            overlap = overlap_length(previous_window, window)
            old_overlap: torch.Tensor | None = None
            if overlap and previous_window is not None and previous_latents is not None:
                previous_offset = window.start - previous_window.start
                old_overlap = previous_latents[:, previous_offset : previous_offset + overlap]
                latents[:, :overlap].copy_(old_overlap)

            self.scheduler.set_timesteps(1, device=device)
            timestep = self.scheduler.timesteps[0]
            model_input = self.scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat((model_input, image_latents), dim=2)
            added_time_ids = self._added_time_ids(dtype=self.dtype, device=device)

            noise_prediction = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]
            predicted = self.scheduler.step(noise_prediction, timestep, latents).prev_sample

            if old_overlap is not None:
                weights = torch.linspace(
                    1.0,
                    0.0,
                    overlap + 2,
                    device=device,
                    dtype=self.dtype,
                )[1:-1].view(1, overlap, 1, 1, 1)
                predicted[:, :overlap] = old_overlap * weights + predicted[:, :overlap] * (1.0 - weights)

            output_cpu[window.start : window.end].copy_(predicted[0].detach().to("cpu"))
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
        self._activate("vae", device, config.offload_mode)
        original_dtype = next(self.vae.parameters()).dtype
        needs_upcast = original_dtype == torch.float16 and bool(getattr(self.vae.config, "force_upcast", False))
        if needs_upcast:
            self.vae.to(dtype=torch.float32)
        compute_dtype = next(self.vae.parameters()).dtype

        if config.output_size == "original":
            out_height, out_width = source.original_height, source.original_width
        else:
            out_height, out_width = source.resized_height, source.resized_width

        output = torch.empty(
            (source.frame_count, out_height, out_width, 3),
            dtype=torch.float32,
            device="cpu",
        )
        scaling_factor = float(self.vae.config.scaling_factor)
        forward_vae = self.vae._orig_mod.forward if hasattr(self.vae, "_orig_mod") else self.vae.forward
        accepts_num_frames = "num_frames" in inspect.signature(forward_vae).parameters
        total = (source.frame_count + config.vae_decode_chunk_size - 1) // config.vae_decode_chunk_size

        try:
            for chunk_index, start in enumerate(
                range(0, source.frame_count, config.vae_decode_chunk_size),
                start=1,
            ):
                end = min(start + config.vae_decode_chunk_size, source.frame_count)
                latents = normal_latents_cpu[start:end].to(device=device, dtype=compute_dtype)
                latents = latents / scaling_factor
                decode_kwargs = {"num_frames": end - start} if accepts_num_frames else {}
                normals = self.vae.decode(latents, **decode_kwargs).sample.float().to("cpu")
                normals = source.crop_padding(normals)

                if (out_height, out_width) != (source.resized_height, source.resized_width):
                    normals = F.interpolate(
                        normals,
                        size=(out_height, out_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                normals = normals.clamp(-1.0, 1.0)
                if config.renormalize_normals:
                    normals = F.normalize(normals, p=2.0, dim=1, eps=1e-6)
                normals = (normals * 0.5 + 0.5).clamp_(0.0, 1.0)
                output[start:end].copy_(normals.permute(0, 2, 3, 1).contiguous())

                del latents, normals
                if progress:
                    progress("VAE decode", chunk_index, total)
        finally:
            if needs_upcast:
                self.vae.to(dtype=original_dtype)

        return output
