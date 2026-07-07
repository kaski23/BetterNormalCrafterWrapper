"""ComfyUI-facing nodes for the clean NormalCrafter implementation.

This module deliberately contains only the thin UI/integration layer. The actual
model lifecycle, inference stages, windowing, preprocessing, and tensor ownership
live in ``normalcrafter_clean``.

That separation is important: ComfyUI may cache node outputs and keep node
instances alive longer than one execution. Keeping model logic out of this file
reduces the risk of accidental global state, duplicate references, or live CUDA
tensors being retained by the graph cache.
"""

from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.utils

from .normalcrafter_clean.engine import InferenceConfig, NormalCrafterModel
from .normalcrafter_clean.windows import build_windows

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Translate the UI dtype choice into a concrete PyTorch dtype.

    ``auto`` uses FP16 on CUDA because that is the practical default for the
    released model weights. CPU inference is forced to FP32: many PyTorch CPU
    kernels either do not implement FP16 efficiently or are numerically fragile.
    """

    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping[name]

    if device.type == "cpu" and dtype != torch.float32:
        logger.warning("CPU inference requested; forcing float32 instead of %s", name)
        return torch.float32

    return dtype


class NormalCrafterCleanLoader:
    """Load all NormalCrafter components into one explicitly-owned model object.

    The loader does not move the model to CUDA. ``NormalCrafterModel`` is created
    on CPU and the generate node later decides whether to stage components one by
    one or keep all of them resident on the selected inference device.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # ComfyUI reads this declarative structure to construct the node UI.
        return {
            "required": {
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
                "attention": (["auto", "default", "xformers"], {"default": "auto"}),
                "local_files_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # These fields are configurable for mirrors, forks, or local repos,
                # while preserving the official public defaults.
                "model_repo": ("STRING", {"default": "Yanrui95/NormalCrafter"}),
                "base_repo": (
                    "STRING",
                    {"default": "stabilityai/stable-video-diffusion-img2vid-xt"},
                ),
            },
        }

    RETURN_TYPES = ("NORMALCRAFTER_CLEAN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "NormalCrafter/Clean"

    def load(
        self,
        dtype: str,
        attention: str,
        local_files_only: bool,
        model_repo: str = "Yanrui95/NormalCrafter",
        base_repo: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    ):
        """Instantiate the model and return it as a ComfyUI custom object type."""

        # Respect ComfyUI's own device selection rather than hard-coding ``cuda``.
        device = comfy.model_management.get_torch_device()
        resolved_dtype = _resolve_dtype(dtype, device)

        model = NormalCrafterModel.from_pretrained(
            model_repo=model_repo.strip(),
            base_repo=base_repo.strip(),
            dtype=resolved_dtype,
            attention_mode=attention,
            local_files_only=local_files_only,
        )
        return (model,)


class NormalCrafterCleanGenerate:
    """Run the four-stage NormalCrafter inference pipeline.

    ComfyUI IMAGE tensors use ``[frames, height, width, channels]`` and values in
    ``[0, 1]``. The engine accepts that exact format and also returns the same
    format on CPU, so the graph never caches a live CUDA output tensor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NORMALCRAFTER_CLEAN_MODEL",),
                "images": ("IMAGE",),
                # The input aspect ratio is preserved. This setting only limits
                # the longest side before symmetric padding to a VAE-safe size.
                "max_resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                # NormalCrafter was trained around 14-frame temporal windows.
                "window_size": ("INT", {"default": 14, "min": 2, "max": 32}),
                # 14 / 10 produces the original regular overlap of four frames.
                "step_size": ("INT", {"default": 10, "min": 1, "max": 32}),
                # Chunk sizes trade transfer overhead for peak VRAM usage.
                "clip_chunk_size": ("INT", {"default": 16, "min": 1, "max": 128}),
                "vae_encode_chunk_size": ("INT", {"default": 7, "min": 1, "max": 32}),
                "vae_decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                # ``staged`` keeps only one heavy component on CUDA at a time.
                # ``resident`` is faster but requires enough VRAM for all models.
                "offload_mode": (["staged", "resident"], {"default": "staged"}),
                "offload_after": ("BOOLEAN", {"default": True}),
                # ``original`` resizes decoded normals back to input dimensions.
                "output_size": (["original", "processed"], {"default": "original"}),
                # Interpolation can shorten vectors; this restores unit length.
                "renormalize_normals": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_maps",)
    FUNCTION = "generate"
    CATEGORY = "NormalCrafter/Clean"

    def generate(
        self,
        model: NormalCrafterModel,
        images: torch.Tensor,
        max_resolution: int,
        window_size: int,
        step_size: int,
        clip_chunk_size: int,
        vae_encode_chunk_size: int,
        vae_decode_chunk_size: int,
        offload_mode: str,
        offload_after: bool,
        output_size: str,
        renormalize_normals: bool,
    ):
        """Validate user settings, build progress accounting, and invoke inference."""

        config = InferenceConfig(
            max_resolution=max_resolution,
            window_size=window_size,
            step_size=step_size,
            clip_chunk_size=clip_chunk_size,
            vae_encode_chunk_size=vae_encode_chunk_size,
            vae_decode_chunk_size=vae_decode_chunk_size,
            offload_mode=offload_mode,
            offload_after=offload_after,
            output_size=output_size,
            renormalize_normals=renormalize_normals,
        )
        config.validate()

        # Short clips are temporally extended by repeating the final frame until
        # one full window is available. The engine removes those synthetic frames
        # again before returning the result.
        frame_count = int(images.shape[0])
        effective_count = max(frame_count, window_size)

        # Progress is counted in actual chunks/windows, not arbitrary percentages.
        # This keeps the bar responsive for both short and long clips.
        total_steps = (
            (effective_count + clip_chunk_size - 1) // clip_chunk_size
            + (effective_count + vae_encode_chunk_size - 1) // vae_encode_chunk_size
            + len(build_windows(effective_count, window_size, step_size))
            + (frame_count + vae_decode_chunk_size - 1) // vae_decode_chunk_size
        )
        progress_bar = comfy.utils.ProgressBar(max(total_steps, 1))

        def progress(_stage: str, _current: int, _total: int) -> None:
            # The callback keeps stage names available for future logging/UI use,
            # while ComfyUI's built-in bar currently only needs one increment.
            progress_bar.update(1)

        device = comfy.model_management.get_torch_device()
        output = model.infer(images, device=device, config=config, progress=progress)

        # The engine already offloads when requested. Asking ComfyUI to release its
        # allocator cache as well makes freed blocks available to other nodes.
        if offload_after:
            comfy.model_management.soft_empty_cache()

        return (output,)


class NormalCrafterCleanOffload:
    """Explicitly move all model components back to CPU.

    This is useful in long graphs where the user wants precise control over when
    NormalCrafter relinquishes VRAM, independent of the generate node setting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("NORMALCRAFTER_CLEAN_MODEL",)}}

    RETURN_TYPES = ("NORMALCRAFTER_CLEAN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "offload"
    CATEGORY = "NormalCrafter/Clean"

    def offload(self, model: NormalCrafterModel):
        model.offload_to_cpu(empty_cache=True)
        comfy.model_management.soft_empty_cache()
        return (model,)


# ComfyUI discovers nodes through these mappings. Internal class names remain
# stable identifiers for serialized workflows; display names may be more readable.
NODE_CLASS_MAPPINGS = {
    "NormalCrafterCleanLoader": NormalCrafterCleanLoader,
    "NormalCrafterCleanGenerate": NormalCrafterCleanGenerate,
    "NormalCrafterCleanOffload": NormalCrafterCleanOffload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalCrafterCleanLoader": "NormalCrafter Clean - Load",
    "NormalCrafterCleanGenerate": "NormalCrafter Clean - Generate Normals",
    "NormalCrafterCleanOffload": "NormalCrafter Clean - Offload",
}
