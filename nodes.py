from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.utils

from .normalcrafter_clean.engine import InferenceConfig, NormalCrafterModel
from .normalcrafter_clean.windows import build_windows

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
                "attention": (["auto", "default", "xformers"], {"default": "auto"}),
                "local_files_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NORMALCRAFTER_CLEAN_MODEL",),
                "images": ("IMAGE",),
                "max_resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "window_size": ("INT", {"default": 14, "min": 2, "max": 32}),
                "step_size": ("INT", {"default": 10, "min": 1, "max": 32}),
                "clip_chunk_size": ("INT", {"default": 16, "min": 1, "max": 128}),
                "vae_encode_chunk_size": ("INT", {"default": 7, "min": 1, "max": 32}),
                "vae_decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "offload_mode": (["staged", "resident"], {"default": "staged"}),
                "offload_after": ("BOOLEAN", {"default": True}),
                "output_size": (["original", "processed"], {"default": "original"}),
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

        frame_count = int(images.shape[0])
        effective_count = max(frame_count, window_size)
        total_steps = (
            (effective_count + clip_chunk_size - 1) // clip_chunk_size
            + (effective_count + vae_encode_chunk_size - 1) // vae_encode_chunk_size
            + len(build_windows(effective_count, window_size, step_size))
            + (frame_count + vae_decode_chunk_size - 1) // vae_decode_chunk_size
        )
        progress_bar = comfy.utils.ProgressBar(max(total_steps, 1))

        def progress(_stage: str, _current: int, _total: int) -> None:
            progress_bar.update(1)

        device = comfy.model_management.get_torch_device()
        output = model.infer(images, device=device, config=config, progress=progress)
        if offload_after:
            comfy.model_management.soft_empty_cache()
        return (output,)


class NormalCrafterCleanOffload:
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
