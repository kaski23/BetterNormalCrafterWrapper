from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _round_size(value: float) -> int:
    return max(1, int(round(value)))


def _ceil_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


@dataclass
class FrameSource:
    """Lazy CPU-side view of a ComfyUI IMAGE batch.

    Frames are resized and padded only for the chunk currently needed. This avoids
    materialising a second full-resolution copy of the whole video.
    """

    frames_bhwc: torch.Tensor
    max_resolution: int
    pad_multiple: int = 64

    def __post_init__(self) -> None:
        if self.frames_bhwc.ndim != 4 or self.frames_bhwc.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, channels>=3]")
        if self.frames_bhwc.shape[0] == 0:
            raise ValueError("images contains no frames")
        if self.max_resolution < 64:
            raise ValueError("max_resolution must be at least 64")

        self.frames_bhwc = (
            self.frames_bhwc[..., :3]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clamp_(0.0, 1.0)
        )

        _, self.original_height, self.original_width, _ = self.frames_bhwc.shape
        scale = min(1.0, self.max_resolution / max(self.original_height, self.original_width))
        self.resized_height = _round_size(self.original_height * scale)
        self.resized_width = _round_size(self.original_width * scale)

        self.padded_height = _ceil_multiple(self.resized_height, self.pad_multiple)
        self.padded_width = _ceil_multiple(self.resized_width, self.pad_multiple)

        total_pad_h = self.padded_height - self.resized_height
        total_pad_w = self.padded_width - self.resized_width
        self.pad_top = total_pad_h // 2
        self.pad_bottom = total_pad_h - self.pad_top
        self.pad_left = total_pad_w // 2
        self.pad_right = total_pad_w - self.pad_left

    @property
    def frame_count(self) -> int:
        return int(self.frames_bhwc.shape[0])

    def effective_frame_count(self, window_size: int) -> int:
        return max(self.frame_count, window_size)

    def get_bchw(self, start: int, end: int) -> torch.Tensor:
        """Return a resized, symmetrically white-padded CPU chunk in [0, 1].

        Indices beyond the real clip repeat its final frame. This is only used to
        satisfy the temporal window for short clips.
        """
        if start < 0 or end <= start:
            raise ValueError(f"invalid frame range [{start}, {end})")

        indices = torch.arange(start, end, dtype=torch.long).clamp_max(self.frame_count - 1)
        chunk = self.frames_bhwc.index_select(0, indices).permute(0, 3, 1, 2).contiguous()

        if (self.resized_height, self.resized_width) != (self.original_height, self.original_width):
            chunk = F.interpolate(
                chunk,
                size=(self.resized_height, self.resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)

        if self.pad_top or self.pad_bottom or self.pad_left or self.pad_right:
            chunk = F.pad(
                chunk,
                (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                mode="constant",
                value=1.0,
            )
        return chunk

    def crop_padding(self, tensor_bchw: torch.Tensor) -> torch.Tensor:
        return tensor_bchw[
            :,
            :,
            self.pad_top : self.pad_top + self.resized_height,
            self.pad_left : self.pad_left + self.resized_width,
        ]
