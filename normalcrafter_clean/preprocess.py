"""CPU-side frame preparation for NormalCrafter inference.

ComfyUI supplies video-like image batches as a single ``BHWC`` tensor:

    [frame, height, width, channel]

The model stack, however, expects ``BCHW`` chunks whose spatial dimensions are
compatible with the VAE. This module bridges those conventions without creating
a second processed copy of the complete clip.

The crucial design choice is laziness: ``FrameSource`` stores the original clip
once on CPU and performs resize, padding, layout conversion, and short-clip frame
repetition only for the range requested by the current inference stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _round_size(value: float) -> int:
    """Round a scaled spatial dimension while guaranteeing at least one pixel."""

    return max(1, int(round(value)))


def _ceil_multiple(value: int, multiple: int) -> int:
    """Return the smallest multiple of ``multiple`` not smaller than ``value``.

    NormalCrafter's VAE/UNet stack repeatedly downsamples spatial dimensions.
    Padding to 64 is conservative and prevents incompatible shapes deep inside
    the network while keeping the original aspect ratio intact.
    """

    return int(math.ceil(value / multiple) * multiple)


@dataclass
class FrameSource:
    """Lazy CPU-side view of a ComfyUI IMAGE batch.

    Parameters
    ----------
    frames_bhwc:
        ComfyUI ``IMAGE`` tensor in ``[frames, height, width, channels]`` layout.
        Values are expected to represent display-linear image data in ``[0, 1]``.
    max_resolution:
        Upper bound for the longest spatial side used by the model. Inputs are
        never enlarged; smaller clips retain their native dimensions.
    pad_multiple:
        Spatial alignment required by the latent model. Padding is symmetric,
        except for the unavoidable one-pixel asymmetry when the total is odd.

    Memory behavior
    ---------------
    Only the source clip is retained. A resized/padded tensor is created solely
    for the requested chunk and becomes collectible after that stage iteration.
    This is central to keeping peak RAM and VRAM independent of total clip length.
    """

    frames_bhwc: torch.Tensor
    max_resolution: int
    pad_multiple: int = 64

    def __post_init__(self) -> None:
        """Validate input and precompute immutable geometry metadata."""

        # ComfyUI IMAGE is BHWC. Alpha or auxiliary channels are allowed, but the
        # released model consumes exactly RGB, so at least three channels exist.
        if self.frames_bhwc.ndim != 4 or self.frames_bhwc.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, channels>=3]")
        if self.frames_bhwc.shape[0] == 0:
            raise ValueError("images contains no frames")
        if self.max_resolution < 64:
            raise ValueError("max_resolution must be at least 64")

        # Break any autograd relationship, discard non-RGB channels, and establish
        # one canonical long-lived representation: contiguous CPU FP32 in [0, 1].
        # Returning to CPU here also prevents a CUDA input tensor from becoming a
        # hidden full-video allocation retained throughout inference.
        self.frames_bhwc = (
            self.frames_bhwc[..., :3]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clamp_(0.0, 1.0)
        )

        # Compute a single aspect-preserving scale. ``min(1.0, ...)`` explicitly
        # forbids upscaling: max_resolution is a ceiling, not a target size.
        _, self.original_height, self.original_width, _ = self.frames_bhwc.shape
        scale = min(1.0, self.max_resolution / max(self.original_height, self.original_width))
        self.resized_height = _round_size(self.original_height * scale)
        self.resized_width = _round_size(self.original_width * scale)

        # The network receives padded dimensions, while output restoration later
        # removes precisely these margins before optional resize to source size.
        self.padded_height = _ceil_multiple(self.resized_height, self.pad_multiple)
        self.padded_width = _ceil_multiple(self.resized_width, self.pad_multiple)

        # Split padding around the image. When an odd number of pixels is needed,
        # bottom/right receive the extra pixel; crop_padding uses the same offsets.
        total_pad_h = self.padded_height - self.resized_height
        total_pad_w = self.padded_width - self.resized_width
        self.pad_top = total_pad_h // 2
        self.pad_bottom = total_pad_h - self.pad_top
        self.pad_left = total_pad_w // 2
        self.pad_right = total_pad_w - self.pad_left

    @property
    def frame_count(self) -> int:
        """Number of real input frames, excluding any temporal repetition."""

        return int(self.frames_bhwc.shape[0])

    def effective_frame_count(self, window_size: int) -> int:
        """Frame count seen by encoders after minimum-window temporal padding.

        NormalCrafter's temporal UNet requires at least one complete window. For a
        shorter clip, ``get_bchw`` repeats the final real frame up to this count.
        The decoder still emits only ``frame_count`` real output frames.
        """

        return max(self.frame_count, window_size)

    def get_bchw(self, start: int, end: int) -> torch.Tensor:
        """Materialize one processed CPU frame range in BCHW layout.

        The returned tensor is:

        * CPU resident;
        * FP32;
        * RGB only;
        * in ``[0, 1]``;
        * resized without changing aspect ratio;
        * symmetrically white-padded to ``pad_multiple``;
        * temporally extended by repeating the final frame when ``end`` exceeds
          the real clip length.

        White padding matches the neutral image-space convention used by the
        original pipeline more closely than zero/black padding.
        """

        if start < 0 or end <= start:
            raise ValueError(f"invalid frame range [{start}, {end})")

        # Clamp synthetic indices to the final real frame. ``index_select`` then
        # handles ordinary ranges and repeated-tail ranges through one code path.
        indices = torch.arange(start, end, dtype=torch.long).clamp_max(self.frame_count - 1)

        # Select BHWC frames, then convert once to the BCHW layout expected by
        # interpolation, CLIP preprocessing, and the VAE encoder.
        chunk = self.frames_bhwc.index_select(0, indices).permute(0, 3, 1, 2).contiguous()

        # Bicubic plus antialiasing is appropriate for RGB downscaling. Clamp after
        # interpolation because bicubic kernels may overshoot slightly outside
        # the nominal [0, 1] input range near high-contrast edges.
        if (self.resized_height, self.resized_width) != (self.original_height, self.original_width):
            chunk = F.interpolate(
                chunk,
                size=(self.resized_height, self.resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)

        # torch.nn.functional.pad expects (left, right, top, bottom) for BCHW.
        # Padding is skipped entirely when the resized dimensions are already safe.
        if self.pad_top or self.pad_bottom or self.pad_left or self.pad_right:
            chunk = F.pad(
                chunk,
                (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                mode="constant",
                value=1.0,
            )
        return chunk

    def crop_padding(self, tensor_bchw: torch.Tensor) -> torch.Tensor:
        """Remove the exact spatial padding previously introduced by ``get_bchw``.

        Channel and batch/frame dimensions are preserved. The method intentionally
        returns a view where possible; callers may later interpolate or copy it.
        """

        return tensor_bchw[
            :,
            :,
            self.pad_top : self.pad_top + self.resized_height,
            self.pad_left : self.pad_left + self.resized_width,
        ]
