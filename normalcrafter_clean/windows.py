"""Deterministic temporal window scheduling for arbitrary-length clips.

NormalCrafter's spatio-temporal UNet processes a fixed number of frames at once.
Long clips are therefore divided into overlapping windows. The scheduler in this
module has no Torch or ComfyUI dependency, making it simple to reason about and
unit-test independently from the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Window:
    """Half-open frame interval ``[start, end)``.

    Frozen instances are safe to share as descriptive values, and ``slots`` keeps
    this tiny object free of an unnecessary per-instance attribute dictionary.
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of frames contained in the interval."""

        return self.end - self.start


def build_windows(num_frames: int, window_size: int, step_size: int) -> list[Window]:
    """Build fixed-size windows that cover every frame without temporal gaps.

    ``step_size`` controls how far each regular window advances. Consequently,
    regular overlap is ``window_size - step_size``. The final window is anchored
    exactly to the clip end, even when this creates a larger-than-regular overlap.

    Example for 25 frames, window 14, step 10::

        [0, 14), [10, 24), [11, 25)

    The last start moves from the regular candidate 20 back to 11 so frames 24 and
    25 are covered by a full-size model window. Short clips must already have an
    effective frame count of at least ``window_size``; ``FrameSource`` provides
    that temporal extension by repeating the final real frame.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if step_size > window_size:
        raise ValueError("step_size cannot exceed window_size; gaps would be created")
    if num_frames < window_size:
        raise ValueError("caller must pad short clips to window_size")

    # A full-size final window beginning here ends exactly at num_frames.
    last_start = num_frames - window_size

    # Generate the regular cadence as far as a full window still fits.
    starts = list(range(0, last_start + 1, step_size))

    # A cadence rarely lands exactly on the clip tail. Append one anchored window
    # instead of accepting an uncovered remainder or creating a shorter window.
    if starts[-1] != last_start:
        starts.append(last_start)

    return [Window(start=s, end=s + window_size) for s in starts]


def overlap_length(previous: Window | None, current: Window) -> int:
    """Return how many leading frames of ``current`` overlap ``previous``.

    Windows are generated in increasing order, so only the previous end and the
    current start are needed. Non-overlapping intervals correctly produce zero.
    """

    if previous is None:
        return 0
    return max(0, previous.end - current.start)
