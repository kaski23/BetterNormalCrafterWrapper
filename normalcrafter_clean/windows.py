from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Window:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def build_windows(num_frames: int, window_size: int, step_size: int) -> list[Window]:
    """Return monotonically increasing, fixed-size windows covering every frame.

    Short clips are expected to be padded by the caller to at least ``window_size``.
    The final window is anchored to the end so no tail frames are skipped.
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

    last_start = num_frames - window_size
    starts = list(range(0, last_start + 1, step_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [Window(start=s, end=s + window_size) for s in starts]


def overlap_length(previous: Window | None, current: Window) -> int:
    if previous is None:
        return 0
    return max(0, previous.end - current.start)
