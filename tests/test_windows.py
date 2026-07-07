"""Fast unit tests for temporal coverage rules.

These tests intentionally avoid importing Torch, Diffusers, or ComfyUI. Window
scheduling is pure infrastructure and should remain testable even on a machine
without model dependencies or CUDA.
"""

from normalcrafter_clean.windows import Window, build_windows, overlap_length


def test_exact_single_window():
    # A clip exactly as long as the model context needs one window and no padding.
    assert build_windows(14, 14, 10) == [Window(0, 14)]


def test_tail_is_anchored():
    # The regular cadence would leave frame 24 uncovered. The scheduler appends a
    # final full-size window beginning at 11 so its end lands exactly on frame 25.
    assert build_windows(25, 14, 10) == [Window(0, 14), Window(10, 24), Window(11, 25)]


def test_overlap():
    # Regular overlap is 14 - 10 = 4. The tail-anchored final window may overlap
    # more strongly; here [10,24) and [11,25) share thirteen frames.
    assert overlap_length(Window(0, 14), Window(10, 24)) == 4
    assert overlap_length(Window(10, 24), Window(11, 25)) == 13
