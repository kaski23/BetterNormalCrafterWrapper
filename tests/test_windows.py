from normalcrafter_clean.windows import Window, build_windows, overlap_length


def test_exact_single_window():
    assert build_windows(14, 14, 10) == [Window(0, 14)]


def test_tail_is_anchored():
    assert build_windows(25, 14, 10) == [Window(0, 14), Window(10, 24), Window(11, 25)]


def test_overlap():
    assert overlap_length(Window(0, 14), Window(10, 24)) == 4
    assert overlap_length(Window(10, 24), Window(11, 25)) == 13
