from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v110_nonempty_segments as nonempty


def test_relocate_empty_tail_window_keeps_length_and_contains_nearest_target() -> None:
    start, end = nonempty._relocate_window(
        duration_s=100.0,
        start_s=70.0,
        end_s=100.0,
        target_times_s=(5.0, 20.0, 40.0),
    )

    assert end - start == 30.0
    assert start <= 40.0 <= end
    assert (start, end) == (25.0, 55.0)


def test_relocate_empty_intro_window_clamps_inside_chart() -> None:
    start, end = nonempty._relocate_window(
        duration_s=100.0,
        start_s=0.0,
        end_s=30.0,
        target_times_s=(60.0, 80.0),
    )

    assert end - start == 30.0
    assert start <= 60.0 <= end
    assert (start, end) == (45.0, 75.0)


def test_relocate_rejects_chart_without_playable_targets() -> None:
    try:
        nonempty._relocate_window(
            duration_s=100.0,
            start_s=0.0,
            end_s=30.0,
            target_times_s=(),
        )
    except ValueError as exc:
        assert "no playable targets" in str(exc)
    else:
        raise AssertionError("expected ValueError")
