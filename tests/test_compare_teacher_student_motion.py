from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_teacher_student_motion as motion


def test_analysis_window_can_focus_chart_start() -> None:
    lo, hi, mode = motion._analysis_window(
        miss_t=1.5,
        last_t=10.0,
        before_s=1.0,
        after_s=0.5,
        start_duration_s=0.2,
    )

    assert lo == 0.0
    assert hi == pytest.approx(0.2)
    assert mode == "chart-start"


def test_analysis_window_preserves_first_miss_mode() -> None:
    lo, hi, mode = motion._analysis_window(
        miss_t=1.5,
        last_t=10.0,
        before_s=1.0,
        after_s=0.5,
        start_duration_s=None,
    )

    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(2.0)
    assert mode == "first-miss"


def test_events_in_window_keeps_exact_keydown_times() -> None:
    events = [
        {"t": 0.068, "key": "left"},
        {"t": 0.205, "key": "right"},
    ]

    assert motion._events_in_window(events, 0.0, 0.2) == [events[0]]
