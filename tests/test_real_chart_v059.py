from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_real_chart_interventions_v059 as diag  # noqa: E402


def _frame(
    index: int,
    *,
    kind: str = "neutral",
    delta: float = 0.0,
) -> diag.InterventionFrame:
    teacher = {
        "press": (1.0, 0.0),
        "release": (-1.0, 0.0),
        "neutral": (0.0, 0.0),
    }[kind]
    return diag.InterventionFrame(
        frame_index=index,
        time_s=index * 0.01,
        target_ordinal=1,
        floor_index=2,
        target_time_s=0.5,
        kind=kind,
        student_left=teacher[0] - delta,
        student_right=teacher[1],
        teacher_left=teacher[0],
        teacher_right=teacher[1],
        max_abs_delta=abs(delta),
    )


def test_action_kind_prefers_press_then_release_then_neutral():
    assert diag._action_kind(1.0, -1.0) == "press"
    assert diag._action_kind(-1.0, 0.0) == "release"
    assert diag._action_kind(0.1, -0.1) == "neutral"


def test_max_action_delta_uses_linf_over_two_motor_channels():
    assert diag._max_action_delta(0.2, -0.4, 1.0, -0.1) == pytest.approx(0.8)


def test_intervention_threshold_is_inclusive():
    frame = _frame(0, kind="press", delta=0.25)
    assert diag._is_intervention(frame, delta_threshold=0.25)
    assert not diag._is_intervention(frame, delta_threshold=0.251)

    with pytest.raises(ValueError):
        diag._is_intervention(frame, delta_threshold=-0.1)


def test_summary_counts_kinds_and_contiguous_runs():
    frames = [
        _frame(0, kind="press", delta=0.4),
        _frame(1, kind="press", delta=0.3),
        _frame(2, kind="neutral", delta=0.1),
        _frame(3, kind="release", delta=0.5),
        _frame(4, kind="neutral", delta=0.25),
    ]
    summary = diag._summarize(frames, total_frames=5, delta_threshold=0.25)

    assert summary.intervention_frames == 4
    assert summary.intervention_runs == 2
    assert summary.longest_run_frames == 2
    assert summary.kind_counts == {"press": 2, "release": 1, "neutral": 1}
    assert summary.intervention_rate == pytest.approx(0.8)
    assert summary.mean_max_abs_delta == pytest.approx((0.4 + 0.3 + 0.5 + 0.25) / 4)
    assert summary.max_abs_delta == pytest.approx(0.5)
