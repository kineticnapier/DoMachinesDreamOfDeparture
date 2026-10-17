from __future__ import annotations

import pytest

from dmdod.evaluation.timing_countdown_v3 import (
    CausalFrameForecast, evaluate_countdown,
)


def frame(now, remaining, *, confidence=.9, present=True, local_id=0):
    due = now + remaining
    return CausalFrameForecast(
        now_s=now, raw_offset_s=remaining, confidence=confidence,
        present=present,
        accepted_offsets_s=(remaining,) if present else (),
        pending=((local_id, due),) if present else (),
    )


def test_perfect_30_20_10ms_countdown_has_zero_drift():
    rows = tuple(frame(now, remaining) for now, remaining in (
        (.070, .030), (.080, .020), (.090, .010),
    ))
    result = evaluate_countdown(rows, (.100,))
    summary = result["aggregate"]
    assert summary["raw_due_mae_ms"] == pytest.approx(0.0, abs=1e-9)
    assert summary["raw_countdown_step_mae_ms"] == pytest.approx(0.0, abs=1e-9)
    assert summary["monotonic_violations"] == 0
    assert summary["failure_attribution"]["time_regression"] is False


def test_constant_remaining_prediction_is_time_regression_failure():
    rows = tuple(frame(now, .030) for now in (.070, .080, .090))
    result = evaluate_countdown(rows, (.100,))["aggregate"]
    assert result["raw_countdown_step_mae_ms"] == pytest.approx(10.0)
    assert result["monotonic_violations"] == 2
    assert result["failure_attribution"]["time_regression"] is True


def test_disappearance_and_reappearance_are_existence_failures():
    rows = (
        frame(.070, .030, present=True),
        frame(.080, .020, confidence=.2, present=False),
        frame(.090, .010, present=True),
    )
    result = evaluate_countdown(rows, (.100,))["aggregate"]
    assert result["disappearances"] == 1
    assert result["reappearances"] == 1
    assert result["failure_attribution"]["event_existence"] is True
    assert result["raw_countdown_step_mae_ms"] == pytest.approx(0.0, abs=1e-9)


def test_local_id_split_is_reassociation_failure_only():
    rows = (
        frame(.070, .030, local_id=4),
        frame(.080, .020, local_id=5),
        frame(.090, .010, local_id=5),
    )
    result = evaluate_countdown(rows, (.100,))["aggregate"]
    assert result["tracker_id_switches_within_20ms"] == 1
    assert result["events_with_multiple_tracker_ids_within_20ms"] == 1
    assert result["failure_attribution"]["forecast_reassociation_observed"] is True
    assert result["failure_attribution"]["forecast_reassociation_independent"] is True
    assert result["failure_attribution"]["time_regression"] is False


def test_initial_acquisition_is_not_reappearance():
    rows = (
        frame(.070, .030, present=False),
        frame(.080, .020, present=False),
        frame(.090, .010, present=True),
    )
    result = evaluate_countdown(rows, (.100,))["aggregate"]
    assert result["reappearances"] == 0
