from __future__ import annotations

from scripts.audit_timing_generalization_phase7_v01 import _evaluate


def _row(time_s, *, pending=(), canceled=(), status="eligible"):
    return {
        "step": round(time_s / 0.010),
        "time_s": time_s,
        "status": status,
        "pending": list(pending),
        "canceled": list(canceled),
    }


def test_interval_containing_truth_is_not_a_false_positive():
    token = {
        "id": "countdown:0",
        "due_s": 1.205,
        "due_lower_s": 1.200,
        "due_upper_s": 1.210,
        "confirmed_now": True,
    }
    result = _evaluate([_row(1.13, pending=(token,))], (1.20000004,))
    assert result["metrics"]["category_counts"]["1"] == 1
    assert result["metrics"]["false_positive_ids"] == 0


def test_cancel_at_control_boundary_is_lifecycle_failure_posthoc():
    token = {"id": "v14:3", "due_s": 4.0500001, "confirmed_now": True}
    rows = [
        _row(3.98, pending=(token,)),
        _row(4.05, canceled=("v14:3",)),
    ]
    result = _evaluate(rows, (4.0500001,))
    assert result["metrics"]["category_counts"]["5"] == 1
    assert result["metrics"]["event_id_continuity_failures"] == 1


def test_known_pause_abstention_is_not_called_estimation_failure():
    rows = [_row(1.15, status="pause_motion_unobservable")]
    result = _evaluate(rows, (1.20,))
    assert result["metrics"]["category_counts"]["3"] == 1
    assert result["metrics"]["category_counts"]["2"] == 0
