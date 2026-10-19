from __future__ import annotations

from scripts.audit_hud_timing_phase8_lifecycle_integration_v01 import (
    _compare_rows,
    _evaluable_rows,
    _motor_contract_audit,
)


def _row(*, due=0.05, uncertainty=0.0, canceled=(), elapsed=()):
    return {
        "step": 0,
        "time_s": 0.0,
        "status": "eligible",
        "public_status": "eligible",
        "pending": [{
            "id": 0,
            "due_s": due,
            "due_lower_s": due - uncertainty,
            "due_upper_s": due + uncertainty,
            "uncertainty_s": uncertainty,
            "confirmed_now": True,
        }],
        "new": [0],
        "reused": [],
        "canceled": list(canceled),
        "elapsed": list(elapsed),
        "raw_pending": [{
            "id": "countdown:0",
            "due_lower_s": due - uncertainty,
            "due_upper_s": due + uncertainty,
            "uncertainty_s": uncertainty,
            "source": "countdown_first_event",
        }],
    }


def test_phase8_comparison_requires_due_uncertainty_and_countdown_invariance():
    same = _compare_rows([_row(uncertainty=0.01)], [_row(uncertainty=0.01)])
    assert all(not values for values in same.values())

    changed_due = _compare_rows([_row()], [_row(due=0.051)])
    assert changed_due["due_differences"]
    assert changed_due["countdown_interval_differences"]

    changed_uncertainty = _compare_rows(
        [_row(uncertainty=0.01)], [_row(uncertainty=0.009)]
    )
    assert changed_uncertainty["uncertainty_differences"]
    assert changed_uncertainty["countdown_interval_differences"]


def test_right_censored_forecast_is_not_counted_as_false_positive_evidence():
    rows = _evaluable_rows([_row(due=5.050)], 5.0)
    assert rows[0]["pending"] == []


def test_elapsed_keeps_armed_motor_intent_and_does_not_claim_keydown():
    result = _motor_contract_audit()
    assert result["pass"] is True
    assert result["phase_after_elapsed"] == "armed_in_flight"
    assert result["retired_reason"] == "elapsed"
    assert result["physical_keydown_observed"] is False
