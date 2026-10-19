from __future__ import annotations

from copy import deepcopy

from scripts import run_hud_timing_phase9_lifecycle_motor_ab_v01 as runner


def _frame(*, new=(0,), reused=(), canceled=(), elapsed=(), due=1.0,
           uncertainty=0.01, reservations=()):
    return {
        "step": 0,
        "pending": [{
            "local_id": local_id,
            "due_local_s": due,
            "uncertainty_s": uncertainty,
        } for local_id in new + reused],
        "new_ids": list(new),
        "reused_ids": list(reused),
        "canceled_ids": list(canceled),
        "elapsed_ids": list(elapsed),
        "reservations_after_command": list(reservations),
        "reservations_after_physics": list(reservations),
    }


def test_lifecycle_validator_accepts_single_retirement_and_rejects_recreation():
    valid = [
        _frame(),
        {**_frame(new=(), elapsed=(0,)), "step": 1},
    ]
    assert runner._lifecycle_contract_violations(valid) == []
    recreated = valid + [{**_frame(), "step": 2}]
    assert runner._lifecycle_contract_violations(recreated)[0]["reasons"] == [
        "id_recreated"
    ]


def test_forecast_comparison_detects_due_and_uncertainty_changes():
    original = [_frame()]
    due_changed = deepcopy(original)
    due_changed[0]["pending"][0]["due_local_s"] += .001
    assert runner._forecast_differences(original, due_changed)["due"]
    uncertainty_changed = deepcopy(original)
    uncertainty_changed[0]["pending"][0]["uncertainty_s"] = .02
    assert runner._forecast_differences(
        original, uncertainty_changed
    )["uncertainty"]


def test_reservation_validator_rejects_double_booked_axis():
    reservations = ({
        "local_id": 0, "axes": ["left_1"], "phase": "reserved",
    }, {
        "local_id": 1, "axes": ["left_1"], "phase": "reserved",
    })
    violations = runner._reservation_violations([
        _frame(reservations=reservations)
    ])
    assert violations[0]["reasons"] == ["axis_double_booked"]


def test_phase3_default_remains_lifecycle_v14():
    assert phase3_default() == runner.LIFECYCLE_V14


def phase3_default():
    import inspect
    from scripts import run_countdown_hud_v3_motor_phase3 as phase3

    return inspect.signature(phase3.simulate).parameters[
        "lifecycle_version"
    ].default
