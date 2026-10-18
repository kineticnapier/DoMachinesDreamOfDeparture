from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.motor.contact_16k import Coupled16ContactMotor
from scripts import audit_fingering_causal_early_v02 as audit


def _finger(axis):
    return audit.teacher.Finger(
        axis, "tip" if axis.endswith("_5") else "single"
    )


def test_causal_extra_lead_from_decision_time_motor_only():
    motor = Coupled16ContactMotor()
    ix = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[ix].activation = -.088
    extra = audit.decision_extra_s(motor, (_finger("right_1"),))
    assert 0.023 < extra < 0.025
    assert audit.decision_extra_s(motor, (_finger("right_5"),)) == 0
    assert audit.decision_extra_s(motor, ()) == 0
    assert audit.decision_extra_s(
        motor, (_finger("right_1"), _finger("left_1"))
    ) == 0


def test_extra_lead_is_zero_without_negative_activation():
    motor = Coupled16ContactMotor()
    assert audit.decision_extra_s(motor, (_finger("right_1"),)) == 0
    ix = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[ix].activation = -.001
    assert audit.decision_extra_s(motor, (_finger("right_1"),)) == 0


def test_adaptive_action_causally_extends_only_committed_finger():
    motor = Coupled16ContactMotor()
    axis = "right_1"
    plan = audit.teacher.PlanItem(10, .065, (_finger(axis),))
    baseline = audit.teacher.action_for(motor, (plan,), 0.0)
    assert baseline.commands[audit.teacher.AXES.index(axis)] == 0
    adapted, early, released = audit.adaptive_action(
        motor, (plan,), 0.0, {10: .025}, enable_release=False
    )
    assert adapted.commands[audit.teacher.AXES.index(axis)] == 1
    assert early == 1 and released == 0
    assert all(v == 0 for i, v in enumerate(adapted.commands)
               if i != audit.teacher.AXES.index(axis))
    assert motor.motor.time_s == 0  # counterfactual control does not mutate Motor


def test_adaptive_action_does_not_bypass_pressed_safety():
    motor = Coupled16ContactMotor()
    ix = audit.teacher.AXES.index("right_1")
    motor.motor.keyboard.states[ix].pressed = True
    plan = audit.teacher.PlanItem(10, .065, (_finger("right_1"),))
    adapted, early, _ = audit.adaptive_action(
        motor, (plan,), 0.0, {10: .025}, enable_release=False
    )
    assert early == 0 and adapted.commands[ix] == -1


def test_adaptive_action_keeps_ordinary_44ms_window_unchanged():
    motor = Coupled16ContactMotor()
    plan = audit.teacher.PlanItem(1, .035, (_finger("left_1"),))
    old = audit.teacher.action_for(motor, (plan,), 0.0)
    updated, early, release = audit.adaptive_action(
        motor, (plan,), 0.0, {1: .025}, enable_release=False
    )
    assert old == updated
    assert early == 0 and release == 0


def test_baseline_checks_actual_saved_network_choices():
    results = {
        "decisions": [
            {"event_index": 10, "decision_ms": 60.0,
             "chosen": "right_1"}
        ]
    }
    saved = {
        "events": [{
            "decisions": {
                "free": {
                    "event_index": 10,
                    "decision_s": .060,
                    "chosen": "right_1",
                }
            }
        }]
    }
    audit.verify_baseline_choices(results, saved, "free")
    wrong = {
        "decisions": [
            {"event_index": 10, "decision_ms": 60.0,
             "chosen": "left_1"}
        ]
    }
    with pytest.raises(RuntimeError, match="decision drift"):
        audit.verify_baseline_choices(wrong, saved, "free")
    with pytest.raises(RuntimeError, match="count drift"):
        audit.verify_baseline_choices({"decisions": []}, saved, "free")


def test_activation_lead_bounded_for_large_negative_state():
    assert audit.activation_extra_s(-.99, .28) == audit.EXTRA_LEAD_S
    assert audit.activation_extra_s(0.0, .28) == 0.0
