from __future__ import annotations

import math

import pytest

from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor
from scripts import audit_fingering_release_lead_v011 as audit


def _zero_action():
    return ContactAction((0.0,) * len(audit.teacher.AXES))


def test_physical_activation_extra_lead_is_bounded():
    assert audit.physical_extra_lead_s(0.0, .28) == 0
    assert audit.physical_extra_lead_s(-.005, .28) == 0
    assert audit.physical_extra_lead_s(-.088, .28) == pytest.approx(.28 * math.log(1.088))
    assert audit.physical_extra_lead_s(-.5, .28) == audit.MAX_EXTRA_LEAD_S
    with pytest.raises(ValueError):
        audit.physical_extra_lead_s(-.088, 0)


def test_release_coast_changes_only_releasing_axis_near_reset():
    motor = Coupled16ContactMotor()
    ix = audit.teacher.AXES.index("right_1")
    state = motor.motor.body.states[ix]
    motor.motor.keyboard.states[ix].pressed = True
    state.position_m = motor.motor.keyboard.config.reset_m + .0001
    state.velocity_m_s = -.05
    values = list(_zero_action().commands)
    values[ix] = -1.0
    old_action = ContactAction(tuple(values))
    new_action, count = audit.release_coast(motor, old_action)
    assert count == 1
    assert new_action.commands[ix] == 0
    assert old_action.commands[ix] == -1
    assert sum(x != y for x, y in zip(new_action.commands, old_action.commands)) == 1
    state.position_m = .004
    unchanged, count = audit.release_coast(motor, old_action)
    assert unchanged == old_action
    assert count == 0


def test_anticipation_is_explicitly_prior_to_saved_decision_time():
    motor = Coupled16ContactMotor()
    ix = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[ix].activation = -.088
    # The allocation is only *known* by this diagnostic, not by live Network.
    future = [{
        "event_index": 12, "decision_s": .020, "due_s": .060,
        "chosen": "right_1", "axes": ["right_1"]
    }]
    base = _zero_action()
    new, armed, new_count = audit.apply_early_lead(
        motor, base, future, 0.0, set(), {}
    )
    assert new_count == 1
    assert armed == {12: "right_1"}
    assert new.commands[ix] == 1.0
    assert base.commands[ix] == 0.0
    # If no future choices are known, lead cannot start.
    none, armed, n = audit.apply_early_lead(motor, base, [], 0.0, set(), {})
    assert none == base and not armed and n == 0


def test_anticipation_is_blocked_when_axis_occupied_or_pressed():
    motor = Coupled16ContactMotor()
    ix = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[ix].activation = -.088
    row = [{
        "event_index": 12, "decision_s": .020, "due_s": .060,
        "chosen": "right_1"
    }]
    base = _zero_action()
    none, armed, n = audit.apply_early_lead(
        motor, base, row, 0.0, {"right_1"}, {}
    )
    assert n == 0 and armed == {} and none == base
    motor.motor.keyboard.states[ix].pressed = True
    none, armed, n = audit.apply_early_lead(
        motor, base, row, 0.0, set(), {}
    )
    assert n == 0 and armed == {} and none == base


def test_anticipation_skips_thumb_and_side_to_preserve_topology():
    motor = Coupled16ContactMotor()
    i = audit.teacher.AXES.index("right_5")
    motor.motor.body.states[i].activation = -.2
    future = [{"event_index": 12, "decision_s": .02,
               "due_s": .06, "chosen": "right_5/tip"}]
    base = _zero_action()
    same, armed, n = audit.apply_early_lead(
        motor, base, future, 0.0, set(), {}
    )
    assert same == base and not armed and n == 0


def test_anticipation_does_not_start_outside_bounded_lead():
    motor = Coupled16ContactMotor()
    i = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[i].activation = -.2
    base = _zero_action()
    far = [{
        "event_index": 20, "decision_s": .060,
        "due_s": .100, "chosen": "right_1"
    }]
    same, armed, n = audit.apply_early_lead(
        motor, base, far, 0.0, set(), {}
    )
    assert same == base and armed == {} and n == 0


def test_anticipation_never_arms_two_events_on_same_axis():
    motor = Coupled16ContactMotor()
    i = audit.teacher.AXES.index("right_1")
    motor.motor.body.states[i].activation = -.088
    choices = [
        {"event_index": 20, "decision_s": .020, "due_s": .060, "chosen": "right_1"},
        {"event_index": 21, "decision_s": .024, "due_s": .063, "chosen": "right_1"},
    ]
    _, armed, n = audit.apply_early_lead(
        motor, _zero_action(), choices, 0.0, set(), {}
    )
    assert n == 1
    assert armed == {20: "right_1"}
