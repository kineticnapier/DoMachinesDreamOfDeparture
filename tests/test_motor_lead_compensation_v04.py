from __future__ import annotations

from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode
from dmdod.planning.motor_lead_compensation_v04 import MotorLeadCompensator


def _commands(**values: float) -> tuple[float, ...]:
    return tuple(values.get(axis, 0.0) for axis in PHYSICAL_AXES)


def test_cold_prediction_is_deterministic_and_does_not_mutate_motor():
    motor = Coupled16ContactMotor()
    before = motor.observe()
    compensator = MotorLeadCompensator()
    left = compensator.predict(motor, axis="left_1", mode="single")
    again = compensator.predict(motor, axis="left_1", mode="single")
    assert left.valid and 0.035 <= left.earliest_arrival_s <= 0.045
    assert left == again
    assert motor.observe() == before


def test_prediction_refuses_pressed_axis_and_uses_legacy_fallback():
    motor = Coupled16ContactMotor()
    action = ContactAction(_commands(left_1=1.0))
    for _ in range(5):
        motor.step(action)
    prediction = MotorLeadCompensator().predict(motor, axis="left_1", mode="single")
    assert not prediction.valid
    assert prediction.reason == "axis_pressed"
    decision = MotorLeadCompensator().decide(
        now_s=.960, due_s=1.0, prediction=prediction)
    assert decision.start
    assert decision.fallback
    assert decision.reason == "legacy_44ms_fallback"


def test_too_early_guard_waits_until_earliest_arrival_reaches_due():
    motor = Coupled16ContactMotor()
    compensator = MotorLeadCompensator()
    prediction = compensator.predict(motor, axis="right_1", mode="single")
    assert not compensator.decide(now_s=.950, due_s=1.0,
                                   prediction=prediction).start
    assert compensator.decide(now_s=.970, due_s=1.0,
                               prediction=prediction).start


def test_thumb_tip_flat_and_edge_contacts_are_forecast_without_topology_mutation():
    motor = Coupled16ContactMotor()
    compensator = MotorLeadCompensator()
    tip = compensator.predict(motor, axis="left_5", mode="tip")
    flat = compensator.predict(motor, axis="left_5", mode="flat")
    edge = compensator.predict(motor, axis="right_6", mode="edge")
    assert tip.valid and tip.contacts == ("left_5",)
    assert flat.valid and flat.contacts == ("left_5", "left_6")
    assert edge.valid and edge.contacts == ("right_7", "right_8")
    assert motor._modes == (ThumbMode.OFF, ThumbMode.OFF)
    assert motor._edge == (False, False)


def test_partial_forward_motion_reduces_remaining_latency():
    motor = Coupled16ContactMotor()
    compensator = MotorLeadCompensator()
    cold = compensator.predict(motor, axis="left_2", mode="single")
    motor.step(ContactAction(_commands(left_2=1.0)))
    warm = compensator.predict(motor, axis="left_2", mode="single")
    assert warm.valid
    assert warm.earliest_arrival_s < cold.earliest_arrival_s
