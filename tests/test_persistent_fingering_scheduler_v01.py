from __future__ import annotations

from dataclasses import fields, replace

import pytest

from dmdod.motor.contact_16k import Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN, TimingEventToken, TimingFrameUpdate,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import (
    AxisPhysics, MotorPhysicalSnapshot, PersistentFingeringScheduler,
    PhysicalFinger, ReservationPhase,
)


RESET = 0.001


def snap(**axes):
    """Invented public Motor physical values, with no evaluator truth."""
    values = []
    for axis in PHYSICAL_AXES:
        overrides = axes.get(axis, {})
        values.append(AxisPhysics(
            axis=axis,
            position_m=overrides.get("position", -0.001),
            velocity_m_s=overrides.get("velocity", -0.001),
            activation=overrides.get("activation", -0.01),
            pressed=overrides.get("pressed", False),
        ))
    return MotorPhysicalSnapshot(tuple(values), RESET)


def token(local_id, due=.10, chord_size=1, *, confirm=True):
    return TimingEventToken(local_id, due, chord_size, confirmed_now=confirm)


def frame(step, *, pending=(), new=(), reused=(), canceled=(), elapsed=(),
          status="eligible", reason=None):
    return TimingFrameUpdate(
        step=step, status=status, covered_until_s=(step + 10) * .010,
        pending=tuple(pending), source="hud_v1.4",
        new_ids=tuple(new), reused_ids=tuple(reused),
        canceled_ids=tuple(canceled), elapsed_ids=tuple(elapsed),
        abstention_reason=reason,
    )


def test_physical_motor_adapter_uses_only_observed_12_axes():
    motor = Coupled16ContactMotor()
    seen = MotorPhysicalSnapshot.from_motor(motor)
    assert tuple(x.axis for x in seen.axes) == PHYSICAL_AXES
    assert all(seen.ready(axis) for axis in PHYSICAL_AXES)
    assert len(seen.axes) == 12
    assert seen.reset_m == motor.motor.keyboard.config.reset_m


@pytest.mark.parametrize("modifier", [
    {"position": RESET + 0.000001},
    {"velocity": +0.000001},
    {"activation": +0.000001},
    {"pressed": True},
])
def test_any_nonreset_positive_motion_or_press_blocks_axis(modifier):
    body = snap(left_5=modifier)
    assert not body.ready("left_5")
    assert body.ready("right_5")


def test_atomic_chord_reservation_and_double_booking_prevention():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .10, 2),
                                               token(1, .12)),
                                   new=(0, 1)), snap())
    chosen = (PhysicalFinger("left_2"), PhysicalFinger("right_1"))
    assert scheduler.reserve(0, chosen, snap()).axes == ("left_2", "right_1")
    assert "right_1" not in scheduler.available_axes(snap())
    with pytest.raises(ValueError, match="reserved or not reset"):
        scheduler.reserve(1, (PhysicalFinger("right_1"),), snap())
    assert len(scheduler.reservations) == 1
    assert scheduler.reservation(0).axes == ("left_2", "right_1")
    assert scheduler.reserve(1, (PhysicalFinger("right_2"),), snap()).axes == ("right_2",)


def test_flat_thumb_and_edge_represent_two_contacts_on_one_axis():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(
        frame(0, pending=(token(0, .10, 2), token(1, .12, 2)),
              new=(0, 1)), snap()
    )
    scheduler.reserve(0, (PhysicalFinger("left_5", 2),), snap())
    assert scheduler.reservation(0).axes == ("left_5",)
    with pytest.raises(ValueError, match="topology"):
        PhysicalFinger("left_1", 2)
    with pytest.raises(ValueError, match="topology"):
        PhysicalFinger("right_6", 1)
    assert PhysicalFinger("right_6", 2).contact_count == 2
    with pytest.raises(ValueError, match="chord size"):
        scheduler.reserve(1, (PhysicalFinger("right_1"),), snap())


def test_quomodocunquize_507_559_inflight_thumb_never_reused():
    """Model the physically observed 507/559 thumb reallocation pattern.

    A target may be consumed by another physical contact, but the scheduler
    receives no evaluator event at all and must keep the moving axis blocked.
    """
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .06),), new=(0,)), snap())
    scheduler.reserve(0, (PhysicalFinger("left_5"),), snap())
    scheduler.arm(0, snap())
    in_flight = snap(left_5={"position": .0016, "velocity": .119,
                             "activation": .132, "pressed": False})
    scheduler.observe_physics(in_flight)
    scheduler.apply_forecast(frame(1, pending=(token(0, .06), token(1, .08)),
                                   reused=(0,), new=(1,)), in_flight)
    with pytest.raises(ValueError, match="reserved or not reset"):
        scheduler.reserve(1, (PhysicalFinger("left_5"),), in_flight)
    # Retire the forecast, not the physically armed stroke.
    scheduler.apply_forecast(frame(2, pending=(token(1, .08),), reused=(1,),
                                   elapsed=(0,)), in_flight)
    assert scheduler.reservation(0).phase is ReservationPhase.ARMED_IN_FLIGHT
    assert scheduler.reservation(0).retired_reason == "elapsed"
    assert "left_5" not in scheduler.available_axes(in_flight)
    # Retained even if a physical KeyDown occurs after forecast retirement.
    pressed = snap(left_5={"position": .0025, "velocity": .1,
                           "activation": .14, "pressed": True})
    scheduler.observe_physics(pressed, (("left_5", KeyEvent.DOWN),))
    assert scheduler.reservation(0).phase is ReservationPhase.KEYDOWN_OBSERVED
    recovering = snap(left_5={"position": .0015, "velocity": -.02,
                              "activation": -.10, "pressed": False})
    scheduler.observe_physics(recovering, (("left_5", KeyEvent.UP),))
    assert 0 in (entry.local_id for entry in scheduler.reservations)
    scheduler.observe_physics(snap())
    assert 0 not in (entry.local_id for entry in scheduler.reservations)
    scheduler.reserve(1, (PhysicalFinger("left_5"),), snap())


def test_quomodocunquize_642_delayed_keydown_cannot_create_second_intent():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .070),
                                               token(1, .090)),
                                   new=(0, 1)), snap())
    scheduler.reserve(0, (PhysicalFinger("right_5"),), snap())
    scheduler.arm(0, snap())
    in_flight = snap(right_5={"position": .00188, "velocity": .088,
                              "activation": .086, "pressed": False})
    scheduler.apply_forecast(frame(1, pending=(token(1, .090),),
                                   reused=(1,), elapsed=(0,)), in_flight)
    with pytest.raises(ValueError, match="reserved or not reset"):
        scheduler.reserve(1, (PhysicalFinger("right_5"),), in_flight)
    scheduler.observe_physics(
        snap(right_5={"position": .002, "velocity": .08,
                      "activation": .1, "pressed": True}),
        (("right_5", KeyEvent.DOWN),),
    )
    assert scheduler.reservation(0).phase is ReservationPhase.KEYDOWN_OBSERVED


def test_unconfirmed_pause_keeps_armed_and_unarmed_reservations():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .05),
                                               token(1, .075)), new=(0, 1)), snap())
    scheduler.reserve(0, (PhysicalFinger("left_1"),), snap())
    scheduler.reserve(1, (PhysicalFinger("left_2"),), snap())
    scheduler.arm(0, snap())
    scheduler.apply_forecast(
        frame(1, status=ABSTAIN, reason="pause_unobservable",
              pending=(token(0, .05, confirm=False),
                       token(1, .075, confirm=False)), reused=(0, 1)), snap()
    )
    assert len(scheduler.reservations) == 2
    assert not scheduler.reservation(0).confirmed
    assert not scheduler.reservation(1).confirmed
    with pytest.raises(ValueError, match="unconfirmed"):
        scheduler.arm(1, snap())
    scheduler.apply_forecast(
        frame(2, pending=(token(0, .05), token(1, .075)), reused=(0, 1)), snap()
    )
    scheduler.arm(1, snap())


def test_armed_retiming_cannot_silently_change_due_or_corrupt_stream():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .08),), new=(0,)), snap())
    scheduler.reserve(0, (PhysicalFinger("left_1"),), snap())
    scheduler.arm(0, snap())
    with pytest.raises(ValueError, match="retime"):
        scheduler.apply_forecast(
            frame(1, pending=(token(0, .09),), reused=(0,)), snap()
        )
    assert scheduler.reservation(0).due_local_s == .08
    # Failed update did not advance the lifecycle validator.
    scheduler.apply_forecast(
        frame(1, pending=(token(0, .08),), reused=(0,)), snap()
    )


def test_ordinary_keydown_keyup_release_requires_actual_motor_reset():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0, .05),), new=(0,)), snap())
    scheduler.reserve(0, (PhysicalFinger("right_1"),), snap())
    scheduler.arm(0, snap())
    pressed = snap(right_1={"position": .003, "velocity": +.2,
                            "activation": +.1, "pressed": True})
    scheduler.observe_physics(pressed, (("right_1", KeyEvent.DOWN),))
    assert scheduler.reservation(0).phase is ReservationPhase.KEYDOWN_OBSERVED
    released_but_not_reset = snap(right_1={"position": RESET + .0001,
                                          "velocity": -.1,
                                          "activation": -.1, "pressed": False})
    scheduler.observe_physics(released_but_not_reset, (("right_1", KeyEvent.UP),))
    assert scheduler.reservation(0).phase is ReservationPhase.RELEASING
    assert "right_1" in scheduler.occupied_axes()
    scheduler.observe_physics(snap())
    assert scheduler.reservations == ()


def test_chord_keydown_partial_and_up_before_final_down_keeps_lifecycle():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(
        frame(0, pending=(token(0, .05, 2),), new=(0,)), snap()
    )
    scheduler.reserve(0, (PhysicalFinger("right_1"), PhysicalFinger("left_1")), snap())
    scheduler.arm(0, snap())
    scheduler.observe_physics(snap(right_1={"pressed": True}),
                              (("right_1", KeyEvent.DOWN),))
    assert scheduler.reservation(0).phase is ReservationPhase.ARMED_IN_FLIGHT
    scheduler.observe_physics(snap(), (("right_1", KeyEvent.UP),))
    scheduler.observe_physics(snap(left_1={"pressed": True}),
                              (("left_1", KeyEvent.DOWN),))
    assert scheduler.reservation(0).phase is ReservationPhase.RELEASING
    scheduler.observe_physics(snap(), (("left_1", KeyEvent.UP),))
    assert scheduler.reservations == ()


def test_cancel_unarmed_is_only_allowed_before_arming():
    scheduler = PersistentFingeringScheduler()
    scheduler.apply_forecast(frame(0, pending=(token(0), token(1, .15)),
                                   new=(0, 1)), snap())
    scheduler.reserve(0, (PhysicalFinger("left_1"),), snap())
    scheduler.cancel_unarmed(0)
    assert "left_1" in scheduler.available_axes(snap())
    scheduler.reserve(1, (PhysicalFinger("right_1"),), snap())
    scheduler.arm(1, snap())
    with pytest.raises(ValueError, match="armed"):
        scheduler.cancel_unarmed(1)


def test_no_evaluator_truth_fields_in_public_scheduler_contract():
    forbidden = {"target_ordinal", "target_index", "floor_index",
                 "target_time_s", "chart_time", "judgement", "_used", "_missed"}
    for klass in (AxisPhysics, MotorPhysicalSnapshot, PhysicalFinger):
        assert not forbidden.intersection(f.name for f in fields(klass))
    assert not forbidden.intersection(
        f.name for f in fields(scheduler_reservation_type())
    )


def scheduler_reservation_type():
    from dmdod.planning.persistent_fingering_scheduler_v01 import PhysicalReservation
    return PhysicalReservation


def test_explicit_forecast_cancellation_brakes_but_keeps_inflight_axis():
    planner = PersistentFingeringScheduler()
    planner.apply_forecast(frame(0, pending=(token(0,.06),), new=(0,)), snap())
    planner.reserve(0, (PhysicalFinger("left_2"),), snap())
    planner.arm(0, snap())
    moving = snap(left_2={"position": .0017, "velocity": .06,
                          "activation": .04})
    planner.apply_forecast(frame(1, canceled=(0,)), moving)
    assert planner.reservation(0).phase is ReservationPhase.CANCELED_IN_FLIGHT
    assert planner.reservation(0).retired_reason == "canceled"
    assert "left_2" not in planner.available_axes(moving)
    planner.observe_physics(snap())
    assert planner.reservations == ()


def test_elapsed_armed_stroke_can_finish_after_predicted_due():
    planner = PersistentFingeringScheduler()
    planner.apply_forecast(frame(0, pending=(token(0,.015),), new=(0,)), snap())
    planner.reserve(0, (PhysicalFinger("left_1"),), snap())
    planner.arm(0, snap())
    still_moving = snap(left_1={"position":.0014, "velocity":.05,
                                "activation":.11})
    # The local forecast contract requires a consecutive 10ms step stream.
    planner.apply_forecast(
        frame(1, pending=(token(0, .015),), reused=(0,)), still_moving
    )
    planner.apply_forecast(frame(2, elapsed=(0,)), still_moving)
    assert planner.reservation(0).phase is ReservationPhase.ARMED_IN_FLIGHT
    assert planner.reservation(0).retired_reason == "elapsed"
    down = snap(left_1={"position":.003, "pressed":True,
                        "velocity":.02, "activation":.08})
    planner.observe_physics(down, (("left_1", KeyEvent.DOWN),))
    assert planner.reservation(0).phase is ReservationPhase.KEYDOWN_OBSERVED
    planner.observe_physics(snap(), (("left_1", KeyEvent.UP),))
    assert planner.reservations == ()


@pytest.mark.parametrize("axis, due, step", [
    ("right_3", 1.93750000238, 189),  # ordinary 480: target29
    ("left_3", .90725807341, 86),    # Pause: target11
])
def test_existing_reservation_arms_despite_passive_motor_spring_velocity(
    axis, due, step,
):
    """Reproduce the observed 0-command cancellation at the arm boundary.

    At selection the axis was strictly idle. Under subsequent zero commands
    its spring produces a small upward velocity but negative activation;
    it is still the SAME reserved axis, not a new competing request.
    """
    planner = PersistentFingeringScheduler()
    initially_idle = snap()
    planner.apply_forecast(
        frame(0, pending=(token(0, due),), new=(0,)), initially_idle
    )
    planner.reserve(0, (PhysicalFinger(axis),), initially_idle)
    passive = snap(**{
        axis: {"position": -.000997, "velocity": .0008, "activation": -.0177}
    })
    assert not passive.ready(axis)
    assert passive.ready_to_arm_reserved(axis)
    assert axis not in planner.available_axes(passive)
    planner.arm(0, passive)
    assert planner.reservation(0).phase is ReservationPhase.ARMED_IN_FLIGHT
    assert axis not in planner.available_axes(passive)


@pytest.mark.parametrize("bad_state", [
    {"pressed": True},
    {"position": RESET + .0001},
    {"activation": .00001},
])
def test_reserved_arm_guard_still_refuses_active_or_unreset_axis(bad_state):
    planner = PersistentFingeringScheduler()
    planner.apply_forecast(
        frame(0, pending=(token(0, .060),), new=(0,)), snap()
    )
    planner.reserve(0, (PhysicalFinger("right_3"),), snap())
    unsafe = snap(right_3=bad_state)
    assert not unsafe.ready_to_arm_reserved("right_3")
    with pytest.raises(ValueError, match="active/unreset stroke"):
        planner.arm(0, unsafe)
    assert planner.reservation(0).phase is ReservationPhase.RESERVED


def test_passive_velocity_does_not_relax_new_reservations():
    moving = snap(right_3={
        "position": -.000997, "velocity": .0008, "activation": -.0177,
    })
    planner = PersistentFingeringScheduler()
    planner.apply_forecast(
        frame(0, pending=(token(0, .060),), new=(0,)), moving
    )
    assert "right_3" not in planner.available_axes(moving)
    with pytest.raises(ValueError, match="reserved or not reset"):
        planner.reserve(0, (PhysicalFinger("right_3"),), moving)


@pytest.mark.parametrize("fingers,chord_size,pressed_axis", [
    ((PhysicalFinger("left_1"),), 1, "left_1"),
    ((PhysicalFinger("left_5", 2),), 2, "left_5"),
    ((PhysicalFinger("left_1"), PhysicalFinger("right_1")), 2, "left_1"),
])
def test_keydown_before_arm_quarantines_axes(fingers, chord_size, pressed_axis):
    planner = PersistentFingeringScheduler()
    planner.apply_forecast(
        frame(0, pending=(token(0, .06, chord_size),), new=(0,)),
        snap(),
    )
    planner.reserve(0, fingers, snap())
    pressed = snap(**{pressed_axis: {
        "position": .003, "velocity": .02,
        "activation": .1, "pressed": True,
    }})
    planner.observe_physics(pressed, ((pressed_axis, KeyEvent.DOWN),))
    entry = planner.reservation(0)
    assert entry.phase is ReservationPhase.CANCELED_IN_FLIGHT
    assert entry.confirmed is False
    assert entry.retired_reason == "unexpected_keydown_before_arm"
    assert all(axis not in planner.available_axes(pressed) for axis in entry.axes)
    with pytest.raises(ValueError, match="only a reserved"):
        planner.arm(0, pressed)
    # Physical reset, not forecast expiration, releases the reservation.
    planner.observe_physics(snap(), ((pressed_axis, KeyEvent.UP),))
    assert planner.reservations == ()
