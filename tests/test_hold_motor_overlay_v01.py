from __future__ import annotations

import pytest

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.motor.contact_16k import (
    ContactAction,
    Coupled16ContactMotor,
    PHYSICAL_AXES,
    ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.hold_motor_overlay_v01 import (
    HoldMotorOverlay,
    HoldMotorPhase,
    HoldMotorPlan,
    HoldTimingProvenance,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import PhysicalFinger


def _neutral() -> ContactAction:
    return ContactAction((0.0,) * len(PHYSICAL_AXES))


def _plan(
    *,
    local_id: int = 7,
    axis: str = "left_1",
    contact_count: int = 1,
    provenance: HoldTimingProvenance = HoldTimingProvenance.CAUSAL_VISIBLE_INTERVAL,
) -> HoldMotorPlan:
    return HoldMotorPlan(
        local_id=local_id,
        finger=PhysicalFinger(axis, contact_count),
        start_due=TimeWindow(0.035, 0.090),
        release_due=TimeWindow(0.180, 0.300),
        start_command_s=0.0,
        release_command_s=0.180,
        provenance=provenance,
    )


def test_unverified_and_oracle_plans_are_fail_closed_by_default() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    with pytest.raises(ValueError, match="unverified"):
        overlay.register(
            _plan(provenance=HoldTimingProvenance.UNVERIFIED), motor
        )
    with pytest.raises(ValueError, match="opt-in"):
        overlay.register(
            _plan(provenance=HoldTimingProvenance.ORACLE_DIAGNOSTIC), motor
        )


def test_oracle_diagnostic_requires_explicit_opt_in_and_remains_labelled() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay(allow_oracle_diagnostic=True)
    plan = _plan(provenance=HoldTimingProvenance.ORACLE_DIAGNOSTIC)
    overlay.register(plan, motor)
    assert overlay.active[0].plan.provenance is HoldTimingProvenance.ORACLE_DIAGNOSTIC


def test_hold_axis_stays_pressed_while_other_axis_can_tap_and_then_releases() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(_plan(), motor)
    physical_edges: list[tuple[float, str, KeyEvent]] = []
    logical_edges: list[tuple[float, str, KeyEvent]] = []

    for step in range(60):
        now_s = step * 0.010
        commands = [0.0] * len(PHYSICAL_AXES)
        # Independent ordinary strike while left_1 remains held.
        if 0.080 <= now_s < 0.130:
            commands[PHYSICAL_AXES.index("right_1")] = 1.0
        elif now_s >= 0.130:
            commands[PHYSICAL_AXES.index("right_1")] = -1.0
        action = overlay.command(
            now_s=now_s,
            base_action=ContactAction(tuple(commands)),
            motor=motor,
        )
        transition = motor.step(action)
        overlay.observe_physics(now_s=now_s + 0.010, motor=motor)
        for event in motor.motor.last_transition.evaluator_events:
            physical_edges.append((event.time_s, event.key, event.event))
        for event in transition.logical_events:
            logical_edges.append((event.time_s, event.key, event.event))

    left = [(t, edge) for t, axis, edge in physical_edges if axis == "left_1"]
    right = [(t, edge) for t, axis, edge in physical_edges if axis == "right_1"]
    assert [edge for _, edge in left] == [KeyEvent.DOWN, KeyEvent.UP]
    assert [edge for _, edge in right] == [KeyEvent.DOWN, KeyEvent.UP]
    assert left[0][0] < right[0][0] < left[1][0]
    assert any(key == "left_1" for _, key, _ in logical_edges)
    assert overlay.completed_ids == (7,)
    assert not overlay.occupied_axes


def test_hold_overrides_base_release_but_not_unoccupied_axis_commands() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(_plan(), motor)
    base = ContactAction(tuple(
        0.75 if axis == "right_2" else -1.0 if axis == "left_1" else 0.0
        for axis in PHYSICAL_AXES
    ))
    command = overlay.command(now_s=0.050, base_action=base, motor=motor)
    values = dict(zip(PHYSICAL_AXES, command.commands))
    assert values["left_1"] == 1.0
    assert values["right_2"] == 0.75


def test_axis_collision_and_early_external_press_are_rejected() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(_plan(), motor)
    with pytest.raises(ValueError, match="occupied"):
        overlay.register(_plan(local_id=8), motor)

    motor2 = Coupled16ContactMotor()
    overlay2 = HoldMotorOverlay()
    future = HoldMotorPlan(
        local_id=9,
        finger=PhysicalFinger("left_1"),
        start_due=TimeWindow(0.200, 0.300),
        release_due=TimeWindow(0.400, 0.500),
        start_command_s=0.150,
        release_command_s=0.400,
        provenance=HoldTimingProvenance.CAUSAL_VISIBLE_INTERVAL,
    )
    overlay2.register(future, motor2)
    positive = ContactAction(tuple(
        1.0 if axis == "left_1" else 0.0 for axis in PHYSICAL_AXES
    ))
    with pytest.raises(ValueError, match="before Hold command"):
        overlay2.command(now_s=0.100, base_action=positive, motor=motor2)


def test_thumb_and_edge_topology_are_explicit_and_stable() -> None:
    motor = Coupled16ContactMotor()
    thumb = HoldMotorOverlay()
    thumb.register(_plan(axis="left_5", contact_count=2), motor)
    action = thumb.command(now_s=0.0, base_action=_neutral(), motor=motor)
    assert action.thumb_modes[0] is ThumbMode.FLAT
    assert not action.edge_enabled[0]

    edge_motor = Coupled16ContactMotor()
    edge = HoldMotorOverlay()
    edge.register(_plan(axis="right_6", contact_count=2), edge_motor)
    action = edge.command(now_s=0.0, base_action=_neutral(), motor=edge_motor)
    assert action.edge_enabled[1]


def test_reset_clears_hold_occupancy_and_is_deterministic() -> None:
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(_plan(), motor)
    first = overlay.command(now_s=0.0, base_action=_neutral(), motor=motor)
    overlay.reset()
    assert not overlay.active
    assert not overlay.occupied_axes
    overlay.register(_plan(), motor)
    second = overlay.command(now_s=0.0, base_action=_neutral(), motor=motor)
    assert first == second


def test_plan_preserves_due_intervals_and_rejects_reversed_commands() -> None:
    plan = _plan()
    assert plan.start_due == TimeWindow(0.035, 0.090)
    assert plan.release_due == TimeWindow(0.180, 0.300)
    with pytest.raises(ValueError, match="release command"):
        HoldMotorPlan(
            local_id=1,
            finger=PhysicalFinger("left_1"),
            start_due=TimeWindow(0.1, 0.2),
            release_due=TimeWindow(0.3, 0.4),
            start_command_s=0.2,
            release_command_s=0.1,
            provenance=HoldTimingProvenance.CAUSAL_VISIBLE_INTERVAL,
        )
