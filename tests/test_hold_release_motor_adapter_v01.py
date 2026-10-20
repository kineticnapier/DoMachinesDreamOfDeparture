from __future__ import annotations

from dataclasses import replace
from math import pi

import pytest

from dmdod.evaluation.hold_judgement_v01 import (
    HoldBehavior,
    HoldInputFrame,
    HoldIntervalContract,
    HoldJudgementState,
    HoldJudgementStateMachine,
    TimeWindow,
)
from dmdod.features.hold_mesh_observability_v01 import HoldDurationObservation
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, LOGICAL_KEYS, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.hold_motor_overlay_v01 import HoldMotorOverlay
from dmdod.planning.hold_release_forecast_v01 import (
    HoldReleaseForecastRequest,
    HoldVisualEvidence,
    MotorReleaseLatency,
    VisibleHoldKinematics,
    forecast_hold_release,
)
from dmdod.planning.hold_release_motor_adapter_v01 import (
    HoldReleaseMotorReservation,
    build_causal_hold_motor_plan,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import PhysicalFinger


def _forecast(*, evidence=HoldVisualEvidence.STOCK_PIXEL_VERIFIED):
    # Synthetic sensor-contract fixture: release due=.500, command=.350.
    duration = HoldDurationObservation(
        observed_at_s=.100,
        rendered_at_s=.090,
        candidates=(0,),
        valid=True,
        confidence=1.0,
        reason="synthetic_pixel_contract_fixture",
    )
    return forecast_hold_release(HoldReleaseForecastRequest(
        hold_start_local_id=7,
        start_due_s=TimeWindow(.050, .050),
        duration=duration,
        evidence=evidence,
        kinematics=VisibleHoldKinematics(
            observed_at_s=.100,
            base_angle_rad=TimeWindow(.90 * pi, .90 * pi),
            real_bpm=TimeWindow(120.0, 120.0),
            valid=True,
            future_transition_free=True,
            source="synthetic_visible_geometry_and_real_bpm",
        ),
        release_latency=MotorReleaseLatency(
            observed_at_s=.100,
            latency_s=TimeWindow(.150, .150),
            valid=True,
        ),
    ))


def _reservation(forecast=None):
    return HoldReleaseMotorReservation(
        local_id=7,
        finger=PhysicalFinger("left_1"),
        start_due_s=TimeWindow(.035, .090),
        start_command_s=0.0,
        release=_forecast() if forecast is None else forecast,
    )


def _neutral():
    return ContactAction((0.0,) * len(PHYSICAL_AXES))


def test_adapter_rejects_proxy_non_point_and_identity_mismatch():
    with pytest.raises(ValueError, match="not Motor-ready"):
        build_causal_hold_motor_plan(_reservation(_forecast(
            evidence=HoldVisualEvidence.DIAGNOSTIC_PROXY
        )))

    ready = _forecast()
    with pytest.raises(ValueError, match="local ID"):
        build_causal_hold_motor_plan(replace(_reservation(ready), local_id=8))

    interval = replace(
        ready,
        release_command_s=TimeWindow(.350, .355),
        release_due_s=TimeWindow(.500, .505),
    )
    with pytest.raises(ValueError, match="non-point"):
        build_causal_hold_motor_plan(_reservation(interval))


def test_causal_release_runs_full_deterministic_hold_chain_and_returns_to_normal():
    plan = build_causal_hold_motor_plan(_reservation())
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(plan, motor)
    judge = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    judge.arm(HoldIntervalContract(
        hold_id="hold:7",
        start_window=plan.start_due,
        # Stock judgement tolerance is evaluator-owned; this broad synthetic
        # fixture checks plumbing, not live-game parity.
        release_window=TimeWindow(.450, .560),
    ))

    left_edges = []
    right_edges = []
    saw_left_held_during_right_down = False
    final = None
    for step in range(80):
        now_s = step * .010
        values = [0.0] * len(PHYSICAL_AXES)
        if .100 <= now_s < .145:
            values[PHYSICAL_AXES.index("right_1")] = 1.0
        elif now_s >= .145:
            values[PHYSICAL_AXES.index("right_1")] = -1.0
        action = overlay.command(
            now_s=now_s,
            base_action=ContactAction(tuple(values)),
            motor=motor,
        )
        transition = motor.step(action)
        observed_at = now_s + .010
        overlay.observe_physics(now_s=observed_at, motor=motor)

        down = frozenset(
            LOGICAL_KEYS.index(event.key)
            for event in transition.logical_events
            if event.event is KeyEvent.DOWN
        )
        up = frozenset(
            LOGICAL_KEYS.index(event.key)
            for event in transition.logical_events
            if event.event is KeyEvent.UP
        )
        for event in transition.logical_events:
            if event.key == "left_1":
                left_edges.append((event.time_s, event.event))
            if event.key == "right_1":
                right_edges.append((event.time_s, event.event))
        if any(event.event is KeyEvent.DOWN for event in transition.logical_events if event.key == "right_1"):
            saw_left_held_during_right_down = motor.observe().pressed("left_1")
        if judge.snapshot().state not in (
            HoldJudgementState.COMPLETED,
            HoldJudgementState.FAILED,
        ):
            final = judge.observe(HoldInputFrame(observed_at, down, up))

    assert final is not None and final.state is HoldJudgementState.COMPLETED
    assert final.ordinary_hit_credit == 0
    assert [edge for _, edge in left_edges] == [KeyEvent.DOWN, KeyEvent.UP]
    assert [edge for _, edge in right_edges] == [KeyEvent.DOWN, KeyEvent.UP]
    assert left_edges[0][0] < right_edges[0][0] < left_edges[1][0]
    assert saw_left_held_during_right_down
    assert overlay.completed_ids == (7,)
    assert not overlay.occupied_axes

    # Normal input resumes after Hold ownership is gone.
    values = tuple(1.0 if axis == "left_2" else 0.0 for axis in PHYSICAL_AXES)
    seen = []
    for _ in range(8):
        seen.extend(motor.step(ContactAction(values)).logical_events)
    assert any(event.key == "left_2" and event.event is KeyEvent.DOWN for event in seen)


def test_consecutive_hold_boundary_remains_fail_closed():
    judge = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    with pytest.raises(NotImplementedError, match="consecutive Hold"):
        judge.arm(HoldIntervalContract(
            hold_id="hold:next",
            start_window=TimeWindow(.1, .2),
            release_window=TimeWindow(.3, .4),
            continues_to_hold=True,
        ))
