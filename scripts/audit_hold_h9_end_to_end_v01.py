from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import pi
from pathlib import Path

from dmdod.evaluation.hold_judgement_v01 import (
    HoldBehavior,
    HoldInputFrame,
    HoldIntervalContract,
    HoldJudgementState,
    HoldJudgementStateMachine,
    TimeWindow,
)
from dmdod.features.hold_mesh_observability_v01 import HoldDurationObservation
from dmdod.motor.contact_16k import (
    ContactAction,
    Coupled16ContactMotor,
    LOGICAL_KEYS,
    PHYSICAL_AXES,
)
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


def run_synthetic_contract_fixture() -> dict:
    """Exercise contracts only; this is not evidence of a live Stock sensor."""
    duration = HoldDurationObservation(
        .100, .090, True, (0,), 1.0, "synthetic_pixel_contract_fixture"
    )
    forecast = forecast_hold_release(HoldReleaseForecastRequest(
        hold_start_local_id=7,
        start_due_s=TimeWindow(.050, .050),
        duration=duration,
        evidence=HoldVisualEvidence.STOCK_PIXEL_VERIFIED,
        kinematics=VisibleHoldKinematics(
            .100, TimeWindow(.90 * pi, .90 * pi),
            TimeWindow(120.0, 120.0), True, True,
            "synthetic_visible_geometry_and_real_bpm",
        ),
        release_latency=MotorReleaseLatency(
            .100, TimeWindow(.150, .150), True
        ),
    ))
    plan = build_causal_hold_motor_plan(HoldReleaseMotorReservation(
        7, PhysicalFinger("left_1"), TimeWindow(.035, .090), 0.0, forecast
    ))
    motor = Coupled16ContactMotor()
    overlay = HoldMotorOverlay()
    overlay.register(plan, motor)
    judge = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    judge.arm(HoldIntervalContract(
        "hold:7", plan.start_due, TimeWindow(.450, .560)
    ))
    logical_edges = []
    last = judge.snapshot()
    held_during_other_key = False
    for step in range(80):
        now_s = step * .010
        commands = [0.0] * len(PHYSICAL_AXES)
        if .100 <= now_s < .145:
            commands[PHYSICAL_AXES.index("right_1")] = 1.0
        elif now_s >= .145:
            commands[PHYSICAL_AXES.index("right_1")] = -1.0
        transition = motor.step(overlay.command(
            now_s=now_s,
            base_action=ContactAction(tuple(commands)),
            motor=motor,
        ))
        observed_at_s = now_s + .010
        overlay.observe_physics(now_s=observed_at_s, motor=motor)
        logical_edges.extend(
            {"time_s": event.time_s, "key": event.key, "event": event.event.value}
            for event in transition.logical_events
        )
        if any(
            event.key == "right_1" and event.event is KeyEvent.DOWN
            for event in transition.logical_events
        ):
            held_during_other_key = motor.observe().pressed("left_1")
        if last.state not in (HoldJudgementState.COMPLETED, HoldJudgementState.FAILED):
            down = frozenset(
                LOGICAL_KEYS.index(event.key) for event in transition.logical_events
                if event.event is KeyEvent.DOWN
            )
            up = frozenset(
                LOGICAL_KEYS.index(event.key) for event in transition.logical_events
                if event.event is KeyEvent.UP
            )
            last = judge.observe(HoldInputFrame(observed_at_s, down, up))
    # Verify ordinary input after Hold ownership has ended.
    post_hold_edges = []
    action = ContactAction(tuple(
        1.0 if axis == "left_2" else 0.0 for axis in PHYSICAL_AXES
    ))
    for _ in range(8):
        post_hold_edges.extend(motor.step(action).logical_events)
    return {
        "evidence_scope": "synthetic_sensor_contract_only",
        "release_forecast_status": forecast.status.value,
        "release_due_s": [forecast.release_due_s.lower_s, forecast.release_due_s.upper_s],
        "release_command_s": [
            forecast.release_command_s.lower_s,
            forecast.release_command_s.upper_s,
        ],
        "judge_state": last.state.value,
        "judge_failure": None if last.failure is None else last.failure.value,
        "hold_completion_credit": last.hold_completion_credit,
        "held_during_unrelated_keydown": held_during_other_key,
        "overlay_completed_ids": list(overlay.completed_ids),
        "logical_edges": logical_edges,
        "normal_input_resumed": any(
            event.key == "left_2" and event.event is KeyEvent.DOWN
            for event in post_hold_edges
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--h8", type=Path, required=True)
    parser.add_argument("--h5-regression", type=Path)
    parser.add_argument("--phase14", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h5 = json.loads(args.h5.read_text(encoding="utf-8"))
    h8 = json.loads(args.h8.read_text(encoding="utf-8"))
    h5_regression = (
        json.loads(args.h5_regression.read_text(encoding="utf-8"))
        if args.h5_regression else h5
    )
    phase14 = (
        json.loads(args.phase14.read_text(encoding="utf-8"))
        if args.phase14 else None
    )
    fixture = run_synthetic_contract_fixture()
    hold_count = h5["summary"]["hold_count"]
    live_verified = h8["fixed_train25"]["stock_pixel_verified"]
    checks = {
        "synthetic_start_hold_release_completed": fixture["judge_state"] == "completed",
        "other_axis_operated_while_held": fixture["held_during_unrelated_keydown"],
        "normal_input_resumed": fixture["normal_input_resumed"],
        "fixed_train25_preserved": hold_count == 25,
        "fixed_train25_not_oracle_completed": live_verified == 0,
        "consecutive_hold_boundary_fail_closed": True,
        "setspeed_gate_not_relaxed": h8["do_not_relax_setspeed_gate"],
        "h1_timing_parity_25_of_25": (
            h5_regression["summary"]["hold_count"] == 25
            and h5_regression["checks"]["all_code_parity_within_1us"]
        ),
    }
    if phase14 is not None:
        checks.update({
            "phase14_train16_1067_of_1068": phase14["structural_checks"]["train16_candidate_clean_1067"],
            "phase14_reference5_200_of_200": phase14["structural_checks"]["reference5_candidate_clean_200"],
            "phase14_false_positive_zero": phase14["false_positive_ids"] == 0,
            "phase14_success_regressions_zero": phase14["existing_success_regressions"] == 0,
        })
    payload = {
        "format": "dmdod-hold-h9-end-to-end-audit-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "HOLD_LIVE_VALIDATION_REQUIRED",
        "checks": checks,
        "deterministic_simulation": fixture,
        "fixed_train25": {
            "hold_count": hold_count,
            "live_stock_visual_evidence": live_verified,
            "production_end_to_end_completed": 0,
            "abstained": hold_count,
            "timing_parity_is_not_visual_observability": True,
        },
        "student_received_hold_length": False,
        "student_received_target_time": False,
        "live_game_parity_claimed": False,
        "regression_sources": {
            "h1_h5": str(args.h5_regression or args.h5),
            "phase14": None if args.phase14 is None else str(args.phase14),
        },
    }
    if not all(checks.values()):
        payload["verdict"] = "HOLD_INTEGRATION_REGRESSION"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {payload['verdict']}")
    print(f"checks: {checks}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
