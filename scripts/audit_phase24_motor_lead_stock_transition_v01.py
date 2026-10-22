from __future__ import annotations

"""Phase 24 research audit; never mutates production Timing or Motor."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.stock_floor_state_machine import (
    StockFloorStateMachine,
    StockRunMode,
    targets_from_playable_segment,
)
from dmdod.adofai.stock_frame_state_machine import (
    StockFrameStateMachine,
    StockQueuedKeyDown,
    first_frame_strictly_after,
)
from dmdod.adofai.stock_judgement_reference import stock_angle_boundaries
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.planning.motor_lead_feasibility_v01 import (
    EvidenceState,
    assess_lead,
    calibrated_nominal_lead_s,
)
from scripts.audit_phase20b_stock_floor_state_v01 import (
    _edge_key,
    _local_ids,
    _logical_keydowns,
)


FORMAT = "dmdod-phase24-motor-lead-stock-transition-v0.1"
CHART = "01645 - First town of this Journey remix - Irin.adofai"
CHART_SHA256 = "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1"
LEADS_S = (0.044, 0.030, 0.020, 0.010, 0.0)
CONTROL_DT_S = 0.010
PHYSICS_DT_S = 0.001
CALIBRATED_INNER_PRESS_S = 0.039
COLD_AXIS_LATENCIES_S = (0.039, 0.041, 0.043)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _window(target, *, pitch: float) -> tuple[float, float]:
    boundary = stock_angle_boundaries(target.bpm, pitch=pitch).counted_deg
    radius = boundary / (3.0 * target.bpm * pitch)
    return target.episode_time_s - radius, target.episode_time_s + radius


def _run_phase(
    targets,
    base: list[dict],
    *,
    phase_s: float,
    candidate_s: float,
    mode: StockRunMode,
) -> dict:
    anchored = mode is StockRunMode.FAIL
    machine = StockFloorStateMachine(
        targets[36:] if anchored else targets,
        run_mode=mode,
        initial_floor=targets[35].floor_index if anchored else 0,
    )
    frames = StockFrameStateMachine(
        machine, frame_dt_s=CONTROL_DT_S, phase_s=phase_s
    )
    edges = [*base, {"input_id": "candidate:event37", "time_s": candidate_s, "contact": "audit"}]
    for edge in edges:
        if anchored and float(edge["time_s"]) <= 3.174:
            continue
        frames.queue_keydown(
            StockQueuedKeyDown(
                str(edge["input_id"]), float(edge["time_s"]), str(edge["contact"])
            )
        )
    rows: list[dict] = []
    for frame_index in range(0 if not anchored else 317, 538):
        update = frames.update(frame_index)
        for transition in update.transitions:
            input_id = next(
                (
                    item.keydown.input_id
                    for item in update.processed_keydowns
                    if item.input_transition is transition
                ),
                None,
            )
            rows.append(
                {
                    "frame_time_s": update.frame_time_s,
                    "input_id": input_id,
                    "kind": transition.kind.value,
                    "target_ordinal": transition.target_ordinal,
                    "judgement": None
                    if transition.judgement is None
                    else getattr(transition.judgement, "value", transition.judgement),
                    "automatic_miss": transition.automatic_miss,
                    "failed": transition.failed,
                }
            )
        if update.failed:
            break
    candidate = next((row for row in rows if row["input_id"] == "candidate:event37"), None)
    auto36 = next(
        (row for row in rows if row["target_ordinal"] == 36 and row["automatic_miss"]),
        None,
    )
    pairs = [
        row
        for row in rows
        if isinstance(row["input_id"], str) and row["input_id"].startswith("local:")
        and row["target_ordinal"] is not None
    ]
    selected = {
        str(ordinal): [row for row in rows if row["target_ordinal"] == ordinal]
        for ordinal in (36, 37, 42, 47, 89)
    }
    return {
        "phase_ms": phase_s * 1000.0,
        "candidate_time_s": candidate_s,
        "mode": mode.value,
        "event36_auto": auto36,
        "candidate_transition": candidate,
        "base_input_mismatch_count": sum(
            int(row["input_id"].split(":", 1)[1]) != row["target_ordinal"]
            for row in pairs
        ),
        "selected_events": selected,
        "failed": machine.failed,
    }


def run(*, phase16_path: Path, phase23_path: Path, train_dir: Path) -> dict:
    chart_path = train_dir / CHART
    if _hash(chart_path) != CHART_SHA256:
        raise ValueError("fixed First town chart missing or changed")
    phase16 = json.loads(phase16_path.read_text(encoding="utf-8"))
    phase23 = json.loads(phase23_path.read_text(encoding="utf-8"))
    arm = phase16["cases"][0]["arms"]["v15_midspin_twirl_checkpoint_on"]
    diagnostic = arm["diagnostic_window_posthoc"]
    segment = build_playable_segment(compile_adofai(load_adofai(chart_path)), end_s=6.0)
    target37 = segment.targets[37]
    stock_window = _window(target37, pitch=segment.pitch_ratio)
    calibration = calibrate_n_key_press_lead(
        12, control_dt_s=CONTROL_DT_S, physics_dt_s=PHYSICS_DT_S
    )
    measured_latencies = dict(calibration.key_latencies_s)

    lead_cases: dict[str, list[dict]] = {}
    scenarios = (
        ("existing_point_forecast_3_190", 3.190, EvidenceState.YES),
        ("hypothetical_exact_due_at_bpm_frame_3_180", 3.180, EvidenceState.UNVERIFIED),
    )
    for name, observed_at, observable in scenarios:
        lead_cases[name] = [
            {
                **asdict(
                    assess_lead(
                        observed_at_s=observed_at,
                        due_s_posthoc=target37.episode_time_s,
                        lead_s=lead,
                        due_observability=observable,
                        stock_window_s=stock_window,
                        # The policy abstained before selecting Event37's axis.
                        # Use the slowest measured cold axis as the primary
                        # projection; retain every measured axis below.
                        cold_press_latency_s=max(measured_latencies.values()),
                        control_dt_s=CONTROL_DT_S,
                        reservation_conflict_clear=EvidenceState.UNVERIFIED,
                    )
                ),
                "due_observability": observable.value,
                "reservation_conflict_clear": EvidenceState.UNVERIFIED.value,
                "physical_scope": "conditional cold/free-axis projection; actual Event37 allocation was abstained and not replayed",
                "cold_axis_sensitivity": [
                    {
                        "latency_ms": latency * 1000.0,
                        "keydown_s": asdict(
                            assess_lead(
                                observed_at_s=observed_at,
                                due_s_posthoc=target37.episode_time_s,
                                lead_s=lead,
                                due_observability=observable,
                                stock_window_s=stock_window,
                                cold_press_latency_s=latency,
                                control_dt_s=CONTROL_DT_S,
                            )
                        )["cold_keydown_s"],
                    }
                    for latency in COLD_AXIS_LATENCIES_S
                ],
            }
            for lead in LEADS_S
        ]

    keydowns = _logical_keydowns(diagnostic, end_s=5.36)
    local_ids = _local_ids(arm, diagnostic)
    base = [
        {
            **row,
            "input_id": f"local:{local_ids[_edge_key(row['time_s'], row['contact'])]}",
        }
        for row in keydowns
    ]
    targets = targets_from_playable_segment(segment)
    phase_values = tuple(value / 1000.0 for value in range(10))
    candidate_values = (3.249, 3.250, 3.251, 3.252, 3.259)
    nofail = [
        _run_phase(targets, base, phase_s=phase, candidate_s=candidate, mode=StockRunMode.NO_FAIL)
        for phase in phase_values
        for candidate in candidate_values
    ]
    fail = [
        _run_phase(targets, base, phase_s=phase, candidate_s=3.251, mode=StockRunMode.FAIL)
        for phase in phase_values
    ]
    boundary = stock_angle_boundaries(targets[36].approach_bpm, pitch=targets[36].pitch).counted_deg
    nofail_crossing = targets[36].due_s + boundary * 1.01 / (
        3.0 * targets[36].approach_bpm * targets[36].pitch
    )
    phase_auto = [
        {
            "phase_ms": phase * 1000.0,
            "first_frame_after_crossing_s": first_frame_strictly_after(
                nofail_crossing, frame_dt_s=CONTROL_DT_S, phase_s=phase
            )[1],
        }
        for phase in phase_values
    ]

    phase23_cf = phase23["observability"]["counterfactual"]
    verdicts = [
        "LEAD_44MS_JUSTIFIED",
        "SHORTER_LEAD_TEMPORALLY_FEASIBLE",
        "SHORTER_LEAD_UNSAFE",
        "NOFAIL_DELAYED_HIT_CONDITIONALLY_VALID",
        "STOCK_RUNTIME_PARITY_UNVERIFIED",
        "NO_SAFE_INTERVENTION_ESTABLISHED",
    ]
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": verdicts,
        "scope": {
            "chart": CHART,
            "chart_sha256": CHART_SHA256,
            "production_changed": False,
            "validation_or_final_used": False,
        },
        "lead_origin": {
            "calibration_key_count": calibration.key_count,
            "individual_cold_axis_ms": {
                key: value * 1000.0 for key, value in calibration.key_latencies_s
            },
            "inner_pair_cold_press_latency_ms": CALIBRATED_INNER_PRESS_S * 1000.0,
            "observed_cold_axis_range_ms": [
                min(measured_latencies.values()) * 1000.0,
                max(measured_latencies.values()) * 1000.0,
            ],
            "control_tick_ms": CONTROL_DT_S * 1000.0,
            "physics_tick_ms": PHYSICS_DT_S * 1000.0,
            "half_control_tick_allowance_ms": CONTROL_DT_S * 500.0,
            "reconstructed_nominal_lead_ms": calibrated_nominal_lead_s(
                cold_press_latency_s=CALIBRATED_INNER_PRESS_S,
                control_dt_s=CONTROL_DT_S,
            ) * 1000.0,
            "classification": "empirical inner-pair cold-press latency plus average half-control-tick allowance; nominal center-first calibration, not a theoretical lower bound or worst-case all-axis proof",
            "evidence": [
                "src/dmdod/motor/capacity.py: calibrate_n_key_press_lead",
                "src/dmdod/legacy/four_key/calibration.py: same convention",
                "tests/test_n_key_capacity.py: every 4-key cold latency is 39ms and lead is 44ms",
                "src/dmdod/planning/timing_fingering_motor_policy_v02.py: frozen NORMAL_LEAD_S=.044",
            ],
            "unresolved": [
                "no worst-case control-phase proof (half tick is an average allowance)",
                "no Event37 physical snapshot or selected axis because policy abstained",
                "12-axis outer cold latencies reach 43ms; the 44ms global policy constant is still calibrated from the 39ms inner pair",
                "no live-hardware measurement; all latencies are simulator measurements",
            ],
        },
        "event37": {
            "due_s_posthoc": target37.episode_time_s,
            "stock_counted_window_s_posthoc": list(stock_window),
            "first_numeric_bpm_frame_s": 3.180,
            "existing_point_forecast_s": 3.190,
            "counterfactual_pre_44ms": {
                "compared_frames": phase23_cf["compared_frames"],
                "last_compared_at_s": phase23_cf["last_compared_at_s"],
                "maximum_hud_difference": phase23_cf["maximum_hud_difference"],
                "alternate_due_s_posthoc": phase23_cf["alternate_due_s_posthoc"],
                "status": phase23_cf["status"],
            },
            "lead_cases": lead_cases,
        },
        "nofail_delayed_candidate": {
            "candidate_is_fixed_safe_time": False,
            "nofail_threshold_crossing_s_posthoc": nofail_crossing,
            "phase_auto_transition": phase_auto,
            "phase_and_microtime_runs": nofail,
            "normal_fail_runs_at_3_251": fail,
            "visible_counter_limit": "automatic FailMiss counter visibility remains unverified; same-capture ordering cannot be reconstructed from a cumulative vector",
            "causal_propagation": "Event37-only delayed input leaves later abstentions at 42 and 47; Event89 still auto-misses in every audited 3.251 NoFail phase",
        },
        "live_stock_plan": {
            "status": "not_executed",
            "steps": [
                "SHA256 the installed Assembly-CSharp.dll and compare with audited 0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD",
                "Record only the Event36/37 interval in NoFail ON and OFF at >=120fps while capturing the full judgement-counter region and an input timestamp trace",
                "Repeat taps bracketing the auto-transition; retain raw frame timestamps and input-clock uncertainty rather than assigning 3.251s as ground truth",
                "Align episode time with two visible anchors, report capture/input latency intervals, and classify target/category only after recording",
                "If frame-order proof is required, collect evaluator-only Unity frameCount logs for CheckPostHoldFail, input dispatch, AddHit, and visible counter update; never feed them to Student",
            ],
            "minimum_outputs": [
                "installed DLL hash and game build identifier",
                "NoFail/Fail setting and FPS/frame-time trace",
                "raw video around Event36/37 with counter labels visible",
                "timestamped KeyDown log and synchronization uncertainty",
                "observed auto-FailMiss visibility and game-stop frame",
            ],
        },
        "student_boundary": {
            "truth_use": "due, target ordinals, Stock windows and crossings are posthoc evaluator labels only",
            "production_modules_modified": [],
            "oracle_exact_due_at_3_180": "explicit hypothetical; not established as HUD-observable",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase16", type=Path, default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"))
    parser.add_argument("--phase23", type=Path, default=Path("reports/phase23-predictive-timing-observability-20261022-r3.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/DMDOD-v140-tuf-p11-p20/Train"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run(phase16_path=args.phase16, phase23_path=args.phase23, train_dir=args.train_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": result["verdicts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
