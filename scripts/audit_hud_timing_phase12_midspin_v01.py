from __future__ import annotations

"""Phase 12 fixed Train-only Midspin causal trace and regression audit.

Every policy stream and visible HUD frame is captured before evaluator target
times are read.  Target indices, floor ordinals, timing errors, and lead are
posthoc fields only and never enter either Forecast Source.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import atan2, pi, remainder
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor,
    StockCountdownRenderer,
)
from dmdod.features.hud_features import HUD_BPM_REFERENCE
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import ContactTimingHUD263Adapter
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    HUD_OFFSET,
    MOTOR_FEATURES,
    visible_slot,
)
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V15,
)

if __package__:
    from scripts.audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows,
        _lifecycle_counts,
        _serialize,
    )
    from scripts.audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE,
        UNSEEN_END_S,
        verify_selection,
    )
    from scripts.audit_timing_generalization_phase7_v01 import _evaluate
else:
    from audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows,
        _lifecycle_counts,
        _serialize,
    )
    from audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE,
        UNSEEN_END_S,
        verify_selection,
    )
    from audit_timing_generalization_phase7_v01 import _evaluate


FORMAT = "dmdod-hud-timing-phase12-midspin-audit-v0.1"
TRACE_FORMAT = "dmdod-hud-timing-phase12-causal-traces-v0.1"
CONTROL_DT_S = .010
TRACE_WINDOW_S = .080
MOTOR_LEAD_S = .044
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
BASELINE = "baseline-v15-identity-on"
CANDIDATE = "candidate-v15-identity-on-midspin"
ARMS = (BASELINE, CANDIDATE)
TARGETS = {
    "02971 - ALL MY FELLAS - RetroBean.adofai": (64, 69, 74, 78),
    "04432 - ADOFAI Regular Dan - 2nd Dan - yzuio & 5 more.adofai": (114,),
    "03096 - Crossing Notes - 피렝.adofai": (23,),
    "01645 - First town of this Journey remix - Irin.adofai": (
        1, 36, 41, 46, 146, 151, 156,
    ),
}
TARGET_HASHES = {
    "02971 - ALL MY FELLAS - RetroBean.adofai": "e142af91b1784dc20c76fbde5a7320de234dbaebf9822f75e733d0b2db1924d5",
    "04432 - ADOFAI Regular Dan - 2nd Dan - yzuio & 5 more.adofai": "2ada1d0a06b5bd4ed259b9b6a480b15a7e964fa2e8ca25e5f0bc50db306b34c0",
    "03096 - Crossing Notes - 피렝.adofai": "3847e47e00162a3794991eb6c37009d15673b290c2261ffa43a1f476b877aa64",
    "01645 - First town of this Journey remix - Irin.adofai": "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1",
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source(*, recovery: bool) -> HUDV3CompositeForecastSource:
    return HUDV3CompositeForecastSource(
        horizon_s=.080,
        lifecycle_version=LIFECYCLE_V15,
        reconcile_countdown_v14_identity=True,
        midspin_recovery=recovery,
    )


def _floor(hud: tuple[float, ...], slot: int) -> dict:
    item = visible_slot(hud, slot)
    return {
        "relative_slot": slot - CURRENT_SLOT,
        "valid": item[-1] > .5,
        "x_normalized": item[0],
        "y_normalized": item[1],
        "entry_angle_rad": atan2(item[2], item[3]),
        "exit_angle_rad": atan2(item[4], item[5]),
        "midspin": item[6] > .5,
        "multiplanet": item[7] > .5,
        "markers": {
            name: item[index] > .5
            for index, name in enumerate((
                "Twirl", "SetSpeed", "MultiPlanet", "Pause",
                "Checkpoint", "SetFloorIcon",
            ), start=8)
        },
    }


def _visible_row(step: int, hud: tuple[float, ...], cue,
                 previous_phase: float | None) -> tuple[dict, float]:
    phase = atan2(hud[MOTOR_FEATURES], hud[MOTOR_FEATURES + 1])
    if previous_phase is None:
        direction = "UNKNOWN_NO_HISTORY"
        phase_delta = None
    else:
        delta = remainder(phase - previous_phase, 2 * pi)
        phase_delta = delta
        if abs(delta) <= 1e-7:
            direction = "STATIONARY_OR_BOUNDARY"
        elif abs(delta) >= pi / 2 - .05:
            direction = "UNKNOWN_ALIASED_OR_ROLE_SWAP"
        else:
            direction = "POSITIVE" if delta > 0 else "NEGATIVE"
    return ({
        "step": step,
        "observed_at_s": step * CONTROL_DT_S,
        "hud263": list(hud),
        "countdown": {
            "valid": cue.valid,
            "kind": cue.kind.value,
            "digit": cue.digit,
            "rendered_at_s": cue.rendered_at_s,
            "latency_s": cue.latency_s,
        },
        "orbit_phase_rad": phase,
        "orbit_phase_delta_rad": phase_delta,
        "causal_direction_from_past_frame": direction,
        "tile_bpm_visible": HUD_BPM_REFERENCE * (2.0 ** hud[HUD_OFFSET]),
        "real_bpm_visible": HUD_BPM_REFERENCE * (2.0 ** hud[HUD_OFFSET + 1]),
        "floors": {
            "current": _floor(hud, CURRENT_SLOT),
            "next": _floor(hud, CURRENT_SLOT + 1),
            "next2": _floor(hud, CURRENT_SLOT + 2),
        },
    }, phase)


def capture(path: Path, end_s: float) -> tuple[dict[str, list[dict]], list[dict], object]:
    """Capture policy inputs/outputs completely before target truth is read."""
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart, end_s=end_s)
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    sources = {
        BASELINE: _source(recovery=False),
        CANDIDATE: _source(recovery=True),
    }
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S,
        physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    idle = ContactAction((0.0,) * 12)
    rows = {arm: [] for arm in ARMS}
    visible_rows = []
    prior_phase = None
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        hud = ContactTimingHUD263Adapter.encode(step, observation).features
        visible, prior_phase = _visible_row(step, hud, cue, prior_phase)
        visible_rows.append(visible)
        for arm, source in sources.items():
            public, raw = source.observe(step, observation, cue)
            serialized = _serialize(public, raw)
            serialized.update({
                "abstention_reason": raw.abstention_reason,
                "countdown_phase_status": raw.countdown_phase.status,
                "countdown_estimate_status": raw.countdown_estimate.status,
                "v14_status": raw.v14_update.status,
                "v14_pending": [{
                    "id": f"v14:{item.local_id}",
                    "due_s": item.due_s,
                    "confirmed_now": item.confirmed_now,
                } for item in raw.v14_update.pending],
                "v14_new": list(raw.v14_update.new_ids),
                "v14_reused": list(raw.v14_update.reused_ids),
                "v14_canceled": [
                    item.local_id for item in raw.v14_update.canceled
                ],
                "v14_elapsed": [
                    item.local_id for item in raw.v14_update.elapsed
                ],
            })
            rows[arm].append(serialized)
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows, visible_rows, segment


def _event_trace(chart: str, event_index: int, truth_s: float,
                 floor_index: int, visible_rows: list[dict],
                 policy_rows: dict[str, list[dict]], evaluations: dict) -> dict:
    selected = []
    for visible in visible_rows:
        now_s = visible["observed_at_s"]
        if truth_s - TRACE_WINDOW_S - 1e-9 <= now_s < truth_s + 1e-9:
            frame = dict(visible)
            frame["evaluator_posthoc"] = {
                "target_time_s": truth_s,
                "delta_to_target_ms": (truth_s - now_s) * 1000.0,
                "at_or_before_44ms_deadline": now_s <= truth_s - MOTOR_LEAD_S + 1e-9,
            }
            frame["timing"] = {
                arm: policy_rows[arm][visible["step"]]
                for arm in ARMS
            }
            selected.append(frame)
    return {
        "chart": chart,
        "event_index_posthoc": event_index,
        "floor_index_posthoc": floor_index,
        "target_time_s_posthoc": truth_s,
        "motor_deadline_s_posthoc": truth_s - MOTOR_LEAD_S,
        "baseline_evaluation": evaluations[BASELINE]["events"][event_index],
        "candidate_evaluation": evaluations[CANDIDATE]["events"][event_index],
        "frames": selected,
    }


def _aggregate(cases: list[dict], arm: str) -> dict:
    events = [event for case in cases for event in case["evaluations"][arm]["events"]]
    clean = [event for event in events if event["category"] == 1]
    return {
        "charts": len(cases),
        "events": len(events),
        "clean": len(clean),
        "categories": {
            str(index): sum(event["category"] == index for event in events)
            for index in range(1, 6)
        },
        "false_positive_ids": sum(
            case["evaluations"][arm]["metrics"]["false_positive_ids"]
            for case in cases
        ),
        "timing_mae_ms_conditional_on_clean": (
            sum(abs(event["first_forecast_error_ms"]) for event in clean) / len(clean)
            if clean else None
        ),
        "mean_forecast_lead_ms_conditional_on_clean": (
            sum(event["first_forecast_lead_ms"] for event in clean) / len(clean)
            if clean else None
        ),
        "lifecycle": {
            key: sum(case["lifecycle"][arm][key] for case in cases)
            for key in ("new", "reused", "canceled", "elapsed")
        },
    }


def _case(path: Path, end_s: float, target_indices=()) -> tuple[dict, list[dict]]:
    rows, visible_rows, segment = capture(path, end_s)
    # Evaluator boundary: no target timestamp was read until capture completed.
    targets = tuple(
        target for target in segment.targets
        if target.episode_time_s <= end_s + 1e-9
    )
    truths = tuple(float(target.episode_time_s) for target in targets)
    evaluations = {
        arm: _evaluate(_evaluable_rows(rows[arm], end_s), truths)
        for arm in ARMS
    }
    result = {
        "chart": path.name,
        "sha256": _hash(path),
        "end_s": end_s,
        "event_count": len(targets),
        "evaluations": evaluations,
        "lifecycle": {arm: _lifecycle_counts(rows[arm]) for arm in ARMS},
        "baseline_success_regressions": [
            index for index, (baseline, candidate) in enumerate(zip(
                evaluations[BASELINE]["events"],
                evaluations[CANDIDATE]["events"],
                strict=True,
            ))
            if baseline["category"] == 1 and candidate["category"] != 1
        ],
    }
    traces = [
        _event_trace(
            path.name,
            index,
            float(targets[index].episode_time_s),
            targets[index].floor_index,
            visible_rows,
            rows,
            evaluations,
        )
        for index in target_indices
    ]
    return result, traces


def run(train_dir: Path, manifest: dict) -> tuple[dict, dict]:
    unseen_paths = verify_selection(manifest, train_dir)
    for name, expected in TARGET_HASHES.items():
        path = train_dir / name
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"fixed Phase 12 chart missing or changed: {name}")

    target_cases = []
    traces = []
    by_name = {path.name: path for path in unseen_paths}
    for name, indices in TARGETS.items():
        case, chart_traces = _case(by_name[name], UNSEEN_END_S, indices)
        target_cases.append(case)
        traces.extend(chart_traces)

    # Reuse captured target cases and run only the remaining frozen charts.
    target_case_by_name = {case["chart"]: case for case in target_cases}
    unseen_cases = []
    for path in unseen_paths:
        if path.name in target_case_by_name:
            unseen_cases.append(target_case_by_name[path.name])
        else:
            case, _ = _case(path, UNSEEN_END_S)
            unseen_cases.append(case)
    reference_cases = [
        _case(train_dir / name, end_s)[0]
        for name, (_, end_s) in REFERENCE.items()
    ]

    target_pairs = [
        (
            trace["baseline_evaluation"],
            trace["candidate_evaluation"],
        )
        for trace in traces
    ]
    recovered = sum(
        baseline["category"] != 1 and candidate["category"] == 1
        for baseline, candidate in target_pairs
    )
    motor_ready = sum(
        candidate["category"] == 1
        and candidate["first_forecast_lead_ms"] >= MOTOR_LEAD_S * 1000.0 - 1e-6
        for _, candidate in target_pairs
    )
    regressions = sum(
        len(case["baseline_success_regressions"])
        for case in unseen_cases + reference_cases
    )
    unseen = {arm: _aggregate(unseen_cases, arm) for arm in ARMS}
    reference = {arm: _aggregate(reference_cases, arm) for arm in ARMS}
    structural = {
        "target_13_present": len(traces) == 13,
        "target_13_lifecycle_recovered": recovered == 13,
        "false_positive_ids_zero": (
            unseen[CANDIDATE]["false_positive_ids"] == 0
            and reference[CANDIDATE]["false_positive_ids"] == 0
        ),
        "existing_success_regressions_zero": regressions == 0,
        "phase11_unseen_baseline_reproduced": (
            unseen[BASELINE]["clean"] == 1052
            and unseen[BASELINE]["events"] == 1068
        ),
        "phase11_reference_baseline_reproduced": (
            reference[BASELINE]["clean"] == 199
            and reference[BASELINE]["events"] == 200
        ),
        "candidate_unseen_expected_clean": unseen[CANDIDATE]["clean"] == 1065,
        # Phase 12 is allowed to recover a causally licensed Midspin in the
        # reference set.  Safety means no previously-clean event regresses;
        # freezing the aggregate at 199 would incorrectly reject a recovery.
        "candidate_reference_not_regressed": (
            reference[CANDIDATE]["clean"] >= reference[BASELINE]["clean"]
        ),
    }
    verdict = (
        "MIDSPIN_CAUSAL_RECOVERY_PASS"
        if all(structural.values()) and motor_ready == 13 else
        "MIDSPIN_PARTIAL_RECOVERY"
        if all(structural.values()) and recovered > 0 else
        "MIDSPIN_INTEGRATION_BLOCKED"
    )
    result = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "structural_checks": structural,
        "target_13": {
            "recovered_clean_lifecycle": recovered,
            "unresolved_clean_lifecycle": 13 - recovered,
            "motor_ready_44ms": motor_ready,
            "not_motor_ready_44ms": 13 - motor_ready,
            "events": [{
                "chart": trace["chart"],
                "event_index": trace["event_index_posthoc"],
                "target_time_s_posthoc": trace["target_time_s_posthoc"],
                "baseline": trace["baseline_evaluation"],
                "candidate": trace["candidate_evaluation"],
            } for trace in traces],
        },
        "unseen_16": unseen,
        "reference_5": reference,
        "existing_success_regressions": regressions,
        "additional_reference_recoveries": [{
            "chart": case["chart"],
            "event_index": index,
            "candidate": candidate,
        } for case in reference_cases for index, (baseline, candidate) in enumerate(zip(
            case["evaluations"][BASELINE]["events"],
            case["evaluations"][CANDIDATE]["events"],
            strict=True,
        )) if baseline["category"] != 1 and candidate["category"] == 1],
        "cases": unseen_cases,
        "reference_cases": reference_cases,
        "causality": {
            "student_inputs": "current/past HUD263 and visible countdown cue only",
            "midspin_rule": "visible zero-arc Midspin followed by visible ordinary marker-free two-planet floor",
            "target_truth_read_after_capture": True,
            "chart_truth_used_by_policy": False,
            "no_motor_fingering_learning_or_checkpoint_run": True,
        },
    }
    trace_result = {
        "format": TRACE_FORMAT,
        "generated_utc": result["generated_utc"],
        "window_s": TRACE_WINDOW_S,
        "control_dt_s": CONTROL_DT_S,
        "motor_lead_requirement_s": MOTOR_LEAD_S,
        "truth_boundary": "posthoc evaluator fields only; policies were fully captured first",
        "events": traces,
    }
    return result, trace_result


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result, traces = run(args.train_dir, manifest)
    _write_new(args.output, result)
    _write_new(args.trace_output, traces)
    print("verdict:", result["verdict"])
    print("target_13:", result["target_13"] | {"events": "omitted"})
    print("unseen_16:", result["unseen_16"])
    print("reference_5:", result["reference_5"])


if __name__ == "__main__":
    main()
