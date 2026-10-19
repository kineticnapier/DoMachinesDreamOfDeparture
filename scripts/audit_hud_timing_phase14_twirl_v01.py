from __future__ import annotations

"""Phase 14 Train-only Twirl causal recovery and frozen regression audit.

Policy capture is completed before evaluator target times/floor ordinals are
read.  The Twirl policy receives only current/past HUD263 and visible countdown
cues.  Truth association, lead, error, and floor indices are posthoc fields.
"""

import argparse
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


FORMAT = "dmdod-hud-timing-phase14-twirl-audit-v0.1"
TRACE_FORMAT = "dmdod-hud-timing-phase14-causal-traces-v0.1"
CONTROL_DT_S = .010
TRACE_WINDOW_S = .080
MOTOR_LEAD_S = .044
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
BASELINE = "baseline-v15-identity-midspin"
CANDIDATE = "candidate-v15-identity-midspin-twirl"
ARMS = (BASELINE, CANDIDATE)
FIRST_TOWN = "01645 - First town of this Journey remix - Irin.adofai"
TRIPLE = "02521 - Triple Cross - Milkyway.adofai"
TARGETS = {FIRST_TOWN: (174, 201), TRIPLE: (23,)}
TARGET_HASHES = {
    FIRST_TOWN: "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1",
    TRIPLE: "5571481d5a0eee9dee9fe70c81702b3bdf6dfd0491a6904c80d19e22610d86d3",
}
PHASE12_TARGETS = {
    "02971 - ALL MY FELLAS - RetroBean.adofai": (64, 69, 74, 78),
    "04432 - ADOFAI Regular Dan - 2nd Dan - yzuio & 5 more.adofai": (114,),
    "03096 - Crossing Notes - 피렝.adofai": (23,),
    FIRST_TOWN: (1, 36, 41, 46, 146, 151, 156),
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source(*, twirl: bool) -> HUDV3CompositeForecastSource:
    return HUDV3CompositeForecastSource(
        horizon_s=.080,
        lifecycle_version=LIFECYCLE_V15,
        reconcile_countdown_v14_identity=True,
        midspin_recovery=True,
        twirl_recovery=twirl,
    )


def _floor(hud: tuple[float, ...], slot: int) -> dict:
    item = visible_slot(hud, slot)
    return {
        "relative_slot": slot - CURRENT_SLOT,
        "valid": item[-1] > .5,
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
                 prior_phase: float | None) -> tuple[dict, float]:
    phase = atan2(hud[MOTOR_FEATURES], hud[MOTOR_FEATURES + 1])
    delta = None if prior_phase is None else remainder(phase - prior_phase, 2*pi)
    if delta is None:
        direction = "UNKNOWN_NO_HISTORY"
    elif abs(delta) <= 1e-7:
        direction = "STATIONARY_OR_BOUNDARY"
    elif abs(delta) >= pi/2-.05:
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
        "orbit_phase_delta_rad": delta,
        "causal_direction_from_past_frame": direction,
        "tile_bpm_visible": HUD_BPM_REFERENCE * (2.0 ** hud[HUD_OFFSET]),
        "floors": [
            _floor(hud, slot) for slot in range(
                CURRENT_SLOT, CURRENT_SLOT + 5
            )
        ],
    }, phase)


def capture(path: Path, end_s: float):
    """Capture both policy streams before exposing evaluator truth."""
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
        BASELINE: _source(twirl=False),
        CANDIDATE: _source(twirl=True),
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


def _aggregate(cases: list[dict], arm: str) -> dict:
    events = [event for case in cases for event in case["evaluations"][arm]["events"]]
    clean = [event for event in events if event["category"] == 1]
    return {
        "charts": len(cases),
        "events": len(events),
        "clean": len(clean),
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


def _preceding_twirl_floor(segment, target_floor: int) -> int:
    for floor_index in range(target_floor - 1, -1, -1):
        floor = segment.chart.floors[floor_index]
        if "Twirl" in floor.event_markers:
            return floor_index
        if not floor.midspin:
            break
    raise ValueError("target has no adjacent visible Twirl chain")


def _trace(chart: str, index: int, target, visible_rows, rows,
           evaluations, segment) -> dict:
    truth_s = float(target.episode_time_s)
    target_floor = target.floor_index
    twirl_floor = _preceding_twirl_floor(segment, target_floor)
    selected = []
    marker_first = None
    target_floor_first = None
    for visible in visible_rows:
        now_s = visible["observed_at_s"]
        chart_time = segment.chart_time_from_episode(now_s)
        current_floor = max(
            floor.index for floor in segment.chart.floors
            if floor.target_time_s <= chart_time + 1e-9
        )
        twirl_relative = twirl_floor - current_floor
        target_relative = target_floor - current_floor
        if 0 <= twirl_relative <= FLOORS.ahead_floors:
            slot = visible_slot(tuple(visible["hud263"]), CURRENT_SLOT + twirl_relative)
            if slot[-1] > .5 and slot[8] > .5 and marker_first is None:
                marker_first = now_s
        if 0 <= target_relative <= FLOORS.ahead_floors:
            slot = visible_slot(tuple(visible["hud263"]), CURRENT_SLOT + target_relative)
            if slot[-1] > .5 and target_floor_first is None:
                target_floor_first = now_s
        if truth_s - TRACE_WINDOW_S - 1e-9 <= now_s < truth_s + 1e-9:
            frame = dict(visible)
            frame["evaluator_posthoc"] = {
                "target_time_s": truth_s,
                "target_floor_index": target_floor,
                "current_floor_index": current_floor,
                "twirl_floor_index": twirl_floor,
                "delta_to_target_ms": (truth_s - now_s) * 1000.0,
                "at_or_before_44ms_deadline": (
                    now_s <= truth_s - MOTOR_LEAD_S + 1e-9
                ),
            }
            frame["timing"] = {arm: rows[arm][visible["step"]] for arm in ARMS}
            selected.append(frame)
    return {
        "chart": chart,
        "event_index_posthoc": index,
        "target_time_s_posthoc": truth_s,
        "target_floor_index_posthoc": target_floor,
        "twirl_floor_index_posthoc": twirl_floor,
        "motor_deadline_s_posthoc": truth_s - MOTOR_LEAD_S,
        "twirl_marker_first_visible_s": marker_first,
        "post_twirl_target_floor_first_visible_s": target_floor_first,
        "marker_lead_ms": None if marker_first is None else (truth_s-marker_first)*1000,
        "target_floor_visible_lead_ms": (
            None if target_floor_first is None else (truth_s-target_floor_first)*1000
        ),
        "baseline_evaluation": evaluations[BASELINE]["events"][index],
        "candidate_evaluation": evaluations[CANDIDATE]["events"][index],
        "frames": selected,
    }


def _case(path: Path, end_s: float, target_indices=()):
    rows, visible_rows, segment = capture(path, end_s)
    targets = tuple(
        target for target in segment.targets
        if target.episode_time_s <= end_s + 1e-9
    )
    truths = tuple(float(target.episode_time_s) for target in targets)
    evaluations = {
        arm: _evaluate(_evaluable_rows(rows[arm], end_s), truths)
        for arm in ARMS
    }
    regressions = [
        index for index, (baseline, candidate) in enumerate(zip(
            evaluations[BASELINE]["events"],
            evaluations[CANDIDATE]["events"], strict=True,
        ))
        if baseline["category"] == 1 and candidate["category"] != 1
    ]
    case = {
        "chart": path.name,
        "sha256": _hash(path),
        "end_s": end_s,
        "event_count": len(targets),
        "evaluations": evaluations,
        "lifecycle": {arm: _lifecycle_counts(rows[arm]) for arm in ARMS},
        "baseline_success_regressions": regressions,
    }
    traces = [
        _trace(path.name, index, targets[index], visible_rows, rows,
               evaluations, segment)
        for index in target_indices
    ]
    return case, traces


def run(train_dir: Path, manifest: dict, *, target_only: bool = False):
    unseen_paths = verify_selection(manifest, train_dir)
    for name, expected in TARGET_HASHES.items():
        path = train_dir / name
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"fixed Phase 14 chart missing or changed: {name}")

    first_case, first_traces = _case(
        train_dir / FIRST_TOWN, UNSEEN_END_S, TARGETS[FIRST_TOWN]
    )
    triple_case, triple_traces = _case(
        train_dir / TRIPLE, REFERENCE[TRIPLE][1], TARGETS[TRIPLE]
    )
    traces = first_traces + triple_traces
    target_cases = [first_case, triple_case]

    target_recovered = sum(
        trace["baseline_evaluation"]["category"] != 1
        and trace["candidate_evaluation"]["category"] == 1
        for trace in traces
    )
    target_candidate_clean = sum(
        trace["candidate_evaluation"]["category"] == 1 for trace in traces
    )
    target_motor_ready = sum(
        trace["candidate_evaluation"]["category"] == 1
        and trace["candidate_evaluation"]["first_forecast_lead_ms"]
        >= MOTOR_LEAD_S * 1000 - 1e-6
        for trace in traces
    )
    predeadline_visible = all(
        trace["twirl_marker_first_visible_s"] is not None
        and trace["twirl_marker_first_visible_s"]
        <= trace["motor_deadline_s_posthoc"] + 1e-9
        for trace in traces
    )

    unseen_cases = []
    reference_cases = []
    if not target_only:
        by_name = {path.name: path for path in unseen_paths}
        for path in unseen_paths:
            if path.name == FIRST_TOWN:
                unseen_cases.append(first_case)
            else:
                unseen_cases.append(_case(path, UNSEEN_END_S)[0])
        for name, (_, end_s) in REFERENCE.items():
            if name == TRIPLE:
                reference_cases.append(triple_case)
            else:
                reference_cases.append(_case(train_dir / name, end_s)[0])

    all_cases = unseen_cases + reference_cases if not target_only else target_cases
    regressions = sum(len(case["baseline_success_regressions"]) for case in all_cases)
    false_positive = sum(
        case["evaluations"][CANDIDATE]["metrics"]["false_positive_ids"]
        for case in all_cases
    )
    checks = {
        "fixed_3_present": len(traces) == 3,
        "twirl_marker_visible_before_44ms_deadline": predeadline_visible,
        "fixed_3_candidate_clean": target_candidate_clean == 3,
        "fixed_3_motor_ready": target_motor_ready == 3,
        "false_positive_ids_zero": false_positive == 0,
        "existing_success_regressions_zero": regressions == 0,
    }
    if not target_only:
        unseen = {arm: _aggregate(unseen_cases, arm) for arm in ARMS}
        reference = {arm: _aggregate(reference_cases, arm) for arm in ARMS}
        unseen_by_name = {case["chart"]: case for case in unseen_cases}
        phase12_events = [
            unseen_by_name[name]["evaluations"][CANDIDATE]["events"][index]
            for name, indices in PHASE12_TARGETS.items()
            for index in indices
        ]
        phase12_clean = sum(event["category"] == 1 for event in phase12_events)
        phase12_motor_ready = sum(
            event["category"] == 1
            and event["first_forecast_lead_ms"] >= MOTOR_LEAD_S*1000-1e-6
            for event in phase12_events
        )
        checks.update({
            # The remaining non-clean Train16 event is the explicitly
            # out-of-scope ヒアソビ Checkpoint event 54.
            "train16_candidate_clean_1067": unseen[CANDIDATE]["clean"] == 1067,
            "reference5_candidate_clean_200": reference[CANDIDATE]["clean"] == 200,
            "phase12_fixed13_clean_13": phase12_clean == 13,
            "phase12_fixed13_motor_ready_at_least_9": phase12_motor_ready >= 9,
            "phase12_baseline_reproduced_train16": unseen[BASELINE]["clean"] == 1065,
            "phase12_baseline_reproduced_reference5": reference[BASELINE]["clean"] == 200,
        })
    else:
        unseen = None
        reference = None
        phase12_clean = None
        phase12_motor_ready = None

    verdict = (
        "TWIRL_CAUSAL_RECOVERY_PASS" if all(checks.values()) else
        "TWIRL_PARTIAL_RECOVERY" if target_recovered > 0 and not regressions
        else "TWIRL_INTEGRATION_BLOCKED"
    )
    now = datetime.now(timezone.utc).isoformat()
    result = {
        "format": FORMAT,
        "generated_utc": now,
        "verdict": verdict,
        "scope": "fixed-3-only" if target_only else "fixed-3-plus-train16-reference5",
        "structural_checks": checks,
        "fixed_3": {
            "new_clean_recoveries": target_recovered,
            "candidate_clean": target_candidate_clean,
            "unresolved": 3 - target_candidate_clean,
            "motor_ready_44ms": target_motor_ready,
            "not_motor_ready_44ms": 3 - target_motor_ready,
            "events": [{
                "chart": trace["chart"],
                "event_index": trace["event_index_posthoc"],
                "marker_lead_ms": trace["marker_lead_ms"],
                "target_floor_visible_lead_ms": trace["target_floor_visible_lead_ms"],
                "baseline": trace["baseline_evaluation"],
                "candidate": trace["candidate_evaluation"],
            } for trace in traces],
        },
        "train16": unseen,
        "reference5": reference,
        "phase12_fixed13": {
            "clean": phase12_clean,
            "motor_ready_44ms": phase12_motor_ready,
        },
        "false_positive_ids": false_positive,
        "existing_success_regressions": regressions,
        "cases": all_cases,
        "causality": {
            "student_inputs": "current/past HUD263 and visible countdown cue only",
            "twirl_rule": "single visible Twirl; direction established by unaliased prior 10ms phase; post-marker visible angle endpoints swapped",
            "marker_counterfactual": "Twirl-present and Twirl-absent worlds differ at the visible marker dimension before the deadline",
            "future_setspeed_gate_relaxed": False,
            "target_truth_read_after_capture": True,
            "chart_truth_used_by_policy": False,
            "no_motor_fingering_learning_or_checkpoint_run": True,
        },
    }
    trace_result = {
        "format": TRACE_FORMAT,
        "generated_utc": now,
        "window_s": TRACE_WINDOW_S,
        "control_dt_s": CONTROL_DT_S,
        "motor_lead_requirement_s": MOTOR_LEAD_S,
        "truth_boundary": "posthoc evaluator fields only; both policies were captured first",
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
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result, traces = run(args.train_dir, manifest, target_only=args.target_only)
    _write_new(args.output, result)
    _write_new(args.trace_output, traces)
    print("verdict:", result["verdict"])
    print("fixed_3:", result["fixed_3"] | {"events": "omitted"})
    print("checks:", result["structural_checks"])


if __name__ == "__main__":
    main()
