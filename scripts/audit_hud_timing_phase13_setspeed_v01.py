from __future__ import annotations

"""Phase 13: future-SetSpeed causal observability, without predictor changes."""

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import floor
from pathlib import Path

from dmdod.adofai.chart import AdoFaiChart, load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor,
    StockCountdownRenderer,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import ContactTimingHUD263Adapter
from dmdod.planning.hud_timing_tracker_midspin_v16 import (
    normalize_visible_midspin_transition,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    HUD_OFFSET,
    visible_slot,
)

if __package__:
    from scripts import audit_hud_timing_phase12_midspin_v01 as phase12
else:
    import audit_hud_timing_phase12_midspin_v01 as phase12


FORMAT = "dmdod-hud-timing-phase13-setspeed-counterfactual-v0.1"
TRACE_FORMAT = "dmdod-hud-timing-phase13-setspeed-causal-traces-v0.1"
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
CONTROL_DT_S = .010
MOTOR_LEAD_S = .044
TRACE_WINDOW_S = .080
INTERVENTION_FACTOR = .125  # Reuse the fixed Phase-4 intervention.
SAME_HUD_TOLERANCE = 0.0
PRIMARY = (
    ("02971 - ALL MY FELLAS - RetroBean.adofai", 64),
    ("01645 - First town of this Journey remix - Irin.adofai", 1),
    ("01645 - First town of this Journey remix - Irin.adofai", 36),
    ("01645 - First town of this Journey remix - Irin.adofai", 146),
)
REFERENCE = ("02521 - Triple Cross - Milkyway.adofai", 23)
EXPECTED_HASHES = {
    **phase12.TARGET_HASHES,
    "02521 - Triple Cross - Milkyway.adofai": (
        "5571481d5a0eee9dee9fe70c81702b3bdf6dfd0491a6904c80d19e22610d86d3"
    ),
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _intervene_numeric_setspeed(
    chart: AdoFaiChart, *, floor_index: int,
) -> tuple[AdoFaiChart, dict]:
    candidates = [
        (index, action) for index, action in enumerate(chart.actions)
        if action.floor == floor_index
        and action.active
        and action.event_type == "SetSpeed"
    ]
    if not candidates:
        raise ValueError("fixed target has no preceding SetSpeed action")
    action_index, original = candidates[-1]
    if (original.speed_type or "").lower() == "bpm":
        old = original.beats_per_minute
        if old is None or old <= 0:
            raise ValueError("invalid fixed SetSpeed BPM")
        changed = replace(
            original, beats_per_minute=old * INTERVENTION_FACTOR
        )
        field = "beatsPerMinute"
    else:
        old = original.bpm_multiplier
        if old is None or old <= 0:
            raise ValueError("invalid fixed SetSpeed multiplier")
        changed = replace(
            original, bpm_multiplier=old * INTERVENTION_FACTOR
        )
        field = "bpmMultiplier"
    actions = list(chart.actions)
    actions[action_index] = changed
    counterfactual = replace(chart, actions=tuple(actions))
    changed_indices = [
        index for index, (left, right) in enumerate(zip(
            chart.actions, counterfactual.actions, strict=True
        )) if left != right
    ]
    if changed_indices != [action_index]:
        raise RuntimeError("counterfactual changed more than one action")
    return counterfactual, {
        "action_source_index": original.source_index,
        "action_sequence_index": action_index,
        "action_floor_posthoc": floor_index,
        "modified_field": field,
        "original_value_posthoc": old,
        "counterfactual_value_posthoc": old * INTERVENTION_FACTOR,
        "factor_predeclared": INTERVENTION_FACTOR,
        "visible_marker_preserved": True,
        "only_one_action_changed": True,
    }


def capture_causal_hud(segment, *, end_s: float) -> list[dict]:
    """Capture only renderer/sensor output; never inspect targets/actions."""
    chart = segment.chart
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
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
    rows = []
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        hud = ContactTimingHUD263Adapter.encode(step, observation).features
        rows.append({
            "step": step,
            "observed_at_s": now_s,
            "hud263": list(hud),
            "countdown": {
                "valid": cue.valid,
                "kind": cue.kind.value,
                "digit": cue.digit,
                "rendered_at_s": cue.rendered_at_s,
                "latency_s": cue.latency_s,
            },
        })
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows


def _current_floor_index(segment, observed_at_s: float) -> int:
    chart_time = segment.chart_time_from_episode(observed_at_s)
    entries = tuple(floor.target_time_s for floor in segment.chart.floors)
    return max(0, min(
        bisect_right(entries, chart_time) - 1,
        len(entries) - 1,
    ))


def _relevant_slot(segment, observed_at_s: float, floor_index: int) -> int:
    return CURRENT_SLOT + floor_index - _current_floor_index(
        segment, observed_at_s
    )


def _floor_visible_with_marker(row: dict, segment, floor_index: int,
                               marker_index: int) -> bool:
    slot = _relevant_slot(segment, row["observed_at_s"], floor_index)
    if not 0 <= slot < FLOORS.floor_slots:
        return False
    floor_features = visible_slot(tuple(row["hud263"]), slot)
    return floor_features[-1] > .5 and floor_features[marker_index] > .5


def _midspin_pair_safe(row: dict, segment, midspin_floor: int) -> bool:
    slot = _relevant_slot(segment, row["observed_at_s"], midspin_floor)
    normalized = normalize_visible_midspin_transition(tuple(row["hud263"]))
    return slot in normalized.recovered_floor_slots


def _first_time(rows: list[dict], predicate) -> float | None:
    for row in rows:
        if predicate(row):
            return row["observed_at_s"]
    return None


def _first_due_forecast(policy_rows: list[dict], truth_s: float) -> dict | None:
    for row in policy_rows:
        if row["time_s"] >= truth_s - 1e-9:
            break
        for token in row["pending"]:
            if (
                token["due_lower_s"] - 1e-9 <= truth_s
                <= token["due_upper_s"] + 1e-9
                or abs(token["due_s"] - truth_s) <= .00075
            ):
                return {
                    "observed_at_s": row["time_s"],
                    "due_s": token["due_s"],
                    "due_lower_s": token["due_lower_s"],
                    "due_upper_s": token["due_upper_s"],
                    "uncertainty_s": token["uncertainty_s"],
                    "local_id": token["id"],
                    "status": row["status"],
                }
    return None


def _compare_prefix(original_rows: list[dict], alternate_rows: list[dict],
                    *, deadline_s: float) -> dict:
    last_step = floor(deadline_s / CONTROL_DT_S + 1e-12)
    left = original_rows[:last_step + 1]
    right = alternate_rows[:last_step + 1]
    if len(left) != len(right) or len(left) != last_step + 1:
        raise RuntimeError("counterfactual capture ended before deadline")
    max_differences = [
        max(abs(a - b) for a, b in zip(
            lrow["hud263"], rrow["hud263"], strict=True
        ))
        for lrow, rrow in zip(left, right, strict=True)
    ]
    cue_equal = all(
        lrow["countdown"] == rrow["countdown"]
        for lrow, rrow in zip(left, right, strict=True)
    )
    distinguishable = [
        index for index, delta in enumerate(max_differences)
        if delta > SAME_HUD_TOLERANCE
    ]
    return {
        "last_compared_step": last_step,
        "last_compared_observation_s": last_step * CONTROL_DT_S,
        "frames_compared": len(left),
        "max_absolute_hud263_difference": max(max_differences),
        "countdown_histories_equal": cue_equal,
        "first_distinguishable_frame_before_deadline": (
            distinguishable[0] if distinguishable else None
        ),
        "indistinguishable_through_deadline": (
            not distinguishable and cue_equal
        ),
    }


def _first_difference(original_rows: list[dict], alternate_rows: list[dict]) -> dict | None:
    for left, right in zip(original_rows, alternate_rows, strict=False):
        differences = [
            index for index, (a, b) in enumerate(zip(
                left["hud263"], right["hud263"], strict=True
            )) if a != b
        ]
        if differences or left["countdown"] != right["countdown"]:
            return {
                "step": left["step"],
                "observed_at_s": left["observed_at_s"],
                "hud263_differing_dimensions": differences,
                "tile_bpm_differs": HUD_OFFSET in differences,
                "real_bpm_differs": HUD_OFFSET + 1 in differences,
                "countdown_differs": left["countdown"] != right["countdown"],
            }
    return None


def _primary_case(path: Path, event_index: int, original_capture) -> tuple[dict, dict]:
    policy_rows, visible_rows, original = original_capture
    raw_chart = load_adofai(path)
    target = original.targets[event_index]
    target_floor = target.floor_index
    midspin_floor = target_floor - 1
    setspeed_floor = target_floor - 2
    if not original.chart.floors[midspin_floor].midspin:
        raise ValueError("fixed target is not immediately after Midspin")
    if "SetSpeed" not in original.chart.floors[setspeed_floor].event_markers:
        raise ValueError("fixed target lacks the expected preceding SetSpeed")
    alternate_chart, intervention = _intervene_numeric_setspeed(
        raw_chart, floor_index=setspeed_floor
    )
    alternate = build_playable_segment(
        compile_adofai(alternate_chart), end_s=phase12.UNSEEN_END_S
    )
    alternate_rows = capture_causal_hud(
        alternate, end_s=phase12.UNSEEN_END_S
    )
    alternate_target = alternate.targets[event_index]
    truth = float(target.episode_time_s)
    alternate_truth = float(alternate_target.episode_time_s)
    deadline = min(truth, alternate_truth) - MOTOR_LEAD_S
    prefix = _compare_prefix(
        visible_rows, alternate_rows, deadline_s=deadline
    )
    first_difference = _first_difference(visible_rows, alternate_rows)
    first_marker = _first_time(
        visible_rows,
        lambda row: _floor_visible_with_marker(
            row, original, setspeed_floor, 9
        ),
    )
    first_midspin_pair = _first_time(
        visible_rows,
        lambda row: _midspin_pair_safe(row, original, midspin_floor),
    )
    first_due = _first_due_forecast(
        policy_rows[phase12.CANDIDATE], truth
    )
    gap_s = abs(alternate_truth - truth)
    classification = (
        "PREDEADLINE_NONIDENTIFIABLE"
        if prefix["indistinguishable_through_deadline"] and gap_s > 1e-9
        else "RECOVERY_IMPLEMENTATION_GAP"
        if first_difference is not None
        and first_difference["observed_at_s"] <= deadline + 1e-9
        else "OBSERVABILITY_UNVERIFIED"
    )
    result = {
        "chart": path.name,
        "chart_sha256": _hash(path),
        "event_index_posthoc": event_index,
        "target_floor_posthoc": target_floor,
        "setspeed_floor_posthoc": setspeed_floor,
        "midspin_floor_posthoc": midspin_floor,
        "classification": classification,
        "intervention_evaluator_only": intervention,
        "target_due_original_s_posthoc": truth,
        "target_due_counterfactual_s_posthoc": alternate_truth,
        "due_gap_ms_posthoc": gap_s * 1000.0,
        "deadline_44ms_s_posthoc": deadline,
        "information_arrival": {
            "future_setspeed_marker_first_visible_s": first_marker,
            "postchange_numeric_bpm_first_distinguishable_s": (
                None if first_difference is None
                else first_difference["observed_at_s"]
            ),
            "midspin_pair_first_safely_visible_s": first_midspin_pair,
            "first_safe_due_forecast": first_due,
            "numeric_bpm_arrival_minus_deadline_ms": (
                None if first_difference is None else
                (first_difference["observed_at_s"] - deadline) * 1000.0
            ),
            "due_forecast_arrival_minus_deadline_ms": (
                None if first_due is None else
                (first_due["observed_at_s"] - deadline) * 1000.0
            ),
        },
        "counterfactual_prefix": prefix,
        "first_full_hud_difference": first_difference,
        "minimal_sound_due_envelope_s_posthoc": [
            min(truth, alternate_truth), max(truth, alternate_truth)
        ],
        "minimal_sound_due_envelope_width_ms_posthoc": gap_s * 1000.0,
        "exact_due_truth_intersection_exists": gap_s <= 1e-9,
        "predeadline_student_derivable_due_interval_s": None,
        "predeadline_due_interval_reason": (
            "The visible marker carries no numeric future speed. The finite "
            "two-world envelope is evaluator-only; under the HUD263 contract "
            "the unobserved positive SetSpeed value has no visible finite "
            "upper-duration bound."
        ),
    }
    trace_frames = []
    for row in visible_rows:
        now = row["observed_at_s"]
        if truth - TRACE_WINDOW_S - 1e-9 <= now < truth + 1e-9:
            frame = dict(row)
            frame["policy"] = policy_rows[phase12.CANDIDATE][row["step"]]
            frame["evaluator_posthoc"] = {
                "target_due_s": truth,
                "delta_to_target_ms": (truth - now) * 1000.0,
                "deadline_s": deadline,
                "at_or_before_deadline": now <= deadline + 1e-9,
                "relevant_setspeed_visible": _floor_visible_with_marker(
                    row, original, setspeed_floor, 9
                ),
                "relevant_midspin_pair_safe": _midspin_pair_safe(
                    row, original, midspin_floor
                ),
                "postchange_numeric_bpm_identified": (
                    first_difference is not None
                    and now >= first_difference["observed_at_s"] - 1e-9
                ),
            }
            trace_frames.append(frame)
    trace = {
        "chart": path.name,
        "event_index_posthoc": event_index,
        "role": "primary",
        "classification": classification,
        "information_arrival": result["information_arrival"],
        "frames": trace_frames,
    }
    return result, trace


def _reference_case(path: Path, event_index: int, original_capture) -> tuple[dict, dict]:
    policy_rows, visible_rows, segment = original_capture
    target = segment.targets[event_index]
    truth = float(target.episode_time_s)
    deadline = truth - MOTOR_LEAD_S
    preceding_floor = target.floor_index - 2
    first_twirl = _first_time(
        visible_rows,
        lambda row: _floor_visible_with_marker(
            row, segment, preceding_floor, 8
        ),
    )
    first_midspin_pair = _first_time(
        visible_rows,
        lambda row: _midspin_pair_safe(
            row, segment, target.floor_index - 1
        ),
    )
    first_due = _first_due_forecast(
        policy_rows[phase12.CANDIDATE], truth
    )
    result = {
        "chart": path.name,
        "chart_sha256": _hash(path),
        "event_index_posthoc": event_index,
        "role": "reference_only_not_setspeed",
        "classification": "OBSERVABILITY_UNVERIFIED",
        "reason": (
            "The preceding special floor is Twirl, not SetSpeed. Its marker "
            "is visible before the deadline, but a causal Twirl-through-floor "
            "estimator is outside this fixed SetSpeed counterfactual audit."
        ),
        "target_due_s_posthoc": truth,
        "deadline_44ms_s_posthoc": deadline,
        "information_arrival": {
            "future_twirl_marker_first_visible_s": first_twirl,
            "midspin_pair_first_safely_visible_s": first_midspin_pair,
            "first_safe_due_forecast": first_due,
        },
    }
    frames = []
    for row in visible_rows:
        now = row["observed_at_s"]
        if truth - TRACE_WINDOW_S - 1e-9 <= now < truth + 1e-9:
            frame = dict(row)
            frame["policy"] = policy_rows[phase12.CANDIDATE][row["step"]]
            frame["evaluator_posthoc"] = {
                "target_due_s": truth,
                "delta_to_target_ms": (truth - now) * 1000.0,
                "deadline_s": deadline,
                "at_or_before_deadline": now <= deadline + 1e-9,
                "relevant_twirl_visible": _floor_visible_with_marker(
                    row, segment, preceding_floor, 8
                ),
                "relevant_midspin_pair_safe": _midspin_pair_safe(
                    row, segment, target.floor_index - 1
                ),
            }
            frames.append(frame)
    return result, {
        "chart": path.name,
        "event_index_posthoc": event_index,
        "role": "reference_only_not_setspeed",
        "classification": result["classification"],
        "information_arrival": result["information_arrival"],
        "frames": frames,
    }


def run(train_dir: Path = TRAIN_DIR) -> tuple[dict, dict]:
    names = {name for name, _ in PRIMARY} | {REFERENCE[0]}
    paths = {name: train_dir / name for name in names}
    for name, path in paths.items():
        if not path.is_file() or _hash(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"fixed Phase 13 chart missing or changed: {name}")

    captures = {
        name: phase12.capture(path, phase12.UNSEEN_END_S)
        for name, path in paths.items()
    }
    cases = []
    traces = []
    for name, event_index in PRIMARY:
        case, trace = _primary_case(
            paths[name], event_index, captures[name]
        )
        cases.append(case)
        traces.append(trace)
    reference, reference_trace = _reference_case(
        paths[REFERENCE[0]], REFERENCE[1], captures[REFERENCE[0]]
    )
    traces.append(reference_trace)
    classifications = Counter(case["classification"] for case in cases)
    if classifications == Counter({"PREDEADLINE_NONIDENTIFIABLE": 4}):
        verdict = "FUTURE_SETSPEED_PREDEADLINE_NONIDENTIFIABLE"
    elif classifications.get("PREDEADLINE_OBSERVABLE", 0) + classifications.get(
        "RECOVERY_IMPLEMENTATION_GAP", 0
    ) and classifications.get("PREDEADLINE_NONIDENTIFIABLE", 0):
        verdict = "FUTURE_SETSPEED_MIXED"
    elif classifications.get("PREDEADLINE_OBSERVABLE", 0) + classifications.get(
        "RECOVERY_IMPLEMENTATION_GAP", 0
    ) == 4:
        verdict = "FUTURE_SETSPEED_PREDEADLINE_OBSERVABLE"
    else:
        verdict = "FUTURE_SETSPEED_OBSERVABILITY_UNVERIFIED"

    phase12_result_path = Path(
        "reports/hud-timing-phase12-midspin-audit-20261020-r3.json"
    )
    phase12_result = json.loads(phase12_result_path.read_text(encoding="utf-8"))
    safety = {
        "phase12_recovered_13_of_13": (
            phase12_result["target_13"]["recovered_clean_lifecycle"] == 13
        ),
        "phase12_motor_ready_9_of_13": (
            phase12_result["target_13"]["motor_ready_44ms"] == 9
        ),
        "phase12_false_positive_zero": (
            phase12_result["unseen_16"][phase12.CANDIDATE]["false_positive_ids"] == 0
        ),
        "phase12_success_regressions_zero": (
            phase12_result["existing_success_regressions"] == 0
        ),
        "predictor_change_made": False,
    }
    result = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "primary_classification_counts": dict(sorted(classifications.items())),
        "primary_cases": cases,
        "reference_case": reference,
        "safety_constraints": safety,
        "phase4_phase5_reuse": {
            "intervention_factor": INTERVENTION_FACTOR,
            "phase4_bool_result": "INDISTINGUISHABLE_BEFORE_44MS_DEADLINE",
            "phase5_stock_icon": "UNVERIFIED",
            "phase5_jrp_numeric_update": "AFTER_DEADLINE",
            "difference_here": (
                "The same numeric-only intervention is applied to the four "
                "Phase-12 late Midspin events and evaluated per event."
            ),
        },
        "student_boundary": {
            "allowed": "current/past HUD263 and visible countdown observations",
            "forbidden": (
                "SetSpeed values, compiled BPM, target due, floor ordinals, "
                "and evaluator associations"
            ),
            "truth_use": "posthoc association and counterfactual evaluation only",
        },
        "implementation_decision": (
            "DO_NOT_RELAX_SETSPEED_GATE: all four primary histories are "
            "non-identifiable before their 44ms deadlines."
        ),
    }
    trace_result = {
        "format": TRACE_FORMAT,
        "generated_utc": result["generated_utc"],
        "control_dt_s": CONTROL_DT_S,
        "window_s": TRACE_WINDOW_S,
        "motor_lead_s": MOTOR_LEAD_S,
        "events": traces,
    }
    return result, trace_result


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    result, traces = run(args.train_dir)
    _write_new(args.output, result)
    _write_new(args.trace_output, traces)
    print("verdict:", result["verdict"])
    for case in result["primary_cases"]:
        print(
            case["chart"], case["event_index_posthoc"],
            case["classification"], case["due_gap_ms_posthoc"],
            case["information_arrival"]["numeric_bpm_arrival_minus_deadline_ms"],
        )
    print("reference:", result["reference_case"]["classification"])


if __name__ == "__main__":
    main()
