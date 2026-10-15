from __future__ import annotations

"""Read-only offline audit of the saved v3.0 Train20 physical fingering.

This is an *evaluator-side* diagnostic of a privileged Teacher, NOT an IRL
training dataset and NOT a Student-observable feature extractor. No rollout,
dataset discovery, checkpoint loading, or policy action is performed.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
import statistics

FINGER_ORDER = (
    "left_4", "left_3", "left_2", "left_1",
    "right_1", "right_2", "right_3", "right_4",
)
SOURCE_FORMAT = "prestige1-stage-c-v30-train-only-finger-phase-annealing"
PLAN_FORMAT = "prestige1-stage-c-v30-train-only-verified-finger-plans"
AUDIT_FORMAT = "prestige1-stage-c-v30-fingering-descriptive-audit-v1"


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "min_ms": ordered[0] if ordered else None,
        "median_ms": statistics.median(ordered) if ordered else None,
        "under_20ms": sum(v < 20. for v in ordered),
        "under_50ms": sum(v < 50. for v in ordered),
        "under_80ms": sum(v < 80. for v in ordered),
        "under_120ms": sum(v < 120. for v in ordered),
    }


def _hit_summary(rows: list[dict]) -> dict:
    key_counts = Counter()
    hand_counts = Counter()
    key_gaps = defaultdict(list)
    hand_gaps = defaultdict(list)
    adjacent_gaps = []
    adjacent_same_key = adjacent_same_hand = 0
    near_20_same_key = near_20_same_hand = 0
    center_pair_transitions = 0
    true_same_time_chords = 0
    repeated_key_details = []
    per_anchor = []
    absolute_errors = []
    judgement_counts = Counter()
    for row in rows:
        anchor = row["anchor"]
        events = row["final_hit_events"]
        if len(events) != row["final_physics"]["hits"]:
            raise ValueError(f"Hit event count differs from physics anchor {anchor}")
        indexed = [event["target_index"] for event in events]
        if indexed != list(range(len(events))):
            raise ValueError(f"Target ordinals missing/duplicated anchor {anchor}")
        by_finger_last = {}
        by_hand_last = {}
        previous = None
        per_key = Counter()
        per_hand = Counter()
        anchor_abs_error = []
        targets = []
        for event in events:
            key = event["key"]
            if key not in FINGER_ORDER:
                raise ValueError(f"Unknown key '{key}' anchor {anchor}")
            t = float(event["time_s"])
            error_ms = float(event["error_ms"])
            if not isfinite(t) or not isfinite(error_ms):
                raise ValueError(f"Nonfinite event time/error anchor {anchor}")
            if previous is not None and t < previous["time_s"] - 1e-7:
                raise ValueError(f"Physical Hit times not monotonic anchor {anchor}")
            hand = key.split("_")[0]
            hand_counts[hand] += 1
            key_counts[key] += 1
            per_key[key] += 1
            per_hand[hand] += 1
            absolute_errors.append(abs(error_ms))
            anchor_abs_error.append(abs(error_ms))
            judgement_counts[event["judgement"]] += 1

            # Evaluator-only reconstruction; NEVER routed to Student.
            target_time = t - error_ms * 0.001
            targets.append(target_time)
            if key in by_finger_last:
                gap = 1000. * (t - by_finger_last[key])
                key_gaps[key].append(gap)
                if gap < 120.:
                    repeated_key_details.append({
                        "anchor": anchor, "key": key, "gap_ms": gap,
                        "target_index": event["target_index"],
                    })
            if hand in by_hand_last:
                hand_gaps[hand].append(1000. * (t - by_hand_last[hand]))
            if previous is not None:
                gap = 1000. * (t - previous["time_s"])
                adjacent_gaps.append(gap)
                same_key = previous["key"] == key
                same_hand = previous["key"].split("_")[0] == hand
                adjacent_same_key += same_key
                adjacent_same_hand += same_hand
                near_20_same_key += gap < 20. and same_key
                near_20_same_hand += gap < 20. and same_hand
                center_pair_transitions += (
                    previous["key"] in ("left_1", "right_1")
                    and key in ("left_1", "right_1")
                    and previous["key"] != key
                )
            by_finger_last[key] = t
            by_hand_last[hand] = t
            previous = event
        # Targets whose evaluator-derived onset times are coincident comprise
        # chords; mere near-simultaneous physical KeyDowns are not chords.
        for prior, later in zip(targets, targets[1:]):
            true_same_time_chords += abs(later - prior) <= 1e-6
        per_anchor.append({
            "anchor": anchor, "chart": row["chart"],
            "hits": len(events),
            "key_hits": {key: per_key[key] for key in FINGER_ORDER},
            "hand_hits": {side: per_hand[side] for side in ("left", "right")},
            "mae_ms": sum(anchor_abs_error) / len(anchor_abs_error) if anchor_abs_error else None,
        })
    if not absolute_errors:
        raise ValueError("No hit events available")
    return {
        "hit_count": len(absolute_errors),
        "finger_order": list(FINGER_ORDER),
        "finger_hit_counts": {key: key_counts[key] for key in FINGER_ORDER},
        "hand_hit_counts": {side: hand_counts[side] for side in ("left", "right")},
        "central_pair_hits": key_counts["left_1"] + key_counts["right_1"],
        "central_pair_hit_fraction": (
            key_counts["left_1"] + key_counts["right_1"]) / len(absolute_errors),
        "mae_ms_from_events": sum(absolute_errors) / len(absolute_errors),
        "judgement_counts": dict(sorted(judgement_counts.items())),
        "adjacent_hit_gaps": _stats(adjacent_gaps),
        "adjacent_same_finger": adjacent_same_key,
        "adjacent_same_hand": adjacent_same_hand,
        "adjacent_same_finger_under_20ms": near_20_same_key,
        "adjacent_same_hand_under_20ms": near_20_same_hand,
        "center_pair_alternations": center_pair_transitions,
        "same_finger_inter_hit_gaps": {
            key: _stats(key_gaps[key]) for key in FINGER_ORDER
        },
        "same_hand_inter_hit_gaps": {
            hand: _stats(hand_gaps[hand]) for hand in ("left", "right")
        },
        "repeated_finger_under_120ms": repeated_key_details,
        "same_target_time_adjacent_pairs": true_same_time_chords,
        "per_anchor": per_anchor,
    }


def _motor_summary(plans: list[dict]) -> dict:
    positive_frames = Counter()
    positive_runs = Counter()
    multi_positive_frames = 0
    max_parallel_positive = 0
    total_frames = 0
    for plan in plans:
        tape = plan["actions_8k"]
        if not tape:
            raise ValueError(f"Empty tape anchor {plan['anchor']}")
        prev_positive = [False] * len(FINGER_ORDER)
        for frame in tape:
            if len(frame) != len(FINGER_ORDER) or any(
                type(x) not in (int, float) or not isfinite(x) or x not in (-1, 0, 1)
                for x in frame
            ):
                raise ValueError(f"Invalid frozen 8-Key action anchor {plan['anchor']}")
            positives = [v == 1 for v in frame]
            count = sum(positives)
            max_parallel_positive = max(max_parallel_positive, count)
            multi_positive_frames += count > 1
            total_frames += 1
            for i, is_positive in enumerate(positives):
                key = FINGER_ORDER[i]
                positive_frames[key] += is_positive
                positive_runs[key] += is_positive and not prev_positive[i]
            prev_positive = positives
    return {
        "total_control_frames": total_frames,
        "positive_command_frames": {
            key: positive_frames[key] for key in FINGER_ORDER
        },
        "positive_command_runs": {
            key: positive_runs[key] for key in FINGER_ORDER
        },
        "frames_with_multiple_positive_commands": multi_positive_frames,
        "maximum_parallel_positive_commands": max_parallel_positive,
    }


def audit_documents(report: dict, plans: dict, *, plan_file_sha256: str) -> dict:
    if (
        report.get("format") != SOURCE_FORMAT
        or report.get("stage") != "C-v3.0"
        or plans.get("format") != PLAN_FORMAT
        or plans.get("stage") != "C-v3.0"
        or report.get("role") != plans.get("role") != "Train20 only"
    ):
        raise ValueError("Unexpected v3.0 source formats/roles")
    required_false = (
        "student_checkpoint_loaded", "validation_evaluated",
        "final_evaluated", "stage_a_gate_overridden",
    )
    for doc in (report, plans):
        if doc.get("no_training") is not True or any(
            doc.get(key) is not False for key in required_false
        ):
            raise ValueError("Source is not frozen Train20 Teacher-only")
    if (
        report.get("original_stage_a_gate", {}).get("passed") is not False
        or report["original_stage_a_gate"].get("may_run_stage_c") is not False
        or report.get("gates", {}).get("full_train20_safe850") is not True
        or report["gates"].get("student_training_permitted") is not False
        or report.get("saved_plan_sha256") != plan_file_sha256
        or report.get("saved_plan_file") != "stage_c_v30_safe_teacher_plans.json"
        or report.get("config_sha256") != plans.get("config_sha256")
        or report.get("train_manifest") != plans.get("train_manifest")
        or plans.get("finger_order") != list(FINGER_ORDER)
        or plans.get("control_dt_s") != .010
        or plans.get("physics_dt_s") != .001
    ):
        raise ValueError("Unsafe or mismatched frozen v3.0 provenance")
    src = report.get("per_anchor", [])
    saved = plans.get("plans", [])
    if len(src) != 20 or len(saved) != 20:
        raise ValueError("Expected exact frozen Train20")
    for i, (row, plan) in enumerate(zip(src, saved), 1):
        stats = row["final_physics"]
        if (
            row.get("anchor") != i or plan.get("anchor") != i
            or row.get("chart_sha256") != plan.get("chart_sha256")
            or row.get("chart") != plan.get("chart")
            or plan.get("independently_replayed") is not True
            or stats["hits"] != stats["targets"] or stats["misses"] != 0
            or stats["too_early"] != 0 or stats["keydowns"] != stats["hits"]
            or stats["unjudged"] != 0 or stats["overload"] is not False
            or plan["verified_hits"] != stats["hits"]
            or abs(plan["verified_mae_ms"] - row["final_timing"]["mae_ms"]) > 1e-6
        ):
            raise ValueError(f"Anchor #{i:02d} physical/provenance mismatch")
    hits = _hit_summary(src)
    if hits["hit_count"] != 850 or abs(
        hits["mae_ms_from_events"] - report["candidate"]["timing"]["mae_ms"]
    ) > 1e-6:
        raise ValueError("Full Train20 Hit/MAE verification failed")
    if (
        report["candidate"]["physics"]["hits"] != 850
        or report["candidate"]["physics"]["keydowns"] != 850
        or report["candidate"]["physics"]["misses"] != 0
        or report["candidate"]["physics"]["too_early"] != 0
        or report["candidate"]["physics"]["overloaded"] != 0
        or report["candidate"]["physics"]["unjudged"] != 0
    ):
        raise ValueError("Aggregate safety mismatch")
    return {
        "format": AUDIT_FORMAT,
        "role": "Train20 only",
        "source_stage": "C-v3.0",
        "no_training": True,
        "no_counterfactual_claims": True,
        "no_human_expert_claims": True,
        "no_student_observations_extracted": True,
        "source_plan_sha256": plan_file_sha256,
        "fingering": hits,
        "motor_commands": _motor_summary(saved),
        "limitations": [
            "The source is one privileged optimized Teacher, not human demonstrations.",
            "An assigned physical key is not uniquely correct for a target.",
            "KeyDown gaps are observed hit intervals, not measured recovery limits.",
            "No alternate-finger action was replayed; counterfactual feasibility unknown.",
            "No Student, IRL, heldout validation, or FINAL evaluation performed.",
        ],
    }


def audit_files(report_path: Path, plan_path: Path) -> dict:
    report_path, plan_path = Path(report_path), Path(plan_path)
    result = audit_documents(
        json.loads(report_path.read_text(encoding="utf-8")),
        json.loads(plan_path.read_text(encoding="utf-8")),
        plan_file_sha256=_sha256(plan_path),
    )
    result["source_report_sha256"] = _sha256(report_path)
    return result


def write_once(path: Path, document: dict) -> None:
    """Never overwrite existing audit artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
