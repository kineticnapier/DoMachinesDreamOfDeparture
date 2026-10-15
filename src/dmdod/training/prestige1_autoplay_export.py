from __future__ import annotations

"""Write-once Train20-only export of exact compiled Autoplay event timestamps.

Evaluates NO policy. Runs NO Student training and NO Validation/FINAL.
The original Stage A v2 FAIL and 850-Hit v3.0 physical report are preserved.
"""

import json
from pathlib import Path
from math import isfinite

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_finger_comparison as v30
from . import prestige1_transfer as transfer
from .prestige1_autoplay_timing import AutoplayTimingTeacher

FORMAT = "prestige1-train20-autoplay-exact-timing-oracle-v1"
OUTPUT = "stage_timing_autoplay_exact_train20.json"
TARGET_TOLERANCE_S = 1e-6


def compare_one_anchor(named, anchor: int, saved_row: dict) -> dict:
    """Cross-check compiled Autoplay truth against v3.0 physical logs.

    The physical KeyDown error is not used to generate Autoplay labels; it is
    used exclusively for checking that two independent sources agree.
    """
    source = AutoplayTimingTeacher.from_segment(named.segment)
    targets = named.segment.targets
    hit_events = saved_row["final_hit_events"]
    physics = saved_row["final_physics"]
    if (
        saved_row["anchor"] != anchor or saved_row["chart"] != named.chart_name
        or saved_row["chart_sha256"] != named.chart_sha256
        or physics["targets"] != len(targets)
        or physics["hits"] != len(targets)
        or physics["misses"] != 0 or physics["too_early"] != 0
        or physics["overload"] is not False or physics["unjudged"] != 0
        or physics["keydowns"] != len(targets)
        or len(hit_events) != len(targets)
    ):
        raise RuntimeError(f"v3.0 physical baseline is not SAFE/identical #{anchor:02d}")
    reconstructed_error = []
    for idx, (hit, target) in enumerate(zip(hit_events, targets)):
        if hit["target_index"] != idx or hit["key"] not in (
            "left_4","left_3","left_2","left_1",
            "right_1","right_2","right_3","right_4",
        ):
            raise RuntimeError(f"v3.0 physical target ordinal/key mismatch #{anchor:02d}")
        keydown_time_s = float(hit["time_s"])
        error_s = float(hit["error_ms"]) / 1000.
        if not isfinite(keydown_time_s) or not isfinite(error_s):
            raise RuntimeError("Nonfinite v3.0 physical timestamps")
        # Cross-source check only. Labels ALWAYS originate from compiled chart.
        target_from_replay_s = keydown_time_s - error_s
        deviation = abs(target_from_replay_s - target.episode_time_s)
        if deviation > TARGET_TOLERANCE_S:
            raise RuntimeError(
                f"Autoplay/physical truth mismatch #{anchor:02d}, "
                f"target={idx}, deviation={1000*deviation:.7f}ms")
        reconstructed_error.append(deviation)
    return {
        "anchor": anchor,
        "chart": named.chart_name,
        "chart_sha256": named.chart_sha256,
        "targets": len(targets),
        "events": len(source.events),
        "simultaneous_extra_targets": len(targets) - len(source.events),
        "max_independent_target_delta_ms": (
            max(reconstructed_error) * 1000. if reconstructed_error else None),
        "pitch_ratio": named.segment.pitch_ratio,
        "segment_start_chart_s": named.segment.start_chart_s,
        "segment_end_chart_s": named.segment.end_chart_s,
        "timeline": [
            {
                "event_index": event.event_index,
                "target_ordinals": list(event.target_ordinals),
                "floor_indices": list(event.floor_indices),
                "episode_time_s": event.episode_time_s,
                "chart_time_s": event.chart_time_s,
                "chord_size": event.chord_size,
            }
            for event in source.events
        ],
    }


def export_train_autoplay_timing(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    out = root / OUTPUT
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite frozen Auto Timing oracle: {out}")

    stage_path = root / p.STAGE_A_REPORT
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if (
        stage.get("format") != p.VERSION or stage.get("stage") != "A"
        or stage.get("metadata", {}).get("config_sha256") != config.signature()
        or stage.get("gate", {}).get("passed") is not False
        or stage.get("gate", {}).get("may_run_stage_c") is not False
        or stage.get("gate", {}).get("gate_data_role") != "train_heldout"
        or stage.get("train_partition", {}).get("validation_used_for_selection") is not False
    ):
        raise RuntimeError("Exact Auto Timing requires frozen failed Stage A provenance")

    report_path = root / v30.REPORT
    plans_path = root / v30.PLANS
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    initial_sha = {
        "stage_a_v2": p.sha256(stage_path),
        "stage_c_v30_finger_phase_annealing": p.sha256(report_path),
        "stage_c_v30_safe_teacher_plans": p.sha256(plans_path),
    }
    if (
        report.get("format") != v30.FORMAT
        or report.get("stage") != "C-v3.0"
        or report.get("role") != "Train20 only"
        or report.get("no_training") is not True
        or report.get("validation_evaluated") is not False
        or report.get("final_evaluated") is not False
        or report.get("original_stage_a_gate") != stage["gate"]
        or report.get("gates", {}).get("full_train20_safe850") is not True
        or report.get("gates", {}).get("student_training_permitted") is not False
        or report.get("config_sha256") != config.signature()
        or report.get("saved_plan_sha256") != initial_sha["stage_c_v30_safe_teacher_plans"]
        or plans.get("format") != v30.PLAN_FORMAT
        or plans.get("stage") != "C-v3.0"
        or plans.get("role") != "Train20 only"
        or plans.get("config_sha256") != config.signature()
        or plans.get("train_manifest") != report.get("train_manifest")
        or len(plans.get("plans", [])) != 20
        or len(report.get("per_anchor", [])) != 20
    ):
        raise RuntimeError("Frozen v3.0 Teacher provenance failed")

    before_old_best = p._protect_old_best(config)
    selected, manifest = audit._train_manifest(config)
    if (
        manifest != stage["metadata"]["dataset_split"]["train"]
        or manifest != report["train_manifest"]
        or manifest != plans["train_manifest"]
    ):
        raise RuntimeError("Train20 manifest changed; abort exact Timing export")

    rows = [
        compare_one_anchor(named, i, saved)
        for i, (named, saved) in enumerate(zip(selected, report["per_anchor"]), 1)
    ]
    if (
        sum(r["targets"] for r in rows) != 850
        or sum(r["events"] for r in rows) > 850
        or sum(r["events"] for r in rows) <= 0
    ):
        raise RuntimeError("Autoplay Train20 expected exactly 850 physical targets")

    after_sha = {
        "stage_a_v2": p.sha256(stage_path),
        "stage_c_v30_finger_phase_annealing": p.sha256(report_path),
        "stage_c_v30_safe_teacher_plans": p.sha256(plans_path),
    }
    if before_old_best != p._protect_old_best(config) or initial_sha != after_sha:
        raise RuntimeError("Frozen artifacts changed during Auto Timing export")

    result = {
        "format": FORMAT,
        "role": "Train20 only",
        "teacher_mode": "autoplay_exact_compiled_event_timestamps",
        "truth_clock": "episode_time_s=(chart_time_s-segment_start_chart_s)/pitch_ratio",
        "time_quantization": "none",
        "timing_truth_clipped_at_200ms": False,
        "near_200ms_is_only_training_mask": True,
        "motor_command_is_not_timestamp_truth": True,
        "fingering_assigned": False,
        "student_inference_receives_privileged_timestamps": False,
        "no_training": True,
        "student_checkpoint_loaded": False,
        "validation_evaluated": False,
        "final_evaluated": False,
        "stage_a_gate_overridden": False,
        "stage_a_original_gate": stage["gate"],
        "student_training_permitted": False,
        "config_sha256": config.signature(),
        "train_manifest": manifest,
        "total_targets": sum(row["targets"] for row in rows),
        "total_distinct_onsets": sum(row["events"] for row in rows),
        "max_cross_source_error_ms": max(
            row["max_independent_target_delta_ms"] or 0. for row in rows),
        "protected_input_sha256": initial_sha,
        "protected_inputs_unchanged": True,
        "per_anchor": rows,
        "note": (
            "This oracle is the simulator's compiled playable event schedule, "
            "not a proof of bit-identical synchronization with external ADOFAI Autoplay. "
            "It is evaluator/label-side privileged truth only; physical Motor still "
            "must account for real 1ms dynamics and finite actuation lead."
        ),
    }
    p._no_replace_json(out, result)
    return result
