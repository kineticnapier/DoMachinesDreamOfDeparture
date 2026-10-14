from __future__ import annotations

"""Stage C v2.4: fixed Train20 baseline vs actuator-aware Teacher replay.

The v2.3 report is the immutable baseline. No Student weights are loaded,
no learning occurs, and Validation/FINAL are never consulted.
"""

import json

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_safety as old
from . import prestige1_transfer as transfer
from .prestige1_actuator_teacher import ActuatorAwareTeacherSafety

FORMAT = "prestige1-stage-c-v24-train-only-actuator-teacher"
REPORT = "stage_c_v24_actuator_teacher_safety.json"


def _aggregate(rows, key):
    return {
        "safe": sum(not row[key]["overload"] for row in rows),
        "overloaded": sum(row[key]["overload"] for row in rows),
        "unsafe_anchors": [row["anchor"] for row in rows if row[key]["overload"]],
        "hits": sum(row[key]["hits"] for row in rows),
        "misses": sum(row[key]["misses"] for row in rows),
        "too_early": sum(row[key]["too_early"] for row in rows),
        "keydowns": sum(row[key]["keydowns"] for row in rows),
        "unjudged": sum(row[key]["unjudged"] for row in rows),
    }


def run_train_actuator_comparison(config):
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Refuse to replace v2.4 candidate report")
    best_before = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Train20 split changed")
    previous_path = transfer.output_dir(config) / old.REPORT
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if (previous.get("format") != old.FORMAT
            or previous.get("role") != "Train20 only"
            or previous.get("train_manifest") != manifest
            or previous.get("config_sha256") != config.signature()
            or previous.get("original_stage_a_gate") != stage_a["gate"]
            or previous.get("stage_a_gate_overridden") is not False
            or previous.get("replay_matches_stage_c_v22") is not True
            or previous.get("protected_inputs_unchanged") is not True
            or previous.get("no_training") is not True
            or previous.get("validation_evaluated") is not False
            or previous.get("final_evaluated") is not False
            or len(previous.get("per_anchor", [])) != 20):
        raise RuntimeError("v2.3 baseline provenance mismatch")
    protected = audit._protected_paths(config, stage_c)
    protected["stage_c_v22_train_audit"] = transfer.output_dir(config) / audit.REPORT
    baseline_hashes = {k: p.sha256(v) for k, v in protected.items()}
    if baseline_hashes != previous["protected_input_sha256"]:
        raise RuntimeError("Protected v2.3 inputs changed")
    protected["stage_c_v23_teacher_safety"] = previous_path
    before = {k: p.sha256(v) for k, v in protected.items()}
    rows = []
    for i, named in enumerate(segments, 1):
        baseline = previous["per_anchor"][i-1]
        if (baseline["anchor"] != i
                or baseline["chart"] != named.chart_name
                or baseline["chart_sha256"] != named.chart_sha256):
            raise RuntimeError(f"Train baseline identity mismatch #{i:02d}")
        adapted = ActuatorAwareTeacherSafety(lead_s=config.lead_s)
        evaluated = old._run_teacher(
            named, config, anchor=i, action_filter=adapted)
        if evaluated["chart_sha256"] != baseline["chart_sha256"]:
            raise RuntimeError("Train chart mismatch during candidate replay")
        stats = evaluated["stats"]
        fields = ("hits", "misses", "targets", "unjudged", "too_early",
                  "keydowns", "overload")
        rows.append({
            "anchor": i,
            "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline": {k: baseline["stats"][k] for k in fields},
            "candidate": {k: stats[k] for k in fields},
            "difference": {
                k: (int(stats[k]) - int(baseline["stats"][k]))
                for k in ("hits", "misses", "too_early", "keydowns", "unjudged")
            },
            "actuator_interventions": adapted.statistics(),
            "candidate_termination": evaluated["termination"],
            "candidate_failure_event": evaluated["failure_event"],
            "candidate_last_12_control_frames": evaluated["last_12_control_frames"],
            "candidate_too_early_last_150ms": evaluated["too_early_last_150ms"],
        })
        print(
            f"Stage C v2.4 Train {i:02d}/20: "
            f"baseline H={baseline['stats']['hits']} early={baseline['stats']['too_early']} "
            f"overload={baseline['stats']['overload']} | "
            f"candidate H={stats['hits']} early={stats['too_early']} "
            f"overload={stats['overload']} changed={adapted.changed_frames}",
            flush=True)
    if ({k: p.sha256(v) for k, v in protected.items()} != before
            or p._protect_old_best(config) != best_before):
        raise RuntimeError("Protected artifacts changed during v2.4 replay")
    baseline_aggregate = _aggregate(rows, "baseline")
    if baseline_aggregate != previous["aggregate"]:
        raise RuntimeError("Baseline aggregation diverges from v2.3")
    candidate_aggregate = _aggregate(rows, "candidate")
    result = {
        "format": FORMAT, "stage": "C-v2.4", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "original_stage_a_gate": stage_a["gate"], "stage_a_gate_overridden": False,
        "config_sha256": config.signature(), "train_manifest": manifest,
        "forecast": ("Fixed 1ms motor-state copy, constant candidate command held "
                     "for lead_s + control_dt_s; not a guaranteed future schedule."),
        "evaluation": ("True simulator physics for Train20; interventions in "
                       "Teacher-only labels, not the Student action path."),
        "baseline": baseline_aggregate, "candidate": candidate_aggregate,
        "per_anchor": rows, "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, result)
    return result
