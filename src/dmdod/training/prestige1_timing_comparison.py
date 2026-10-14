from __future__ import annotations

"""Stage C v2.6 Train20-only timing-centered Teacher comparison.

Frozen v2.4 safety results and v2.5 physical timing traces are the baseline.
No Student runs, checkpoint writes, Validation/FINAL or hyperparameter search.
"""

import json

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_safety as old
from . import prestige1_actuator_comparison as v24
from . import prestige1_teacher_quality as v25
from . import prestige1_transfer as transfer
from .prestige1_timing_teacher import TimingCenteredActuatorTeacher

FORMAT = "prestige1-stage-c-v26-train-only-timing-centered-teacher"
REPORT = "stage_c_v26_timing_centered_teacher.json"


def _check_baseline(v24_report, v25_report, old_report, stage_a, config, manifest):
    v25._check_v24_report(v24_report, old_report, stage_a, config, manifest)
    if (
        v25_report.get("format") != v25.FORMAT
        or v25_report.get("stage") != "C-v2.5"
        or v25_report.get("role") != "Train20 only"
        or v25_report.get("no_training") is not True
        or v25_report.get("read_only") is not True
        or v25_report.get("checkpoint_modified") is not False
        or v25_report.get("student_checkpoint_loaded") is not False
        or v25_report.get("validation_evaluated") is not False
        or v25_report.get("final_evaluated") is not False
        or v25_report.get("validation_used_for_selection") is not False
        or v25_report.get("stage_a_gate_overridden") is not False
        or v25_report.get("original_stage_a_gate") != stage_a["gate"]
        or v25_report.get("config_sha256") != config.signature()
        or v25_report.get("train_manifest") != manifest
        or v25_report.get("protected_inputs_unchanged") is not True
        or v25_report.get("physics_candidate") != v24_report.get("candidate")
        or len(v25_report.get("per_anchor", [])) != 20
    ):
        raise RuntimeError("Frozen v2.5 timing baseline provenance mismatch")
    full = v25_report["timing_quality_candidate"]
    if (full.get("hits") != v24_report["candidate"]["hits"]
            or v24_report["candidate"] != {
                "safe": 20, "overloaded": 0, "unsafe_anchors": [],
                "hits": 850, "misses": 0, "too_early": 0,
                "keydowns": 850, "unjudged": 0}):
        raise RuntimeError("Frozen v2.4 SAFE850 baseline mismatch")
    reconstructed = v25.summarize_hit_quality([
        hit
        for row in v25_report["per_anchor"]
        for hit in row["candidate"]["hit_events"]
    ])
    if reconstructed != full:
        raise RuntimeError("v2.5 Hit-event aggregate differs from frozen baseline")


def _paired_analysis(previous, current):
    prior = {event["target_index"]: event["error_ms"] for event in previous}
    now = {event["target_index"]: event["error_ms"] for event in current}
    common = set(prior) & set(now)
    improved = sum(abs(now[i]) + 1e-6 < abs(prior[i]) for i in common)
    worse = sum(abs(now[i]) > abs(prior[i]) + 1e-6 for i in common)
    same = len(common) - improved - worse
    return {
        "paired_hits": len(common),
        "baseline_only_targets": len(prior.keys() - now.keys()),
        "candidate_only_targets": len(now.keys() - prior.keys()),
        "improved": improved, "worsened": worse, "unchanged": same,
        "paired_baseline_mae_ms": (
            sum(abs(prior[i]) for i in common)/len(common) if common else None),
        "paired_candidate_mae_ms": (
            sum(abs(now[i]) for i in common)/len(common) if common else None),
    }


def _safety_gate(candidate, baseline):
    return (
        candidate["safe"] == 20
        and candidate["unsafe_anchors"] == []
        and candidate["hits"] == baseline["hits"]
        and candidate["keydowns"] == baseline["keydowns"]
        and candidate["misses"] == 0
        and candidate["too_early"] == 0
        and candidate["unjudged"] == 0
    )


def run_train_timing_comparison(config):
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Refuse to overwrite v2.6 timing audit")

    best_before = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 split changed")

    root = transfer.output_dir(config)
    source_v23 = root / old.REPORT
    source_v24 = root / v24.REPORT
    source_v25 = root / v25.REPORT
    prev23 = json.loads(source_v23.read_text(encoding="utf-8"))
    prev24 = json.loads(source_v24.read_text(encoding="utf-8"))
    prev25 = json.loads(source_v25.read_text(encoding="utf-8"))
    _check_baseline(prev24, prev25, prev23, stage_a, config, manifest)

    protected = audit._protected_paths(config, stage_c)
    protected["stage_c_v22_train_audit"] = root / audit.REPORT
    protected["stage_c_v23_teacher_safety"] = source_v23
    protected["stage_c_v24_actuator_teacher_safety"] = source_v24
    prior_hashes = {name: p.sha256(path) for name, path in protected.items()}
    if prior_hashes != prev25["protected_input_sha256"]:
        raise RuntimeError("Frozen v2.5 protected files changed")
    protected["stage_c_v25_teacher_quality"] = source_v25
    before = {name: p.sha256(path) for name, path in protected.items()}

    rows = []
    all_hit_events = []
    paired = CounterPair()
    for index, named in enumerate(segments, 1):
        baseline = prev25["per_anchor"][index - 1]
        previous_v24 = prev24["per_anchor"][index - 1]
        if (baseline["anchor"] != index
                or previous_v24["anchor"] != index
                or baseline["chart"] != named.chart_name
                or previous_v24["chart"] != named.chart_name
                or baseline["chart_sha256"] != named.chart_sha256
                or previous_v24["chart_sha256"] != named.chart_sha256
                or baseline["physics_stats"] != previous_v24["candidate"]):
            raise RuntimeError(f"Frozen Train anchor mismatch #{index:02d}")

        controller = TimingCenteredActuatorTeacher(lead_s=config.lead_s)
        replay = old._run_teacher(
            named, config, anchor=index, action_filter=controller)
        stats = replay["stats"]
        if controller.env is None or controller.counts["frames"] != (
                sum(replay["phase_frame_counts"].values())):
            raise RuntimeError(f"v2.6 action/trace frame mismatch #{index:02d}")

        reconstructed = v25.classify_physical_keydowns(
            replay["all_physical_keydowns"], controller.env, stats)
        quality = reconstructed["quality"]
        all_hit_events.extend(reconstructed["hit_events"])
        same_notes = _paired_analysis(
            baseline["candidate"]["hit_events"],
            reconstructed["hit_events"])
        paired.add(same_notes)

        rows.append({
            "anchor": index, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline_physics": baseline["physics_stats"],
            "candidate_physics": stats,
            "baseline_timing": baseline["candidate"]["quality"],
            "candidate_timing": quality,
            "paired_timing": same_notes,
            "timing_controller": controller.statistics(),
            "candidate_termination": replay["termination"],
            "candidate_hit_events": reconstructed["hit_events"],
            "candidate_too_early_keydowns": reconstructed["too_early_keydowns"],
        })
        print(
            f"Stage C v2.6 Train {index:02d}/20: "
            f"H={stats['hits']}/{stats['targets']} "
            f"miss={stats['misses']} early={stats['too_early']} "
            f"overload={stats['overload']} "
            f"MAE={quality['mae_ms'] if quality['mae_ms'] is not None else 'n/a'} "
            f"deferred={controller.counts['deferred_key_commands']}",
            flush=True)

    if ({name: p.sha256(path) for name, path in protected.items()} != before
            or p._protect_old_best(config) != best_before):
        raise RuntimeError("Protected checkpoints and frozen reports changed")

    aggregate = v24._aggregate(
        [{"anchor": row["anchor"], "candidate": row["candidate_physics"]}
         for row in rows], "candidate")
    candidate_quality = v25.summarize_hit_quality(all_hit_events)
    baseline_quality = prev25["timing_quality_candidate"]
    safety_ok = _safety_gate(aggregate, prev24["candidate"])
    precision_ok = bool(
        candidate_quality["mae_ms"] is not None
        and candidate_quality["mae_ms"] < baseline_quality["mae_ms"]
        and candidate_quality["perfect_family_rate"]
        >= baseline_quality["perfect_family_rate"]
    )
    gates = {
        "physical_safety_not_worse_than_v24": safety_ok,
        "timing_mae_improved_and_perfect_family_not_worse": precision_ok,
        "eligible_for_later_student_design": safety_ok and precision_ok,
        "automatically_adopted": False,
        "stage_a_gate_overridden": False,
    }
    report = {
        "format": FORMAT, "stage": "C-v2.6", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "original_stage_a_gate": stage_a["gate"],
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(), "train_manifest": manifest,
        "forecast_assumption": (
            "Copy frozen motor, first 10ms alternative command, then hold "
            "v2.4 shield action for lead_s + control_dt_s. This is not "
            "a full future Teacher policy rollout."
        ),
        "reference": {
            "v24_physics": prev24["candidate"],
            "v25_timing": baseline_quality,
        },
        "candidate": {
            "physics": aggregate, "timing": candidate_quality,
            "paired_hit_analysis": paired.summary(),
        },
        "gates": gates,
        "per_anchor": rows,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report


class CounterPair:
    """Accumulate pairing by sums; handle missing target identities safely."""

    def __init__(self):
        self.total = {
            field: 0 for field in (
                "paired_hits", "baseline_only_targets", "candidate_only_targets",
                "improved", "worsened", "unchanged")
        }
        self.baseline_sum_abs = 0.
        self.candidate_sum_abs = 0.

    def add(self, row):
        for key in self.total:
            self.total[key] += row[key]
        n = row["paired_hits"]
        if n:
            self.baseline_sum_abs += n * row["paired_baseline_mae_ms"]
            self.candidate_sum_abs += n * row["paired_candidate_mae_ms"]

    def summary(self):
        n = self.total["paired_hits"]
        return {
            **self.total,
            "paired_baseline_mae_ms": self.baseline_sum_abs / n if n else None,
            "paired_candidate_mae_ms": self.candidate_sum_abs / n if n else None,
        }
