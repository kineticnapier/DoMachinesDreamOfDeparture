from __future__ import annotations

"""Prestige I Stage C v2.7: protected Train20-only global SA comparison."""

import json

from dmdod.motor.n_key import NKeyAction

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_safety as old
from . import prestige1_actuator_comparison as v24
from . import prestige1_teacher_quality as v25
from . import prestige1_timing_comparison as v26
from . import prestige1_transfer as transfer
from .prestige1_actuator_teacher import ActuatorAwareTeacherSafety
from .prestige1_global_annealing import (
    INITIAL_TEMPERATURE_MS, FINAL_TEMPERATURE_MS,
    MAX_DELAY_TICKS, PROPOSALS_PER_ANCHOR, SNAPSHOT_STRIDE,
    anneal_chart,
)

FORMAT = "prestige1-stage-c-v27-train-only-global-motor-annealing"
PLAN_FORMAT = "prestige1-stage-c-v27-train-only-open-loop-teacher-plans"
REPORT = "stage_c_v27_global_annealing.json"
PLANS = "stage_c_v27_safe_teacher_plans.json"


class FrozenSafetyCapture:
    """Run the original v2.4 shield and capture its *executed* 10ms actions."""

    def __init__(self, lead_s):
        self.shield = ActuatorAwareTeacherSafety(lead_s=lead_s)
        self.commands = []

    def __call__(self, env, original: NKeyAction) -> NKeyAction:
        action = self.shield(env, original)
        if len(action.values) != p.KEY_COUNT:
            raise RuntimeError("Shield action width changed")
        self.commands.append(tuple(float(v) for v in action.values))
        return action


def _verify_frozen_provenance(config, stage_a, manifest, prev23, prev24, prev25, prev26):
    v26._check_baseline(prev24, prev25, prev23, stage_a, config, manifest)
    if (
        prev26.get("format") != v26.FORMAT
        or prev26.get("stage") != "C-v2.6"
        or prev26.get("role") != "Train20 only"
        or prev26.get("no_training") is not True
        or prev26.get("read_only") is not True
        or prev26.get("checkpoint_modified") is not False
        or prev26.get("student_checkpoint_loaded") is not False
        or prev26.get("validation_evaluated") is not False
        or prev26.get("final_evaluated") is not False
        or prev26.get("validation_used_for_selection") is not False
        or prev26.get("stage_a_gate_overridden") is not False
        or prev26.get("original_stage_a_gate") != stage_a["gate"]
        or prev26.get("config_sha256") != config.signature()
        or prev26.get("train_manifest") != manifest
        or prev26.get("protected_inputs_unchanged") is not True
        or len(prev26.get("per_anchor", [])) != 20
        or prev26.get("reference", {}).get("v24_physics") != prev24["candidate"]
        or prev26.get("reference", {}).get("v25_timing") != prev25["timing_quality_candidate"]
        or prev26.get("gates", {}).get("physical_safety_not_worse_than_v24") is not False
    ):
        raise RuntimeError("Stage C v2.6 source provenance changed")
    if stage_a["gate"].get("passed") is not False or (
            stage_a["gate"].get("may_run_stage_c") is not False):
        raise RuntimeError("Stage A frozen FAIL/may_run_stage_c=False required")


def _all_safe_aggregate(rows):
    return v24._aggregate([
        {"anchor": row["anchor"], "candidate": row["best_physics"]}
        for row in rows
    ], "candidate")


def _report_gate(physical, timing, baseline):
    return {
        "full_train20_safe850": v26._safety_gate(physical, baseline["physics"]),
        "timing_mae_below_v24": (
            timing["hits"] == baseline["timing"]["hits"]
            and timing["mae_ms"] is not None
            and timing["mae_ms"] < baseline["timing"]["mae_ms"]
        ),
        "student_training_permitted": False,
        "original_stage_a_gate_overridden": False,
        "automatically_adopted": False,
    }


def run_train_global_annealing(config):
    root = transfer.output_dir(config)
    output = root / REPORT
    plan_path = root / PLANS
    if output.exists() or plan_path.exists():
        raise FileExistsError("Refuse to overwrite v2.7 report or verified plans")

    best_before = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 split changed")

    source_paths = {
        "stage_c_v23_teacher_safety": root / old.REPORT,
        "stage_c_v24_actuator_teacher_safety": root / v24.REPORT,
        "stage_c_v25_teacher_quality": root / v25.REPORT,
        "stage_c_v26_timing_centered_teacher": root / v26.REPORT,
    }
    prev23, prev24, prev25, prev26 = (
        json.loads(source_paths[key].read_text(encoding="utf-8"))
        for key in source_paths
    )
    _verify_frozen_provenance(
        config, stage_a, manifest, prev23, prev24, prev25, prev26)

    protected = audit._protected_paths(config, stage_c)
    protected["stage_c_v22_train_audit"] = root / audit.REPORT
    protected.update({key: value for key, value in source_paths.items()
                      if key != "stage_c_v26_timing_centered_teacher"})
    prior_hashes = {key: p.sha256(path) for key, path in protected.items()}
    if prior_hashes != prev26["protected_input_sha256"]:
        raise RuntimeError("Stage C v2.6 protected input SHA256 mismatch")
    protected["stage_c_v26_timing_centered_teacher"] = source_paths[
        "stage_c_v26_timing_centered_teacher"]
    before = {key: p.sha256(path) for key, path in protected.items()}

    rows = []
    plans = []
    best_events = []
    for anchor, named in enumerate(segments, 1):
        frozen = prev24["per_anchor"][anchor-1]
        quality_ref = prev25["per_anchor"][anchor-1]
        failed_ref = prev26["per_anchor"][anchor-1]
        if any(
            row["anchor"] != anchor
            or row["chart"] != named.chart_name
            or row["chart_sha256"] != named.chart_sha256
            for row in (frozen, quality_ref, failed_ref)
        ) or (
            frozen["candidate"] != quality_ref["physics_stats"]
            or failed_ref["baseline_physics"] != frozen["candidate"]
            or frozen["candidate"]["hits"] != frozen["candidate"]["targets"]
            or frozen["candidate"]["misses"] != 0
        ):
            raise RuntimeError(f"Original Train safety baseline changed #{anchor:02d}")

        recorder = FrozenSafetyCapture(config.lead_s)
        teacher_replay = old._run_teacher(
            named, config, anchor=anchor, action_filter=recorder)
        if (teacher_replay["stats"] != frozen["candidate"]
                or recorder.shield.statistics() != frozen["actuator_interventions"]
                or len(recorder.commands) != sum(
                    teacher_replay["phase_frame_counts"].values())):
            raise RuntimeError(f"Frozen v2.4 Teacher command capture diverged #{anchor:02d}")

        optimized = anneal_chart(
            named, tuple(recorder.commands),
            seed=int(config.seed) * 1009 + anchor)
        initial, best = optimized["baseline"], optimized["best"]
        if (initial.stats != frozen["candidate"]
                or initial.timing != quality_ref["candidate"]["quality"]
                or initial.hit_events != quality_ref["candidate"]["hit_events"]):
            raise RuntimeError(f"v2.4 open-loop 1ms baseline failed identity #{anchor:02d}")
        if not best.safe or best.objective_ms > initial.objective_ms + 1e-9:
            raise RuntimeError("SA selected an invalid or inferior best trajectory")
        best_events.extend(best.hit_events)

        modifications = [
            {
                "finger": named_key,
                "original_start_frame": run.start,
                "original_end_frame_exclusive": run.end,
                "delay_frames": delay,
                "effective_start_frame": run.start + delay,
            }
            for run, delay in zip(optimized["runs"], optimized["best_delays"])
            if delay
            for named_key in [best.env.key_names[run.key_index]]
        ]
        rows.append({
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline_physics": initial.stats,
            "best_physics": best.stats,
            "baseline_timing": initial.timing,
            "best_timing": best.timing,
            "best_hit_events": best.hit_events,
            "baseline_command_frames": len(recorder.commands),
            "best_termination": best.termination,
            "scheduled_press_runs": len(optimized["runs"]),
            "changed_press_runs": len(modifications),
            "annealing": optimized["counts"],
            "candidate_v26_physics": failed_ref["candidate_physics"],
            "candidate_v26_mae_ms": failed_ref["candidate_timing"]["mae_ms"],
        })
        plans.append({
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "initial_seed": optimized["seed"],
            "actions_8k": [list(frame) for frame in optimized["best_commands"]],
            "modified_press_runs": modifications,
            "independently_replayed": True,
            "verified_hits": best.stats["hits"],
            "verified_mae_ms": best.objective_ms,
        })
        print(
            f"Stage C v2.7 Train {anchor:02d}/20: "
            f"SAFE H={best.stats['hits']}/{best.stats['targets']} "
            f"MAE={initial.objective_ms:.3f}->{best.objective_ms:.3f}ms "
            f"feasible={optimized['counts']['physically_safe']}/"
            f"{optimized['counts']['proposals']} "
            f"best_updates={optimized['counts']['new_bests']}",
            flush=True,
        )

    if ({key: p.sha256(path) for key, path in protected.items()} != before
            or p._protect_old_best(config) != best_before):
        raise RuntimeError("Frozen reports or checkpoints changed during global annealing")
    aggregate = _all_safe_aggregate(rows)
    baseline = {
        "physics": prev24["candidate"],
        "timing": prev25["timing_quality_candidate"],
    }
    if aggregate != prev24["candidate"]:
        raise RuntimeError("Global best violated the frozen SAFE850 baseline")
    all_quality = v25.summarize_hit_quality(best_events)
    gates = _report_gate(aggregate, all_quality, baseline)

    plans_document = {
        "format": PLAN_FORMAT, "stage": "C-v2.7", "role": "Train20 only",
        "no_training": True, "student_checkpoint_loaded": False,
        "validation_evaluated": False, "final_evaluated": False,
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(),
        "train_manifest": manifest,
        "control_dt_s": p.CONTROL_DT, "physics_dt_s": p.PHYSICS_DT,
        "plans": plans,
    }
    # Both outputs must be absent from the beginning; writes are non-replacing.
    # Plan is written first so the report can bind its exact SHA256.
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(plan_path, plans_document)
    saved_plan_sha = p.sha256(plan_path)
    report = {
        "format": FORMAT, "stage": "C-v2.7", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "original_stage_a_gate": stage_a["gate"],
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(), "train_manifest": manifest,
        "search": {
            "kind": "simulated_annealing_fixed_protocol",
            "variable": "bounded per-finger positive-pulse onset delay, 10ms ticks",
            "mutation": "single or neighboring 2-3 press-run delays",
            "candidate_validation": "true full-chart 1ms motor physics",
            "feasibility": "full Hit, no Miss/TooEarly/Overload/extra KeyDown",
            "anneal_uphill": "only physically feasible worse-MAE candidates",
            "start_temperature_ms": INITIAL_TEMPERATURE_MS,
            "end_temperature_ms": FINAL_TEMPERATURE_MS,
            "proposals_per_anchor": PROPOSALS_PER_ANCHOR,
            "max_delay_ticks": MAX_DELAY_TICKS,
            "snapshot_stride_frames": SNAPSHOT_STRIDE,
            "full_independent_final_replay": True,
            "hyperparameter_sweep": False,
        },
        "baseline_v24": baseline,
        "reference_v26": {
            "physics": prev26["candidate"]["physics"],
            "timing": prev26["candidate"]["timing"],
        },
        "candidate": {
            "physics": aggregate,
            "timing": all_quality,
            "sum_proposals": sum(row["annealing"]["proposals"] for row in rows),
            "sum_safe_proposals": sum(
                row["annealing"]["physically_safe"] for row in rows),
            "sum_modified_press_runs": sum(
                row["changed_press_runs"] for row in rows),
        },
        "gates": gates,
        "per_anchor": rows,
        "saved_plan_file": PLANS,
        "saved_plan_sha256": saved_plan_sha,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    p._no_replace_json(output, report)
    return report
