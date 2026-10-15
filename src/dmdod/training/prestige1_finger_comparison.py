from __future__ import annotations

"""Stage C v3.0 Train20-only per-finger phase vector + whole-pulse SA."""

import json

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_quality as quality
from . import prestige1_timing_comparison as v26
from . import prestige1_global_comparison as v27
from . import prestige1_phase_comparison as v29
from . import prestige1_transfer as transfer
from .prestige1_finger_phase import (
    FINGER_PHASE_LIMIT, PHASE_CHOICES, LOCAL_PROPOSALS,
    optimize_finger_phases_and_pulses,
)

FORMAT = "prestige1-stage-c-v30-train-only-finger-phase-annealing"
PLAN_FORMAT = "prestige1-stage-c-v30-train-only-verified-finger-plans"
REPORT = "stage_c_v30_finger_phase_annealing.json"
PLANS = "stage_c_v30_safe_teacher_plans.json"


def _v29_sources(config, stage_a, stage_c, manifest):
    root = transfer.output_dir(config)
    prior_v28, _, protected = v29._sources(
        config, stage_a, stage_c, manifest)
    report_file, plan_file = root / v29.REPORT, root / v29.PLANS
    old = json.loads(report_file.read_text(encoding="utf-8"))
    plans = json.loads(plan_file.read_text(encoding="utf-8"))
    if (
        old.get("format") != v29.FORMAT
        or old.get("stage") != "C-v2.9"
        or old.get("role") != "Train20 only"
        or old.get("no_training") is not True
        or old.get("read_only") is not True
        or old.get("checkpoint_modified") is not False
        or old.get("student_checkpoint_loaded") is not False
        or old.get("validation_evaluated") is not False
        or old.get("final_evaluated") is not False
        or old.get("validation_used_for_selection") is not False
        or old.get("stage_a_gate_overridden") is not False
        or old.get("original_stage_a_gate") != stage_a["gate"]
        or old.get("original_stage_a_gate", {}).get("may_run_stage_c") is not False
        or old.get("config_sha256") != config.signature()
        or old.get("train_manifest") != manifest
        or old.get("protected_inputs_unchanged") is not True
        or old.get("protected_input_sha256") != {
            name: p.sha256(path) for name, path in protected.items()
        }
        or old.get("saved_plan_file") != v29.PLANS
        or old.get("saved_plan_sha256") != p.sha256(plan_file)
        or old.get("baseline_v28", {}).get("physics") != prior_v28["candidate"]["physics"]
        or old.get("baseline_v28", {}).get("timing") != prior_v28["candidate"]["timing"]
        or old.get("candidate", {}).get("physics") != prior_v28["candidate"]["physics"]
        or old.get("candidate", {}).get("timing", {}).get("hits") != 850
        or old.get("gates", {}).get("full_train20_safe850") is not True
        or old.get("gates", {}).get("timing_mae_below_v28") is not True
        or old.get("gates", {}).get("student_training_permitted") is not False
        or old.get("gates", {}).get("automatically_adopted") is not False
        or len(old.get("per_anchor", ())) != 20
    ):
        raise RuntimeError("Stage C v2.9 frozen report provenance mismatch")
    if (
        plans.get("format") != v29.PLAN_FORMAT
        or plans.get("stage") != "C-v2.9"
        or plans.get("role") != "Train20 only"
        or plans.get("no_training") is not True
        or plans.get("student_checkpoint_loaded") is not False
        or plans.get("validation_evaluated") is not False
        or plans.get("final_evaluated") is not False
        or plans.get("stage_a_gate_overridden") is not False
        or plans.get("config_sha256") != config.signature()
        or plans.get("train_manifest") != manifest
        or len(plans.get("plans", ())) != 20
    ):
        raise RuntimeError("Stage C v2.9 frozen action plans mismatch")
    protected["stage_c_v29_phase_aligned_annealing"] = report_file
    protected["stage_c_v29_safe_teacher_plans"] = plan_file
    return old, plans, protected


def run_train_finger_phase_annealing(config):
    root = transfer.output_dir(config)
    report_path, plans_path = root / REPORT, root / PLANS
    if report_path.exists() or plans_path.exists():
        raise FileExistsError("Refuse to overwrite Stage C v3.0 artifacts")
    old_best = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 manifest mismatch")

    old, plans, protected = _v29_sources(config, stage_a, stage_c, manifest)
    before = {name: p.sha256(path) for name, path in protected.items()}
    rows, saved, all_hit_events = [], [], []
    for anchor, named in enumerate(segments, 1):
        previous = old["per_anchor"][anchor - 1]
        plan = plans["plans"][anchor - 1]
        if (
            previous["anchor"] != anchor
            or plan["anchor"] != anchor
            or previous["chart"] != named.chart_name
            or plan["chart"] != named.chart_name
            or previous["chart_sha256"] != named.chart_sha256
            or plan["chart_sha256"] != named.chart_sha256
            or plan["verified_hits"] != previous["final_physics"]["hits"]
            or plan["independently_replayed"] is not True
            or previous["final_physics"]["hits"] != previous["final_physics"]["targets"]
            or previous["final_physics"]["misses"] != 0
            or previous["final_physics"]["too_early"] != 0
            or previous["final_physics"]["overload"] is not False
        ):
            raise RuntimeError(f"v2.9 Train anchor #{anchor:02d} provenance mismatch")
        original = tuple(tuple(float(x) for x in frame) for frame in plan["actions_8k"])
        if not original:
            raise RuntimeError("Empty frozen v2.9 action tape")
        selected = optimize_finger_phases_and_pulses(
            named, original, seed=int(config.seed) * 1021 + anchor)
        initial, phase, best = (
            selected["initial"], selected["phase_best"], selected["best"])
        if (
            initial.stats != previous["final_physics"]
            or initial.timing != previous["final_timing"]
            or initial.hit_events != previous["final_hit_events"]
            or initial.termination != previous["final_termination"]
            or abs(initial.objective_ms - plan["verified_mae_ms"]) > 1e-9
        ):
            raise RuntimeError(
                f"v2.9 independent full-chart replay diverged #{anchor:02d}")
        if (
            not best.safe or best.stats != initial.stats
            or best.objective_ms > initial.objective_ms + 1e-9
        ):
            raise RuntimeError(f"v3.0 final replay is unsafe or regressed #{anchor:02d}")
        all_hit_events.extend(best.hit_events)
        phase_data = selected["finger_phase"]
        local_data = selected["pulse_annealing"]
        moved = [
            {
                "finger": best.env.key_names[window.key_index],
                "positive_start_frame": window.start,
                "positive_end_frame_exclusive": window.positive_end,
                "release_end_frame_exclusive": window.tail_end,
                "offset_frames": shift,
            }
            for window, shift in zip(
                local_data["pulse_windows"], local_data["best_shifts"])
            if shift
        ]
        rows.append({
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline_physics": initial.stats, "final_physics": best.stats,
            "baseline_timing": initial.timing,
            "finger_phase_timing": phase.timing,
            "final_timing": best.timing,
            "final_hit_events": best.hit_events,
            "final_termination": best.termination,
            "finger_phase_offsets": list(phase_data.offsets),
            "finger_phase_checked": phase_data.checked,
            "finger_phase_safe": phase_data.physically_safe,
            "finger_phase_unsafe_miss": phase_data.unsafe_miss,
            "finger_phase_unsafe_too_early": phase_data.unsafe_too_early,
            "finger_phase_unsafe_overload": phase_data.unsafe_overload,
            "finger_phase_invalid_boundary": phase_data.invalid_boundary,
            "finger_phase_improved_keys": phase_data.improved_keys,
            "finger_phase_active_fingers": phase_data.active_fingers,
            "eligible_local_pulses": len(local_data["pulse_windows"]),
            "moved_local_pulses": len(moved),
            "pulse_annealing": local_data["counts"],
        })
        saved.append({
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "seed": selected["seed"],
            "actions_8k": [list(frame) for frame in selected["best_commands"]],
            "finger_phase_offsets": list(phase_data.offsets),
            "moved_local_pulses": moved,
            "independently_replayed": True,
            "verified_hits": best.stats["hits"],
            "verified_mae_ms": best.objective_ms,
        })
        print(
            f"Stage C v3.0 Train {anchor:02d}/20: "
            f"SAFE H={best.stats['hits']}/{best.stats['targets']} "
            f"MAE={initial.objective_ms:.3f}->"
            f"{phase.objective_ms:.3f}->{best.objective_ms:.3f}ms "
            f"finger_offsets={list(phase_data.offsets)} "
            f"phase_safe={phase_data.physically_safe}/{phase_data.checked} "
            f"local_pulses={len(moved)}", flush=True,
        )

    if ({name: p.sha256(path) for name, path in protected.items()} != before
            or p._protect_old_best(config) != old_best):
        raise RuntimeError("Protected provenance/checkpoints changed in v3.0")
    physical = v27._all_safe_aggregate([
        {"anchor": row["anchor"], "best_physics": row["final_physics"]}
        for row in rows
    ])
    if physical != old["candidate"]["physics"]:
        raise RuntimeError("Full Train20 850-Hit physical invariant violated")
    timing = quality.summarize_hit_quality(all_hit_events)
    reference = old["candidate"]["timing"]
    if timing["mae_ms"] is None or timing["mae_ms"] > reference["mae_ms"] + 1e-9:
        raise RuntimeError("v3.0 aggregate timing regressed below safe v2.9")

    plans_document = {
        "format": PLAN_FORMAT, "stage": "C-v3.0", "role": "Train20 only",
        "no_training": True, "student_checkpoint_loaded": False,
        "validation_evaluated": False, "final_evaluated": False,
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(), "train_manifest": manifest,
        "control_dt_s": p.CONTROL_DT, "physics_dt_s": p.PHYSICS_DT,
        "finger_order": ["left_4", "left_3", "left_2", "left_1",
                         "right_1", "right_2", "right_3", "right_4"],
        "plans": saved,
    }
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(plans_path, plans_document)
    result = {
        "format": FORMAT, "stage": "C-v3.0", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "stage_a_gate_overridden": False,
        "original_stage_a_gate": stage_a["gate"],
        "config_sha256": config.signature(), "train_manifest": manifest,
        "search": {
            "kind": "per_finger_waveform_coordinate_phase_then_full_pulse_sa",
            "finger_phase_limit_ticks": FINGER_PHASE_LIMIT,
            "offset_candidates": list(PHASE_CHOICES),
            "sweep_passes": 1,
            "finger_order": "descending source positive command frames",
            "local_proposals_per_anchor": LOCAL_PROPOSALS,
            "local_optimizer": "v2.8 full-pulse SA + physical miss repair",
            "candidate_scoring": "complete chart 1ms physics, feasible-only MAE",
            "independent_full_episode_verification": True,
            "hyperparameter_sweep": False,
        },
        "baseline_v29": {
            "physics": old["candidate"]["physics"],
            "timing": reference,
            "plan_sha256": old["saved_plan_sha256"],
        },
        "candidate": {
            "physics": physical, "timing": timing,
            "finger_phase_checked": sum(x["finger_phase_checked"] for x in rows),
            "finger_phase_safe": sum(x["finger_phase_safe"] for x in rows),
            "finger_phase_improved_keys": sum(x["finger_phase_improved_keys"] for x in rows),
            "finger_phase_invalid_boundary": sum(
                x["finger_phase_invalid_boundary"] for x in rows),
            "local_main_proposals": sum(
                x["pulse_annealing"]["main_proposals"] for x in rows),
            "local_physically_safe": sum(
                x["pulse_annealing"]["physically_safe_main"] for x in rows),
            "local_repaired_safe": sum(
                x["pulse_annealing"]["repair_safe"] for x in rows),
            "moved_local_pulses": sum(x["moved_local_pulses"] for x in rows),
        },
        "gates": {
            "full_train20_safe850": v26._safety_gate(
                physical, old["candidate"]["physics"]),
            "timing_mae_below_v29": timing["mae_ms"] < reference["mae_ms"],
            "student_training_permitted": False,
            "original_stage_a_gate_overridden": False,
            "automatically_adopted": False,
        },
        "per_anchor": rows,
        "saved_plan_file": PLANS,
        "saved_plan_sha256": p.sha256(plans_path),
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    p._no_replace_json(report_path, result)
    return result
