from __future__ import annotations

"""Stage C v2.9: original Train20-only full-phase + local-pulse SA evaluation."""

import json

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_quality as quality
from . import prestige1_timing_comparison as v26
from . import prestige1_global_comparison as v27
from . import prestige1_pulse_comparison as v28
from . import prestige1_transfer as transfer
from .prestige1_phase_annealing import (
    LOCAL_PROPOSALS, MAX_GLOBAL_PHASE_TICKS, optimize_phase_and_pulses,
)

FORMAT = "prestige1-stage-c-v29-train-only-phase-aligned-annealing"
PLAN_FORMAT = "prestige1-stage-c-v29-train-only-verified-phase-plans"
REPORT = "stage_c_v29_phase_aligned_annealing.json"
PLANS = "stage_c_v29_safe_teacher_plans.json"


def _sources(config, stage_a, stage_c, manifest):
    root = transfer.output_dir(config)
    v27_report, v27_plans, protected = v28._load_and_protect_sources(
        config, stage_a, stage_c, manifest)
    report_file = root / v28.REPORT
    plan_file = root / v28.PLANS
    prior = json.loads(report_file.read_text(encoding="utf-8"))
    prior_plans = json.loads(plan_file.read_text(encoding="utf-8"))
    if (
        prior.get("format") != v28.FORMAT
        or prior.get("stage") != "C-v2.8"
        or prior.get("role") != "Train20 only"
        or prior.get("no_training") is not True
        or prior.get("read_only") is not True
        or prior.get("checkpoint_modified") is not False
        or prior.get("student_checkpoint_loaded") is not False
        or prior.get("validation_evaluated") is not False
        or prior.get("final_evaluated") is not False
        or prior.get("validation_used_for_selection") is not False
        or prior.get("stage_a_gate_overridden") is not False
        or prior.get("original_stage_a_gate") != stage_a["gate"]
        or prior.get("config_sha256") != config.signature()
        or prior.get("train_manifest") != manifest
        or prior.get("protected_inputs_unchanged") is not True
        or prior.get("protected_input_sha256") != {
            key: p.sha256(path) for key, path in protected.items()
        }
        or prior.get("gates", {}).get("full_train20_safe850") is not True
        or prior.get("gates", {}).get("timing_mae_below_v27") is not True
        or prior.get("gates", {}).get("student_training_permitted") is not False
        or prior.get("gates", {}).get("automatically_adopted") is not False
        or prior.get("baseline_v27", {}).get("physics") != v27_report["candidate"]["physics"]
        or prior.get("baseline_v27", {}).get("timing") != v27_report["candidate"]["timing"]
        or prior.get("candidate", {}).get("physics") != v27_report["candidate"]["physics"]
        or len(prior.get("per_anchor", [])) != 20
        or prior.get("saved_plan_file") != v28.PLANS
        or prior.get("saved_plan_sha256") != p.sha256(plan_file)
        or prior.get("original_stage_a_gate", {}).get("may_run_stage_c") is not False
    ):
        raise RuntimeError("Stage C v2.8 frozen provenance mismatch")
    if (
        prior_plans.get("format") != v28.PLAN_FORMAT
        or prior_plans.get("stage") != "C-v2.8"
        or prior_plans.get("role") != "Train20 only"
        or prior_plans.get("no_training") is not True
        or prior_plans.get("student_checkpoint_loaded") is not False
        or prior_plans.get("validation_evaluated") is not False
        or prior_plans.get("final_evaluated") is not False
        or prior_plans.get("stage_a_gate_overridden") is not False
        or prior_plans.get("config_sha256") != config.signature()
        or prior_plans.get("train_manifest") != manifest
        or len(prior_plans.get("plans", [])) != 20
    ):
        raise RuntimeError("Stage C v2.8 action plan provenance mismatch")
    protected["stage_c_v28_full_pulse_annealing"] = report_file
    protected["stage_c_v28_safe_teacher_plans"] = plan_file
    return prior, prior_plans, protected


def run_train_phase_aligned_annealing(config):
    root = transfer.output_dir(config)
    output = root / REPORT
    saved_plans = root / PLANS
    if output.exists() or saved_plans.exists():
        raise FileExistsError("v2.9 reports and plans may never be replaced")
    old_best_sha = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 manifest changed")

    prior, plans, protected = _sources(config, stage_a, stage_c, manifest)
    before = {name: p.sha256(path) for name, path in protected.items()}

    rows = []
    saved = []
    hit_events = []
    for anchor, named in enumerate(segments, 1):
        previous = prior["per_anchor"][anchor - 1]
        plan = plans["plans"][anchor - 1]
        if (
            previous["anchor"] != anchor
            or plan["anchor"] != anchor
            or previous["chart"] != named.chart_name
            or plan["chart"] != named.chart_name
            or previous["chart_sha256"] != named.chart_sha256
            or plan["chart_sha256"] != named.chart_sha256
            or plan["verified_hits"] != previous["best_physics"]["hits"]
            or plan["verified_mae_ms"] is None
            or previous["best_physics"]["hits"] != previous["best_physics"]["targets"]
            or previous["best_physics"]["misses"] != 0
            or previous["best_physics"]["too_early"] != 0
            or previous["best_physics"]["overload"] is not False
            or plan["independently_replayed"] is not True
        ):
            raise RuntimeError(f"Frozen v2.8 anchor identity mismatch #{anchor:02d}")
        tape = tuple(tuple(float(v) for v in frame) for frame in plan["actions_8k"])
        if len(tape) < 1:
            raise RuntimeError("Empty frozen Stage C v2.8 action tape")
        solution = optimize_phase_and_pulses(
            named, tape, seed=int(config.seed) * 1019 + anchor)
        initial, phase, best = (
            solution["initial"], solution["phase_best"], solution["best"])
        if (
            initial.stats != previous["best_physics"]
            or initial.hit_events != previous["best_hit_events"]
            or initial.timing != previous["best_timing"]
            or initial.termination != previous["best_termination"]
            or abs(initial.objective_ms - plan["verified_mae_ms"]) > 1e-9
        ):
            raise RuntimeError(
                f"Independent v2.8 physical baseline mismatch #{anchor:02d}")
        if (not best.safe or best.stats != initial.stats
                or best.objective_ms > initial.objective_ms + 1e-9):
            raise RuntimeError(f"v2.9 best invalid or worse than v2.8 #{anchor:02d}")
        hit_events.extend(best.hit_events)

        pulse_data = solution["pulse_annealing"]
        moved = [
            {
                "finger": best.env.key_names[window.key_index],
                "positive_start_frame": window.start,
                "positive_end_frame_exclusive": window.positive_end,
                "release_end_frame_exclusive": window.tail_end,
                "offset_frames": shift,
            }
            for window, shift in zip(
                pulse_data["pulse_windows"], pulse_data["best_shifts"])
            if shift
        ]
        row = {
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline_physics": initial.stats,
            "final_physics": best.stats,
            "baseline_timing": initial.timing,
            "phase_timing": phase.timing,
            "final_timing": best.timing,
            "final_hit_events": best.hit_events,
            "final_termination": best.termination,
            "baseline_command_frames": len(tape),
            "global_phase_ticks": solution["phase_ticks"],
            "global_phase_candidates": solution["phase_candidates"],
            "global_phase_safe_candidates": solution["phase_safe_candidates"],
            "global_phase_unsafe_miss": solution["phase_unsafe_miss_candidates"],
            "global_phase_unsafe_early": solution["phase_unsafe_early_candidates"],
            "global_phase_unsafe_overload": solution["phase_unsafe_overload_candidates"],
            "eligible_local_pulses": len(pulse_data["pulse_windows"]),
            "moved_local_pulses": len(moved),
            "pulse_annealing": pulse_data["counts"],
        }
        rows.append(row)
        saved.append({
            "anchor": anchor, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "seed": solution["seed"],
            "actions_8k": [list(frame) for frame in solution["best_commands"]],
            "global_phase_ticks": solution["phase_ticks"],
            "moved_local_pulses": moved,
            "independently_replayed": True,
            "verified_hits": best.stats["hits"],
            "verified_mae_ms": best.objective_ms,
        })
        print(
            f"Stage C v2.9 Train {anchor:02d}/20: "
            f"H={best.stats['hits']}/{best.stats['targets']} "
            f"MAE={initial.objective_ms:.3f}->"
            f"{phase.objective_ms:.3f}->{best.objective_ms:.3f}ms "
            f"phase={solution['phase_ticks']}t, "
            f"phase_safe={solution['phase_safe_candidates']}/"
            f"{solution['phase_candidates']}, "
            f"pulses={len(moved)}/{len(pulse_data['pulse_windows'])}",
            flush=True,
        )

    if ({name: p.sha256(path) for name, path in protected.items()} != before
            or p._protect_old_best(config) != old_best_sha):
        raise RuntimeError("v2.9 protected inputs/checkpoints changed")
    aggregate = v27._all_safe_aggregate([
        {"anchor": row["anchor"], "best_physics": row["final_physics"]}
        for row in rows
    ])
    if aggregate != prior["candidate"]["physics"]:
        raise RuntimeError("Train20 whole-chart SAFE850 invariant violated")
    timing = quality.summarize_hit_quality(hit_events)
    old_timing = prior["candidate"]["timing"]
    if timing["mae_ms"] > old_timing["mae_ms"] + 1e-9:
        raise RuntimeError("Global phase/SA selected an inferior objective")

    documents = {
        "format": PLAN_FORMAT, "stage": "C-v2.9", "role": "Train20 only",
        "no_training": True, "student_checkpoint_loaded": False,
        "validation_evaluated": False, "final_evaluated": False,
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(), "train_manifest": manifest,
        "control_dt_s": p.CONTROL_DT, "physics_dt_s": p.PHYSICS_DT,
        "plans": saved,
    }
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(saved_plans, documents)
    plan_sha = p.sha256(saved_plans)
    report = {
        "format": FORMAT, "stage": "C-v2.9", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "stage_a_gate_overridden": False,
        "original_stage_a_gate": stage_a["gate"],
        "config_sha256": config.signature(), "train_manifest": manifest,
        "search": {
            "kind": "hierarchical_global_phase_then_full_pulse_sa",
            "global_variable": "whole 8-key 10ms waveform onset phase",
            "global_phase_max_ticks": MAX_GLOBAL_PHASE_TICKS,
            "global_search": "exhaustive single decision variable; no hyperparameter sweep",
            "local_proposals_per_anchor": LOCAL_PROPOSALS,
            "local_optimizer": "v2.8 full-pulse constrained SA with repair",
            "feasibility": "full-chart 1ms physics, Hit all, no early/overload/extra KeyDown",
            "independent_final_full_chart_replay": True,
            "student_training": False,
            "validation_or_final_selection": False,
            "hyperparameter_sweep": False,
        },
        "baseline_v28": {
            "physics": prior["candidate"]["physics"],
            "timing": old_timing,
            "plan_sha256": prior["saved_plan_sha256"],
        },
        "candidate": {
            "physics": aggregate, "timing": timing,
            "global_phase_candidates": sum(row["global_phase_candidates"] for row in rows),
            "global_phase_safe": sum(row["global_phase_safe_candidates"] for row in rows),
            "selected_global_phase_ticks": [row["global_phase_ticks"] for row in rows],
            "local_main_proposals": sum(
                row["pulse_annealing"]["main_proposals"] for row in rows),
            "local_physically_safe": sum(
                row["pulse_annealing"]["physically_safe_main"] for row in rows),
            "local_repair_safe": sum(
                row["pulse_annealing"]["repair_safe"] for row in rows),
            "moved_local_pulses": sum(row["moved_local_pulses"] for row in rows),
        },
        "gates": {
            "full_train20_safe850": v26._safety_gate(
                aggregate, prior["candidate"]["physics"]),
            "timing_mae_below_v28": timing["mae_ms"] < old_timing["mae_ms"],
            "student_training_permitted": False,
            "original_stage_a_gate_overridden": False,
            "automatically_adopted": False,
        },
        "per_anchor": rows,
        "saved_plan_file": PLANS,
        "saved_plan_sha256": plan_sha,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    p._no_replace_json(output, report)
    return report
