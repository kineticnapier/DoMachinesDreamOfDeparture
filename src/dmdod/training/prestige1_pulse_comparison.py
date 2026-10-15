from __future__ import annotations

"""Stage C v2.8 Train20-only safeguarded full-pulse annealing comparison."""

import json

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_teacher_safety as v23
from . import prestige1_actuator_comparison as v24
from . import prestige1_teacher_quality as v25
from . import prestige1_timing_comparison as v26
from . import prestige1_global_comparison as v27
from . import prestige1_transfer as transfer
from .prestige1_pulse_annealing import (
    START_TEMPERATURE_MS, END_TEMPERATURE_MS,
    MAX_SHIFT_TICKS, PROPOSALS_PER_ANCHOR, REPAIR_TRIALS_PER_UNSAFE,
    search_chart,
)

FORMAT = "prestige1-stage-c-v28-train-only-full-pulse-annealing"
PLAN_FORMAT = "prestige1-stage-c-v28-train-only-verified-pulse-plans"
REPORT = "stage_c_v28_full_pulse_annealing.json"
PLANS = "stage_c_v28_safe_teacher_plans.json"


def _load_and_protect_sources(config, stage_a, stage_c, manifest):
    root = transfer.output_dir(config)
    source_files = {
        "stage_c_v23_teacher_safety": root / v23.REPORT,
        "stage_c_v24_actuator_teacher_safety": root / v24.REPORT,
        "stage_c_v25_teacher_quality": root / v25.REPORT,
        "stage_c_v26_timing_centered_teacher": root / v26.REPORT,
        "stage_c_v27_global_annealing": root / v27.REPORT,
        "stage_c_v27_safe_teacher_plans": root / v27.PLANS,
    }
    prior = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in source_files.items()
    }
    v27._verify_frozen_provenance(
        config, stage_a, manifest,
        prior["stage_c_v23_teacher_safety"],
        prior["stage_c_v24_actuator_teacher_safety"],
        prior["stage_c_v25_teacher_quality"],
        prior["stage_c_v26_timing_centered_teacher"],
    )
    report = prior["stage_c_v27_global_annealing"]
    plans = prior["stage_c_v27_safe_teacher_plans"]
    if (
        report.get("format") != v27.FORMAT
        or report.get("stage") != "C-v2.7"
        or report.get("role") != "Train20 only"
        or report.get("no_training") is not True
        or report.get("read_only") is not True
        or report.get("checkpoint_modified") is not False
        or report.get("student_checkpoint_loaded") is not False
        or report.get("validation_evaluated") is not False
        or report.get("final_evaluated") is not False
        or report.get("validation_used_for_selection") is not False
        or report.get("stage_a_gate_overridden") is not False
        or report.get("original_stage_a_gate") != stage_a["gate"]
        or report.get("config_sha256") != config.signature()
        or report.get("train_manifest") != manifest
        or report.get("protected_inputs_unchanged") is not True
        or report.get("gates", {}).get("full_train20_safe850") is not True
        or report.get("gates", {}).get("student_training_permitted") is not False
        or report.get("gates", {}).get("automatically_adopted") is not False
        or len(report.get("per_anchor", [])) != 20
        or report.get("saved_plan_file") != v27.PLANS
        or report.get("saved_plan_sha256") != p.sha256(source_files[
            "stage_c_v27_safe_teacher_plans"])
    ):
        raise RuntimeError("Frozen Stage C v2.7 report provenance mismatch")
    if (
        plans.get("format") != v27.PLAN_FORMAT
        or plans.get("stage") != "C-v2.7"
        or plans.get("role") != "Train20 only"
        or plans.get("no_training") is not True
        or plans.get("student_checkpoint_loaded") is not False
        or plans.get("validation_evaluated") is not False
        or plans.get("final_evaluated") is not False
        or plans.get("stage_a_gate_overridden") is not False
        or plans.get("config_sha256") != config.signature()
        or plans.get("train_manifest") != manifest
        or len(plans.get("plans", [])) != 20
        or report["candidate"]["physics"] !=
            prior["stage_c_v24_actuator_teacher_safety"]["candidate"]
        or report["candidate"]["timing"]["hits"] != 850
    ):
        raise RuntimeError("Stage C v2.7 safe action plan provenance mismatch")

    protected = audit._protected_paths(config, stage_c)
    protected["stage_c_v22_train_audit"] = root / audit.REPORT
    for name, path in source_files.items():
        if name not in (
            "stage_c_v27_global_annealing",
            "stage_c_v27_safe_teacher_plans",
        ):
            protected[name] = path
    frozen_prior = {key: p.sha256(path) for key, path in protected.items()}
    if frozen_prior != report["protected_input_sha256"]:
        raise RuntimeError("Frozen v2.7 protected inputs have changed")
    protected["stage_c_v27_global_annealing"] = source_files[
        "stage_c_v27_global_annealing"]
    protected["stage_c_v27_safe_teacher_plans"] = source_files[
        "stage_c_v27_safe_teacher_plans"]
    return report, plans, protected


def run_train_full_pulse_annealing(config):
    root = transfer.output_dir(config)
    output = root / REPORT
    saved_plans = root / PLANS
    if output.exists() or saved_plans.exists():
        raise FileExistsError("Refuse to replace v2.8 global report or plan")
    original_best = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 data split changed")
    previous, plans, protected = _load_and_protect_sources(
        config, stage_a, stage_c, manifest)
    before = {name: p.sha256(path) for name, path in protected.items()}

    rows = []
    saved = []
    all_hits = []
    for anchor, named in enumerate(segments, 1):
        prior_row = previous["per_anchor"][anchor - 1]
        prior_plan = plans["plans"][anchor - 1]
        if (
            prior_row["anchor"] != anchor
            or prior_plan["anchor"] != anchor
            or prior_row["chart"] != named.chart_name
            or prior_plan["chart"] != named.chart_name
            or prior_row["chart_sha256"] != named.chart_sha256
            or prior_plan["chart_sha256"] != named.chart_sha256
            or prior_plan["verified_hits"] != prior_row["best_physics"]["hits"]
            or prior_row["best_physics"]["misses"] != 0
            or prior_row["best_physics"]["too_early"] != 0
            or prior_row["best_physics"]["overload"] is not False
        ):
            raise RuntimeError(f"Frozen v2.7 Train anchor provenance mismatch #{anchor:02d}")
        tape = tuple(
            tuple(float(v) for v in frame)
            for frame in prior_plan["actions_8k"]
        )
        if len(tape) < 1:
            raise RuntimeError("Empty v2.7 safe action plan")

        solution = search_chart(
            named, tape, seed=int(config.seed) * 1013 + anchor)
        initial = solution["baseline"]
        best = solution["best"]
        if (
            initial.stats != prior_row["best_physics"]
            or initial.timing != prior_row["best_timing"]
            or initial.hit_events != prior_row["best_hit_events"]
            or initial.termination != prior_row["best_termination"]
            or abs(initial.objective_ms - prior_plan["verified_mae_ms"]) > 1e-9
        ):
            raise RuntimeError(
                f"Open-loop v2.7 initial action tape failed identity #{anchor:02d}")
        if not best.safe or best.stats != initial.stats:
            raise RuntimeError(
                f"v2.8 must keep all physical v2.7 guarantees #{anchor:02d}")
        if best.objective_ms > initial.objective_ms + 1e-9:
            raise RuntimeError(f"v2.8 regressed timing objective #{anchor:02d}")
        all_hits.extend(best.hit_events)

        moved = [
            {
                "finger": best.env.key_names[pulse.key_index],
                "source_start_frame": pulse.start,
                "positive_end_frame_exclusive": pulse.positive_end,
                "release_end_frame_exclusive": pulse.tail_end,
                "offset_frames": shift,
                "shifted_start_frame": pulse.start + shift,
            }
            for pulse, shift in zip(
                solution["pulse_windows"], solution["best_shifts"])
            if shift
        ]
        rows.append({
            "anchor": anchor,
            "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "baseline_physics": initial.stats,
            "best_physics": best.stats,
            "baseline_timing": initial.timing,
            "best_timing": best.timing,
            "best_hit_events": best.hit_events,
            "best_termination": best.termination,
            "baseline_command_frames": len(tape),
            "eligible_shiftable_pulses": len(solution["pulse_windows"]),
            "moved_pulses": len(moved),
            "annealing": solution["counts"],
        })
        saved.append({
            "anchor": anchor,
            "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "seed": solution["seed"],
            "actions_8k": [list(frame) for frame in solution["best_commands"]],
            "moved_pulses": moved,
            "independently_replayed": True,
            "verified_hits": best.stats["hits"],
            "verified_mae_ms": best.objective_ms,
        })
        print(
            f"Stage C v2.8 Train {anchor:02d}/20: "
            f"SAFE H={best.stats['hits']}/{best.stats['targets']} "
            f"MAE={initial.objective_ms:.3f}->{best.objective_ms:.3f}ms "
            f"pulse={len(moved)}/{len(solution['pulse_windows'])} "
            f"safe={solution['counts']['physically_safe_main']}/"
            f"{solution['counts']['main_proposals']} "
            f"repaired={solution['counts']['repair_selected']}",
            flush=True,
        )

    if ({name: p.sha256(path) for name, path in protected.items()} != before
            or p._protect_old_best(config) != original_best):
        raise RuntimeError("Protected source files/checkpoint changed during v2.8")

    aggregate = v27._all_safe_aggregate(rows)
    if aggregate != previous["candidate"]["physics"]:
        raise RuntimeError("Full Train20 physical SAFE850 invariant violated")
    timing = v25.summarize_hit_quality(all_hits)
    old_timing = previous["candidate"]["timing"]
    if timing["mae_ms"] is None or timing["mae_ms"] > old_timing["mae_ms"] + 1e-9:
        raise RuntimeError("v2.8 returned an inferior aggregate solution")

    safe_plans_document = {
        "format": PLAN_FORMAT, "stage": "C-v2.8", "role": "Train20 only",
        "no_training": True,
        "student_checkpoint_loaded": False,
        "validation_evaluated": False,
        "final_evaluated": False,
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(),
        "train_manifest": manifest,
        "control_dt_s": p.CONTROL_DT,
        "physics_dt_s": p.PHYSICS_DT,
        "plans": saved,
    }
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(saved_plans, safe_plans_document)
    plans_sha = p.sha256(saved_plans)
    physical_gate = v26._safety_gate(aggregate, previous["candidate"]["physics"])
    report = {
        "format": FORMAT, "stage": "C-v2.8", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "stage_a_gate_overridden": False,
        "original_stage_a_gate": stage_a["gate"],
        "config_sha256": config.signature(), "train_manifest": manifest,
        "search": {
            "kind": "constrained_full_pulse_sa_repair",
            "variable": "complete +1 waveform and contiguous -1 release, 10ms shift",
            "coupled_move": "up to three consecutive full pulses",
            "repair": "first physical miss, nearby pulse +/-10ms",
            "acceptance": "only 1ms-physics SAFE; feasible uphill MAE allowed",
            "proposals_per_anchor": PROPOSALS_PER_ANCHOR,
            "max_pulse_shift_frames": MAX_SHIFT_TICKS,
            "repair_trials_per_unsafe": REPAIR_TRIALS_PER_UNSAFE,
            "start_temperature_ms": START_TEMPERATURE_MS,
            "end_temperature_ms": END_TEMPERATURE_MS,
            "hyperparameter_sweep": False,
            "independent_full_chart_best_replay": True,
        },
        "baseline_v27": {
            "physics": previous["candidate"]["physics"],
            "timing": old_timing,
            "plan_sha256": previous["saved_plan_sha256"],
        },
        "candidate": {
            "physics": aggregate, "timing": timing,
            "main_proposals": sum(row["annealing"]["main_proposals"] for row in rows),
            "physically_safe_main": sum(
                row["annealing"]["physically_safe_main"] for row in rows),
            "repair_attempts": sum(
                row["annealing"]["repair_attempts"] for row in rows),
            "repaired_safe": sum(
                row["annealing"]["repair_safe"] for row in rows),
            "moved_pulses": sum(row["moved_pulses"] for row in rows),
        },
        "gates": {
            "full_train20_safe850": physical_gate,
            "timing_mae_below_v27": timing["mae_ms"] < old_timing["mae_ms"],
            "student_training_permitted": False,
            "original_stage_a_gate_overridden": False,
            "automatically_adopted": False,
        },
        "per_anchor": rows,
        "saved_plan_file": PLANS,
        "saved_plan_sha256": plans_sha,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    p._no_replace_json(output, report)
    return report
