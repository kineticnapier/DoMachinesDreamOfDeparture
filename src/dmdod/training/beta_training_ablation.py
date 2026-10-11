from __future__ import annotations

"""Controlled beta DAgger training ablation from the same selected-best state.

This is an experimental training runner, not curriculum progression:
* each arm restores the identical model weights AND AdamW moments;
* all arms use the same Train anchors, expert data, batch schedule seed,
  two optimizer steps, and fixed Validation anchors;
* beta is used ONLY in Train collection, never in evaluation;
* one arm caps beta=0 student failure-continuation at one second;
* no arm is automatically selected/promoted. Existing best is untouched.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch

from dmdod.training.beta_collection_probe import (
    BetaAnchorStats, format_beta_summary, summarize_rollout,
)
from dmdod.training.dagger_continuation import (
    collect_n_key_intervention_dagger_sequence_with_continuation,
)
from dmdod.training.human_visible_curriculum import (
    HumanVisibleTrajectory,
    _check_validation_context,
    _check_validation_summary,
    _collect_expert_trajectories,
    _load_checked_optimizer,
    build_balanced_epoch_plan,
    build_human_visible_optimizer,
    build_replay_windows,
    curriculum_checkpoint_role,
    stable_seed,
    train_human_visible_epoch,
    validation_summary,
)
from dmdod.training.real_chart import (
    clone_model_state, evaluate_role_continuous,
)


@dataclass(frozen=True, slots=True)
class BetaTrainingArm:
    name: str
    beta: float
    max_post_failure_s: float | None


ARMS = (
    BetaTrainingArm("beta1_full", 1.0, None),
    BetaTrainingArm("beta05_full", 0.5, None),
    BetaTrainingArm("beta0_full", 0.0, None),
    BetaTrainingArm("beta0_post1s", 0.0, 1.0),
)


def _validate_experiment(prepared, config, *, updates: int) -> int:
    if config.mode != "human_visible_curriculum_dagger":
        raise ValueError("beta ablation requires human_visible_curriculum_dagger")
    if len(prepared.anchors) != 20 or len(prepared.validation) != 20:
        raise ValueError("beta training ablation requires 20 Train and 20 fixed Validation anchors")
    if updates != 2 or config.human_visible.updates_per_epoch != 2:
        raise ValueError("beta training ablation requires exactly two optimizer updates")
    if config.human_visible.alpha_sweep_enabled:
        raise ValueError("alpha sweep must be disabled in beta training ablation")
    parent = prepared.parent
    if curriculum_checkpoint_role(parent) != "selected-best":
        raise ValueError("beta training source must be a selected-best curriculum checkpoint")
    if not isinstance(parent.get("human_visible_best_validation_summary"), dict):
        raise ValueError("source checkpoint lacks recorded best Validation")
    _check_validation_context(parent, config, prepared)
    epoch = int(parent.get("human_visible_best_epoch", parent.get("human_visible_epoch", -1))) + 1
    if epoch <= 0:
        raise ValueError("source checkpoint lacks valid epoch")
    return epoch


def _candidate_paths(report: Path) -> dict[str, Path]:
    return {
        arm.name: report.with_name(f"{report.stem}.{arm.name}.pt")
        for arm in ARMS
    }


def preflight_outputs(report: Path, *, save_candidates: bool, source_checkpoint: Path) -> dict[str, Path]:
    """Fail before work if a previous ablation report/checkpoint could be overwritten."""
    paths = _candidate_paths(report) if save_candidates else {}
    if report.resolve() == source_checkpoint.resolve() or report.exists():
        raise FileExistsError(f"beta ablation report path already exists or conflicts: {report}")
    all_paths = [report, *paths.values()]
    if len({p.resolve() for p in all_paths}) != len(all_paths):
        raise ValueError("beta ablation output paths must be unique")
    for path in paths.values():
        if path.exists() or path.resolve() == source_checkpoint.resolve():
            raise FileExistsError(f"beta ablation candidate path already exists or conflicts: {path}")
    return paths


def _collect_variant(
    prepared, model, arm: BetaTrainingArm, *, epoch: int, seed: int,
):
    trajectories: list[HumanVisibleTrajectory] = []
    summaries: list[BetaAnchorStats] = []
    for anchor_id, named in enumerate(prepared.anchors):
        # Same per-anchor coin-flip stream across betas. Use the original
        # curriculum seed convention to reproduce beta=1 reference training.
        rollout = collect_n_key_intervention_dagger_sequence_with_continuation(
            model, named.segment,
            lead_s=prepared.lead_s,
            beta=arm.beta,
            seed=stable_seed(seed, "beta", epoch, anchor_id),
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            source=f"ablation-{arm.name}-e{epoch}-{anchor_id}-{named.chart_name}",
            action_mode="continuous",
            continue_after_failure=True,
            max_post_failure_s=arm.max_post_failure_s,
        )
        trajectories.append(
            HumanVisibleTrajectory(
                anchor_id=anchor_id,
                source_kind="dagger",
                observations=rollout.sequence.observations,
                teacher_actions=rollout.sequence.teacher_actions,
                student_actions=rollout.student_actions,
            )
        )
        summaries.append(
            summarize_rollout(
                rollout, beta=arm.beta, anchor_index=anchor_id + 1,
                chart_name=named.chart_name,
            )
        )
        print(
            f"{arm.name} collect {anchor_id + 1:02d}/{len(prepared.anchors)} "
            f"H={rollout.stats.hits}/{rollout.stats.targets} "
            f"overload={int(rollout.stats.overloaded)} "
            f"frames={rollout.sequence.frames} "
            f"post={summaries[-1].post_terminal_frames}"
        )
    return trajectories, tuple(summaries)


def _assert_beta_reference(summary: dict, expected: tuple[int, int, int, int] | None) -> None:
    if expected is None:
        return
    fields = ("safe", "hits", "early", "keydowns")
    observed = tuple(int(summary[field]) for field in fields)
    if observed != expected:
        raise RuntimeError(
            f"beta=1 reference candidate mismatch: actual={observed}, "
            f"expected={expected} ([SAFE, Hits, TooEarly, Keydowns]). "
            "Do not interpret the other arms before checking reproducibility."
        )


def run_beta_training_ablation(
    prepared,
    config,
    *,
    report_path: Path,
    save_candidates: bool = False,
    expected_beta1: tuple[int, int, int, int] | None = (9, 1546, 250, 1796),
    updates: int = 2,
) -> dict:
    """Evaluate each trained arm once; report all arms without choosing a winner."""
    epoch = _validate_experiment(prepared, config, updates=updates)
    paths = preflight_outputs(
        report_path, save_candidates=save_candidates,
        source_checkpoint=prepared.source_checkpoint,
    )
    seed = int(config.human_visible.seed)
    model = prepared.model
    initial_state = clone_model_state(model)
    visible = config.human_visible
    print("=== DMDOD beta training ablation (NOT curriculum selection) ===")
    print(
        f"source={prepared.source_checkpoint} "
        f"Train={len(prepared.anchors)} fixed-Validation={len(prepared.validation)} "
        f"epoch={epoch} updates/arm={updates} device={prepared.device}"
    )

    # This baseline verifies that the common starting checkpoint and fixed
    # Validation set reproduce their recorded scores BEFORE any optimizer step.
    baseline_results = evaluate_role_continuous(
        model, prepared.validation, label="ablation-start",
        control_dt_s=prepared.control_dt_s, physics_dt_s=prepared.physics_dt_s,
        device=prepared.device, verbose=False,
    )
    baseline = validation_summary(baseline_results)
    _check_validation_summary(
        baseline, prepared.parent["human_visible_best_validation_summary"]
    )
    print(
        f"verified baseline SAFE={baseline['safe']}/{baseline['anchors']} "
        f"H={baseline['hits']}/{baseline['targets']}"
    )

    # Expert trajectories do not depend on beta or model parameters.
    expert = _collect_expert_trajectories(prepared)
    print(f"static expert frames={sum(t.frames for t in expert)}")
    results: list[dict] = []
    candidate_files: list[Path] = []
    for arm in ARMS:
        model.load_state_dict(initial_state, strict=True)
        model.prepare_recurrent_runtime()
        optimizer = build_human_visible_optimizer(model, visible)
        _load_checked_optimizer(optimizer, prepared.parent, model)
        random.seed(stable_seed(seed, "ablation-random", epoch))
        torch.manual_seed(stable_seed(seed, "ablation-torch", epoch) % (2 ** 63))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stable_seed(seed, "ablation-cuda", epoch) % (2 ** 63))

        print(f"=== arm={arm.name} beta={arm.beta:.2f} post-limit={arm.max_post_failure_s} ===")
        dagger, collected = _collect_variant(
            prepared, model, arm, epoch=epoch, seed=seed,
        )
        for line in format_beta_summary(collected):
            print(line)
        windows = build_replay_windows(
            model, [*expert, *dagger],
            burn_in_steps=visible.burn_in_steps,
            supervised_steps=visible.supervised_steps,
        )
        plan = build_balanced_epoch_plan(
            windows, anchor_ids=list(range(len(prepared.anchors))),
            updates_per_epoch=updates, anchors_per_batch=visible.anchors_per_batch,
            seed=stable_seed(seed, "plan", epoch),
        )
        metrics = train_human_visible_epoch(
            model, windows, plan, optimizer=optimizer,
            grad_clip=visible.grad_clip, control_dt_s=prepared.control_dt_s,
        )
        if metrics.updates != updates:
            raise RuntimeError("optimizer update count mismatch")

        # Evaluation is strictly student-only (beta is not passed to evaluator).
        validation_results = evaluate_role_continuous(
            model, prepared.validation,
            label=f"ablation-val-{arm.name}",
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s, device=prepared.device,
            verbose=False,
        )
        summary = validation_summary(validation_results)
        print(
            f"RESULT {arm.name}: SAFE={summary['safe']}/{summary['anchors']} "
            f"H={summary['hits']}/{summary['targets']} "
            f"Early={summary['early']} Keydowns={summary['keydowns']} "
            f"updates={metrics.updates}"
        )
        if arm.name == "beta1_full":
            _assert_beta_reference(summary, expected_beta1)

        arm_data = {
            "name": arm.name,
            "beta": arm.beta,
            "max_post_failure_s": arm.max_post_failure_s,
            "training": asdict(metrics),
            "collection": [asdict(row) for row in collected],
            "validation": summary,
            "replay_windows": len(windows),
            "selected_for_checkpoint": False,
        }
        if save_candidates:
            path = paths[arm.name]
            path.parent.mkdir(parents=True, exist_ok=True)
            # Experimental checkpoint deliberately cannot be mistaken for a
            # selected-best curriculum checkpoint.
            checkpoint = dict(prepared.parent)
            checkpoint.update(model.checkpoint_metadata())
            checkpoint.update({
                "model_state": clone_model_state(model),
                "training_mode": "human_visible_beta_ablation_candidate",
                "human_visible_checkpoint_role": "diagnostic-candidate",
                "beta_ablation_arm": arm.name,
                "beta_ablation_beta": arm.beta,
                "beta_ablation_max_post_failure_s": arm.max_post_failure_s,
                "beta_ablation_validation_summary": summary,
                "beta_ablation_updates": updates,
                "beta_ablation_source_checkpoint": str(prepared.source_checkpoint),
            })
            with path.open("xb") as handle:
                torch.save(checkpoint, handle)
            candidate_files.append(path)
            arm_data["candidate_checkpoint"] = str(path)
        results.append(arm_data)
        del optimizer, windows, dagger, collected

    report = {
        "kind": "human_visible_beta_training_ablation_v1",
        "source_checkpoint": str(prepared.source_checkpoint),
        "source_epoch": epoch - 1,
        "training_epoch": epoch,
        "seed": seed,
        "anchors": len(prepared.anchors),
        "validation_anchors": len(prepared.validation),
        "optimizer_updates_per_arm": updates,
        "max_post_failure_s_applied_to": "beta0_post1s",
        "baseline_validation": baseline,
        "arms": results,
        "notes": [
            "All arms start from the same selected-best model and saved AdamW moments.",
            "All arms use the same fixed Train anchors, expert data, plan seed and update count.",
            "Training collects teacher labels along executed mixed-policy trajectories.",
            "Fixed Validation contains student-only rollouts and was not used for updates.",
            "The full failure continuation trajectory is diagnostic beyond ordinary termination.",
            "No arm was automatically promoted or selected. Holdout remains untouched.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"ablation report: {report_path}")
    if candidate_files:
        print("experimental checkpoints (not selected best): " + ", ".join(map(str, candidate_files)))
    print("No changes made to the source selected-best checkpoint or FINAL holdout.")
    return report
