from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch

from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import NKeyBCSequence, collect_n_key_expert_sequence

from .action_trust import (
    actor_parameter_names,
    build_action_trust_sequences,
    freeze_actor_only,
    train_actor_action_trust,
)
from .budget import NonBestRestartController, TrustRadiusController
from .config import TrainingConfig
from .human_visible import (
    build_human_visible_replay_chunks,
    freeze_human_visible_controller,
    human_visible_parameter_names,
    train_human_visible_replay,
)
from .real_chart import (
    aggregate,
    build_anchor_segments,
    build_connectome_policy_from_checkpoint,
    build_human_visible_policy_from_parent_checkpoint,
    build_validation_segments,
    clone_model_state,
    collect_student_state_sequences,
    compile_role,
    device_from_arg,
    evaluate_role_continuous,
    evaluate_role_survival_guarded,
    safe_anchor_count,
    save_checkpoint,
    selection_key,
    summarize,
    survival_selection_key,
    train_safety_guard,
)
from .trajectory_trust import (
    build_probe_candidate,
    collect_probe_training_sequences,
    format_probe_result,
    probe_anchor_pair,
    save_probe_report,
)


TRAINER_VERSION = "3.0.0-human-visible-controller"
CHECKPOINT_FORMAT_VERSION = 34


@dataclass(slots=True)
class PreparedRun:
    model: object
    parent: dict
    source_checkpoint: Path
    output_checkpoint: Path
    anchors: list
    validation: list
    round_index: int
    key_count: int
    input_dim: int
    lead_s: float
    control_dt_s: float
    physics_dt_s: float
    chunk_steps: int
    device: torch.device


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_checkpoint_path(output: Path) -> Path:
    return output.with_name(output.stem + ".progress" + output.suffix)


def _prepare(config: TrainingConfig) -> PreparedRun:
    device = device_from_arg(config.run.device)
    source_checkpoint = Path(config.run.checkpoint)
    output_checkpoint = Path(config.run.output)
    parent = torch.load(
        source_checkpoint,
        map_location=device,
        weights_only=False,
    )

    key_count = int(parent["key_count"])
    input_dim = int(parent["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match "
            f"{key_count}K expected {expected_input}"
        )

    if config.mode == "human_visible_dagger":
        model = build_human_visible_policy_from_parent_checkpoint(
            parent,
            device=device,
        )
        freeze_human_visible_controller(model)
    else:
        model = build_connectome_policy_from_checkpoint(
            parent,
            device=device,
        )
        freeze_actor_only(model)

    calibration = parent.get("calibration") or {}
    if "lead_s" not in calibration:
        raise SystemExit("checkpoint calibration is missing lead_s")

    chunk_steps = int(
        config.data.chunk_steps
        or parent.get("chunk_steps", 192)
    )
    dataset = discover_multichart_dataset(config.run.dataset)
    train_charts = compile_role(dataset.train)
    validation_charts = compile_role(dataset.validation)

    anchors = build_anchor_segments(
        train_charts,
        window_s=float(parent["train_window"]),
        anchors_per_chart=int(parent["anchors_per_chart"]),
    )
    validation = build_validation_segments(
        validation_charts,
        window_s=float(parent["validation_window"]),
    )

    anchor_limit = (
        config.data.anchor_limit
        if config.data.anchor_limit is not None
        else parent.get("anchor_limit")
    )
    if anchor_limit is not None:
        anchors = anchors[: int(anchor_limit)]

    validation_limit = (
        config.data.validation_limit
        if config.data.validation_limit is not None
        else parent.get("validation_limit")
    )
    if validation_limit is not None:
        validation = validation[: int(validation_limit)]

    if not anchors:
        raise SystemExit("no Train anchors selected")

    return PreparedRun(
        model=model,
        parent=parent,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        anchors=anchors,
        validation=validation,
        round_index=int(parent.get("dagger_round", 0)) + 1,
        key_count=key_count,
        input_dim=input_dim,
        lead_s=float(calibration["lead_s"]),
        control_dt_s=float(parent["control_dt"]),
        physics_dt_s=float(parent.get("physics_dt", 0.001)),
        chunk_steps=chunk_steps,
        device=device,
    )


def _checkpoint_payload(
    prepared: PreparedRun,
    config: TrainingConfig,
    *,
    model_state: dict[str, torch.Tensor],
    start_time: float,
    trial_count: int,
    accepted_steps: int,
    selected_step: int,
    stopped_reason: str,
    current_action_rms: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
) -> dict:
    payload = dict(prepared.parent)
    payload.update(prepared.model.checkpoint_metadata())
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model_state,
            "training_mode": config.mode,
            "training_config": config.as_dict(),
            "dagger_round": int(prepared.round_index),
            "dagger_action_mode": "continuous",
            "dagger_source_checkpoint": str(prepared.source_checkpoint),
            "dagger_output_checkpoint": str(prepared.output_checkpoint),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_student_state_frame_history": [
                int(frames) for frames in student_frame_history
            ],
            "dagger_student_state_refresh": "after-each-safe-action-trust-step",
            "dagger_selection_uses_validation": False,
            "dagger_trust_alphas": (),
            "dagger_trust_history": [],
            "action_trust_optimizer_semantics": (
                "actor-only-plain-sgd-fixed-reference-action-trust-v1"
            ),
            "action_trust_trainable_parameters": actor_parameter_names(
                prepared.model
            ),
            "action_trust_frozen_feature_extractor": True,
            "action_trust_reference": (
                "accepted-policy-actions-on-current-aggregate-dataset"
            ),
            "action_trust_trial_count": int(trial_count),
            "action_trust_accepted_steps": int(accepted_steps),
            "action_trust_selected_step": int(selected_step),
            "action_trust_current_rms": float(current_action_rms),
            "action_trust_history": list(history),
            "survival_guard_enabled": config.mode in {
                "budget_survival_trust",
                "budget_boundary_trust",
            },
            "survival_guard_semantics": (
                "reject-current-safe-anchor-to-overload-v1"
                if config.mode in {
                    "budget_survival_trust",
                    "budget_boundary_trust",
                }
                else None
            ),
            "survival_selection_semantics": (
                "safe-anchor-count-then-hits-x-early-keydowns-v1"
                if config.mode in {
                    "budget_survival_trust",
                    "budget_boundary_trust",
                }
                else None
            ),
            "survival_restart_limit": int(
                config.budget.max_nonbest_accepts
            ),
            "survival_restart_count": sum(
                1
                for item in history
                if item.get("continuation_restart")
            ),
            "boundary_trust_enabled": config.mode == "budget_boundary_trust",
            "boundary_trust_semantics": (
                "deprecated-alias-of-survival-guard-v1"
                if config.mode == "budget_boundary_trust"
                else None
            ),
            "budget_requested_hours": float(config.budget.hours),
            "budget_elapsed_seconds": float(time.monotonic() - start_time),
            "budget_stopped_reason": str(stopped_reason),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    return payload


def _save(
    prepared: PreparedRun,
    config: TrainingConfig,
    *,
    model_state: dict[str, torch.Tensor],
    start_time: float,
    trial_count: int,
    accepted_steps: int,
    selected_step: int,
    stopped_reason: str,
    current_action_rms: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
    path: Path | None = None,
    checkpoint_role: str = "selected-best",
) -> None:
    payload = _checkpoint_payload(
        prepared,
        config,
        model_state=model_state,
        start_time=start_time,
        trial_count=trial_count,
        accepted_steps=accepted_steps,
        selected_step=selected_step,
        stopped_reason=stopped_reason,
        current_action_rms=current_action_rms,
        expert_frames=expert_frames,
        dagger_frames=dagger_frames,
        student_frame_history=student_frame_history,
        history=history,
    )
    payload["checkpoint_role"] = str(checkpoint_role)
    save_checkpoint(path or prepared.output_checkpoint, payload)


def _save_human_visible(
    prepared: PreparedRun,
    config: TrainingConfig,
    *,
    model_state: dict[str, torch.Tensor],
    start_time: float,
    trial_count: int,
    accepted_steps: int,
    selected_step: int,
    stopped_reason: str,
    current_lr: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
    path: Path | None = None,
    checkpoint_role: str = "selected-best",
) -> None:
    payload = dict(prepared.parent)
    payload.update(prepared.model.checkpoint_metadata())
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model_state,
            "training_mode": config.mode,
            "training_config": config.as_dict(),
            "checkpoint_role": str(checkpoint_role),
            "dagger_round": int(prepared.round_index),
            "dagger_action_mode": "continuous",
            "dagger_source_checkpoint": str(prepared.source_checkpoint),
            "dagger_output_checkpoint": str(prepared.output_checkpoint),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_student_state_frame_history": [
                int(frames) for frames in student_frame_history
            ],
            "dagger_student_state_refresh": "after-each-safe-human-visible-step",
            "dagger_selection_uses_validation": False,
            "action_trust_optimizer_semantics": None,
            "action_trust_trainable_parameters": (),
            "action_trust_frozen_feature_extractor": None,
            "human_visible_training_semantics": (
                "fixed-parent-connectome+human-visible-residual-adamw-dagger-v1"
            ),
            "human_visible_parent_backend": str(
                prepared.parent.get("n_key_policy_backend", "unknown")
            ),
            "human_visible_trainable_parameters": human_visible_parameter_names(
                prepared.model
            ),
            "human_visible_trial_count": int(trial_count),
            "human_visible_accepted_steps": int(accepted_steps),
            "human_visible_selected_step": int(selected_step),
            "human_visible_current_lr": float(current_lr),
            "human_visible_history": list(history),
            "survival_guard_enabled": True,
            "survival_guard_semantics": "reject-current-safe-anchor-to-overload-v1",
            "survival_selection_semantics": (
                "safe-anchor-count-then-hits-x-early-keydowns-v1"
            ),
            "survival_restart_limit": int(config.budget.max_nonbest_accepts),
            "survival_restart_count": sum(
                1 for item in history if item.get("continuation_restart")
            ),
            "budget_requested_hours": float(config.budget.hours),
            "budget_elapsed_seconds": float(time.monotonic() - start_time),
            "budget_stopped_reason": str(stopped_reason),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    save_checkpoint(path or prepared.output_checkpoint, payload)


def _collect_expert_sequences(
    prepared: PreparedRun,
) -> tuple[list[NKeyBCSequence], int]:
    sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(prepared.anchors, 1):
        rollout = collect_n_key_expert_sequence(
            named.segment,
            key_count=prepared.key_count,
            lead_s=prepared.lead_s,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            source=(
                f"dagger{prepared.round_index}-expert-"
                f"{index}-{named.chart_name}"
            ),
        )
        sequences.append(rollout.sequence)
    return sequences, sum(sequence.frames for sequence in sequences)


def run_budget_action_trust(
    config: TrainingConfig,
    *,
    use_survival_guard: bool = False,
) -> None:
    prepared = _prepare(config)
    model = prepared.model
    trust = config.action_trust
    budget = config.budget

    total_budget_s = budget.hours * 3600.0
    reserve_s = budget.reserve_minutes * 60.0
    train_budget_s = total_budget_s - reserve_s
    start_time = time.monotonic()

    print(
        "=== DMDOD configured budgeted actor "
        + ("survival trust" if use_survival_guard else "action-trust")
        + " DAgger ==="
    )
    print(
        f"source={prepared.source_checkpoint} "
        f"output={prepared.output_checkpoint} "
        f"round={prepared.round_index} backend={model.backend_name} "
        f"keys={prepared.key_count} input={prepared.input_dim}D "
        f"device={prepared.device}"
    )
    print("key-order: " + ",".join(n_key_names(prepared.key_count)))
    print(
        f"budget={budget.hours:g}h reserve={budget.reserve_minutes:g}m "
        f"anchors={len(prepared.anchors)} "
        f"validation={len(prepared.validation)} "
        f"actor-steps={trust.actor_steps} lr={trust.lr:g} "
        f"stay={trust.stay_coef:g} "
        f"action-rms={trust.initial_action_rms:g}->{trust.min_action_rms:g} "
        f"reject-shrink={budget.reject_shrink:g} "
        f"safe-grow={budget.safe_grow:g} "
        f"nonbest-restart={budget.max_nonbest_accepts}"
    )
    print(
        "Trainable: actor_mean only. Selection is Train-only; "
        "Validation runs after selection and Final is untouched."
    )
    if use_survival_guard:
        print(
            "Survival guard: reject only SAFE->overload on Train anchors. "
            "Hit/X/early/keydown changes are allowed and ranked separately."
        )

    print("=== pre-budget continuous Train / fixed safety baseline ===")
    initial_results = evaluate_role_continuous(
        model,
        prepared.anchors,
        label="pre-train",
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    safety_reference_results = initial_results
    continuation_results = initial_results
    continuation_state = clone_model_state(model)
    best_state = continuation_state
    best_results = initial_results
    best_step = 0

    expert_sequences, expert_frames = _collect_expert_sequences(prepared)
    dagger_sequences, dagger_frames = collect_student_state_sequences(
        model,
        prepared.anchors,
        round_index=prepared.round_index,
        collection_index=0,
        lead_s=prepared.lead_s,
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    student_frame_history = [dagger_frames]
    training_sequences = [*expert_sequences, *dagger_sequences]
    print(
        f"aggregate-data generation=0 expert={expert_frames} "
        f"student-state={dagger_frames} "
        f"total={expert_frames + dagger_frames} frames"
    )

    radius = TrustRadiusController.create(
        initial=trust.initial_action_rms,
        minimum=trust.min_action_rms,
        reject_shrink=budget.reject_shrink,
        safe_grow=budget.safe_grow,
    )

    history: list[dict] = []
    trial_count = 0
    accepted_steps = 0
    restart = NonBestRestartController.create(
        limit=budget.max_nonbest_accepts
    )
    stopped_reason = "time-budget"
    cached = None
    progress_checkpoint = (
        _progress_checkpoint_path(prepared.output_checkpoint)
        if use_survival_guard
        else None
    )

    try:
        while trial_count < budget.max_trials:
            elapsed = time.monotonic() - start_time
            remaining_train = train_budget_s - elapsed
            if remaining_train <= 0.0:
                stopped_reason = "time-budget"
                break

            if cached is None:
                model.load_state_dict(continuation_state)
                model.prepare_recurrent_runtime()
                freeze_actor_only(model)
                print(
                    f"feature-cache: build generation={accepted_steps} "
                    f"remaining={_format_duration(remaining_train)}"
                )
                cached = build_action_trust_sequences(
                    model,
                    training_sequences,
                )

            trial_count += 1
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            actor_parameters = freeze_actor_only(model)
            optimizer = torch.optim.SGD(
                actor_parameters,
                lr=trust.lr,
            )

            print(
                f"=== budget trial {trial_count} accepted={accepted_steps} "
                f"radius={radius.current:.6g} "
                f"elapsed={_format_duration(time.monotonic() - start_time)} "
                f"remaining={_format_duration(train_budget_s - (time.monotonic() - start_time))} ==="
            )

            metrics = train_actor_action_trust(
                model,
                cached,
                optimizer=optimizer,
                actor_steps=trust.actor_steps,
                chunk_steps=prepared.chunk_steps,
                base_lr=trust.lr,
                stay_coef=trust.stay_coef,
                max_action_rms=radius.current,
                lr_backoffs=trust.lr_backoffs,
                min_lr=trust.min_lr,
            )
            metrics_dict = metrics.as_dict()

            if metrics.accepted_inner_steps == 0:
                history.append(
                    {
                        "trial": int(trial_count),
                        "accepted_step": int(accepted_steps),
                        "radius": float(radius.current),
                        **metrics_dict,
                        "guard_accepted": False,
                        "guard_reasons": ("action-trust-no-op",),
                    }
                )
                if not radius.reject():
                    stopped_reason = "action-rms-floor"
                    print(
                        f"budget stop: action-trust no-op at radius floor "
                        f"{radius.minimum:.6g}"
                    )
                    break
                print(
                    f"budget retry: action-trust no-op; "
                    f"radius -> {radius.current:.6g}"
                )
                continue

            guard_reference = (
                continuation_results
                if use_survival_guard
                else safety_reference_results
            )
            if use_survival_guard:
                candidate_results, safe, reasons = (
                    evaluate_role_survival_guarded(
                        model,
                        prepared.anchors,
                        guard_reference,
                        label=f"budget-{trial_count:04d}",
                        control_dt_s=prepared.control_dt_s,
                        physics_dt_s=prepared.physics_dt_s,
                        device=prepared.device,
                    )
                )
            else:
                candidate_results = evaluate_role_continuous(
                    model,
                    prepared.anchors,
                    label=f"budget-{trial_count:04d}",
                    control_dt_s=prepared.control_dt_s,
                    physics_dt_s=prepared.physics_dt_s,
                    device=prepared.device,
                )
                safe, reasons = train_safety_guard(
                    guard_reference,
                    candidate_results,
                )
            rank_key = (
                survival_selection_key
                if use_survival_guard
                else selection_key
            )
            selected_best = safe and (
                rank_key(candidate_results)
                > rank_key(best_results)
            )
            summary = summarize(candidate_results)
            history.append(
                {
                    "trial": int(trial_count),
                    "accepted_step": int(accepted_steps),
                    "radius": float(radius.current),
                    **metrics_dict,
                    "survival_guard_enabled": bool(use_survival_guard),
                    "survival_guard_accepted": bool(safe) if use_survival_guard else None,
                    "guard_accepted": bool(safe),
                    "guard_reasons": tuple(reasons),
                    "selected_best_when_evaluated": bool(selected_best),
                    "evaluation_complete": bool(
                        len(candidate_results) == len(prepared.anchors)
                    ),
                    "hits": int(summary.hits),
                    "targets": int(summary.targets),
                    "x_accuracy_percent": float(
                        summary.x_accuracy_percent
                    ),
                    "early": int(summary.early),
                    "overloaded": bool(summary.overloaded),
                    "keydowns": int(summary.keydowns),
                }
            )

            status = (
                "SAFE+BEST"
                if selected_best
                else "SAFE"
                if safe
                else "REJECT"
            )
            detail = "" if safe else " " + "; ".join(reasons)
            survival_detail = ""
            if use_survival_guard:
                parent_safe = safe_anchor_count(guard_reference)
                if safe:
                    candidate_safe = safe_anchor_count(candidate_results)
                    recovered = max(0, candidate_safe - parent_safe)
                    survival_detail = (
                        f" safe-anchors={candidate_safe}/{len(prepared.anchors)}"
                        f" recovered={recovered}"
                    )
                    history[-1]["safe_anchor_count"] = int(candidate_safe)
                    history[-1]["recovered_safe_anchors"] = int(recovered)
                else:
                    survival_detail = (
                        f" protected-death=1 "
                        f"parent-safe={parent_safe}/{len(prepared.anchors)}"
                    )
                    history[-1]["safe_anchor_count"] = None
                    history[-1]["recovered_safe_anchors"] = 0
            print(
                f"budget guard={status}:{survival_detail} "
                f"{aggregate(candidate_results)}{detail}"
            )

            if not safe:
                model.load_state_dict(continuation_state)
                model.prepare_recurrent_runtime()
                if not radius.reject():
                    stopped_reason = "action-rms-floor"
                    print(
                        f"budget stop: rejected at radius floor "
                        f"{radius.minimum:.6g}"
                    )
                    break
                print(
                    f"budget retry: rollback accepted={accepted_steps}; "
                    f"radius -> {radius.current:.6g}"
                )
                continue

            candidate_state = clone_model_state(model)
            continuation_state = candidate_state
            continuation_results = candidate_results
            accepted_steps += 1

            if selected_best:
                best_state = candidate_state
                best_results = candidate_results
                best_step = accepted_steps
                _save(
                    prepared,
                    config,
                    model_state=best_state,
                    start_time=start_time,
                    trial_count=trial_count,
                    accepted_steps=accepted_steps,
                    selected_step=best_step,
                    stopped_reason="running",
                    current_action_rms=radius.current,
                    expert_frames=expert_frames,
                    dagger_frames=dagger_frames,
                    student_frame_history=student_frame_history,
                    history=history,
                )
                print(
                    f"autosave best: accepted={accepted_steps} "
                    f"trial={trial_count} {prepared.output_checkpoint}"
                )
            radius.accept()

            restart_to_best = (
                use_survival_guard
                and restart.observe(selected_best=selected_best)
            )
            if restart_to_best:
                continuation_state = {
                    name: tensor.clone()
                    for name, tensor in best_state.items()
                }
                continuation_results = best_results
                if not radius.at_floor:
                    radius.reject()
                history[-1]["continuation_restart"] = True
                history[-1]["restart_count"] = int(restart.restarts)
                print(
                    f"budget continuation: RESTART best={best_step} "
                    f"after {budget.max_nonbest_accepts} non-best accepts; "
                    f"radius={radius.current:.6g}"
                )
            else:
                history[-1]["continuation_restart"] = False
                history[-1]["restart_count"] = int(restart.restarts)
                print(
                    f"budget continuation: ACCEPT step={accepted_steps} "
                    f"{aggregate(candidate_results)} "
                    f"next-radius={radius.current:.6g}"
                )

            if progress_checkpoint is not None:
                _save(
                    prepared,
                    config,
                    model_state=continuation_state,
                    start_time=start_time,
                    trial_count=trial_count,
                    accepted_steps=accepted_steps,
                    selected_step=best_step,
                    stopped_reason="running-progress",
                    current_action_rms=radius.current,
                    expert_frames=expert_frames,
                    dagger_frames=dagger_frames,
                    student_frame_history=student_frame_history,
                    history=history,
                    path=progress_checkpoint,
                    checkpoint_role="continuation-progress",
                )
                print(f"autosave progress: {progress_checkpoint}")

            del cached
            cached = None

            if time.monotonic() - start_time >= train_budget_s:
                stopped_reason = "time-budget"
                break

            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            dagger_sequences, dagger_frames = collect_student_state_sequences(
                model,
                prepared.anchors,
                round_index=prepared.round_index,
                collection_index=accepted_steps,
                lead_s=prepared.lead_s,
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s,
                device=prepared.device,
            )
            student_frame_history.append(dagger_frames)
            training_sequences = [*expert_sequences, *dagger_sequences]
            print(
                f"aggregate-data generation={accepted_steps} "
                f"expert={expert_frames} student-state={dagger_frames} "
                f"total={expert_frames + dagger_frames} frames"
            )
        else:
            stopped_reason = "max-trials"
    finally:
        if cached is not None:
            del cached

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    freeze_actor_only(model)

    _save(
        prepared,
        config,
        model_state=best_state,
        start_time=start_time,
        trial_count=trial_count,
        accepted_steps=accepted_steps,
        selected_step=best_step,
        stopped_reason=stopped_reason,
        current_action_rms=radius.current,
        expert_frames=expert_frames,
        dagger_frames=dagger_frames,
        student_frame_history=student_frame_history,
        history=history,
    )

    print("=== selected Train-safe configured checkpoint ===")
    print(
        f"stop={stopped_reason} "
        f"elapsed={_format_duration(time.monotonic() - start_time)} "
        f"trials={trial_count} accepted={accepted_steps} "
        f"restarts={restart.restarts} selected={best_step}: "
        f"{aggregate(best_results)}"
    )

    print("=== selected checkpoint continuous Validation ===")
    evaluate_role_continuous(
        model,
        prepared.validation,
        label="validation",
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    if progress_checkpoint is not None and progress_checkpoint.exists():
        progress_checkpoint.unlink()
        print(f"removed completed progress checkpoint: {progress_checkpoint}")

    print(
        f"budget final "
        f"elapsed={_format_duration(time.monotonic() - start_time)} "
        f"checkpoint={prepared.output_checkpoint}"
    )



def run_human_visible_dagger(config: TrainingConfig) -> None:
    """Train the relaxed human-visible residual controller with guarded DAgger."""

    prepared = _prepare(config)
    model = prepared.model
    visible = config.human_visible
    budget = config.budget

    total_budget_s = budget.hours * 3600.0
    reserve_s = budget.reserve_minutes * 60.0
    train_budget_s = total_budget_s - reserve_s
    start_time = time.monotonic()

    print("=== DMDOD human-visible residual controller DAgger ===")
    print(
        f"source={prepared.source_checkpoint} "
        f"output={prepared.output_checkpoint} "
        f"round={prepared.round_index} backend={model.backend_name} "
        f"keys={prepared.key_count} input={prepared.input_dim}D "
        f"device={prepared.device}"
    )
    print("key-order: " + ",".join(n_key_names(prepared.key_count)))
    print(
        f"budget={budget.hours:g}h reserve={budget.reserve_minutes:g}m "
        f"anchors={len(prepared.anchors)} validation={len(prepared.validation)} "
        f"updates/trial={visible.updates_per_trial} "
        f"lr={visible.lr:g}->{visible.min_lr:g} "
        f"weight-decay={visible.weight_decay:g} grad-clip={visible.grad_clip:g} "
        f"nonbest-restart={budget.max_nonbest_accepts}"
    )
    print(
        "Information contract: existing human-visible 263D only; direct floor/motor/HUD "
        "paths plus controller memory are trainable. Parent sensory/connectome/actor stay fixed."
    )
    print(
        "Safety: reject only current SAFE->overload on Train anchors. "
        "No action-RMS trust radius."
    )

    print("=== pre-train continuous Train / exact residual warm start ===")
    initial_results = evaluate_role_continuous(
        model,
        prepared.anchors,
        label="pre-train",
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    continuation_results = initial_results
    continuation_state = clone_model_state(model)
    best_results = initial_results
    best_state = continuation_state
    best_step = 0

    expert_sequences, expert_frames = _collect_expert_sequences(prepared)
    dagger_sequences, dagger_frames = collect_student_state_sequences(
        model,
        prepared.anchors,
        round_index=prepared.round_index,
        collection_index=0,
        lead_s=prepared.lead_s,
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    student_frame_history = [dagger_frames]
    training_sequences = [*expert_sequences, *dagger_sequences]
    print(
        f"aggregate-data generation=0 expert={expert_frames} "
        f"student-state={dagger_frames} total={expert_frames + dagger_frames} frames"
    )

    history: list[dict] = []
    trial_count = 0
    accepted_steps = 0
    current_lr = float(visible.lr)
    restart = NonBestRestartController.create(
        limit=budget.max_nonbest_accepts
    )
    replay_chunks = None
    stopped_reason = "time-budget"
    progress_checkpoint = _progress_checkpoint_path(prepared.output_checkpoint)

    try:
        while trial_count < budget.max_trials:
            elapsed = time.monotonic() - start_time
            remaining_train = train_budget_s - elapsed
            if remaining_train <= 0.0:
                stopped_reason = "time-budget"
                break

            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            freeze_human_visible_controller(model)

            if replay_chunks is None:
                print(
                    f"replay-cache: build generation={accepted_steps} "
                    f"remaining={_format_duration(remaining_train)}"
                )
                replay_chunks = build_human_visible_replay_chunks(
                    model,
                    training_sequences,
                    chunk_steps=prepared.chunk_steps,
                )
                print(
                    f"replay-cache: chunks={len(replay_chunks)} "
                    f"chunk-steps={prepared.chunk_steps}"
                )

            trial_count += 1
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            trainable = freeze_human_visible_controller(model)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=current_lr,
                weight_decay=visible.weight_decay,
            )

            print(
                f"=== human-visible trial {trial_count} accepted={accepted_steps} "
                f"lr={current_lr:.6g} "
                f"elapsed={_format_duration(time.monotonic() - start_time)} "
                f"remaining={_format_duration(train_budget_s - (time.monotonic() - start_time))} ==="
            )
            metrics = train_human_visible_replay(
                model,
                replay_chunks,
                optimizer=optimizer,
                updates=visible.updates_per_trial,
                grad_clip=visible.grad_clip,
                seed=visible.seed + trial_count,
            )
            print(
                f"  train: updates={metrics.updates} "
                f"loss={metrics.mean_loss:.6f}->{metrics.final_loss:.6f} "
                f"grad={metrics.mean_grad_norm:.6g} max={metrics.max_grad_norm:.6g}"
            )

            candidate_results, safe, reasons = evaluate_role_survival_guarded(
                model,
                prepared.anchors,
                continuation_results,
                label=f"visible-{trial_count:04d}",
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s,
                device=prepared.device,
            )
            selected_best = safe and (
                survival_selection_key(candidate_results)
                > survival_selection_key(best_results)
            )
            summary = summarize(candidate_results)
            parent_safe = safe_anchor_count(continuation_results)
            candidate_safe = (
                safe_anchor_count(candidate_results)
                if safe
                else None
            )
            recovered = (
                max(0, int(candidate_safe) - parent_safe)
                if candidate_safe is not None
                else 0
            )
            history.append(
                {
                    "trial": int(trial_count),
                    "accepted_step": int(accepted_steps),
                    "lr": float(current_lr),
                    **metrics.as_dict(),
                    "guard_accepted": bool(safe),
                    "guard_reasons": tuple(reasons),
                    "selected_best_when_evaluated": bool(selected_best),
                    "evaluation_complete": bool(
                        len(candidate_results) == len(prepared.anchors)
                    ),
                    "safe_anchor_count": (
                        int(candidate_safe)
                        if candidate_safe is not None
                        else None
                    ),
                    "recovered_safe_anchors": int(recovered),
                    "hits": int(summary.hits),
                    "targets": int(summary.targets),
                    "x_accuracy_percent": float(summary.x_accuracy_percent),
                    "early": int(summary.early),
                    "overloaded": bool(summary.overloaded),
                    "keydowns": int(summary.keydowns),
                }
            )

            if not safe:
                model.load_state_dict(continuation_state)
                model.prepare_recurrent_runtime()
                current_lr = max(
                    float(visible.min_lr),
                    current_lr * float(budget.reject_shrink),
                )
                history[-1]["continuation_restart"] = False
                history[-1]["restart_count"] = int(restart.restarts)
                print(
                    f"human-visible guard=REJECT protected-death=1 "
                    f"parent-safe={parent_safe}/{len(prepared.anchors)} "
                    f"{aggregate(candidate_results)} "
                    + "; ".join(reasons)
                )
                print(
                    f"human-visible retry: rollback accepted={accepted_steps}; "
                    f"lr -> {current_lr:.6g}"
                )
                continue

            print(
                f"human-visible guard={'SAFE+BEST' if selected_best else 'SAFE'} "
                f"safe-anchors={candidate_safe}/{len(prepared.anchors)} "
                f"recovered={recovered} {aggregate(candidate_results)}"
            )
            candidate_state = clone_model_state(model)
            continuation_state = candidate_state
            continuation_results = candidate_results
            accepted_steps += 1

            if selected_best:
                best_state = candidate_state
                best_results = candidate_results
                best_step = accepted_steps
                _save_human_visible(
                    prepared,
                    config,
                    model_state=best_state,
                    start_time=start_time,
                    trial_count=trial_count,
                    accepted_steps=accepted_steps,
                    selected_step=best_step,
                    stopped_reason="running",
                    current_lr=current_lr,
                    expert_frames=expert_frames,
                    dagger_frames=dagger_frames,
                    student_frame_history=student_frame_history,
                    history=history,
                )
                print(
                    f"autosave best: accepted={accepted_steps} "
                    f"trial={trial_count} {prepared.output_checkpoint}"
                )

            current_lr = min(
                float(visible.lr),
                current_lr * float(budget.safe_grow),
            )
            restart_to_best = restart.observe(selected_best=selected_best)
            if restart_to_best:
                continuation_state = {
                    name: tensor.clone()
                    for name, tensor in best_state.items()
                }
                continuation_results = best_results
                current_lr = max(
                    float(visible.min_lr),
                    current_lr * float(budget.reject_shrink),
                )
                history[-1]["continuation_restart"] = True
                history[-1]["restart_count"] = int(restart.restarts)
                print(
                    f"human-visible continuation: RESTART best={best_step} "
                    f"after {budget.max_nonbest_accepts} non-best accepts; "
                    f"lr={current_lr:.6g}"
                )
            else:
                history[-1]["continuation_restart"] = False
                history[-1]["restart_count"] = int(restart.restarts)
                print(
                    f"human-visible continuation: ACCEPT step={accepted_steps} "
                    f"next-lr={current_lr:.6g}"
                )

            _save_human_visible(
                prepared,
                config,
                model_state=continuation_state,
                start_time=start_time,
                trial_count=trial_count,
                accepted_steps=accepted_steps,
                selected_step=best_step,
                stopped_reason="running-progress",
                current_lr=current_lr,
                expert_frames=expert_frames,
                dagger_frames=dagger_frames,
                student_frame_history=student_frame_history,
                history=history,
                path=progress_checkpoint,
                checkpoint_role="continuation-progress",
            )
            print(f"autosave progress: {progress_checkpoint}")

            if time.monotonic() - start_time >= train_budget_s:
                stopped_reason = "time-budget"
                break

            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            dagger_sequences, dagger_frames = collect_student_state_sequences(
                model,
                prepared.anchors,
                round_index=prepared.round_index,
                collection_index=accepted_steps,
                lead_s=prepared.lead_s,
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s,
                device=prepared.device,
            )
            student_frame_history.append(dagger_frames)
            training_sequences = [*expert_sequences, *dagger_sequences]
            replay_chunks = None
            print(
                f"aggregate-data generation={accepted_steps} "
                f"expert={expert_frames} student-state={dagger_frames} "
                f"total={expert_frames + dagger_frames} frames"
            )
        else:
            stopped_reason = "max-trials"
    finally:
        replay_chunks = None

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    freeze_human_visible_controller(model)
    _save_human_visible(
        prepared,
        config,
        model_state=best_state,
        start_time=start_time,
        trial_count=trial_count,
        accepted_steps=accepted_steps,
        selected_step=best_step,
        stopped_reason=stopped_reason,
        current_lr=current_lr,
        expert_frames=expert_frames,
        dagger_frames=dagger_frames,
        student_frame_history=student_frame_history,
        history=history,
    )

    print("=== selected human-visible Train-safe checkpoint ===")
    print(
        f"stop={stopped_reason} "
        f"elapsed={_format_duration(time.monotonic() - start_time)} "
        f"trials={trial_count} accepted={accepted_steps} "
        f"restarts={restart.restarts} selected={best_step}: "
        f"{aggregate(best_results)}"
    )
    print("=== selected checkpoint continuous Validation ===")
    evaluate_role_continuous(
        model,
        prepared.validation,
        label="validation",
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
    )
    if progress_checkpoint.exists():
        progress_checkpoint.unlink()
        print(f"removed completed progress checkpoint: {progress_checkpoint}")
    print(
        f"human-visible final elapsed="
        f"{_format_duration(time.monotonic() - start_time)} "
        f"checkpoint={prepared.output_checkpoint}"
    )


def run_trajectory_probe(config: TrainingConfig) -> None:
    prepared = _prepare(config)
    baseline_model = prepared.model
    trust = config.action_trust
    probe = config.trajectory_probe

    print("=== DMDOD closed-loop trajectory divergence probe ===")
    print(
        f"source={prepared.source_checkpoint} output={prepared.output_checkpoint} "
        f"anchors={len(prepared.anchors)} candidate-action-rms="
        f"{probe.candidate_action_rms:g} device={prepared.device}"
    )

    training_sequences, expert_frames, student_frames = (
        collect_probe_training_sequences(
            baseline_model,
            prepared.anchors,
            round_index=prepared.round_index,
            lead_s=prepared.lead_s,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
        )
    )
    print(
        f"probe-data expert={expert_frames} student-state={student_frames} "
        f"total={expert_frames + student_frames} frames"
    )

    candidate_model, candidate_metrics = build_probe_candidate(
        prepared.parent,
        baseline_model,
        training_sequences,
        action_rms=probe.candidate_action_rms,
        actor_steps=trust.actor_steps,
        lr=trust.lr,
        stay_coef=trust.stay_coef,
        lr_backoffs=trust.lr_backoffs,
        min_lr=trust.min_lr,
        chunk_steps=prepared.chunk_steps,
        device=prepared.device,
    )
    del training_sequences

    print(
        "candidate: "
        f"inner={candidate_metrics.accepted_inner_steps}/{trust.actor_steps} "
        f"open-loop-action-rms={candidate_metrics.action_rms:.6g} "
        f"action-max={candidate_metrics.action_max:.6g} "
        f"lr={candidate_metrics.final_lr:.3g}"
    )
    if candidate_metrics.accepted_inner_steps == 0:
        raise SystemExit(
            "trajectory probe candidate could not fit inside the configured "
            "action RMS bound"
        )

    results = []
    for index, named in enumerate(prepared.anchors, 1):
        result = probe_anchor_pair(
            baseline_model,
            candidate_model,
            named,
            anchor_index=index,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
        )
        results.append(result)
        print(format_probe_result(result))

    save_probe_report(
        prepared.output_checkpoint,
        candidate_metrics=candidate_metrics,
        results=results,
        source_checkpoint=str(prepared.source_checkpoint),
        configured_action_rms=probe.candidate_action_rms,
    )
    print(f"trajectory probe report: {prepared.output_checkpoint}")


def run_budget_survival_trust(config: TrainingConfig) -> None:
    run_budget_action_trust(config, use_survival_guard=True)


def run_budget_boundary_trust(config: TrainingConfig) -> None:
    """Backward-compatible alias for the survival-only guard mode."""
    run_budget_survival_trust(config)


_RUNNERS = {
    "human_visible_dagger": run_human_visible_dagger,
    "budget_action_trust": run_budget_action_trust,
    "budget_survival_trust": run_budget_survival_trust,
    "budget_boundary_trust": run_budget_boundary_trust,
    "trajectory_probe": run_trajectory_probe,
}


def run_training(config: TrainingConfig) -> None:
    try:
        runner = _RUNNERS[config.mode]
    except KeyError as exc:
        choices = ", ".join(sorted(_RUNNERS))
        raise ValueError(
            f"unknown training mode {config.mode!r}; expected one of: {choices}"
        ) from exc
    runner(config)
