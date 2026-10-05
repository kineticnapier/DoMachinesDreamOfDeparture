from __future__ import annotations

"""v1.8.1: SGD micro-step DAgger with an explicit update-norm cap.

v1.8.0 established that even one fresh Adam step can move a sensitive recurrent
closed-loop policy far enough to destroy previously safe anchors.  Adam's first
step is approximately sign-normalized per parameter, so a small scalar learning
rate does not imply a small global parameter-space move.

v1.8.1 keeps the aggregate-gradient micro-step design but replaces Adam with
plain SGD (no momentum or weight decay).  Before the single optimizer step, the
global gradient L2 norm is clipped so the resulting SGD parameter delta has a
known upper bound: ||delta theta||_2 <= --max-update-norm.

There is still no proposal interpolation and no alpha.  Every accepted step is
evaluated immediately against the fixed pre-run Train safety baseline.  Unsafe
steps are rolled back and end the run; safe steps refresh student-state DAgger
trajectories before the next micro-step.  Validation is evaluated only after
Train-only selection and Final remains untouched.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
import train_real_chart_v171_n_key_connectome_failure_continuation_dagger as v171
from dmdod.fly_connectome_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
    NKeyFlyConnectomeActorCritic,
)
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_dagger_continuation import (
    collect_n_key_dagger_sequence_with_continuation,
)
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_expert_sequence,
    n_key_actuation_loss,
)
from dmdod.random_connectome_policy import (
    N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    NKeyRandomConnectomeActorCritic,
)


TRAINER_VERSION = "1.8.1-n-key-connectome-sgd-microstep-dagger"
CHECKPOINT_FORMAT_VERSION = 30
MICROSTEP_OPTIMIZER_SEMANTICS = "aggregate-gradient-one-sgd-step-update-norm-cap-v1"
DEFAULT_MICRO_STEPS = 4
DEFAULT_LR = 3e-6
DEFAULT_MAX_UPDATE_NORM = 1e-5


def _clone_optimizer_state(optimizer_or_state: torch.optim.Optimizer | dict) -> dict:
    if isinstance(optimizer_or_state, torch.optim.Optimizer):
        state = optimizer_or_state.state_dict()
    else:
        state = optimizer_or_state
    return deepcopy(state)


def _restore_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    optimizer.load_state_dict(deepcopy(state))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _can_resume_microstep_optimizer(parent: dict) -> bool:
    return bool(
        parent.get("microstep_optimizer_state") is not None
        and parent.get("microstep_optimizer_semantics")
        == MICROSTEP_OPTIMIZER_SEMANTICS
    )


def _train_aggregate_microstep(
    model,
    sequences: list[NKeyBCSequence],
    *,
    optimizer: torch.optim.Optimizer,
    chunk_steps: int,
    max_update_norm: float,
) -> tuple[float, float, float, float]:
    """Accumulate one dataset gradient, then perform one norm-bounded SGD step.

    Truncated recurrent state boundaries match the existing BC trainer, but
    optimizer.step() is deliberately delayed until every chunk has contributed
    to the gradient. Chunk losses are frame-weighted so the accumulated gradient
    represents the whole aggregate dataset rather than one arbitrary final chunk.

    The optimizer must be plain SGD with one shared positive learning rate.
    Clipping the global gradient norm to max_update_norm / lr then guarantees
    the actual SGD parameter delta has global L2 norm <= max_update_norm.
    """

    if not sequences:
        raise ValueError("at least one N-key training sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    if max_update_norm <= 0.0:
        raise ValueError("max_update_norm must be positive")
    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("v1.8.1 micro-step optimizer must be torch.optim.SGD")
    if any(
        float(group.get("momentum", 0.0)) != 0.0
        or float(group.get("weight_decay", 0.0)) != 0.0
        or bool(group.get("nesterov", False))
        for group in optimizer.param_groups
    ):
        raise ValueError("v1.8.1 requires plain SGD without momentum/weight_decay/nesterov")
    lr_values = {float(group["lr"]) for group in optimizer.param_groups}
    if len(lr_values) != 1:
        raise ValueError("all SGD parameter groups must use the same learning rate")
    step_lr = lr_values.pop()
    if step_lr <= 0.0:
        raise ValueError("SGD learning rate must be positive")
    total_frames = sum(int(sequence.frames) for sequence in sequences)
    if total_frames <= 0:
        raise ValueError("training sequences must contain at least one frame")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0

    connectome = isinstance(
        model,
        (NKeyFlyConnectomeActorCritic, NKeyRandomConnectomeActorCritic),
    )
    original_critic_forward = model.critic.forward if connectome else None
    if connectome:
        # The BC objective consumes only actor means. Avoid evaluating the
        # discarded 4096->1 critic for every frame, matching the v1.7 fast path.
        model.critic.forward = lambda features: features[:, :1]

    try:
        for sequence in sequences:
            if int(sequence.key_count) != int(model.key_count):
                raise ValueError("sequence key_count does not match model key_count")
            state = model.initial_state(sequence.observations.device)
            for start in range(0, sequence.frames, chunk_steps):
                end = min(sequence.frames, start + chunk_steps)
                state = state.detach()
                means, _, state = model.forward_sequence(
                    sequence.observations[start:end],
                    state,
                )
                predicted = torch.tanh(means)
                target = sequence.teacher_actions[start:end]
                loss = n_key_actuation_loss(predicted, target)
                frames = end - start
                weight = float(frames) / float(total_frames)
                (loss * weight).backward()
                loss_sum += float(loss.detach().item()) * frames
    finally:
        if connectome and original_critic_forward is not None:
            model.critic.forward = original_critic_forward

    max_grad_norm = float(max_update_norm) / step_lr
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
        error_if_nonfinite=True,
    )
    grad_norm = float(grad_norm_tensor.detach().item())
    if grad_norm <= max_grad_norm or grad_norm == 0.0:
        grad_scale = 1.0
        update_norm = step_lr * grad_norm
    else:
        grad_scale = max_grad_norm / grad_norm
        update_norm = float(max_update_norm)

    optimizer.step()
    return (
        loss_sum / float(total_frames),
        grad_norm,
        update_norm,
        grad_scale,
    )


def _collect_student_state_sequences(
    model,
    anchors,
    *,
    round_index: int,
    collection_index: int,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
) -> tuple[list[NKeyBCSequence], int]:
    sequences: list[NKeyBCSequence] = []
    total = len(anchors)
    label = f"collect-m{collection_index}"
    for index, named in enumerate(anchors, 1):
        rollout = collect_n_key_dagger_sequence_with_continuation(
            model,
            named.segment,
            lead_s=lead_s,
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=(
                f"dagger{round_index}-microstep-student-m{collection_index}-"
                f"{index}-{named.chart_name}"
            ),
            action_mode="continuous",
            continue_after_failure=True,
        )
        sequences.append(rollout.sequence)
        print(
            f"{label} {index:02d}/{total} {named.chart_name}: "
            f"frames={rollout.sequence.frames} H={rollout.stats.hits}/{rollout.stats.targets} "
            f"X={rollout.stats.x_accuracy_percent:.2f}% early={rollout.stats.too_early_presses} "
            f"over={rollout.stats.overloaded} keydowns={rollout.physical_keydowns}"
        )
    frames = sum(sequence.frames for sequence in sequences)
    print(f"{label} aggregate: student-state={frames} frames")
    return sequences, frames


def _microstep_record(
    *,
    step: int,
    loss: float,
    grad_norm: float,
    update_norm: float,
    grad_scale: float,
    safe: bool,
    reasons: tuple[str, ...],
    continued: bool,
    selected_best: bool,
    results: list[tuple[object, int]],
) -> dict:
    summary = v161._summarize(results)
    return {
        "step": int(step),
        "loss": float(loss),
        "grad_norm_before_clip": float(grad_norm),
        "update_norm_l2": float(update_norm),
        "gradient_scale": float(grad_scale),
        "guard_accepted": bool(safe),
        "accepted_for_continuation": bool(continued),
        "selected_best_when_evaluated": bool(selected_best),
        "guard_reasons": tuple(reasons),
        "hits": int(summary.hits),
        "targets": int(summary.targets),
        "x_accuracy_percent": float(summary.x_accuracy_percent),
        "early": int(summary.early),
        "overloaded": bool(summary.overloaded),
        "keydowns": int(summary.keydowns),
    }


def _checkpoint_payload(
    parent: dict,
    *,
    model,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
    source_checkpoint: Path,
    output_checkpoint: Path,
    round_index: int,
    requested_steps: int,
    attempted_steps: int,
    accepted_steps: int,
    selected_step: int,
    stopped_unsafe: bool,
    lr: float,
    max_update_norm: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    losses: list[float],
    history: list[dict],
    optimizer_resumed: bool,
) -> dict:
    payload = dict(parent)
    payload.update(model.checkpoint_metadata())
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model_state,
            # The old proposal/interpolation optimizer state has different
            # semantics. Do not advertise a v1.8 state through that old key.
            "dagger_optimizer_state": None,
            "dagger_round": int(round_index),
            "dagger_action_mode": "continuous",
            "dagger_press_threshold": None,
            "dagger_release_threshold": None,
            "dagger_lr": float(lr),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_student_state_frame_history": [
                int(frames) for frames in student_frame_history
            ],
            "dagger_student_state_refresh": "after-each-safe-microstep",
            "dagger_loss_history": list(losses),
            "dagger_trust_alphas": (),
            "dagger_trust_history": [],
            "dagger_trust_continuation": "removed-v1.8-direct-microstep",
            "dagger_trust_optimizer_continuation": "not-applicable",
            "dagger_selection_uses_validation": False,
            "dagger_source_checkpoint": str(source_checkpoint),
            "dagger_output_checkpoint": str(output_checkpoint),
            "microstep_optimizer_state": optimizer_state,
            "microstep_optimizer_semantics": MICROSTEP_OPTIMIZER_SEMANTICS,
            "microstep_optimizer_resumed": bool(optimizer_resumed),
            "microstep_requested_steps": int(requested_steps),
            "microstep_attempted_steps": int(attempted_steps),
            "microstep_accepted_steps": int(accepted_steps),
            "microstep_selected_step": int(selected_step),
            "microstep_stopped_unsafe": bool(stopped_unsafe),
            "microstep_safety_reference": "fixed-pre-run-train-anchors",
            "microstep_selection": "best-train-safe-selection-key",
            "microstep_gradient": "frame-weighted-aggregate-before-one-sgd-step",
            "microstep_max_update_norm": float(max_update_norm),
            "microstep_update_norm_control": "global-l2-gradient-clip-for-plain-sgd-delta",
            "microstep_history": list(history),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    return payload


def _default_output_path(source: Path, round_index: int) -> Path:
    suffix = source.suffix or ".pt"
    stem = source.name[: -len(suffix)] if source.name.endswith(suffix) else source.name
    return source.with_name(f"{stem}_microstep{round_index}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct connectome micro-step DAgger: one norm-bounded aggregate-"
            "gradient plain-SGD step at a time, followed immediately by fixed-"
            "baseline Train safety."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--micro-steps", type=int, default=DEFAULT_MICRO_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--max-update-norm",
        type=float,
        default=DEFAULT_MAX_UPDATE_NORM,
        help="global L2 upper bound for each plain-SGD parameter delta",
    )
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.micro_steps <= 0:
        raise SystemExit("--micro-steps must be positive")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be positive")
    if args.max_update_norm <= 0.0:
        raise SystemExit("--max-update-norm must be positive")
    if args.chunk_steps is not None and args.chunk_steps <= 0:
        raise SystemExit("--chunk-steps must be positive")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")

    device = v161._device_from_arg(args.device)
    source_checkpoint = Path(args.checkpoint)
    parent = torch.load(source_checkpoint, map_location=device, weights_only=False)
    backend = str(parent.get("n_key_policy_backend", "gru"))
    if backend not in {
        N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
        N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    }:
        raise SystemExit(
            "v1.8 micro-step trainer currently requires fly_connectome or random_connectome"
        )

    key_count = int(parent["key_count"])
    input_dim = int(parent["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected {expected_input}"
        )
    key_names = n_key_names(key_count)
    model = v171._build_policy_from_checkpoint(parent, device=device)

    calibration = parent.get("calibration") or {}
    if "lead_s" not in calibration:
        raise SystemExit("checkpoint calibration is missing lead_s")
    lead_s = float(calibration["lead_s"])
    control_dt_s = float(parent["control_dt"])
    physics_dt_s = float(parent.get("physics_dt", 0.001))
    chunk_steps = int(args.chunk_steps or parent.get("chunk_steps", 192))

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    validation_charts = v080._compile_role(dataset.validation)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=float(parent["train_window"]),
        anchors_per_chart=int(parent["anchors_per_chart"]),
    )
    validation = v080._build_validation_segments(
        validation_charts,
        window_s=float(parent["validation_window"]),
    )

    parent_anchor_limit = parent.get("anchor_limit")
    anchor_limit = args.anchor_limit if args.anchor_limit is not None else parent_anchor_limit
    if anchor_limit is not None:
        anchors = anchors[: int(anchor_limit)]
    parent_validation_limit = parent.get("validation_limit")
    validation_limit = (
        args.validation_limit
        if args.validation_limit is not None
        else parent_validation_limit
    )
    if validation_limit is not None:
        validation = validation[: int(validation_limit)]
    if not anchors:
        raise SystemExit("no Train anchors selected")

    round_index = int(parent.get("dagger_round", 0)) + 1
    output_checkpoint = Path(
        args.output or _default_output_path(source_checkpoint, round_index)
    )

    print("=== DMDOD v1.8.1 N-Key Connectome SGD Micro-Step DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"backend={model.backend_name} keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchors)} validation={len(validation)} micro-steps={args.micro_steps} "
        f"action=continuous lr={args.lr:g} max-update-norm={args.max_update_norm:g} "
        f"chunk={chunk_steps} | FINAL untouched"
    )
    print(
        "No proposal interpolation / no alpha / no Adam. Each step accumulates the whole "
        "current BC dataset gradient, caps the global plain-SGD parameter delta, then runs "
        "fixed-baseline Train safety."
    )

    print("=== pre-microstep continuous Train / fixed safety baseline ===")
    initial_results = v162._evaluate_role_continuous(
        model,
        anchors,
        label="pre-train",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    safety_reference_results = initial_results
    continuation_results = initial_results
    continuation_state = v161._clone_model_state(model)

    expert_sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(anchors, 1):
        expert = collect_n_key_expert_sequence(
            named.segment,
            key_count=key_count,
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"dagger{round_index}-microstep-expert-{index}-{named.chart_name}",
        )
        expert_sequences.append(expert.sequence)
    expert_frames = sum(sequence.frames for sequence in expert_sequences)

    dagger_sequences, dagger_frames = _collect_student_state_sequences(
        model,
        anchors,
        round_index=round_index,
        collection_index=0,
        lead_s=lead_s,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    student_frame_history = [dagger_frames]
    training_sequences = [*expert_sequences, *dagger_sequences]
    print(
        f"aggregate-data generation=0 expert={expert_frames} student-state={dagger_frames} "
        f"total={expert_frames + dagger_frames} frames"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer_resumed = _can_resume_microstep_optimizer(parent)
    if optimizer_resumed:
        _restore_optimizer_state(optimizer, parent["microstep_optimizer_state"])
        _set_optimizer_lr(optimizer, args.lr)
        print("optimizer: resumed compatible v1.8.1 plain-SGD micro-step state")
    else:
        print(
            "optimizer: fresh plain SGD state "
            "(legacy Adam/proposal optimizer states intentionally ignored)"
        )

    continuation_optimizer_state = _clone_optimizer_state(optimizer)
    best_state = continuation_state
    best_optimizer_state = _clone_optimizer_state(continuation_optimizer_state)
    best_results = initial_results
    best_step = 0

    losses: list[float] = []
    history: list[dict] = []
    attempted_steps = 0
    accepted_steps = 0
    stopped_unsafe = False

    for step in range(1, args.micro_steps + 1):
        attempted_steps = step
        model.load_state_dict(continuation_state)
        model.prepare_recurrent_runtime()
        _restore_optimizer_state(optimizer, continuation_optimizer_state)
        _set_optimizer_lr(optimizer, args.lr)

        loss, grad_norm, update_norm, grad_scale = _train_aggregate_microstep(
            model,
            training_sequences,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            max_update_norm=args.max_update_norm,
        )
        losses.append(loss)
        candidate_state = v161._clone_model_state(model)
        candidate_optimizer_state = _clone_optimizer_state(optimizer)
        print(
            f"microstep {step:03d}/{args.micro_steps}: loss={loss:.6f} "
            f"grad-norm={grad_norm:.6g} update-norm={update_norm:.6g} "
            f"grad-scale={grad_scale:.6g}"
        )

        candidate_results = v162._evaluate_role_continuous(
            model,
            anchors,
            label=f"microstep-{step:03d}",
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
        )
        safe, reasons = v161._train_safety_guard(
            safety_reference_results,
            candidate_results,
        )
        selected_best = safe and (
            v161._selection_key(candidate_results) > v161._selection_key(best_results)
        )
        status = "SAFE+BEST" if selected_best else "SAFE" if safe else "REJECT"
        detail = "" if safe else " " + "; ".join(reasons)
        print(
            f"microstep guard={status}: {v161._aggregate(candidate_results)}{detail}"
        )

        history.append(
            _microstep_record(
                step=step,
                loss=loss,
                grad_norm=grad_norm,
                update_norm=update_norm,
                grad_scale=grad_scale,
                safe=safe,
                reasons=reasons,
                continued=safe,
                selected_best=selected_best,
                results=candidate_results,
            )
        )

        if not safe:
            stopped_unsafe = True
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            _restore_optimizer_state(optimizer, continuation_optimizer_state)
            print(
                f"microstep-loop: STOP at rejected step={step}; rolled back to last safe state"
            )
            break

        continuation_state = candidate_state
        continuation_optimizer_state = candidate_optimizer_state
        continuation_results = candidate_results
        accepted_steps += 1

        if selected_best:
            best_state = candidate_state
            best_optimizer_state = _clone_optimizer_state(candidate_optimizer_state)
            best_results = candidate_results
            best_step = step

        print(
            f"microstep-continuation: ACCEPT step={step} "
            + v161._aggregate(continuation_results)
        )

        if step < args.micro_steps:
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            dagger_sequences, dagger_frames = _collect_student_state_sequences(
                model,
                anchors,
                round_index=round_index,
                collection_index=step,
                lead_s=lead_s,
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            student_frame_history.append(dagger_frames)
            training_sequences = [*expert_sequences, *dagger_sequences]
            print(
                f"aggregate-data generation={step} expert={expert_frames} "
                f"student-state={dagger_frames} "
                f"total={expert_frames + dagger_frames} frames"
            )

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    _restore_optimizer_state(optimizer, best_optimizer_state)
    _set_optimizer_lr(optimizer, args.lr)

    print("=== selected Train-safe micro-step checkpoint ===")
    print(
        f"selected step={best_step}/{args.micro_steps}: "
        + v161._aggregate(best_results)
    )

    v161._save_checkpoint(
        output_checkpoint,
        _checkpoint_payload(
            parent,
            model=model,
            model_state=best_state,
            optimizer_state=best_optimizer_state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            requested_steps=args.micro_steps,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            selected_step=best_step,
            stopped_unsafe=stopped_unsafe,
            lr=args.lr,
            max_update_norm=args.max_update_norm,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            student_frame_history=student_frame_history,
            losses=losses,
            history=history,
            optimizer_resumed=optimizer_resumed,
        ),
    )

    print("=== selected checkpoint continuous Validation ===")
    v162._evaluate_role_continuous(
        model,
        validation,
        label="validation",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    print(f"checkpoint final: {output_checkpoint}")


if __name__ == "__main__":
    main()
