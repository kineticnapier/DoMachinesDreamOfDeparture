from __future__ import annotations

"""v1.9.0: actor-only action-space-trust DAgger for connectome policies.

The recurrent connectome, sensory adapter, projection, critic, and log_std are
frozen.  Only actor_mean learns.  For each outer micro-step, deterministic
connectome features are cached for the current expert + student-state dataset,
and the accepted policy's actions on those same states become a fixed action
reference.

The actor is then optimized for several full-dataset SGD steps with

    teacher BC loss + stay_coef * MSE(current_action, reference_action)

The stay term becomes meaningful after the first inner step because the
reference remains fixed for the whole outer micro-step.  Every tentative inner
step is measured in action space.  If aggregate action RMS drift would exceed
--max-action-rms, that inner step is rolled back and retried with a halved
learning rate.  No parameter interpolation or weight-distance trust is used.

After the actor update, the complete policy is evaluated on Train anchors
against the fixed pre-run safety baseline.  Unsafe outer steps are rolled back
and end the run.  Safe outer steps refresh student-state trajectories before the
next micro-step.  Validation is used only after Train-only selection; Final is
untouched.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import train_real_chart_v080 as v080
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
import train_real_chart_v171_n_key_connectome_failure_continuation_dagger as v171
import train_real_chart_v181_n_key_connectome_sgd_microstep_dagger as v181
from dmdod.fly_connectome_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
    NKeyFlyConnectomeActorCritic,
)
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_expert_sequence,
    n_key_actuation_loss,
)
from dmdod.random_connectome_policy import N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME


TRAINER_VERSION = "1.9.0-n-key-connectome-actor-action-trust-dagger"
CHECKPOINT_FORMAT_VERSION = 31
ACTION_TRUST_OPTIMIZER_SEMANTICS = "actor-only-plain-sgd-fixed-reference-action-trust-v1"

DEFAULT_MICRO_STEPS = 4
DEFAULT_ACTOR_STEPS = 8
DEFAULT_LR = 3e-4
DEFAULT_STAY_COEF = 20.0
DEFAULT_MAX_ACTION_RMS = 0.01
DEFAULT_LR_BACKOFFS = 10
DEFAULT_MIN_LR = 1e-8


@dataclass(slots=True)
class ActorTrustSequence:
    features: torch.Tensor
    teacher_actions: torch.Tensor
    reference_actions: torch.Tensor
    source: str

    @property
    def frames(self) -> int:
        return int(self.features.shape[0])


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    return deepcopy(optimizer.state_dict())


def _restore_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    optimizer.load_state_dict(deepcopy(state))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _actor_parameter_names(model) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in model.named_parameters()
        if name in {"actor_mean.weight", "actor_mean.bias"}
    )


def _freeze_actor_only(model) -> tuple[torch.nn.Parameter, ...]:
    actor_names = {"actor_mean.weight", "actor_mean.bias"}
    selected: list[torch.nn.Parameter] = []
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        trainable = name in actor_names
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
            seen.add(name)
    missing = actor_names - seen
    if missing:
        raise RuntimeError(
            "connectome policy is missing actor parameters: " + ", ".join(sorted(missing))
        )
    return tuple(selected)


@torch.inference_mode()
def _extract_connectome_features(
    model: NKeyFlyConnectomeActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Return deterministic recurrent states used by actor_mean.

    With the sensory/connectome side frozen, these states are fixed features.
    CUDA no-grad execution intentionally goes through the deterministic dense
    recurrent runtime selected by NKeyFlyConnectomeActorCritic.
    """

    if observations.ndim != 2 or observations.shape[1] != model.input_dim:
        raise ValueError(
            f"observations must have shape [T, {model.input_dim}], got "
            f"{tuple(observations.shape)}"
        )
    if observations.shape[0] == 0:
        return observations.new_empty((0, model.hidden_dim))

    model.eval()
    model.prepare_recurrent_runtime()
    encoded = model.sensory(observations)
    encoded.tanh_()
    # Each row is mutated in place by _advance_injected into that tick's state,
    # so this tensor itself becomes the feature matrix without a second [T,H]
    # allocation.
    features = torch.matmul(encoded, model._input_projection_transpose_runtime)
    state = model.initial_state(observations.device)
    for row in features:
        state = model._advance_injected(row, state)
    return features


@torch.inference_mode()
def _build_action_trust_sequences(
    model: NKeyFlyConnectomeActorCritic,
    sequences: list[NKeyBCSequence],
) -> list[ActorTrustSequence]:
    cached: list[ActorTrustSequence] = []
    for sequence in sequences:
        features = _extract_connectome_features(model, sequence.observations)
        reference_actions = torch.tanh(model.actor_mean(features))
        cached.append(
            ActorTrustSequence(
                features=features,
                teacher_actions=sequence.teacher_actions,
                reference_actions=reference_actions,
                source=sequence.source,
            )
        )
    return cached


@torch.inference_mode()
def _action_drift(
    model,
    sequences: list[ActorTrustSequence],
    *,
    chunk_steps: int,
) -> tuple[float, float]:
    squared = 0.0
    count = 0
    maximum = 0.0
    for sequence in sequences:
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            current = torch.tanh(model.actor_mean(sequence.features[start:end]))
            delta = current - sequence.reference_actions[start:end]
            squared += float(torch.sum(delta * delta).item())
            count += int(delta.numel())
            if delta.numel():
                maximum = max(maximum, float(delta.abs().max().item()))
    rms = (squared / max(1, count)) ** 0.5
    return rms, maximum


def _actor_full_dataset_gradient(
    model,
    sequences: list[ActorTrustSequence],
    *,
    chunk_steps: int,
    stay_coef: float,
) -> tuple[float, float, float, float]:
    if not sequences:
        raise ValueError("at least one actor trust sequence is required")
    total_frames = sum(sequence.frames for sequence in sequences)
    if total_frames <= 0:
        raise ValueError("actor trust sequences must contain at least one frame")

    actor_parameters = tuple(model.actor_mean.parameters())
    for parameter in actor_parameters:
        parameter.grad = None

    teacher_sum = 0.0
    stay_sum = 0.0
    objective_sum = 0.0

    model.train()
    for sequence in sequences:
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            features = sequence.features[start:end]
            target = sequence.teacher_actions[start:end]
            reference = sequence.reference_actions[start:end]
            predicted = torch.tanh(model.actor_mean(features))
            teacher_loss = n_key_actuation_loss(predicted, target)
            stay_loss = F.mse_loss(predicted, reference)
            objective = teacher_loss + float(stay_coef) * stay_loss
            frames = end - start
            weight = float(frames) / float(total_frames)
            (objective * weight).backward()
            teacher_sum += float(teacher_loss.detach().item()) * frames
            stay_sum += float(stay_loss.detach().item()) * frames
            objective_sum += float(objective.detach().item()) * frames

    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        actor_parameters,
        10.0,
        error_if_nonfinite=True,
    )
    grad_norm = float(grad_norm_tensor.detach().item())
    denominator = float(total_frames)
    return (
        objective_sum / denominator,
        teacher_sum / denominator,
        stay_sum / denominator,
        grad_norm,
    )


def _clone_actor_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.actor_mean.state_dict().items()
    }


def _restore_actor_state(model, state: dict[str, torch.Tensor]) -> None:
    model.actor_mean.load_state_dict(state)


def _train_actor_action_trust(
    model,
    sequences: list[ActorTrustSequence],
    *,
    optimizer: torch.optim.Optimizer,
    actor_steps: int,
    chunk_steps: int,
    base_lr: float,
    stay_coef: float,
    max_action_rms: float,
    lr_backoffs: int,
    min_lr: float,
) -> dict:
    """Train actor_mean while enforcing trust in actual squashed actions."""

    if actor_steps <= 0:
        raise ValueError("actor_steps must be positive")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    if base_lr <= 0.0 or min_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if stay_coef < 0.0:
        raise ValueError("stay_coef must be non-negative")
    if max_action_rms <= 0.0:
        raise ValueError("max_action_rms must be positive")
    if lr_backoffs < 0:
        raise ValueError("lr_backoffs must be non-negative")
    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("v1.9 action-trust optimizer must be torch.optim.SGD")

    accepted_inner = 0
    current_lr = float(base_lr)
    last_objective = 0.0
    last_teacher = 0.0
    last_stay = 0.0
    last_grad_norm = 0.0
    action_rms, action_max = _action_drift(model, sequences, chunk_steps=chunk_steps)

    for inner in range(1, actor_steps + 1):
        before_actor = _clone_actor_state(model)
        before_optimizer = _clone_optimizer_state(optimizer)

        objective, teacher_loss, stay_loss, grad_norm = _actor_full_dataset_gradient(
            model,
            sequences,
            chunk_steps=chunk_steps,
            stay_coef=stay_coef,
        )
        last_objective = objective
        last_teacher = teacher_loss
        last_stay = stay_loss
        last_grad_norm = grad_norm

        accepted_this_inner = False
        trial_lr = current_lr
        for _ in range(lr_backoffs + 1):
            if trial_lr < min_lr:
                break
            _restore_actor_state(model, before_actor)
            _restore_optimizer_state(optimizer, before_optimizer)
            _set_optimizer_lr(optimizer, trial_lr)
            # Gradients survive load_state_dict/optimizer restore. Plain SGD has
            # no momentum, so retrying with a smaller LR follows the same local
            # gradient direction without parameter interpolation.
            optimizer.step()
            trial_rms, trial_max = _action_drift(
                model,
                sequences,
                chunk_steps=chunk_steps,
            )
            if trial_rms <= max_action_rms:
                action_rms = trial_rms
                action_max = trial_max
                current_lr = trial_lr
                accepted_inner += 1
                accepted_this_inner = True
                print(
                    f"  actor-step {inner:02d}/{actor_steps}: "
                    f"objective={objective:.6f} teacher={teacher_loss:.6f} "
                    f"stay={stay_loss:.6f} grad={grad_norm:.6g} lr={trial_lr:.3g} "
                    f"action-rms={trial_rms:.6g} action-max={trial_max:.6g}"
                )
                break
            trial_lr *= 0.5

        if not accepted_this_inner:
            _restore_actor_state(model, before_actor)
            _restore_optimizer_state(optimizer, before_optimizer)
            print(
                f"  actor-step {inner:02d}/{actor_steps}: ACTION-TRUST STOP "
                f"(cannot stay <= rms {max_action_rms:g} above min-lr {min_lr:g})"
            )
            break

    return {
        "objective": float(last_objective),
        "teacher_loss": float(last_teacher),
        "stay_loss": float(last_stay),
        "grad_norm": float(last_grad_norm),
        "action_rms": float(action_rms),
        "action_max": float(action_max),
        "accepted_inner_steps": int(accepted_inner),
        "final_lr": float(current_lr),
    }


def _checkpoint_payload(
    parent: dict,
    *,
    model,
    model_state: dict[str, torch.Tensor],
    source_checkpoint: Path,
    output_checkpoint: Path,
    round_index: int,
    requested_steps: int,
    attempted_steps: int,
    accepted_steps: int,
    selected_step: int,
    stopped_unsafe: bool,
    actor_steps: int,
    lr: float,
    stay_coef: float,
    max_action_rms: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
) -> dict:
    payload = dict(parent)
    payload.update(model.checkpoint_metadata())
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model_state,
            "dagger_optimizer_state": None,
            "dagger_round": int(round_index),
            "dagger_action_mode": "continuous",
            "dagger_lr": float(lr),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_student_state_frame_history": [
                int(frames) for frames in student_frame_history
            ],
            "dagger_student_state_refresh": "after-each-safe-action-trust-step",
            "dagger_trust_alphas": (),
            "dagger_trust_history": [],
            "dagger_trust_continuation": "removed-v1.9-action-space-trust",
            "dagger_selection_uses_validation": False,
            "dagger_source_checkpoint": str(source_checkpoint),
            "dagger_output_checkpoint": str(output_checkpoint),
            "action_trust_optimizer_semantics": ACTION_TRUST_OPTIMIZER_SEMANTICS,
            "action_trust_trainable_parameters": _actor_parameter_names(model),
            "action_trust_frozen_feature_extractor": True,
            "action_trust_reference": "accepted-policy-actions-on-current-aggregate-dataset",
            "action_trust_stay_coef": float(stay_coef),
            "action_trust_max_rms": float(max_action_rms),
            "action_trust_actor_steps_per_guard": int(actor_steps),
            "action_trust_requested_steps": int(requested_steps),
            "action_trust_attempted_steps": int(attempted_steps),
            "action_trust_accepted_steps": int(accepted_steps),
            "action_trust_selected_step": int(selected_step),
            "action_trust_stopped_unsafe": bool(stopped_unsafe),
            "action_trust_safety_reference": "fixed-pre-run-train-anchors",
            "action_trust_history": list(history),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    return payload


def _default_output_path(source: Path, round_index: int) -> Path:
    suffix = source.suffix or ".pt"
    stem = source.name[: -len(suffix)] if source.name.endswith(suffix) else source.name
    return source.with_name(f"{stem}_actiontrust{round_index}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run actor-only connectome DAgger with fixed-reference action-space trust "
            "and fixed-baseline Train safety."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--micro-steps", type=int, default=DEFAULT_MICRO_STEPS)
    parser.add_argument("--actor-steps", type=int, default=DEFAULT_ACTOR_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--stay-coef", type=float, default=DEFAULT_STAY_COEF)
    parser.add_argument(
        "--max-action-rms",
        type=float,
        default=DEFAULT_MAX_ACTION_RMS,
    )
    parser.add_argument("--lr-backoffs", type=int, default=DEFAULT_LR_BACKOFFS)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.micro_steps <= 0:
        raise SystemExit("--micro-steps must be positive")
    if args.actor_steps <= 0:
        raise SystemExit("--actor-steps must be positive")
    if args.lr <= 0.0 or args.min_lr <= 0.0:
        raise SystemExit("--lr/--min-lr must be positive")
    if args.stay_coef < 0.0:
        raise SystemExit("--stay-coef must be non-negative")
    if args.max_action_rms <= 0.0:
        raise SystemExit("--max-action-rms must be positive")
    if args.lr_backoffs < 0:
        raise SystemExit("--lr-backoffs must be non-negative")
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
            "v1.9 action-trust trainer requires fly_connectome or random_connectome"
        )

    key_count = int(parent["key_count"])
    input_dim = int(parent["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected "
            f"{expected_input}"
        )

    key_names = n_key_names(key_count)
    model = v171._build_policy_from_checkpoint(parent, device=device)
    actor_parameters = _freeze_actor_only(model)

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

    print("=== DMDOD v1.9.0 N-Key Connectome Actor Action-Trust DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"backend={model.backend_name} keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchors)} validation={len(validation)} micro-steps={args.micro_steps} "
        f"actor-steps={args.actor_steps} lr={args.lr:g} stay={args.stay_coef:g} "
        f"max-action-rms={args.max_action_rms:g} chunk={chunk_steps} | FINAL untouched"
    )
    print(
        "Trainable: actor_mean only. Sensory + fixed connectome + projection + critic + "
        "log_std are frozen. Trust is measured on squashed actions, not weight distance."
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
            source=f"dagger{round_index}-actiontrust-expert-{index}-{named.chart_name}",
        )
        expert_sequences.append(expert.sequence)
    expert_frames = sum(sequence.frames for sequence in expert_sequences)

    dagger_sequences, dagger_frames = v181._collect_student_state_sequences(
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

    best_state = continuation_state
    best_results = initial_results
    best_step = 0
    history: list[dict] = []
    attempted_steps = 0
    accepted_steps = 0
    stopped_unsafe = False

    for step in range(1, args.micro_steps + 1):
        attempted_steps = step
        model.load_state_dict(continuation_state)
        model.prepare_recurrent_runtime()
        _freeze_actor_only(model)

        print(f"=== action-trust actor update {step}/{args.micro_steps} ===")
        cached = _build_action_trust_sequences(model, training_sequences)
        optimizer = torch.optim.SGD(actor_parameters, lr=args.lr)
        train_info = _train_actor_action_trust(
            model,
            cached,
            optimizer=optimizer,
            actor_steps=args.actor_steps,
            chunk_steps=chunk_steps,
            base_lr=args.lr,
            stay_coef=args.stay_coef,
            max_action_rms=args.max_action_rms,
            lr_backoffs=args.lr_backoffs,
            min_lr=args.min_lr,
        )

        print(
            f"microstep {step:03d}/{args.micro_steps}: "
            f"loss={train_info['objective']:.6f} "
            f"grad-norm={train_info['grad_norm']:.6g} "
            f"action-rms={train_info['action_rms']:.6g} "
            f"action-max={train_info['action_max']:.6g} "
            f"inner={train_info['accepted_inner_steps']}/{args.actor_steps} "
            f"lr={train_info['final_lr']:.3g}"
        )

        if int(train_info["accepted_inner_steps"]) == 0:
            stopped_unsafe = True
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            print(
                f"microstep-loop: STOP at action-trust no-op step={step}; "
                "no actor update fit inside the action RMS bound"
            )
            break

        candidate_state = v161._clone_model_state(model)
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

        summary = v161._summarize(candidate_results)
        history.append(
            {
                "step": int(step),
                **train_info,
                "guard_accepted": bool(safe),
                "guard_reasons": tuple(reasons),
                "selected_best_when_evaluated": bool(selected_best),
                "hits": int(summary.hits),
                "targets": int(summary.targets),
                "x_accuracy_percent": float(summary.x_accuracy_percent),
                "early": int(summary.early),
                "overloaded": bool(summary.overloaded),
                "keydowns": int(summary.keydowns),
            }
        )

        if not safe:
            stopped_unsafe = True
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            print(
                f"microstep-loop: STOP at rejected step={step}; "
                "rolled back to last safe actor"
            )
            break

        continuation_state = candidate_state
        continuation_results = candidate_results
        accepted_steps += 1

        if selected_best:
            best_state = candidate_state
            best_results = candidate_results
            best_step = step

        print(
            f"microstep-continuation: ACCEPT step={step} "
            + v161._aggregate(continuation_results)
        )

        # Release the [frames, 4096] feature cache before collecting the next
        # generation.
        del cached

        if step < args.micro_steps:
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            dagger_sequences, dagger_frames = v181._collect_student_state_sequences(
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
    _freeze_actor_only(model)

    print("=== selected Train-safe action-trust checkpoint ===")
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
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            requested_steps=args.micro_steps,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            selected_step=best_step,
            stopped_unsafe=stopped_unsafe,
            actor_steps=args.actor_steps,
            lr=args.lr,
            stay_coef=args.stay_coef,
            max_action_rms=args.max_action_rms,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            student_frame_history=student_frame_history,
            history=history,
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
