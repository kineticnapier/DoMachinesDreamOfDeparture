from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from dmdod.connectome.fly_policy import NKeyFlyConnectomeActorCritic
from dmdod.n_key_training import NKeyBCSequence, n_key_actuation_loss


@dataclass(slots=True)
class ActorTrustSequence:
    features: torch.Tensor
    teacher_actions: torch.Tensor
    reference_actions: torch.Tensor
    source: str

    @property
    def frames(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class ActionTrustMetrics:
    objective: float
    teacher_loss: float
    stay_loss: float
    grad_norm: float
    action_rms: float
    action_max: float
    accepted_inner_steps: int
    final_lr: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def actor_parameter_names(model) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in model.named_parameters()
        if name in {"actor_mean.weight", "actor_mean.bias"}
    )


def freeze_actor_only(model) -> tuple[torch.nn.Parameter, ...]:
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
            "connectome policy is missing actor parameters: "
            + ", ".join(sorted(missing))
        )
    return tuple(selected)


@torch.no_grad()
def extract_connectome_features(
    model: NKeyFlyConnectomeActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
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
    features = torch.matmul(
        encoded,
        model._input_projection_transpose_runtime,
    )
    state = model.initial_state(observations.device)
    for row in features:
        state = model._advance_injected(row, state)
    return features


@torch.no_grad()
def build_action_trust_sequences(
    model: NKeyFlyConnectomeActorCritic,
    sequences: list[NKeyBCSequence],
) -> list[ActorTrustSequence]:
    result: list[ActorTrustSequence] = []
    for sequence in sequences:
        features = extract_connectome_features(model, sequence.observations)
        result.append(
            ActorTrustSequence(
                features=features,
                teacher_actions=sequence.teacher_actions,
                reference_actions=torch.tanh(model.actor_mean(features)),
                source=sequence.source,
            )
        )
    return result


@torch.no_grad()
def action_drift(
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
            current = torch.tanh(
                model.actor_mean(sequence.features[start:end])
            )
            delta = current - sequence.reference_actions[start:end]
            squared += float(torch.sum(delta * delta).item())
            count += int(delta.numel())
            if delta.numel():
                maximum = max(maximum, float(delta.abs().max().item()))

    return (squared / max(1, count)) ** 0.5, maximum


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

    objective_sum = 0.0
    teacher_sum = 0.0
    stay_sum = 0.0
    model.train()

    for sequence in sequences:
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            predicted = torch.tanh(
                model.actor_mean(sequence.features[start:end])
            )
            teacher_loss = n_key_actuation_loss(
                predicted,
                sequence.teacher_actions[start:end],
            )
            stay_loss = F.mse_loss(
                predicted,
                sequence.reference_actions[start:end],
            )
            objective = teacher_loss + float(stay_coef) * stay_loss
            frames = end - start
            weight = float(frames) / float(total_frames)
            (objective * weight).backward()

            objective_sum += float(objective.detach().item()) * frames
            teacher_sum += float(teacher_loss.detach().item()) * frames
            stay_sum += float(stay_loss.detach().item()) * frames

    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        actor_parameters,
        10.0,
        error_if_nonfinite=True,
    )
    denominator = float(total_frames)
    return (
        objective_sum / denominator,
        teacher_sum / denominator,
        stay_sum / denominator,
        float(grad_norm_tensor.detach().item()),
    )


def _clone_actor_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.actor_mean.state_dict().items()
    }


def _restore_actor_state(
    model,
    state: dict[str, torch.Tensor],
) -> None:
    model.actor_mean.load_state_dict(state)


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    import copy
    return copy.deepcopy(optimizer.state_dict())


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: dict,
) -> None:
    import copy
    optimizer.load_state_dict(copy.deepcopy(state))


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_actor_action_trust(
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
) -> ActionTrustMetrics:
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
        raise TypeError("action-trust optimizer must be torch.optim.SGD")

    accepted_inner = 0
    current_lr = float(base_lr)
    last_objective = 0.0
    last_teacher = 0.0
    last_stay = 0.0
    last_grad_norm = 0.0
    action_rms, action_max = action_drift(
        model,
        sequences,
        chunk_steps=chunk_steps,
    )

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
            optimizer.step()

            trial_rms, trial_max = action_drift(
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
                    f"objective={objective:.6f} "
                    f"teacher={teacher_loss:.6f} stay={stay_loss:.6f} "
                    f"grad={grad_norm:.6g} lr={trial_lr:.3g} "
                    f"action-rms={trial_rms:.6g} "
                    f"action-max={trial_max:.6g}"
                )
                break
            trial_lr *= 0.5

        if not accepted_this_inner:
            _restore_actor_state(model, before_actor)
            _restore_optimizer_state(optimizer, before_optimizer)
            print(
                f"  actor-step {inner:02d}/{actor_steps}: ACTION-TRUST STOP "
                f"(cannot stay <= rms {max_action_rms:g} "
                f"above min-lr {min_lr:g})"
            )
            break

    return ActionTrustMetrics(
        objective=float(last_objective),
        teacher_loss=float(last_teacher),
        stay_loss=float(last_stay),
        grad_norm=float(last_grad_norm),
        action_rms=float(action_rms),
        action_max=float(action_max),
        accepted_inner_steps=int(accepted_inner),
        final_lr=float(current_lr),
    )
