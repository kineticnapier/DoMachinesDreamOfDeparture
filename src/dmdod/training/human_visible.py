from __future__ import annotations

"""Training helpers for the human-visible residual controller."""

from dataclasses import asdict, dataclass
import random

import torch

from dmdod.connectome.human_visible_policy import (
    NKeyHumanVisibleControllerActorCritic,
)
from dmdod.training.n_key import NKeyBCSequence, n_key_actuation_loss


TRAINABLE_PREFIXES = (
    "connectome_context.",
    "floor_encoder.",
    "motor_encoder.",
    "hud_encoder.",
    "controller.",
    "controller_post.",
    "controller_delta.",
)


@dataclass(slots=True)
class HumanVisibleReplayChunk:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    initial_state: torch.Tensor
    source: str
    start: int

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class HumanVisibleTrainMetrics:
    updates: int
    mean_loss: float
    final_loss: float
    mean_grad_norm: float
    max_grad_norm: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def human_visible_parameter_names(model) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in model.named_parameters()
        if name.startswith(TRAINABLE_PREFIXES)
    )


def freeze_human_visible_controller(
    model: NKeyHumanVisibleControllerActorCritic,
) -> tuple[torch.nn.Parameter, ...]:
    """Train the new controller while preserving the inherited baseline path."""

    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)

    if not selected:
        raise RuntimeError("human-visible controller has no trainable parameters")
    return tuple(selected)


@torch.no_grad()
def build_human_visible_replay_chunks(
    model: NKeyHumanVisibleControllerActorCritic,
    sequences: list[NKeyBCSequence],
    *,
    chunk_steps: int,
) -> list[HumanVisibleReplayChunk]:
    """Cache realistic recurrent state at each truncated-BPTT chunk boundary."""

    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    model.eval()
    model.prepare_recurrent_runtime()
    chunks: list[HumanVisibleReplayChunk] = []
    for sequence in sequences:
        state = model.initial_state(sequence.observations.device)
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            chunks.append(
                HumanVisibleReplayChunk(
                    observations=sequence.observations[start:end],
                    teacher_actions=sequence.teacher_actions[start:end],
                    initial_state=state.detach().clone(),
                    source=sequence.source,
                    start=start,
                )
            )
            _, _, state = model.forward_sequence(
                sequence.observations[start:end],
                state,
            )
            state = state.detach()
    if not chunks:
        raise ValueError("at least one non-empty replay chunk is required")
    return chunks


def train_human_visible_replay(
    model: NKeyHumanVisibleControllerActorCritic,
    chunks: list[HumanVisibleReplayChunk],
    *,
    optimizer: torch.optim.Optimizer,
    updates: int,
    grad_clip: float,
    seed: int,
) -> HumanVisibleTrainMetrics:
    """Run ordinary AdamW updates on cached student/expert recurrent states.

    Candidate diversity comes from shuffled replay chunks rather than from a
    single full-dataset gradient direction.  The fixed MaleCNS path and legacy
    baseline actor remain frozen; only the new human-visible residual controller
    learns.
    """

    if updates <= 0:
        raise ValueError("updates must be positive")
    if grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if not chunks:
        raise ValueError("replay chunks must not be empty")

    trainable = freeze_human_visible_controller(model)
    model.train()
    rng = random.Random(int(seed))
    order = list(range(len(chunks)))
    rng.shuffle(order)

    losses: list[float] = []
    grad_norms: list[float] = []
    for update in range(updates):
        if update and update % len(order) == 0:
            rng.shuffle(order)
        chunk = chunks[order[update % len(order)]]

        optimizer.zero_grad(set_to_none=True)
        means, _, _ = model.forward_sequence(
            chunk.observations,
            chunk.initial_state.detach(),
        )
        loss = n_key_actuation_loss(
            torch.tanh(means),
            chunk.teacher_actions,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(grad_clip),
            error_if_nonfinite=True,
        )
        optimizer.step()

        loss_value = float(loss.detach().item())
        grad_value = float(grad_norm.detach().item())
        losses.append(loss_value)
        grad_norms.append(grad_value)

    return HumanVisibleTrainMetrics(
        updates=int(updates),
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
        mean_grad_norm=sum(grad_norms) / len(grad_norms),
        max_grad_norm=max(grad_norms),
    )
