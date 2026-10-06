from __future__ import annotations

import torch
from torch import nn

from dmdod.training.action_trust import (
    ActorTrustSequence,
    extract_connectome_features,
    freeze_actor_only,
    train_actor_action_trust,
)


class _TinyPolicy(nn.Module):
    input_dim = 3
    hidden_dim = 4
    key_count = 2

    def __init__(self) -> None:
        super().__init__()
        self.sensory = nn.Linear(3, 2)
        self.actor_mean = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)
        self.log_std = nn.Parameter(torch.zeros(2))
        projection = torch.tensor(
            [
                [0.5, -0.2],
                [0.1, 0.3],
                [-0.4, 0.2],
                [0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("input_projection", projection)
        self._input_projection_transpose_runtime = (
            self.input_projection.transpose(0, 1)
        )

    def prepare_recurrent_runtime(self) -> None:
        self._input_projection_transpose_runtime = (
            self.input_projection.transpose(0, 1)
        )

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

    def _advance_injected(
        self,
        injected: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        injected.add_(0.2 * state)
        return injected.tanh_()


def _sequence(model: _TinyPolicy, frames: int = 24) -> ActorTrustSequence:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(frames, model.hidden_dim, generator=generator)
    with torch.no_grad():
        reference = torch.tanh(model.actor_mean(features))
    teacher = torch.where(
        torch.randn(frames, model.key_count, generator=generator) > 0,
        torch.ones(frames, model.key_count),
        -torch.ones(frames, model.key_count),
    )
    return ActorTrustSequence(
        features=features,
        teacher_actions=teacher,
        reference_actions=reference,
        source="tiny",
    )


def test_freeze_actor_only_selects_exact_readout() -> None:
    model = _TinyPolicy()
    selected = freeze_actor_only(model)

    names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert names == ("actor_mean.weight", "actor_mean.bias")
    assert len(selected) == 2


def test_cached_features_are_actor_independent_and_autograd_usable() -> None:
    torch.manual_seed(3)
    model = _TinyPolicy()
    observations = torch.randn(11, model.input_dim)

    first = extract_connectome_features(model, observations)
    with torch.no_grad():
        model.actor_mean.weight.add_(10.0)
        model.actor_mean.bias.sub_(4.0)
    second = extract_connectome_features(model, observations)

    assert torch.equal(first, second)

    freeze_actor_only(model)
    model.actor_mean.zero_grad(set_to_none=True)
    torch.tanh(model.actor_mean(first)).sum().backward()
    assert model.actor_mean.weight.grad is not None


def test_action_trust_backoff_respects_rms_bound() -> None:
    torch.manual_seed(11)
    model = _TinyPolicy()
    actor = freeze_actor_only(model)
    optimizer = torch.optim.SGD(actor, lr=1.0)

    metrics = train_actor_action_trust(
        model,
        [_sequence(model)],
        optimizer=optimizer,
        actor_steps=3,
        chunk_steps=5,
        base_lr=1.0,
        stay_coef=20.0,
        max_action_rms=1e-3,
        lr_backoffs=30,
        min_lr=1e-10,
    )

    assert metrics.accepted_inner_steps >= 1
    assert metrics.action_rms <= 1e-3 * (1.0 + 1e-6)
    assert metrics.final_lr < 1.0
