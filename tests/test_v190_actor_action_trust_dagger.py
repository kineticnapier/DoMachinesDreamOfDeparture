from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v190_n_key_connectome_actor_action_trust_dagger as v190


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
        self._input_projection_transpose_runtime = self.input_projection.transpose(0, 1)

    def prepare_recurrent_runtime(self) -> None:
        self._input_projection_transpose_runtime = self.input_projection.transpose(0, 1)

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

    def _advance_injected(self, injected: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        injected.add_(0.2 * state)
        return injected.tanh_()


def _trust_sequence(model: _TinyPolicy, frames: int = 24) -> v190.ActorTrustSequence:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(frames, model.hidden_dim, generator=generator)
    with torch.no_grad():
        reference = torch.tanh(model.actor_mean(features))
    teacher = torch.where(
        torch.randn(frames, model.key_count, generator=generator) > 0,
        torch.ones(frames, model.key_count),
        -torch.ones(frames, model.key_count),
    )
    return v190.ActorTrustSequence(
        features=features,
        teacher_actions=teacher,
        reference_actions=reference,
        source="tiny",
    )


def test_freeze_actor_only_selects_exact_readout_parameters() -> None:
    model = _TinyPolicy()
    selected = v190._freeze_actor_only(model)

    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    assert trainable_names == ("actor_mean.weight", "actor_mean.bias")
    assert len(selected) == 2
    assert all(
        a is b for a, b in zip(selected, model.actor_mean.parameters())
    )
    assert not model.sensory.weight.requires_grad
    assert not model.critic.weight.requires_grad
    assert not model.log_std.requires_grad


def test_feature_extractor_is_deterministic_and_actor_independent() -> None:
    torch.manual_seed(3)
    model = _TinyPolicy()
    observations = torch.randn(11, model.input_dim)

    first = v190._extract_connectome_features(model, observations)
    with torch.no_grad():
        model.actor_mean.weight.add_(10.0)
        model.actor_mean.bias.sub_(4.0)
    second = v190._extract_connectome_features(model, observations)

    assert first.shape == (11, model.hidden_dim)
    assert torch.equal(first, second)

    v190._freeze_actor_only(model)
    model.actor_mean.zero_grad(set_to_none=True)
    torch.tanh(model.actor_mean(first)).sum().backward()
    assert model.actor_mean.weight.grad is not None
    assert model.actor_mean.bias.grad is not None


def test_action_trust_backoff_keeps_rms_inside_bound() -> None:
    torch.manual_seed(11)
    model = _TinyPolicy()
    actor_parameters = v190._freeze_actor_only(model)
    sequence = _trust_sequence(model)
    optimizer = torch.optim.SGD(actor_parameters, lr=1.0)

    info = v190._train_actor_action_trust(
        model,
        [sequence],
        optimizer=optimizer,
        actor_steps=3,
        chunk_steps=5,
        base_lr=1.0,
        stay_coef=20.0,
        max_action_rms=1e-3,
        lr_backoffs=30,
        min_lr=1e-10,
    )

    assert int(info["accepted_inner_steps"]) >= 1
    assert float(info["action_rms"]) <= 1e-3 * (1.0 + 1e-6)
    assert float(info["final_lr"]) < 1.0


def test_actor_training_does_not_change_frozen_parameters() -> None:
    torch.manual_seed(23)
    model = _TinyPolicy()
    actor_parameters = v190._freeze_actor_only(model)
    before_sensory = {
        name: tensor.detach().clone()
        for name, tensor in model.sensory.state_dict().items()
    }
    before_critic = {
        name: tensor.detach().clone()
        for name, tensor in model.critic.state_dict().items()
    }
    before_log_std = model.log_std.detach().clone()

    sequence = _trust_sequence(model)
    optimizer = torch.optim.SGD(actor_parameters, lr=1e-3)
    v190._train_actor_action_trust(
        model,
        [sequence],
        optimizer=optimizer,
        actor_steps=2,
        chunk_steps=6,
        base_lr=1e-3,
        stay_coef=20.0,
        max_action_rms=0.5,
        lr_backoffs=4,
        min_lr=1e-8,
    )

    for name, tensor in model.sensory.state_dict().items():
        assert torch.equal(tensor, before_sensory[name])
    for name, tensor in model.critic.state_dict().items():
        assert torch.equal(tensor, before_critic[name])
    assert torch.equal(model.log_std, before_log_std)


def test_checkpoint_payload_records_action_space_trust_semantics() -> None:
    class _FakeModel:
        def checkpoint_metadata(self):
            return {"n_key_policy_backend": "fly_connectome"}

        def named_parameters(self):
            yield "sensory.weight", nn.Parameter(torch.zeros(1), requires_grad=False)
            yield "actor_mean.weight", nn.Parameter(torch.zeros(1), requires_grad=True)
            yield "actor_mean.bias", nn.Parameter(torch.zeros(1), requires_grad=True)

    payload = v190._checkpoint_payload(
        {"dagger_trust_alphas": (0.5,)},
        model=_FakeModel(),
        model_state={"actor_mean.weight": torch.zeros(1)},
        source_checkpoint=Path("source.pt"),
        output_checkpoint=Path("output.pt"),
        round_index=4,
        requested_steps=3,
        attempted_steps=2,
        accepted_steps=1,
        selected_step=1,
        stopped_unsafe=True,
        actor_steps=8,
        lr=3e-4,
        stay_coef=20.0,
        max_action_rms=0.01,
        expert_frames=100,
        dagger_frames=120,
        student_frame_history=[120, 121],
        history=[{"step": 1}],
    )

    assert payload["format_version"] == 31
    assert payload["trainer_version"] == v190.TRAINER_VERSION
    assert payload["dagger_trust_alphas"] == ()
    assert payload["action_trust_frozen_feature_extractor"] is True
    assert payload["action_trust_trainable_parameters"] == (
        "actor_mean.weight",
        "actor_mean.bias",
    )
    assert payload["action_trust_max_rms"] == 0.01
    assert payload["action_trust_stay_coef"] == 20.0
    assert payload["action_trust_safety_reference"] == "fixed-pre-run-train-anchors"
