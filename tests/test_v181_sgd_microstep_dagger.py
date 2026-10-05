from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v181_n_key_connectome_sgd_microstep_dagger as v181


class _TinyPolicy(nn.Module):
    key_count = 2

    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(3, 2)

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, device=device)

    def forward_sequence(self, observations: torch.Tensor, state: torch.Tensor):
        means = self.actor(observations)
        values = observations.new_zeros((observations.shape[0],))
        return means, values, state


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, *, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _sequence(frames: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    observations = torch.randn(frames, 3, generator=generator)
    teacher_actions = torch.where(
        torch.randn(frames, 2, generator=generator) >= 0,
        torch.ones(frames, 2),
        -torch.ones(frames, 2),
    )
    return SimpleNamespace(
        observations=observations,
        teacher_actions=teacher_actions,
        key_count=2,
        frames=frames,
    )


def test_aggregate_microstep_performs_one_norm_bounded_sgd_step() -> None:
    torch.manual_seed(7)
    model = _TinyPolicy()
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    max_update_norm = 1e-5

    loss, grad_norm, update_norm, grad_scale = v181._train_aggregate_microstep(
        model,
        [_sequence(7, 1), _sequence(5, 2)],
        optimizer=optimizer,
        chunk_steps=2,
        max_update_norm=max_update_norm,
    )

    assert optimizer.step_calls == 1
    assert loss >= 0.0
    assert grad_norm >= 0.0
    assert 0.0 < grad_scale <= 1.0
    assert update_norm <= max_update_norm * (1.0 + 1e-6)

    squared_delta = 0.0
    for name, value in model.state_dict().items():
        delta = value.detach() - before[name]
        squared_delta += float(torch.sum(delta * delta).item())
    actual_update_norm = squared_delta ** 0.5
    assert actual_update_norm > 0.0
    assert actual_update_norm <= max_update_norm * 1.01
    assert abs(actual_update_norm - update_norm) <= max(1e-8, max_update_norm * 0.02)


def test_legacy_optimizer_state_is_not_resumed() -> None:
    assert not v181._can_resume_microstep_optimizer(
        {
            "dagger_optimizer_state": {"legacy": True},
            "microstep_optimizer_state": {"adam-v180": True},
            "microstep_optimizer_semantics": "aggregate-gradient-one-adam-step-v1",
        }
    )
    assert v181._can_resume_microstep_optimizer(
        {
            "microstep_optimizer_state": {"sgd-v181": True},
            "microstep_optimizer_semantics": v181.MICROSTEP_OPTIMIZER_SEMANTICS,
        }
    )


def test_checkpoint_payload_marks_alpha_interpolation_removed() -> None:
    class _FakeModel:
        def checkpoint_metadata(self):
            return {"n_key_policy_backend": "fly_connectome"}

    payload = v181._checkpoint_payload(
        {"dagger_optimizer_state": {"legacy": True}, "dagger_trust_alphas": (0.5,)},
        model=_FakeModel(),
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"state": {}, "param_groups": []},
        source_checkpoint=Path("source.pt"),
        output_checkpoint=Path("output.pt"),
        round_index=4,
        requested_steps=5,
        attempted_steps=2,
        accepted_steps=1,
        selected_step=1,
        stopped_unsafe=True,
        lr=3e-6,
        max_update_norm=1e-5,
        expert_frames=100,
        dagger_frames=120,
        student_frame_history=[120],
        losses=[0.5, 0.4],
        history=[{"step": 1}],
        optimizer_resumed=False,
    )

    assert payload["format_version"] == 30
    assert payload["trainer_version"] == v181.TRAINER_VERSION
    assert payload["dagger_optimizer_state"] is None
    assert payload["dagger_trust_alphas"] == ()
    assert payload["dagger_trust_continuation"] == "removed-v1.8-direct-microstep"
    assert payload["microstep_optimizer_semantics"] == v181.MICROSTEP_OPTIMIZER_SEMANTICS
    assert payload["microstep_safety_reference"] == "fixed-pre-run-train-anchors"
    assert payload["microstep_max_update_norm"] == 1e-5
    assert payload["microstep_gradient"] == "frame-weighted-aggregate-before-one-sgd-step"
    assert payload["microstep_stopped_unsafe"] is True


def test_microstep_rejects_adam_optimizer() -> None:
    model = _TinyPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    try:
        v181._train_aggregate_microstep(
            model,
            [_sequence(3, 9)],
            optimizer=optimizer,
            chunk_steps=2,
            max_update_norm=1e-5,
        )
    except TypeError as exc:
        assert "torch.optim.SGD" in str(exc)
    else:
        raise AssertionError("Adam must not be accepted by the v1.8.1 micro-step path")
