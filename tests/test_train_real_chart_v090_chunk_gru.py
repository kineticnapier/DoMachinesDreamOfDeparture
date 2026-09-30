from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v057 as v057
import train_real_chart_v090_chunk_gru as chunk_gru
from dmdod.recurrent_policy import RecurrentActorCritic


def _stable_sequence(*, frames: int = 11, input_dim: int = 5):
    observations = torch.randn(frames, input_dim)
    teacher_actions = torch.tanh(torch.randn(frames, 2))
    per_frame = torch.linspace(0.5, 1.5, frames).unsqueeze(1)
    loss_weights = per_frame.expand(-1, 2).clone()
    return SimpleNamespace(
        sequence=SimpleNamespace(
            observations=observations,
            teacher_actions=teacher_actions,
            frames=frames,
            source="test-sequence",
        ),
        loss_weights=loss_weights,
    )


def test_batched_bc_uses_one_forward_sequence_call_per_chunk(monkeypatch) -> None:
    torch.manual_seed(20260930)
    model = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    stable = _stable_sequence()

    calls: list[int] = []
    original_forward_sequence = model.forward_sequence

    def traced_forward_sequence(chunk, state):
        calls.append(int(chunk.shape[0]))
        return original_forward_sequence(chunk, state)

    def forbidden_forward_step(*_args, **_kwargs):
        raise AssertionError("batched BC must not call forward_step inside a chunk")

    monkeypatch.setattr(model, "forward_sequence", traced_forward_sequence)
    monkeypatch.setattr(model, "forward_step", forbidden_forward_step)

    optimizer = torch.optim.Adam(v057._policy_parameters(model), lr=1e-4)
    loss = chunk_gru._batched_progress_train_one_epoch(
        model,
        [stable],
        optimizer=optimizer,
        chunk_steps=4,
        reverse_order=False,
    )

    assert calls == [4, 4, 3]
    assert math.isfinite(loss)


def test_batched_bc_matches_legacy_step_loop_after_optimizer_updates() -> None:
    torch.manual_seed(20260930)
    legacy = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    batched = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    batched.load_state_dict(legacy.state_dict())
    stable = _stable_sequence(frames=23)

    legacy_optimizer = torch.optim.Adam(v057._policy_parameters(legacy), lr=1e-4)
    batched_optimizer = torch.optim.Adam(v057._policy_parameters(batched), lr=1e-4)

    legacy_loss = v057._train_one_epoch(
        legacy,
        [stable],
        optimizer=legacy_optimizer,
        chunk_steps=7,
        reverse_order=False,
    )
    batched_loss = chunk_gru._batched_progress_train_one_epoch(
        batched,
        [stable],
        optimizer=batched_optimizer,
        chunk_steps=7,
        reverse_order=False,
    )

    assert math.isclose(batched_loss, legacy_loss, rel_tol=1e-6, abs_tol=1e-7)
    legacy_state = legacy.state_dict()
    batched_state = batched.state_dict()
    assert legacy_state.keys() == batched_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(
            batched_state[key],
            legacy_state[key],
            rtol=2e-5,
            atol=2e-6,
            msg=lambda message, key=key: f"{key}: {message}",
        )
