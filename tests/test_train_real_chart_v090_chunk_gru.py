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


def test_batched_bc_uses_one_forward_sequence_call_per_chunk(monkeypatch) -> None:
    torch.manual_seed(20260930)
    model = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    observations = torch.randn(11, 5)
    teacher_actions = torch.tanh(torch.randn(11, 2))
    stable = SimpleNamespace(
        sequence=SimpleNamespace(
            observations=observations,
            teacher_actions=teacher_actions,
            frames=11,
            source="test-sequence",
        ),
        loss_weights=torch.ones(11),
    )

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
