from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v057 as v057
import train_real_chart_v090_chunk_gru as chunk_gru
import train_real_chart_v090_cuda_bc as cuda_bc
from dmdod.recurrent_policy import RecurrentActorCritic


def _stable_sequence(*, frames: int = 19, input_dim: int = 5):
    observations = torch.randn(frames, input_dim)
    teacher_actions = torch.tanh(torch.randn(frames, 2))
    loss_weights = torch.rand(frames, 2) + 0.25
    return SimpleNamespace(
        sequence=SimpleNamespace(
            observations=observations,
            teacher_actions=teacher_actions,
            frames=frames,
            source="cuda-bc-test",
        ),
        loss_weights=loss_weights,
    )


def test_cuda_requested_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DMDOD_BC_CUDA", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_bc.cuda_requested() is False


def test_cuda_requested_auto_tracks_torch(monkeypatch) -> None:
    monkeypatch.setenv("DMDOD_BC_CUDA", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_bc.cuda_requested() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reverse_cuda_bc_matches_cpu_training_closely() -> None:
    torch.manual_seed(20260930)
    stable = _stable_sequence()

    cpu_model = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    gpu_model = RecurrentActorCritic(input_dim=5, hidden_dim=7)
    gpu_model.load_state_dict(cpu_model.state_dict())

    cpu_optimizer = torch.optim.Adam(v057._policy_parameters(cpu_model), lr=1e-4)
    gpu_source_optimizer = torch.optim.Adam(v057._policy_parameters(gpu_model), lr=1e-4)

    cpu_loss = chunk_gru._batched_progress_train_one_epoch(
        cpu_model,
        [stable],
        optimizer=cpu_optimizer,
        chunk_steps=6,
        reverse_order=True,
    )
    gpu_loss = cuda_bc.train_reverse_on_cuda(
        gpu_model,
        [stable],
        optimizer=gpu_source_optimizer,
        chunk_steps=6,
    )

    assert math.isclose(gpu_loss, cpu_loss, rel_tol=2e-3, abs_tol=2e-4)
    assert not gpu_source_optimizer.state
    assert all(parameter.device.type == "cpu" for parameter in gpu_model.parameters())

    cpu_state = cpu_model.state_dict()
    gpu_state = gpu_model.state_dict()
    assert cpu_state.keys() == gpu_state.keys()
    for key in cpu_state:
        torch.testing.assert_close(
            gpu_state[key],
            cpu_state[key],
            rtol=5e-4,
            atol=5e-5,
            msg=lambda message, key=key: f"{key}: {message}",
        )
