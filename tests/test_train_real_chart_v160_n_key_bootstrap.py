from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_real_chart_v160_n_key_bootstrap as trainer
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import NKeyBCSequence


def test_n_key_bootstrap_epoch_trains_263d_to_8d_policy() -> None:
    torch.manual_seed(11)
    input_dim = n_key_hud_real_chart_input_dim(8)
    model = NKeyRecurrentActorCritic(
        input_dim=input_dim,
        key_count=8,
        hidden_dim=16,
    )
    observations = torch.zeros((12, input_dim), dtype=torch.float32)
    observations[:, 0] = torch.linspace(0.0, 1.0, 12)
    actions = torch.zeros((12, 8), dtype=torch.float32)
    actions[2:6, 3] = 1.0
    actions[6:9, 3] = -1.0
    sequence = NKeyBCSequence(observations, actions, 8, "unit")

    before = model.actor_mean.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = trainer._train_bc_epoch(
        model,
        [sequence],
        optimizer=optimizer,
        chunk_steps=5,
    )

    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.actor_mean.weight.detach())


def test_n_key_bootstrap_rejects_sequence_for_different_body() -> None:
    model = NKeyRecurrentActorCritic(
        input_dim=n_key_hud_real_chart_input_dim(8),
        key_count=8,
        hidden_dim=8,
    )
    sequence = NKeyBCSequence(
        observations=torch.zeros((2, n_key_hud_real_chart_input_dim(6))),
        teacher_actions=torch.zeros((2, 6)),
        key_count=6,
        source="6k",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="sequence key_count"):
        trainer._train_bc_epoch(model, [sequence], optimizer=optimizer, chunk_steps=8)


def test_n_key_bootstrap_identity_and_8k_dimensions() -> None:
    assert trainer.TRAINER_VERSION == "1.6.0-n-key-bootstrap"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 17
    assert n_key_hud_real_chart_input_dim(8) == 263
