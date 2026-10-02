from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import train_real_chart_v150_four_key_bootstrap as trainer
from dmdod.four_key_policy import FourKeyRecurrentActorCritic
from dmdod.four_key_real_chart import FOUR_KEY_HUD_REAL_CHART_INPUT_DIM
from dmdod.four_key_training import FourKeyDAggerSequence


def test_four_key_bootstrap_epoch_trains_251d_to_4d_policy() -> None:
    torch.manual_seed(7)
    model = FourKeyRecurrentActorCritic(
        input_dim=FOUR_KEY_HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=16,
    )
    observations = torch.zeros((12, FOUR_KEY_HUD_REAL_CHART_INPUT_DIM), dtype=torch.float32)
    observations[:, 0] = torch.linspace(0.0, 1.0, 12)
    actions = torch.zeros((12, 4), dtype=torch.float32)
    actions[2:6, 1] = 1.0
    actions[6:9, 1] = -1.0
    sequence = FourKeyDAggerSequence(observations, actions, "unit")

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


def test_four_key_bootstrap_aggregate_uses_xacc_components() -> None:
    first = SimpleNamespace(
        targets=10,
        hits=8,
        too_early_presses=2,
        overloaded=False,
        x_accuracy_points=7.5,
        x_accuracy_denominator=10.0,
    )
    second = SimpleNamespace(
        targets=20,
        hits=15,
        too_early_presses=3,
        overloaded=True,
        x_accuracy_points=10.0,
        x_accuracy_denominator=20.0,
    )

    text = trainer._aggregate([(first, 9), (second, 18)])

    assert "H=23/30" in text
    assert "X=58.33%" in text
    assert "early=5" in text
    assert "over=True" in text
    assert "keydowns=27" in text


def test_four_key_bootstrap_rejects_empty_sequences() -> None:
    model = FourKeyRecurrentActorCritic(
        input_dim=FOUR_KEY_HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="at least one four-key training sequence"):
        trainer._train_bc_epoch(model, [], optimizer=optimizer, chunk_steps=8)


def test_four_key_bootstrap_checkpoint_identity_is_distinct_from_two_key() -> None:
    assert trainer.TRAINER_VERSION == "1.5.0-four-key-bootstrap"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 16
    assert FOUR_KEY_HUD_REAL_CHART_INPUT_DIM == 251
