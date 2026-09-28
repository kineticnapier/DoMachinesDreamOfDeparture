from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v038 as trainer  # noqa: E402
from dmdod.recurrent_policy import RecurrentActorCritic  # noqa: E402


def test_v038_migration_preserves_old_policy_columns_and_zeros_new_motion_columns():
    source = RecurrentActorCritic(input_dim=10, hidden_dim=16)
    target = trainer.PredictiveRecurrentActorCritic(input_dim=14, hidden_dim=16)

    copied = trainer._copy_recurrent_weights(target, source.state_dict())

    assert "input_layer.weight[0:10]" in copied
    assert torch.equal(
        target.input_layer.weight[:, :10],
        source.input_layer.weight,
    )
    assert torch.count_nonzero(target.input_layer.weight[:, 10:]).item() == 0
    assert torch.equal(target.gru.weight_ih, source.gru.weight_ih)
    assert torch.equal(target.actor_mean.weight, source.actor_mean.weight)


def test_prediction_targets_are_observation_deltas_not_privileged_values():
    assert trainer.MOTION_FEATURE_START == 10
    assert trainer.MOTION_FEATURE_END == 14
    assert trainer.MOTION_GEOMETRY_INPUT_DIM == 14
