from __future__ import annotations

import torch

from dmdod.predictive_recurrent_policy import PredictiveRecurrentActorCritic
from dmdod.toy_policy import MOTION_GEOMETRY_INPUT_DIM


def test_predictive_sequence_shapes() -> None:
    model = PredictiveRecurrentActorCritic(
        input_dim=MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=16,
    )
    observations = torch.randn(7, MOTION_GEOMETRY_INPUT_DIM)
    latents = torch.randn(7, 2)

    log_probs, values, entropies, predictions = (
        model.evaluate_latent_sequence_predictive(observations, latents)
    )

    assert log_probs.shape == (7,)
    assert values.shape == (7,)
    assert entropies.shape == (7,)
    assert predictions.shape == (7, 4)


def test_predictor_receives_gradient_from_next_delta_loss() -> None:
    model = PredictiveRecurrentActorCritic(
        input_dim=MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=16,
    )
    observations = torch.randn(6, MOTION_GEOMETRY_INPUT_DIM)
    latents = torch.randn(6, 2)
    _, _, _, predictions = model.evaluate_latent_sequence_predictive(observations, latents)

    target = observations[1:, 10:14]
    loss = ((predictions[:-1] - target) ** 2).mean()
    loss.backward()

    assert model.motion_predictor.weight.grad is not None
    assert model.gru.weight_hh.grad is not None
