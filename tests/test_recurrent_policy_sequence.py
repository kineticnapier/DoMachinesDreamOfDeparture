from __future__ import annotations

import torch

from dmdod.recurrent_policy import RecurrentActorCritic


def _forward_step_reference(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
):
    state = initial_state
    means: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for x in observations:
        mean, _std, value, state = model.forward_step(x, state)
        means.append(mean)
        values.append(value)
    return torch.stack(means), torch.stack(values), state


def test_forward_sequence_matches_step_loop_for_nonzero_initial_state() -> None:
    torch.manual_seed(20260930)
    model = RecurrentActorCritic(input_dim=17, hidden_dim=23)
    observations = torch.randn(37, 17)
    initial_state = torch.randn(23)

    expected_means, expected_values, expected_state = _forward_step_reference(
        model,
        observations,
        initial_state,
    )

    actual_means, actual_values, actual_state = model.forward_sequence(
        observations,
        initial_state,
    )

    torch.testing.assert_close(actual_means, expected_means, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_values, expected_values, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-6, atol=1e-6)
