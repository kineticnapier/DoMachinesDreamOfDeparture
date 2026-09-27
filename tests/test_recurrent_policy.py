from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dmdod.recurrent_policy import RecurrentActorCritic


def test_recurrent_policy_state_changes_across_frames() -> None:
    model = RecurrentActorCritic(input_dim=10, hidden_dim=16)
    model.eval()
    device = torch.device("cpu")
    state0 = model.initial_state(device)
    frame = torch.zeros(10, dtype=torch.float32)

    _, state1 = model.deterministic_action(frame, state0)
    _, state2 = model.deterministic_action(frame, state1)

    assert state0.shape == (16,)
    assert not torch.equal(state0, state1)
    assert not torch.equal(state1, state2)


def test_recurrent_policy_reset_is_zero_and_deterministic() -> None:
    model = RecurrentActorCritic(input_dim=10, hidden_dim=16)
    model.eval()
    device = torch.device("cpu")
    frame = torch.linspace(-1.0, 1.0, 10)

    state_a = model.initial_state(device)
    action_a, next_a = model.deterministic_action(frame, state_a)
    state_b = model.initial_state(device)
    action_b, next_b = model.deterministic_action(frame, state_b)

    assert torch.equal(state_a, torch.zeros_like(state_a))
    assert action_a == action_b
    assert torch.allclose(next_a, next_b)


def test_recurrent_ppo_replay_matches_collected_log_probs() -> None:
    torch.manual_seed(123)
    model = RecurrentActorCritic(input_dim=10, hidden_dim=16)
    model.eval()
    device = torch.device("cpu")
    observations = torch.stack(
        [
            torch.linspace(-1.0, 1.0, 10),
            torch.linspace(1.0, -1.0, 10),
            torch.zeros(10),
        ]
    )
    state = model.initial_state(device)
    latents = []
    collected_log_probs = []

    for frame in observations:
        _, latent, log_prob, _, _, state = model.sample_action_latent(frame, state)
        latents.append(latent.detach())
        collected_log_probs.append(log_prob.detach())

    replay_log_probs, values, entropies = model.evaluate_latent_sequence(
        observations,
        torch.stack(latents),
    )

    assert torch.allclose(replay_log_probs, torch.stack(collected_log_probs), atol=1e-6)
    assert values.shape == (3,)
    assert entropies.shape == (3,)
