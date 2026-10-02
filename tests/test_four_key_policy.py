import torch

from dmdod.four_key_motor import FOUR_KEY_ACTION_DIM, FourKeyAction
from dmdod.four_key_policy import FourKeyRecurrentActorCritic


def test_four_key_policy_outputs_four_motor_channels() -> None:
    policy = FourKeyRecurrentActorCritic(input_dim=17, hidden_dim=32)
    state = policy.initial_state(torch.device("cpu"))
    x = torch.zeros(17, dtype=torch.float32)

    mean, std, value, next_state = policy.forward_step(x, state)

    assert mean.shape == (FOUR_KEY_ACTION_DIM,)
    assert std.shape == (FOUR_KEY_ACTION_DIM,)
    assert value.ndim == 0
    assert next_state.shape == (32,)
    assert policy.actor_mean.out_features == 4
    assert policy.log_std.shape == (4,)


def test_four_key_policy_deterministic_action_returns_physical_order() -> None:
    policy = FourKeyRecurrentActorCritic(input_dim=5, hidden_dim=16)
    state = policy.initial_state(torch.device("cpu"))
    x = torch.zeros(5, dtype=torch.float32)

    action, next_state = policy.deterministic_action(x, state)

    assert isinstance(action, FourKeyAction)
    assert len(action.as_tuple()) == 4
    assert all(-1.0 <= value <= 1.0 for value in action.as_tuple())
    assert next_state.shape == (16,)


def test_four_key_forward_sequence_keeps_four_action_shape() -> None:
    policy = FourKeyRecurrentActorCritic(input_dim=3, hidden_dim=8)
    observations = torch.zeros((7, 3), dtype=torch.float32)
    state = policy.initial_state(torch.device("cpu"))

    means, values, final_state = policy.forward_sequence(observations, state)

    assert means.shape == (7, 4)
    assert values.shape == (7,)
    assert final_state.shape == (8,)

    empty_means, empty_values, empty_state = policy.forward_sequence(
        observations[:0], state
    )
    assert empty_means.shape == (0, 4)
    assert empty_values.shape == (0,)
    assert torch.equal(empty_state, state)


def test_four_key_checkpoint_metadata_records_action_dimension() -> None:
    policy = FourKeyRecurrentActorCritic(input_dim=251, hidden_dim=128)
    metadata = policy.checkpoint_metadata()

    assert metadata["input_dim"] == 251
    assert metadata["hidden_dim"] == 128
    assert metadata["action_dim"] == 4
