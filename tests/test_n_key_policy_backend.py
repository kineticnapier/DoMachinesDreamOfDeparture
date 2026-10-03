from __future__ import annotations

import pytest
import torch

from dmdod.n_key_policy import (
    N_KEY_POLICY_BACKEND_GRU,
    N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR,
    NKeyFixedSparseReservoirActorCritic,
    NKeyPolicyBase,
    NKeyRecurrentActorCritic,
    build_n_key_policy,
    n_key_policy_backend_from_checkpoint,
)


def test_gru_factory_exposes_backend_neutral_contract() -> None:
    model = build_n_key_policy(
        backend=N_KEY_POLICY_BACKEND_GRU,
        input_dim=263,
        key_count=8,
        hidden_dim=16,
    )

    assert isinstance(model, NKeyPolicyBase)
    assert isinstance(model, NKeyRecurrentActorCritic)
    assert model.backend_name == "gru"
    assert model.input_dim == 263
    assert model.action_dim == 8

    model.prepare_recurrent_runtime()
    state = model.initial_state(torch.device("cpu"))
    mean, std, value, next_state = model.forward_step(
        torch.zeros(263, dtype=torch.float32),
        state,
    )

    assert mean.shape == (8,)
    assert std.shape == (8,)
    assert value.ndim == 0
    assert next_state.shape == (16,)


def test_gru_checkpoint_metadata_records_backend() -> None:
    model = build_n_key_policy(
        backend="gru",
        input_dim=263,
        key_count=8,
        hidden_dim=16,
    )

    metadata = model.checkpoint_metadata()

    assert metadata["n_key_policy_backend"] == "gru"
    assert metadata["n_key_policy_version"] == "n-key-gru-v1"
    assert metadata["key_count"] == 8
    assert metadata["hidden_dim"] == 16


def test_sparse_reservoir_factory_matches_policy_contract() -> None:
    model = build_n_key_policy(
        backend=N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR,
        input_dim=263,
        key_count=8,
        hidden_dim=32,
        reservoir_density=0.20,
        reservoir_gain=0.75,
        reservoir_seed=123,
    )

    assert isinstance(model, NKeyPolicyBase)
    assert isinstance(model, NKeyFixedSparseReservoirActorCritic)
    assert model.backend_name == "sparse_reservoir"
    assert model.input_dim == 263
    assert model.action_dim == 8

    model.prepare_recurrent_runtime()
    state = model.initial_state(torch.device("cpu"))
    mean, std, value, next_state = model.forward_step(
        torch.zeros(263, dtype=torch.float32),
        state,
    )

    assert mean.shape == (8,)
    assert std.shape == (8,)
    assert value.ndim == 0
    assert next_state.shape == (32,)


def test_sparse_reservoir_recurrent_core_is_fixed_and_sparse() -> None:
    model = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=12,
        key_count=4,
        hidden_dim=64,
        reservoir_density=0.10,
        reservoir_seed=99,
    )
    assert isinstance(model, NKeyFixedSparseReservoirActorCritic)

    recurrent = model.reservoir.weight_hh_l0
    identity_input = model.reservoir.weight_ih_l0
    nonzero_fraction = float(torch.count_nonzero(recurrent)) / float(recurrent.numel())

    assert recurrent.requires_grad is False
    assert identity_input.requires_grad is False
    assert torch.equal(identity_input, torch.eye(64))
    assert 0.05 < nonzero_fraction < 0.15

    before = recurrent.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    observations = torch.randn(5, 12)
    state = model.initial_state(torch.device("cpu"))
    means, values, _ = model.forward_sequence(observations, state)
    loss = means.square().mean() + values.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.equal(model.reservoir.weight_hh_l0, before)
    assert model.input_layer.weight.grad is not None
    assert model.actor_mean.weight.grad is not None


def test_sparse_reservoir_step_and_sequence_are_consistent() -> None:
    torch.manual_seed(7)
    model = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=10,
        key_count=4,
        hidden_dim=24,
        reservoir_density=0.25,
        reservoir_seed=321,
    )
    observations = torch.randn(6, 10)
    initial_state = model.initial_state(torch.device("cpu"))

    sequence_means, sequence_values, sequence_final = model.forward_sequence(
        observations,
        initial_state,
    )

    state = initial_state
    step_means = []
    step_values = []
    for observation in observations:
        mean, _, value, state = model.forward_step(observation, state)
        step_means.append(mean)
        step_values.append(value)

    assert torch.allclose(sequence_means, torch.stack(step_means), atol=1e-6, rtol=1e-5)
    assert torch.allclose(sequence_values, torch.stack(step_values), atol=1e-6, rtol=1e-5)
    assert torch.allclose(sequence_final, state, atol=1e-6, rtol=1e-5)


def test_sparse_reservoir_checkpoint_metadata_records_topology() -> None:
    model = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=263,
        key_count=8,
        hidden_dim=32,
        reservoir_density=0.125,
        reservoir_gain=0.8,
        reservoir_seed=456,
    )

    metadata = model.checkpoint_metadata()

    assert metadata["n_key_policy_backend"] == "sparse_reservoir"
    assert metadata["n_key_policy_version"] == "n-key-fixed-sparse-reservoir-v1"
    assert metadata["reservoir_density"] == pytest.approx(0.125)
    assert metadata["reservoir_gain"] == pytest.approx(0.8)
    assert metadata["reservoir_seed"] == 456


def test_sparse_reservoir_seed_reproduces_fixed_topology() -> None:
    first = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=8,
        key_count=4,
        hidden_dim=20,
        reservoir_seed=42,
    )
    second = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=8,
        key_count=4,
        hidden_dim=20,
        reservoir_seed=42,
    )
    third = build_n_key_policy(
        backend="sparse_reservoir",
        input_dim=8,
        key_count=4,
        hidden_dim=20,
        reservoir_seed=43,
    )

    assert isinstance(first, NKeyFixedSparseReservoirActorCritic)
    assert isinstance(second, NKeyFixedSparseReservoirActorCritic)
    assert isinstance(third, NKeyFixedSparseReservoirActorCritic)
    assert torch.equal(first.reservoir.weight_hh_l0, second.reservoir.weight_hh_l0)
    assert not torch.equal(first.reservoir.weight_hh_l0, third.reservoir.weight_hh_l0)


def test_legacy_checkpoint_defaults_to_gru_backend() -> None:
    assert n_key_policy_backend_from_checkpoint({"hidden_dim": 128}) == "gru"


def test_sparse_reservoir_checkpoint_backend_is_recognized() -> None:
    assert (
        n_key_policy_backend_from_checkpoint(
            {"n_key_policy_backend": "sparse_reservoir"}
        )
        == "sparse_reservoir"
    )


def test_unknown_policy_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown N-key policy backend"):
        build_n_key_policy(
            backend="fly",
            input_dim=263,
            key_count=8,
            hidden_dim=16,
        )

    with pytest.raises(ValueError, match="unsupported N-key policy backend"):
        n_key_policy_backend_from_checkpoint({"n_key_policy_backend": "fly"})
