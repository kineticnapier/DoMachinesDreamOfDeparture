from __future__ import annotations

import pytest
import torch

from dmdod.n_key_policy import (
    N_KEY_POLICY_BACKEND_GRU,
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


def test_legacy_checkpoint_defaults_to_gru_backend() -> None:
    assert n_key_policy_backend_from_checkpoint({"hidden_dim": 128}) == "gru"


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
