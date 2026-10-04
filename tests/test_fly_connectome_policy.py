from __future__ import annotations

import torch

from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic
from dmdod.malecns_connectome import build_malecns_weighted_core, save_malecns_core
from dmdod.n_key_policy import NKeyPolicyBase


def _core(tmp_path):
    artifact = build_malecns_weighted_core(
        torch.tensor([1, 2, 3, 4, 1, 3]),
        torch.tensor([2, 3, 4, 1, 3, 1]),
        torch.tensor([5, 6, 7, 8, 9, 10]),
        node_limit=4,
        min_weight=1,
    )
    path = tmp_path / "fly-core.pt"
    save_malecns_core(path, artifact)
    return path, artifact


def _assert_sparse_equal(first: torch.Tensor, second: torch.Tensor) -> None:
    first = first.coalesce()
    second = second.coalesce()
    assert first.shape == second.shape
    assert torch.equal(first.indices(), second.indices())
    assert torch.equal(first.values(), second.values())


def test_fly_connectome_policy_matches_backend_contract(tmp_path) -> None:
    path, _ = _core(tmp_path)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=12,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.8,
        projection_seed=42,
    )

    assert isinstance(model, NKeyPolicyBase)
    assert model.backend_name == "fly_connectome"
    assert model.hidden_dim == 4
    assert model.input_dim == 12
    assert model.action_dim == 4

    state = model.initial_state(torch.device("cpu"))
    mean, std, value, next_state = model.forward_step(torch.zeros(12), state)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert value.ndim == 0
    assert next_state.shape == (4,)


def test_fly_connectome_core_and_projection_are_fixed(tmp_path) -> None:
    path, _ = _core(tmp_path)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=8,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        projection_seed=7,
    )
    recurrent_before = model.recurrent_weight.detach().clone()
    projection_before = model.input_projection.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    observations = torch.randn(5, 8)
    means, values, _ = model.forward_sequence(
        observations,
        model.initial_state(torch.device("cpu")),
    )
    loss = means.square().mean() + values.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _assert_sparse_equal(model.recurrent_weight, recurrent_before)
    assert torch.equal(model.input_projection, projection_before)
    assert model.sensory.weight.grad is not None
    assert model.actor_mean.weight.grad is not None


def test_fixed_sparse_recurrent_backward_matches_dense_reference(tmp_path) -> None:
    path, _ = _core(tmp_path)
    torch.manual_seed(17)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=6,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.8,
        projection_seed=11,
    )

    injected = torch.randn(model.hidden_dim, requires_grad=True)
    state = torch.randn(model.hidden_dim, requires_grad=True)
    probe = torch.randn(model.hidden_dim)
    actual = model._advance_injected(injected, state)
    (actual * probe).sum().backward()
    actual_injected_grad = injected.grad.detach().clone()
    actual_state_grad = state.grad.detach().clone()

    injected_ref = injected.detach().clone().requires_grad_(True)
    state_ref = state.detach().clone().requires_grad_(True)
    dense_weight = model.recurrent_weight.to_dense()
    expected = torch.tanh(injected_ref + torch.mv(dense_weight, state_ref))
    (expected * probe).sum().backward()

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual_injected_grad, injected_ref.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual_state_grad, state_ref.grad, atol=1e-7, rtol=1e-6)
    state_dict = model.state_dict()
    assert "_recurrent_weight_runtime" not in state_dict
    assert "_recurrent_weight_transpose_runtime" not in state_dict


def test_fly_connectome_step_and_sequence_are_consistent(tmp_path) -> None:
    path, _ = _core(tmp_path)
    torch.manual_seed(5)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=6,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        projection_seed=11,
    )
    observations = torch.randn(7, 6)
    initial_state = model.initial_state(torch.device("cpu"))

    sequence_means, sequence_values, final_state = model.forward_sequence(
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
    assert torch.allclose(final_state, state, atol=1e-6, rtol=1e-5)


def test_fly_connectome_metadata_records_malecns_topology(tmp_path) -> None:
    path, artifact = _core(tmp_path)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=10,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.75,
        projection_seed=19,
    )

    metadata = model.checkpoint_metadata()
    assert metadata["n_key_policy_backend"] == "fly_connectome"
    assert metadata["n_key_policy_version"] == "n-key-malecns-fixed-connectome-v1"
    assert metadata["fly_connectome_dataset"] == "male-cns:v1.0"
    assert metadata["fly_connectome_node_count"] == 4
    assert metadata["fly_connectome_edge_count"] == artifact["metadata"]["edge_count"]
    assert metadata["fly_connectome_signed"] is False
    assert metadata["fly_connectome_sensory_dim"] == 3
    assert metadata["fly_connectome_recurrent_gain"] == 0.75
    assert metadata["fly_connectome_projection_seed"] == 19


def test_fly_connectome_projection_seed_is_reproducible(tmp_path) -> None:
    path, _ = _core(tmp_path)
    first = NKeyFlyConnectomeActorCritic(
        input_dim=5, key_count=4, core_path=path, sensory_dim=3, projection_seed=123
    )
    second = NKeyFlyConnectomeActorCritic(
        input_dim=5, key_count=4, core_path=path, sensory_dim=3, projection_seed=123
    )
    third = NKeyFlyConnectomeActorCritic(
        input_dim=5, key_count=4, core_path=path, sensory_dim=3, projection_seed=124
    )

    assert torch.equal(first.input_projection, second.input_projection)
    assert not torch.equal(first.input_projection, third.input_projection)
    _assert_sparse_equal(first.recurrent_weight, second.recurrent_weight)
