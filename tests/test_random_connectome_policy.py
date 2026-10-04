from __future__ import annotations

import torch

from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic
from dmdod.malecns_connectome import build_malecns_weighted_core, save_malecns_core
from dmdod.random_connectome_policy import NKeyRandomConnectomeActorCritic


def _core(tmp_path):
    artifact = build_malecns_weighted_core(
        torch.tensor([1, 2, 3, 4, 5, 1, 2, 4, 5, 2]),
        torch.tensor([2, 3, 4, 5, 1, 3, 4, 1, 2, 5]),
        torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        node_limit=5,
        min_weight=1,
    )
    path = tmp_path / "core.pt"
    save_malecns_core(path, artifact)
    return path, artifact


def _indices(model):
    return model.recurrent_weight.coalesce().indices()


def test_random_connectome_matches_fly_trainable_parameter_count(tmp_path) -> None:
    path, _ = _core(tmp_path)
    kwargs = dict(input_dim=12, key_count=4, core_path=path, sensory_dim=3, projection_seed=17)
    fly = NKeyFlyConnectomeActorCritic(**kwargs)
    random_model = NKeyRandomConnectomeActorCritic(**kwargs, topology_seed=23)

    fly_trainable = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    random_trainable = sum(p.numel() for p in random_model.parameters() if p.requires_grad)
    assert fly_trainable == random_trainable
    assert torch.equal(fly.input_projection, random_model.input_projection)


def test_random_connectome_preserves_target_fanin_and_row_sums(tmp_path) -> None:
    path, _ = _core(tmp_path)
    fly = NKeyFlyConnectomeActorCritic(
        input_dim=8, key_count=4, core_path=path, sensory_dim=3, recurrent_gain=0.8
    )
    random_model = NKeyRandomConnectomeActorCritic(
        input_dim=8,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.8,
        topology_seed=99,
    )

    fly_sparse = fly.recurrent_weight.coalesce()
    random_sparse = random_model.recurrent_weight.coalesce()
    assert fly_sparse._nnz() == random_sparse._nnz()

    for post in range(fly.hidden_dim):
        fly_mask = fly_sparse.indices()[0] == post
        random_mask = random_sparse.indices()[0] == post
        assert int(fly_mask.sum()) == int(random_mask.sum())
        assert torch.allclose(
            fly_sparse.values()[fly_mask].sum(),
            random_sparse.values()[random_mask].sum(),
            atol=1e-7,
        )

    indices = random_sparse.indices()
    assert not torch.any(indices[0] == indices[1])


def test_random_connectome_seed_is_reproducible_and_changes_topology(tmp_path) -> None:
    path, _ = _core(tmp_path)
    kwargs = dict(input_dim=8, key_count=4, core_path=path, sensory_dim=3)
    first = NKeyRandomConnectomeActorCritic(**kwargs, topology_seed=101)
    second = NKeyRandomConnectomeActorCritic(**kwargs, topology_seed=101)
    third = NKeyRandomConnectomeActorCritic(**kwargs, topology_seed=102)

    assert torch.equal(_indices(first), _indices(second))
    assert not torch.equal(_indices(first), _indices(third))


def test_random_connectome_forward_and_metadata(tmp_path) -> None:
    path, artifact = _core(tmp_path)
    model = NKeyRandomConnectomeActorCritic(
        input_dim=9,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.7,
        projection_seed=31,
        topology_seed=41,
    )
    state = model.initial_state(torch.device("cpu"))
    mean, std, value, next_state = model.forward_step(torch.zeros(9), state)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert value.ndim == 0
    assert next_state.shape == (5,)

    metadata = model.checkpoint_metadata()
    assert metadata["n_key_policy_backend"] == "random_connectome"
    assert metadata["n_key_policy_version"] == "n-key-degree-matched-random-connectome-v1"
    assert metadata["fly_connectome_edge_count"] == artifact["metadata"]["edge_count"]
    assert metadata["random_connectome_topology_seed"] == 41
    assert metadata["random_connectome_matching"] == "per-target-fanin-and-incoming-weight-multiset"
