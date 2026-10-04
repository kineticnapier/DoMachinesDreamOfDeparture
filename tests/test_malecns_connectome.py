from __future__ import annotations

import torch

from dmdod.malecns_connectome import (
    MALECNS_DATASET,
    build_malecns_weighted_core,
    load_malecns_core,
    save_malecns_core,
)


def test_malecns_weighted_core_selects_strongest_bodies_deterministically() -> None:
    artifact = build_malecns_weighted_core(
        torch.tensor([10, 10, 20, 20, 30, 40, 50, 60, 99]),
        torch.tensor([20, 30, 30, 40, 40, 20, 60, 50, 99]),
        torch.tensor([5, 5, 10, 10, 10, 10, 5, 5, 100]),
        node_limit=3,
        min_weight=5,
    )

    # Body 99's self-edge is discarded.  The weighted-strength top three are
    # 20, 40, 30; state order is then stabilized by ascending body ID.
    assert torch.equal(artifact["body_ids"], torch.tensor([20, 30, 40]))
    assert artifact["metadata"]["dataset"] == MALECNS_DATASET
    assert artifact["metadata"]["selection"] == "weighted-strength-topk"
    assert artifact["metadata"]["signed"] is False

    edge_index = artifact["edge_index"]
    edge_weight = artifact["edge_weight"]
    by_edge = {
        tuple(edge_index[:, index].tolist()): float(edge_weight[index])
        for index in range(edge_weight.numel())
    }
    # COO is [post(target), pre(source)] with body IDs [20,30,40].
    assert by_edge == {
        (1, 0): 10.0,  # 20 -> 30
        (2, 0): 10.0,  # 20 -> 40
        (2, 1): 10.0,  # 30 -> 40
        (0, 2): 10.0,  # 40 -> 20
    }


def test_malecns_weighted_core_coalesces_duplicate_connections() -> None:
    artifact = build_malecns_weighted_core(
        torch.tensor([1, 1, 2]),
        torch.tensor([2, 2, 1]),
        torch.tensor([5, 7, 6]),
        node_limit=2,
        min_weight=1,
    )

    edge_index = artifact["edge_index"]
    edge_weight = artifact["edge_weight"]
    by_edge = {
        tuple(edge_index[:, index].tolist()): float(edge_weight[index])
        for index in range(edge_weight.numel())
    }
    assert by_edge[(1, 0)] == 12.0
    assert by_edge[(0, 1)] == 6.0
    assert artifact["metadata"]["edge_count"] == 2


def test_malecns_core_round_trip(tmp_path) -> None:
    artifact = build_malecns_weighted_core(
        torch.tensor([1, 2, 3]),
        torch.tensor([2, 3, 1]),
        torch.tensor([5, 6, 7]),
        node_limit=3,
        min_weight=1,
    )
    path = tmp_path / "core.pt"

    save_malecns_core(path, artifact)
    loaded = load_malecns_core(path)

    assert loaded["metadata"] == artifact["metadata"]
    assert torch.equal(loaded["body_ids"], artifact["body_ids"])
    assert torch.equal(loaded["edge_index"], artifact["edge_index"])
    assert torch.equal(loaded["edge_weight"], artifact["edge_weight"])
