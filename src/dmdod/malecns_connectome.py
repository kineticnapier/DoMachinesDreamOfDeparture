from __future__ import annotations

"""MaleCNS v1.0 connectivity preprocessing for fixed recurrent cores.

The raw Janelia flat connectome is intentionally kept outside the repository.
This module turns the official segment-to-segment Feather edge table into a
small, deterministic torch artifact that can later be consumed by an N-key
connectome backend.
"""

from pathlib import Path

import torch


MALECNS_DATASET = "male-cns:v1.0"
MALECNS_WEIGHTS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
MALECNS_WEIGHTS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
    + MALECNS_WEIGHTS_FILENAME
)
MALECNS_CORE_FORMAT_VERSION = 1
DEFAULT_MALECNS_CORE_NODES = 4096
DEFAULT_MALECNS_MIN_WEIGHT = 5


def _as_int64_vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device="cpu")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tensor


def build_malecns_weighted_core(
    body_pre: torch.Tensor,
    body_post: torch.Tensor,
    weight: torch.Tensor,
    *,
    node_limit: int = DEFAULT_MALECNS_CORE_NODES,
    min_weight: int = DEFAULT_MALECNS_MIN_WEIGHT,
) -> dict[str, object]:
    """Build a deterministic weighted-strength top-k MaleCNS subgraph.

    Ranking uses total retained synapse strength (incoming + outgoing) after
    the minimum edge-weight filter. Ties are resolved by ascending body ID.
    The saved COO orientation is recurrent-matrix friendly: edge_index[0] is
    the postsynaptic/target node and edge_index[1] is the presynaptic/source
    node, so W[target, source] carries the connection strength.
    """

    if node_limit <= 0:
        raise ValueError("node_limit must be positive")
    if min_weight <= 0:
        raise ValueError("min_weight must be positive")

    pre = _as_int64_vector(body_pre, name="body_pre")
    post = _as_int64_vector(body_post, name="body_post")
    weights = _as_int64_vector(weight, name="weight")
    if pre.numel() != post.numel() or pre.numel() != weights.numel():
        raise ValueError("body_pre, body_post, and weight must have equal lengths")
    if torch.any(weights < 0):
        raise ValueError("MaleCNS connection weights must be non-negative")

    retained = (weights >= int(min_weight)) & (pre != post)
    pre = pre[retained]
    post = post[retained]
    weights = weights[retained]
    if pre.numel() == 0:
        raise ValueError("no edges remain after min_weight/self-edge filtering")

    all_bodies = torch.cat((pre, post))
    all_strengths = torch.cat((weights, weights))
    body_ids, inverse = torch.unique(all_bodies, sorted=True, return_inverse=True)
    strength = torch.zeros(body_ids.numel(), dtype=torch.int64)
    strength.scatter_add_(0, inverse, all_strengths)

    selected_count = min(int(node_limit), int(body_ids.numel()))
    ranking = torch.argsort(strength, descending=True, stable=True)[:selected_count]
    selected_body_ids = torch.sort(body_ids[ranking]).values

    pre_pos = torch.searchsorted(selected_body_ids, pre)
    post_pos = torch.searchsorted(selected_body_ids, post)
    pre_probe = pre_pos.clamp(max=selected_count - 1)
    post_probe = post_pos.clamp(max=selected_count - 1)
    internal = (
        (pre_pos < selected_count)
        & (post_pos < selected_count)
        & (selected_body_ids[pre_probe] == pre)
        & (selected_body_ids[post_probe] == post)
    )

    source = pre_pos[internal]
    target = post_pos[internal]
    internal_weights = weights[internal].to(dtype=torch.float32)
    edge_index = torch.stack((target, source), dim=0)
    sparse = torch.sparse_coo_tensor(
        edge_index,
        internal_weights,
        (selected_count, selected_count),
        dtype=torch.float32,
    ).coalesce()
    edge_index = sparse.indices().cpu()
    edge_weight = sparse.values().cpu()
    if edge_weight.numel() == 0:
        raise ValueError("selected MaleCNS core contains no internal edges")

    possible_edges = float(selected_count * selected_count)
    metadata = {
        "format_version": MALECNS_CORE_FORMAT_VERSION,
        "dataset": MALECNS_DATASET,
        "selection": "weighted-strength-topk",
        "node_limit": int(node_limit),
        "node_count": int(selected_count),
        "min_weight": int(min_weight),
        "filtered_edge_count": int(pre.numel()),
        "edge_count": int(edge_weight.numel()),
        "edge_density": float(edge_weight.numel() / possible_edges),
        "edge_orientation": "edge_index[0]=post(target), edge_index[1]=pre(source)",
        "edge_weight_semantics": "raw-synapse-count",
        "signed": False,
    }
    return {
        "metadata": metadata,
        "body_ids": selected_body_ids.cpu(),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }


def build_malecns_weighted_core_from_feather(
    path: str | Path,
    *,
    node_limit: int = DEFAULT_MALECNS_CORE_NODES,
    min_weight: int = DEFAULT_MALECNS_MIN_WEIGHT,
) -> dict[str, object]:
    """Read the official MaleCNS v1.0 weight table and build a core artifact."""

    try:
        import pyarrow.compute as pc
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover - optional preprocessing dependency
        raise RuntimeError(
            "pyarrow is required for MaleCNS Feather preprocessing; "
            "install it with e.g. `uv pip install pyarrow`"
        ) from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    table = feather.read_table(
        source,
        columns=["body_pre", "body_post", "weight"],
        memory_map=True,
    )
    required = {"body_pre", "body_post", "weight"}
    if not required.issubset(table.column_names):
        missing = ", ".join(sorted(required.difference(table.column_names)))
        raise ValueError(f"MaleCNS weight table is missing columns: {missing}")
    source_rows = int(table.num_rows)

    retained = pc.and_(
        pc.greater_equal(table["weight"], int(min_weight)),
        pc.not_equal(table["body_pre"], table["body_post"]),
    )
    table = table.filter(retained)
    filtered_rows = int(table.num_rows)
    if filtered_rows == 0:
        raise ValueError("no MaleCNS edges remain after filtering")

    def tensor_column(name: str) -> torch.Tensor:
        array = table[name].combine_chunks().to_numpy(zero_copy_only=False)
        return torch.tensor(array, dtype=torch.int64)

    artifact = build_malecns_weighted_core(
        tensor_column("body_pre"),
        tensor_column("body_post"),
        tensor_column("weight"),
        node_limit=node_limit,
        min_weight=min_weight,
    )
    metadata = dict(artifact["metadata"])
    metadata.update(
        {
            "source_filename": source.name,
            "source_rows": source_rows,
            "source_rows_after_filter": filtered_rows,
        }
    )
    artifact["metadata"] = metadata
    return artifact


def save_malecns_core(path: str | Path, artifact: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)


def load_malecns_core(path: str | Path) -> dict[str, object]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError("MaleCNS core artifact must be a dict")
    metadata = artifact.get("metadata")
    body_ids = artifact.get("body_ids")
    edge_index = artifact.get("edge_index")
    edge_weight = artifact.get("edge_weight")
    if not isinstance(metadata, dict):
        raise ValueError("MaleCNS core artifact is missing metadata")
    if int(metadata.get("format_version", -1)) != MALECNS_CORE_FORMAT_VERSION:
        raise ValueError("unsupported MaleCNS core artifact format")
    if metadata.get("dataset") != MALECNS_DATASET:
        raise ValueError("MaleCNS core artifact uses an unexpected dataset")
    if not isinstance(body_ids, torch.Tensor) or body_ids.ndim != 1:
        raise ValueError("MaleCNS core artifact has invalid body_ids")
    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("MaleCNS core artifact has invalid edge_index")
    if not isinstance(edge_weight, torch.Tensor) or edge_weight.ndim != 1:
        raise ValueError("MaleCNS core artifact has invalid edge_weight")
    if edge_index.shape[1] != edge_weight.shape[0]:
        raise ValueError("MaleCNS core artifact edge arrays disagree")
    if int(metadata.get("node_count", -1)) != body_ids.numel():
        raise ValueError("MaleCNS core artifact node_count disagrees with body_ids")
    return artifact
