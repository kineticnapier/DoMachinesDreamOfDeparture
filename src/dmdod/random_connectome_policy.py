from __future__ import annotations

from pathlib import Path

import torch

from .fly_connectome_policy import (
    DEFAULT_FLY_CONNECTOME_GAIN,
    DEFAULT_FLY_CONNECTOME_PROJECTION_SEED,
    DEFAULT_FLY_CONNECTOME_SENSORY_DIM,
    NKeyFlyConnectomeActorCritic,
)
from .malecns_connectome import load_malecns_core


N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME = "random_connectome"
N_KEY_RANDOM_CONNECTOME_VERSION = "n-key-degree-matched-random-connectome-v1"
DEFAULT_RANDOM_CONNECTOME_TOPOLOGY_SEED = 2718


class NKeyRandomConnectomeActorCritic(NKeyFlyConnectomeActorCritic):
    """Fly-architecture null model with a randomized recurrent topology.

    Everything outside the recurrent graph matches NKeyFlyConnectomeActorCritic:
    trainable sensory bottleneck, fixed seeded sensory projection, tanh state,
    and trainable actor/critic heads. The null model preserves each target
    neuron's incoming edge count and incoming raw-weight multiset from the
    reference MaleCNS artifact, but replaces the presynaptic neuron identities
    with deterministic random unique sources (excluding self edges).

    This makes node count, per-target fan-in, row-normalization, recurrent gain,
    raw incoming weights, and trainable parameter count match the biological
    backend while destroying the biological source->target pairing.
    """

    backend_name = N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME
    policy_version = N_KEY_RANDOM_CONNECTOME_VERSION

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        core_path: str | Path,
        sensory_dim: int = DEFAULT_FLY_CONNECTOME_SENSORY_DIM,
        recurrent_gain: float = DEFAULT_FLY_CONNECTOME_GAIN,
        projection_seed: int = DEFAULT_FLY_CONNECTOME_PROJECTION_SEED,
        topology_seed: int = DEFAULT_RANDOM_CONNECTOME_TOPOLOGY_SEED,
        initial_log_std: float = -1.20,
    ) -> None:
        self.topology_seed = int(topology_seed)
        super().__init__(
            input_dim=input_dim,
            key_count=key_count,
            core_path=core_path,
            sensory_dim=sensory_dim,
            recurrent_gain=recurrent_gain,
            projection_seed=projection_seed,
            initial_log_std=initial_log_std,
        )

        artifact = load_malecns_core(core_path)
        edge_index = torch.as_tensor(artifact["edge_index"], dtype=torch.int64, device="cpu")
        raw_weight = torch.as_tensor(artifact["edge_weight"], dtype=torch.float32, device="cpu")
        target = edge_index[0]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.topology_seed)
        random_source = torch.empty_like(edge_index[1])

        # MaleCNS core artifacts are emitted in target-major order.  Exploit that
        # layout to avoid rescanning every edge once per target neuron.  Keep the
        # old lookup as a compatibility fallback for externally produced cores.
        target_is_sorted = bool(target.numel() < 2 or torch.all(target[1:] >= target[:-1]).item())
        if target_is_sorted:
            fan_in = torch.bincount(target, minlength=self.hidden_dim)
            target_offsets = torch.empty(self.hidden_dim + 1, dtype=torch.int64)
            target_offsets[0] = 0
            torch.cumsum(fan_in, dim=0, out=target_offsets[1:])

        for post in range(self.hidden_dim):
            if target_is_sorted:
                start = int(target_offsets[post].item())
                end = int(target_offsets[post + 1].item())
                count = end - start
                positions = slice(start, end)
            else:
                positions = torch.nonzero(target == post, as_tuple=False).flatten()
                count = int(positions.numel())
            if count == 0:
                continue
            if count > self.hidden_dim - 1:
                raise ValueError(
                    f"target neuron {post} has fan-in {count}, which cannot be randomized "
                    f"without self edges among {self.hidden_dim} nodes"
                )
            candidates = torch.randperm(self.hidden_dim - 1, generator=generator)[:count]
            candidates = candidates + (candidates >= post).to(candidates.dtype)
            random_source[positions] = candidates

        incoming = torch.zeros(self.hidden_dim, dtype=torch.float32)
        incoming.scatter_add_(0, target, raw_weight)
        denominator = incoming[target].clamp_min(1.0)
        normalized_weight = raw_weight * (self.recurrent_gain / denominator)
        randomized_index = torch.stack((target, random_source), dim=0)
        randomized = torch.sparse_coo_tensor(
            randomized_index,
            normalized_weight,
            (self.hidden_dim, self.hidden_dim),
            dtype=torch.float32,
            check_invariants=True,
        ).coalesce()
        if randomized._nnz() != raw_weight.numel():
            raise RuntimeError(
                "degree-matched randomization unexpectedly changed edge count: "
                f"expected {raw_weight.numel()}, got {randomized._nnz()}"
            )
        self.recurrent_weight = randomized
        self._refresh_recurrent_runtime_weight()

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "random_connectome_reference_dataset": self.core_metadata.get("dataset"),
                "random_connectome_reference_core_path": self.core_path,
                "random_connectome_topology_seed": self.topology_seed,
                "random_connectome_matching": "per-target-fanin-and-incoming-weight-multiset",
            }
        )
        return metadata
