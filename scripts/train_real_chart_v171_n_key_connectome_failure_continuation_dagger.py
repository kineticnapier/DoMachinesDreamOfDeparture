from __future__ import annotations

"""v1.7.1: failure-continuation trust DAgger for connectome checkpoints.

This is a thin compatibility layer over v1.6.6.  It preserves the existing
failure-continuation collection, trust-region selection, evaluation semantics,
and CLI while teaching the checkpoint loader how to reconstruct FlyConnectome
and RandomConnectome policies from checkpoint metadata.
"""

import torch

import train_real_chart_v166_n_key_failure_continuation_dagger as v166
from dmdod.fly_connectome_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
    NKeyFlyConnectomeActorCritic,
)
from dmdod.random_connectome_policy import (
    N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    NKeyRandomConnectomeActorCritic,
)


TRAINER_VERSION = "1.7.1-n-key-connectome-failure-continuation-dagger"
CHECKPOINT_FORMAT_VERSION = 27

_ORIGINAL_BUILD_POLICY_FROM_CHECKPOINT = v166.base._build_policy_from_checkpoint


def _required(checkpoint: dict, name: str):
    if name not in checkpoint:
        raise SystemExit(f"connectome checkpoint is missing {name}")
    return checkpoint[name]


def _build_policy_from_checkpoint(
    checkpoint: dict,
    *,
    device: torch.device,
):
    backend = str(checkpoint.get("n_key_policy_backend", "gru"))
    if backend not in {
        N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
        N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    }:
        return _ORIGINAL_BUILD_POLICY_FROM_CHECKPOINT(checkpoint, device=device)

    common = dict(
        input_dim=int(checkpoint["input_dim"]),
        key_count=int(checkpoint["key_count"]),
        core_path=str(_required(checkpoint, "fly_connectome_core_path")),
        sensory_dim=int(_required(checkpoint, "fly_connectome_sensory_dim")),
        recurrent_gain=float(_required(checkpoint, "fly_connectome_recurrent_gain")),
        projection_seed=int(_required(checkpoint, "fly_connectome_projection_seed")),
    )
    if backend == N_KEY_POLICY_BACKEND_FLY_CONNECTOME:
        model = NKeyFlyConnectomeActorCritic(**common)
    else:
        model = NKeyRandomConnectomeActorCritic(
            **common,
            topology_seed=int(_required(checkpoint, "random_connectome_topology_seed")),
        )

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.prepare_recurrent_runtime()
    return model


def main() -> None:
    v166.TRAINER_VERSION = TRAINER_VERSION
    v166.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v166.base._build_policy_from_checkpoint = _build_policy_from_checkpoint
    print("=== DMDOD v1.7.1 N-Key Connectome Failure-Continuation Trust DAgger ===")
    print(
        "FlyConnectome/RandomConnectome checkpoints are reconstructed from checkpoint "
        "metadata; v1.6.6 collection, trust selection, and evaluation semantics are unchanged."
    )
    v166.main()


if __name__ == "__main__":
    main()
