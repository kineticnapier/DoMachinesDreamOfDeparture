from __future__ import annotations

"""v1.7.0: bootstrap a degree-matched random-connectome null model."""

from pathlib import Path
import sys

import train_real_chart_v160_n_key_bootstrap as v160
from dmdod.fly_connectome_policy import (
    DEFAULT_FLY_CONNECTOME_GAIN,
    DEFAULT_FLY_CONNECTOME_PROJECTION_SEED,
    DEFAULT_FLY_CONNECTOME_SENSORY_DIM,
)
from dmdod.malecns_connectome import load_malecns_core
from dmdod.modern_cli_bootstrap import run_with_modern_console
from dmdod.random_connectome_policy import (
    DEFAULT_RANDOM_CONNECTOME_TOPOLOGY_SEED,
    N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    NKeyRandomConnectomeActorCritic,
)


TRAINER_VERSION = "1.7.0-n-key-random-connectome-bootstrap"
CHECKPOINT_FORMAT_VERSION = 27
DEFAULT_REFERENCE_CORE = Path("data/malecns/male-cns-v1.0-core4096-w5.pt")


def _pop_option(argv: list[str], name: str, default: str | None) -> str | None:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise SystemExit(f"{name} requires a value")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def main() -> None:
    argv = list(sys.argv)
    core_path = Path(
        _pop_option(argv, "--reference-core", str(DEFAULT_REFERENCE_CORE))
        or DEFAULT_REFERENCE_CORE
    )
    sensory_dim = int(
        _pop_option(argv, "--sensory-dim", str(DEFAULT_FLY_CONNECTOME_SENSORY_DIM))
        or DEFAULT_FLY_CONNECTOME_SENSORY_DIM
    )
    recurrent_gain = float(
        _pop_option(argv, "--gain", str(DEFAULT_FLY_CONNECTOME_GAIN))
        or DEFAULT_FLY_CONNECTOME_GAIN
    )
    projection_seed = int(
        _pop_option(argv, "--projection-seed", str(DEFAULT_FLY_CONNECTOME_PROJECTION_SEED))
        or DEFAULT_FLY_CONNECTOME_PROJECTION_SEED
    )
    topology_seed = int(
        _pop_option(argv, "--topology-seed", str(DEFAULT_RANDOM_CONNECTOME_TOPOLOGY_SEED))
        or DEFAULT_RANDOM_CONNECTOME_TOPOLOGY_SEED
    )
    if sensory_dim <= 0:
        raise SystemExit("--sensory-dim must be positive")
    if recurrent_gain <= 0.0:
        raise SystemExit("--gain must be positive")

    artifact = load_malecns_core(core_path)
    node_count = int(artifact["metadata"]["node_count"])
    edge_count = int(artifact["metadata"]["edge_count"])

    if "--backend" in argv:
        index = argv.index("--backend")
        if index + 1 >= len(argv) or argv[index + 1] != N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME:
            raise SystemExit("v1.7.0 wrapper only supports --backend random_connectome")
    else:
        argv.extend(["--backend", N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME])
    if "--hidden" not in argv:
        argv.extend(["--hidden", str(node_count)])

    original_builder = v160._build_bootstrap_policy

    def build_policy(
        *,
        backend: str,
        input_dim: int,
        key_count: int,
        hidden_dim: int,
        reservoir_density: float,
        reservoir_gain: float,
        reservoir_seed: int,
        device,
    ):
        if backend == N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME:
            if hidden_dim != node_count:
                raise SystemExit(
                    f"--hidden={hidden_dim} must equal reference core node_count={node_count}"
                )
            model = NKeyRandomConnectomeActorCritic(
                input_dim=input_dim,
                key_count=key_count,
                core_path=core_path,
                sensory_dim=sensory_dim,
                recurrent_gain=recurrent_gain,
                projection_seed=projection_seed,
                topology_seed=topology_seed,
                initial_log_std=-1.20,
            ).to(device)
            model.prepare_recurrent_runtime()
            return model
        return original_builder(
            backend=backend,
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=hidden_dim,
            reservoir_density=reservoir_density,
            reservoir_gain=reservoir_gain,
            reservoir_seed=reservoir_seed,
            device=device,
        )

    v160.N_KEY_POLICY_BACKENDS = tuple(v160.N_KEY_POLICY_BACKENDS) + (
        N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    )
    v160._build_bootstrap_policy = build_policy
    v160.TRAINER_VERSION = TRAINER_VERSION
    v160.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    sys.argv = argv

    print("=== DMDOD v1.7.0 N-Key Degree-Matched Random Connectome Bootstrap ===")
    print(
        f"reference-core={core_path} nodes={node_count} edges={edge_count} "
        f"sensory={sensory_dim} gain={recurrent_gain:g} "
        f"projection-seed={projection_seed} topology-seed={topology_seed}"
    )
    print(
        "null topology=randomized presynaptic identities; preserves per-target fan-in "
        "and incoming raw-weight multiset"
    )
    v160.main()


if __name__ == "__main__":
    run_with_modern_console(main)
