from __future__ import annotations

"""Build a deterministic fixed-core artifact from MaleCNS v1.0 connectivity."""

import argparse
from pathlib import Path

from dmdod.connectome.malecns import (
    DEFAULT_MALECNS_CORE_NODES,
    DEFAULT_MALECNS_MIN_WEIGHT,
    MALECNS_DATASET,
    build_malecns_weighted_core_from_feather,
    save_malecns_core,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the official MaleCNS v1.0 connection-weight Feather table "
            "into a deterministic weighted-strength top-k torch core artifact."
        )
    )
    parser.add_argument("weights", help="MaleCNS connectome-weights .feather file")
    parser.add_argument("output", help="output .pt core artifact")
    parser.add_argument("--nodes", type=int, default=DEFAULT_MALECNS_CORE_NODES)
    parser.add_argument("--min-weight", type=int, default=DEFAULT_MALECNS_MIN_WEIGHT)
    args = parser.parse_args()

    if args.nodes <= 0:
        raise SystemExit("--nodes must be positive")
    if args.min_weight <= 0:
        raise SystemExit("--min-weight must be positive")

    source = Path(args.weights)
    output = Path(args.output)
    print("=== DMDOD MaleCNS v1.0 Core Builder ===")
    print(
        f"dataset={MALECNS_DATASET} source={source} output={output} "
        f"nodes={args.nodes} min-weight={args.min_weight}"
    )
    print(
        "selection=weighted-strength-topk (incoming+outgoing retained synapse count); "
        "weights remain unsigned/raw in this first topology artifact"
    )

    artifact = build_malecns_weighted_core_from_feather(
        source,
        node_limit=args.nodes,
        min_weight=args.min_weight,
    )
    save_malecns_core(output, artifact)
    metadata = artifact["metadata"]
    print(
        f"core nodes={metadata['node_count']} edges={metadata['edge_count']} "
        f"density={metadata['edge_density']:.6f} "
        f"filtered-source-edges={metadata['filtered_edge_count']}"
    )
    print(f"artifact: {output}")


if __name__ == "__main__":
    main()
