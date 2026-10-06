from __future__ import annotations

import argparse

from dmdod.training.config import load_training_config
from dmdod.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the configured DMDOD real-chart training pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="TOML training configuration",
    )
    args = parser.parse_args()
    run_training(load_training_config(args.config))


if __name__ == "__main__":
    main()
