from __future__ import annotations

"""Opt-in read-only beta collection comparison, no checkpoint output."""

import argparse

from dmdod.training.beta_collection_probe import run_beta_collection_probe
from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare beta=1.0/0.5/0.0 DAgger collection on fixed Train anchors."
    )
    parser.add_argument(
        "--config",
        default="configs/training/human_visible_beta_collection_probe.toml",
        help="Curriculum-format TOML; its run.output is never written.",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    if config.mode != "human_visible_curriculum_dagger":
        raise SystemExit("beta collection probe requires human_visible_curriculum_dagger")
    prepared = _prepare(config)
    run_beta_collection_probe(
        prepared, prepared.model, seed=config.human_visible.seed,
    )


if __name__ == "__main__":
    main()
