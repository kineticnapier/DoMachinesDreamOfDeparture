from __future__ import annotations

import argparse
from pathlib import Path

from dmdod.training.beta_training_ablation import run_beta_training_ablation
from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled 2-update beta DAgger training ablation; no promotion."
    )
    parser.add_argument(
        "--config",
        default="configs/training/human_visible_beta_training_ablation.toml",
    )
    parser.add_argument(
        "--report",
        default="reports/human_visible_beta_training_ablation.json",
        help="New JSON report path; refuses to overwrite.",
    )
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help="Also save ALL experimental arm models as separate non-best .pt files.",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    prepared = _prepare(config)
    run_beta_training_ablation(
        prepared, config, report_path=Path(args.report),
        save_candidates=args.save_candidates,
    )


if __name__ == "__main__":
    main()
