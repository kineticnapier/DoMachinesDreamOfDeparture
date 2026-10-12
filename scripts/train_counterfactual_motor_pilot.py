from __future__ import annotations

"""Train-only causal counterfactual motor teacher and controlled A/B pilot.

No FINAL, checkpoint promotion, source checkpoint mutation or Validation labels.
"""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor import run_counterfactual_motor_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/human_visible_onset_pilot.toml"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("diagnostics/train_counterfactual_motor_v1"),
    )
    parser.add_argument("--max-cases-per-anchor", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty")
    config = load_training_config(args.config)
    prepared = _prepare(config)
    report = run_counterfactual_motor_pilot(
        prepared, config, output_dir=args.output_dir,
        max_cases_per_anchor=args.max_cases_per_anchor,
    )
    print("=== Train Counterfactual Motor Pilot ===", flush=True)
    print(f"Train corrections: {report['train_anchors_with_correction']}/20", flush=True)
    print(f"Train candidate evaluations: {report['train_candidate_evaluations']}", flush=True)
    print(f"Comparison: {args.output_dir/'pilot.json'}", flush=True)


if __name__ == "__main__":
    main()
