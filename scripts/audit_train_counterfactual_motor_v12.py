from __future__ import annotations

"""Frozen Best, Train-only motor v1.2 diagnosis. NEVER updates checkpoint."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor_v12 import run_v12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/human_visible_onset_pilot.toml")
    parser.add_argument("--previous", type=Path,
                        default=Path("diagnostics/train_counterfactual_motor_v11"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("diagnostics/train_counterfactual_motor_v12"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output must not exist: {args.output_dir}")
    config = load_training_config(args.config)
    if config.mode != "human_visible_curriculum_dagger":
        raise SystemExit("requires human_visible_curriculum_dagger")
    # _prepare reads fixed Train/Validation manifests, but ONLY Train rollout
    # is performed. No Validation evaluator is invoked.
    prepared = _prepare(config)
    report = run_v12(
        prepared, previous=args.previous, output=args.output_dir,
        seed=int(config.human_visible.seed),
    )
    print("=== Train Counterfactual Motor v1.2 ===")
    print(f"151 reproduced: {report['v11_candidates_reproduced']}/151")
    print(f"Student-relative corrections: {report['student_relative_improvement_count']}")
    print(f"Distinct improvement patterns: {report['unique_improvement_patterns']}")
    print("No Student training, teacher adoption, Validation or FINAL execution.")
    print(f"Report: {args.output_dir/'diagnostic.json'}")


if __name__ == "__main__":
    main()
