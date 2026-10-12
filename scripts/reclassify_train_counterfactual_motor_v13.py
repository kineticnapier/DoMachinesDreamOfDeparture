from __future__ import annotations

"""Offline v1.2 diagnostic reclassification; no model, CUDA, or Trainer load."""

import argparse
from pathlib import Path

from dmdod.training.train_counterfactual_motor_v13 import run_v13


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path,
                        default=Path('diagnostics/train_counterfactual_motor_v12'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('diagnostics/train_counterfactual_motor_v13'))
    args = parser.parse_args()
    result = run_v13(args.previous, args.output_dir)
    print('=== Train Counterfactual Motor v1.3 (offline) ===')
    print(f"151 input candidates: {result['original_v11_candidates']}")
    print(f"Safe manual-review candidates /151: {result['safe_improvement_original_151']}")
    print(f"Conditional-review candidates /151: {result['conditional_review_original_151']}")
    print(f"Independent Train anchors: {len(result['qualified_train_anchors_original_151'])}")
    print(f"Outcome patterns: {result['unique_qualified_outcome_patterns']}")
    print(f"Output: {args.output_dir / 'diagnostic.json'}")


if __name__ == '__main__':
    main()
