from __future__ import annotations

"""Train-only balanced counterfactual diagnostic. No learning or validation selection."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor_v11 import run_train_counterfactual_v11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/human_visible_onset_pilot.toml')
    parser.add_argument('--output-dir',type=Path,
                        default=Path('diagnostics/train_counterfactual_motor_v11'))
    parser.add_argument('--max-cases-per-anchor',type=int,default=128)
    parser.add_argument('--followup-seconds',type=float,default=0.800)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'output path must not exist: {args.output_dir}')
    config=load_training_config(args.config)
    prepared=_prepare(config)
    report=run_train_counterfactual_v11(
        prepared,config,output_dir=args.output_dir,
        max_cases_per_anchor=args.max_cases_per_anchor,
        followup_s=args.followup_seconds,
    )
    print('=== DMDOD Train Counterfactual Motor v1.1 diagnostic ===',flush=True)
    print(f"Train case evaluations: {report['train_candidate_evaluations']}",flush=True)
    print(f"800ms qualification: {report['qualified_followup_cases']} cases on "
          f"{report['train_anchors_with_followup_qualification']}/20 Train anchors",flush=True)
    print('No learning performed. No Validation/FINAL evaluated.',flush=True)
    print(f"Output: {args.output_dir/'diagnostic.json'}",flush=True)


if __name__=='__main__':
    main()
