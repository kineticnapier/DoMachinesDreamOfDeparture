from __future__ import annotations

"""Train-only v1.4 continuation/robustness audit. No training or Validation."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor_v14 import run_v14


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/human_visible_onset_pilot.toml')
    parser.add_argument('--v12',type=Path,default=Path('diagnostics/train_counterfactual_motor_v12'))
    parser.add_argument('--v13',type=Path,default=Path('diagnostics/train_counterfactual_motor_v13'))
    parser.add_argument('--output-dir',type=Path,default=Path('diagnostics/train_counterfactual_motor_v14'))
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'v1.4 output already exists: {args.output_dir}')
    config=load_training_config(args.config)
    if config.mode!='human_visible_curriculum_dagger':
        raise SystemExit('v1.4 requires existing human-visible curriculum config')
    prepared=_prepare(config)
    report=run_v14(prepared,v12_dir=args.v12,v13_dir=args.v13,output=args.output_dir)
    print('=== Train Counterfactual Motor v1.4 ===',flush=True)
    print(f"20 Train baseline replay PASS; {len(report['stage_a'])} saved cases reproduced",flush=True)
    print('No optimizer, DAgger, Validation, FINAL, or teacher adoption used.',flush=True)
    print(f"Report: {args.output_dir/'diagnostic.json'}",flush=True)


if __name__=='__main__':
    main()
