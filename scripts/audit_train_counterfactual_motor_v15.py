from __future__ import annotations

"""Train-only 3.2-second v1.5 counterfactual durability/boundary audit."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor_v15 import run_v15


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/human_visible_onset_pilot.toml')
    parser.add_argument('--v12',type=Path,default=Path('diagnostics/train_counterfactual_motor_v12'))
    parser.add_argument('--v13',type=Path,default=Path('diagnostics/train_counterfactual_motor_v13'))
    parser.add_argument('--v14',type=Path,default=Path('diagnostics/train_counterfactual_motor_v14'))
    parser.add_argument('--output-dir',type=Path,default=Path('diagnostics/train_counterfactual_motor_v15'))
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'v1.5 output path already exists: {args.output_dir}')
    config=load_training_config(args.config)
    if config.mode!='human_visible_curriculum_dagger':
        raise SystemExit('v1.5 requires the existing 263D human-visible curriculum mode')
    prepared=_prepare(config)
    result=run_v15(prepared,v12_dir=args.v12,v13_dir=args.v13,v14_dir=args.v14,
                   output=args.output_dir)
    print('=== Train Counterfactual Motor v1.5 ===',flush=True)
    print('Reproduction: PASS for 20 Train baselines and 9 v1.4 candidates',flush=True)
    print(f"#06 boundary: {result['boundary_06']['successful_grid_points']} / 18",flush=True)
    print(f"#18 long grid: {result['late_drift_18']['tested_grid_points']} trials",flush=True)
    print('No optimizer, training, Validation/FINAL, or teacher adoption.',flush=True)
    print(f"Output: {args.output_dir/'diagnostic.json'}",flush=True)


if __name__=='__main__':
    main()
