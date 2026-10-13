from __future__ import annotations

"""Train-only v1.6 state-conditioned motor diagnostic; no training or model selection."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.train_counterfactual_motor_v16 import run_v16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/human_visible_onset_pilot.toml')
    for v in (12,13,14,15):
        parser.add_argument(f'--v{v}',type=Path,
                            default=Path(f'diagnostics/train_counterfactual_motor_v{v}'))
    parser.add_argument('--output-dir',type=Path,
                        default=Path('diagnostics/train_counterfactual_motor_v16'))
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'v1.6 output path already exists: {args.output_dir}')
    config=load_training_config(args.config)
    if config.mode!='human_visible_curriculum_dagger':
        raise SystemExit('v1.6 requires existing human-visible 263D curriculum')
    prepared=_prepare(config)
    report=run_v16(prepared,v12_dir=args.v12,v13_dir=args.v13,v14_dir=args.v14,
                   v15_dir=args.v15,output=args.output_dir)
    print('=== DMDOD Train Counterfactual Motor v1.6 ===',flush=True)
    print(f"Reproduction: {report['reproduction_status']}",flush=True)
    print(f"Discovery coverage: {report['discovery_probe_count']}/288",flush=True)
    print(f"Independent success anchors: {report['independent_evidence']['successful_anchor_count']}",flush=True)
    print(f"Generalization evidence: {report['independent_evidence']['generalization_status']}",flush=True)
    print('No training/DAgger/optimizer, no Validation/FINAL, no teacher adoption.',flush=True)
    print(f"Output: {args.output_dir/'diagnostic.json'}",flush=True)


if __name__=='__main__':
    main()
