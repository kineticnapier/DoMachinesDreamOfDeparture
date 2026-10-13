from __future__ import annotations
"""Prestige I: Stage A observability; Stage C gated fresh-Student BC."""
import argparse
from dmdod.training.prestige1 import PrestigeConfig, run_stage_a, run_stage_c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('stage-a','stage-c'))
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    parser.add_argument('--resume',action='store_true',help='Strict read-only completed-checkpoint verification; no rewriting')
    args=parser.parse_args()
    if args.resume and args.stage!='stage-c':parser.error('--resume only applies to stage-c')
    config=PrestigeConfig.load(args.config)
    if args.stage=='stage-a':
        result=run_stage_a(config)
        print('Stage A gate:',result['gate'],' | motor updates: 0',flush=True)
    else:
        result=run_stage_c(config,resume=args.resume)
        print('Stage C fixed-budget Control vs Prestige:',
              {k:v['validation'] for k,v in result['models'].items()},flush=True)

if __name__=='__main__':main()
