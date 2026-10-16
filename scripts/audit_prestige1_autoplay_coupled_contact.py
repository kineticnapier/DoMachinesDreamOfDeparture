from __future__ import annotations

"""Read-only Stage T 12-axis/16-contact human-inspired contact topology."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_coupled_contact_audit import REPORT, run_audit


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    args=parser.parse_args()
    result=run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT}; mode cases: {len(result["cases"])}')
    print('Contact-model smoke diagnostic only: not human BPM/skill validation.')


if __name__=='__main__':
    main()
