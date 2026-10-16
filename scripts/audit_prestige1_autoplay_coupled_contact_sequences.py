from __future__ import annotations

"""Read-only Stage T human-inspired 16-contact scripted sequence diagnostic."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_coupled_contact_sequences_audit import REPORT, run_audit


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    args=parser.parse_args()
    report=run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT}; sequences: {len(report["cases"])}')
    print('Only motor contact sequences; no ADOFAI evaluator, Student training or policy change.')


if __name__=='__main__':
    main()
