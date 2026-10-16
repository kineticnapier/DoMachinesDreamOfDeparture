from __future__ import annotations

"""Minimal read-only persistent Motor test; 10ms actions and 1ms physics."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_persistent_motor_audit import REPORT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    report = run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT} / {len(report["cases"])} fixed experiments')
    print('No Student training, frozen Stage A changes, Validation or FINAL.')


if __name__ == '__main__':
    main()
