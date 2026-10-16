from __future__ import annotations

"""Read-only Stage T own-action shadow physics for predictive 8-key arming."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_state_aware_motor_audit import REPORT,run_audit


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    args=parser.parse_args()
    report=run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT}; fixed comparisons: {len(report["cases"])}')
    print('No Student/Geometry/Motor changes; Stage A FAIL and Validation/FINAL preserved.')


if __name__ == '__main__':
    main()
