from __future__ import annotations

"""Read-only Stage T visible-prefix persistent Motor timing attribution."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_prefix_motor_timing_audit import REPORT,run_audit


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/training/prestige1.toml')
    args=ap.parse_args()
    report=run_audit(PrestigeConfig.load(args.config))
    print(f"Saved {len(report['cases'])} fixed diagnostic cases: {REPORT}")
    print("Original Planner, checkpoints, Stage A, Validation and FINAL untouched.")


if __name__ == '__main__':
    main()
