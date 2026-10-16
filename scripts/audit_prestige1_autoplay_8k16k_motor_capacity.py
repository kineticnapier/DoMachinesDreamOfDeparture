from __future__ import annotations

"""Read-only Stage T 8K vs 16K *simulated actuator* comparison (oracle Motor)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_8k16k_motor_capacity_audit import REPORT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    report = run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT}; fixture sets: {len(report["cases"])}')
    print('Motor-only privileged teacher; NOT human 16K fingering or HUD/Student capability.')


if __name__ == '__main__':
    main()
