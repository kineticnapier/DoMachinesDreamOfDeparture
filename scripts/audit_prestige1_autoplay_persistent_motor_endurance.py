from __future__ import annotations

"""Read-only long-chart persistent 8K motor failure-attribution diagnostic."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_persistent_motor_endurance_audit import REPORT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    result = run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT} | fixed cases: {len(result["cases"])}')
    print('No training, changed original policy, Validation or FINAL.')


if __name__ == '__main__':
    main()
