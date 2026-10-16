from __future__ import annotations

"""Read-only Stage T visible-prefix vs full 70ms horizon Motor comparison."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_visible_prefix_audit import REPORT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    result = run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT} | cases: {len(result["cases"])}')
    print('Read-only diagnostic: no Student training, Validation, or FINAL.')


if __name__ == '__main__':
    main()
