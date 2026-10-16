from __future__ import annotations

"""Read-only 12-axis/16-contact varied-angle timing attribution diagnostic."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_coupled_chart_patterns_audit import REPORT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    result = run_audit(PrestigeConfig.load(args.config))
    print(f'Report: {REPORT}; fixed chart+mode combinations: {len(result["cases"])}')
    print('Generated patterns, privileged oracle teacher, not HUD-only Student or human 16K.')


if __name__ == '__main__':
    main()
