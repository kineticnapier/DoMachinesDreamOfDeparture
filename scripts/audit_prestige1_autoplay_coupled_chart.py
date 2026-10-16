from __future__ import annotations

"""Physical 12-axis/16-contact real-chart integration, oracle Motor only."""

import argparse
from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_coupled_chart_audit import run_audit, REPORT


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    config=PrestigeConfig.load(parser.parse_args().config)
    result=run_audit(config)
    print(f'{REPORT}: {len(result["cases"])} deterministic oracle-only cases')


if __name__=='__main__':
    main()
