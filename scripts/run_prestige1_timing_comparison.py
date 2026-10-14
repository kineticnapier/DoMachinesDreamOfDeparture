from __future__ import annotations

"""Stage C v2.6 Train20-only timing Teacher replay."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_timing_comparison import run_train_timing_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/prestige1.toml')
    args = parser.parse_args()
    result = run_train_timing_comparison(PrestigeConfig.load(args.config))
    print('Stage C v2.6 Train20', result['candidate']['physics'], result['gates'], flush=True)


if __name__ == '__main__':
    main()
