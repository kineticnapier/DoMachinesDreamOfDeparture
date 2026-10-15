from __future__ import annotations

"""Prestige I Stage C v2.8: Train20 constrained full-pulse SA + physical repair."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_pulse_comparison import run_train_full_pulse_annealing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/prestige1.toml",
        help="Frozen Prestige I configuration")
    args = parser.parse_args()
    result = run_train_full_pulse_annealing(PrestigeConfig.load(args.config))
    prior = result["baseline_v27"]["timing"]["mae_ms"]
    current = result["candidate"]["timing"]["mae_ms"]
    physics = result["candidate"]["physics"]
    print(
        f"Stage C v2.8 Train20 SAFE={physics['safe']}/20 "
        f"H={physics['hits']}/850 Miss={physics['misses']} "
        f"TooEarly={physics['too_early']} "
        f"MAE={prior:.3f}->{current:.3f}ms "
        f"gate={result['gates']['full_train20_safe850']} "
        f"protected={result['protected_inputs_unchanged']}",
        flush=True)


if __name__ == "__main__":
    main()
