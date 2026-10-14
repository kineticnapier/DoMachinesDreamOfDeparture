from __future__ import annotations

"""Prestige I Stage C v2.4 Train20-only actuator Teacher comparison."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_actuator_comparison import run_train_actuator_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = run_train_actuator_comparison(PrestigeConfig.load(args.config))
    old, new = result["baseline"], result["candidate"]
    print(
        f"Train20 old SAFE={old['safe']}/20 H={old['hits']} "
        f"miss={old['misses']} early={old['too_early']} | "
        f"candidate SAFE={new['safe']}/20 H={new['hits']} "
        f"miss={new['misses']} early={new['too_early']}",
        flush=True)


if __name__ == "__main__":
    main()
