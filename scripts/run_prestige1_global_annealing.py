from __future__ import annotations

"""Prestige I Stage C v2.7 — fixed Train20 global Teacher annealing."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_global_comparison import run_train_global_annealing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_train_global_annealing(PrestigeConfig.load(args.config))
    base = report["baseline_v24"]["timing"]["mae_ms"]
    optimized = report["candidate"]["timing"]["mae_ms"]
    physics = report["candidate"]["physics"]
    print(
        f"Stage C v2.7 Train20 SAFE={physics['safe']}/20 "
        f"H={physics['hits']}/{report['baseline_v24']['physics']['hits']} "
        f"Miss={physics['misses']} TooEarly={physics['too_early']} "
        f"MAE={base:.3f}->{optimized:.3f}ms "
        f"physical_gate={report['gates']['full_train20_safe850']} "
        f"protected={report['protected_inputs_unchanged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
