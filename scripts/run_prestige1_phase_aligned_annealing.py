from __future__ import annotations

"""DMDOD Prestige I Stage C v2.9: Global Phase Alignment + full-pulse SA."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_phase_comparison import run_train_phase_aligned_annealing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_train_phase_aligned_annealing(PrestigeConfig.load(args.config))
    original = report["baseline_v28"]["timing"]["mae_ms"]
    optimized = report["candidate"]["timing"]["mae_ms"]
    physics = report["candidate"]["physics"]
    print(
        f"Stage C v2.9 Train20 SAFE={physics['safe']}/20 "
        f"Hit={physics['hits']}/850 Miss={physics['misses']} "
        f"TooEarly={physics['too_early']} "
        f"MAE={original:.3f}->{optimized:.3f}ms "
        f"safe_gate={report['gates']['full_train20_safe850']} "
        f"inputs_unchanged={report['protected_inputs_unchanged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
