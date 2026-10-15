from __future__ import annotations

"""DMDOD Prestige I Stage C v3.0: safe per-finger phase + pulse SA."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_finger_comparison import (
    run_train_finger_phase_annealing,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_train_finger_phase_annealing(PrestigeConfig.load(args.config))
    source = report["baseline_v29"]["timing"]["mae_ms"]
    current = report["candidate"]["timing"]["mae_ms"]
    physics = report["candidate"]["physics"]
    print(
        f"Stage C v3.0 Train20 SAFE={physics['safe']}/20 "
        f"Hit={physics['hits']}/850 Miss={physics['misses']} "
        f"TooEarly={physics['too_early']} "
        f"MAE={source:.3f}->{current:.3f}ms "
        f"safe_gate={report['gates']['full_train20_safe850']} "
        f"protected={report['protected_inputs_unchanged']}", flush=True)


if __name__ == "__main__":
    main()
