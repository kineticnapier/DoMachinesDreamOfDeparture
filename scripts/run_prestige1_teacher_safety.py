from __future__ import annotations

"""Prestige I Stage C v2.3 Train20-only Teacher safety audit."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_teacher_safety import run_train_teacher_safety


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = run_train_teacher_safety(PrestigeConfig.load(args.config))
    print(f"Stage C v2.3 Train-only audit: "
          f"SAFE={result['aggregate']['safe']}/20 "
          f"Overload={result['aggregate']['unsafe_anchors']} "
          f"originals-unchanged={result['protected_inputs_unchanged']}", flush=True)


if __name__ == "__main__":
    main()
