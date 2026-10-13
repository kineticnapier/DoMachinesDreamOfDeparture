from __future__ import annotations
"""Stage B/C v2: Stage A timing transfer and exploratory matched Motor comparison."""

import argparse
from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_transfer import run_stage_b, run_stage_c_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage-b", "stage-c-v2"))
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    config = PrestigeConfig.load(args.config)
    if args.stage == "stage-b":
        report = run_stage_b(config)
        print("Stage B transfer parity:", report["parity"], flush=True)
    else:
        report = run_stage_c_v2(config)
        print("Stage C v2 exploratory physics:",
              {arm: item["validation"] for arm, item in report["results"].items()},
              flush=True)


if __name__ == "__main__":
    main()
