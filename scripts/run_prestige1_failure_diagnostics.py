from __future__ import annotations
"""Prestige I Stage C v2.1: frozen checkpoint failure diagnostics; no training."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_failure_diagnostics import (
    DEFAULT_ANCHORS, run_readonly_diagnostics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    parser.add_argument("--anchors", type=int, nargs="+",
                        default=list(DEFAULT_ANCHORS),
                        help="Existing Validation anchor IDs (default: 7 11 18)")
    args = parser.parse_args()
    config = PrestigeConfig.load(args.config)
    result = run_readonly_diagnostics(config, anchors=args.anchors)
    print("Stage C v2.1 read-only diagnostics complete:",
          result["anchors"], "| original artifacts unchanged:",
          result["original_artifacts_unchanged"], flush=True)


if __name__ == "__main__":
    main()
