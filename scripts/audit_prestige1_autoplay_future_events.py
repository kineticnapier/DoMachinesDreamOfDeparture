from __future__ import annotations

"""Read-only HUD-only future 10ms event sequence diagnostic through 20k BPM."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_future_events_audit import REPORT,run_audit


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/training/prestige1.toml")
    args=ap.parse_args()
    result=run_audit(PrestigeConfig.load(args.config))
    print(f"Report: {REPORT}, cases={len(result['cases'])}",flush=True)
    print("No Student training, Stage A gate change, Validation or FINAL.",flush=True)


if __name__=="__main__":
    main()
