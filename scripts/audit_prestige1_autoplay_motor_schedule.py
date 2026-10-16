from __future__ import annotations

"""Stage T read-only HUD event -> frozen 8-Key Motor shadow schedule (up to 20k)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_motor_schedule_audit import REPORT, run_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/training/prestige1.toml")
    args = ap.parse_args()
    report = run_audit(PrestigeConfig.load(args.config))
    print(f"Report: {REPORT}; {len(report['results'])} fixed fixture runs")
    print("No Student / Motor changes; the 1ms commands are a reference-only shadow simulation.")


if __name__ == "__main__":
    main()
