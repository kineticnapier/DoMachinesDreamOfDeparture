from __future__ import annotations

"""Deterministic REAL 8-key ADOFAI/HUD simulation, 600-20000 BPM (read-only)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_high_bpm_simulator_audit import (
    REPORT,run_simulator_20k_audit,
)


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/training/prestige1.toml")
    args=ap.parse_args()
    report=run_simulator_20k_audit(PrestigeConfig.load(args.config))
    print("=== REAL 8K simulator HUD 1ms/10ms through 20000 BPM ===",flush=True)
    print("All 26 fixed BPM/direction fixtures captured with v2 BPM input.",flush=True)
    for c in report["cases"]:
        one=c["phase_1ms"]
        ten=c["phase_10ms"]
        target=c["next_target_timing_10ms"]
        print(
            f"{c['bpm']:8.0f} {c['direction']}: "
            f"1ms angular correct={one['correct_within_0p05deg']}/"
            f"{one['posthoc_same_floor_pairs']} "
            f"(eligible={one['eligible']}, wrong={one['wrong_above_0p05deg']}), "
            f"10ms samefloor={ten['posthoc_same_floor_pairs']}, "
            f"10ms multi-target={target['pairs_crossing_two_or_more_targets']}/"
            f"{target['pairs']} max/interval={target['max_targets_crossed_per_sample']}",
            flush=True)
        for name,data in target["candidates"].items():
            print(
                f"    {name}: eligible={data['eligible_with_next_target']} "
                f"Timing MAE={data['mae_ms']}ms "
                f"wrong>0.05ms={data['wrong_over_0p05ms']} "
                f"unsafe multi-target proposals={data['eligible_on_multi_target_interval']}",
                flush=True)
    print(
        f"Report={REPORT} unsafe_for_adoption={report['unsafe_for_adoption']}; "
        "NO Stage A change, training, Motor, Validation or FINAL.",flush=True)


if __name__=="__main__":
    main()
