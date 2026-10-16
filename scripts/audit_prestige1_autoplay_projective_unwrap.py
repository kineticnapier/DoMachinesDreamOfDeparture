from __future__ import annotations

"""Read-only Stage T projective angular unwrapping from unclipped v2 HUD."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_projective_unwrap_audit import (
    CANDIDATES, BPM_BINS, run_unwrap_audit,
)


def metric(m:dict)->str:
    return "n/a" if m["mae_ms"] is None else (
        f"MAE={m['mae_ms']:.6f}ms "
        f"P95={m['p95_abs_ms']:.6f}ms max-availability={m['count']}"
    )


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/training/prestige1.toml")
    args=parser.parse_args()
    report=run_unwrap_audit(PrestigeConfig.load(args.config))
    print("=== Stage T v2 unclipped BPM / RP1 high-speed unwrapping ===",flush=True)
    for split in ("train_fit","train_heldout"):
        out=report["results"][split]
        print(f"{split}: oracle switches={out['oracle_pair_counts']['switch']} "
              f"same-event pairs={out['oracle_pair_counts']['same_event']}",
              flush=True)
        print(f"  Above 1920 BPM observed switch pairs="
              f"{out['above_1920_pairs'].get('switch',0)}; "
              f"same-event={out['above_1920_pairs'].get('same_event',0)}",
              flush=True)
        for name in CANDIDATES:
            rows=out["candidate_results"][name]
            print(f"  {name}: switch={rows['switch']['eligible']}/"
                  f"{rows['switch']['total']}, "
                  f"{metric(rows['switch']['timing'])}; "
                  f"same-event eligible={rows['same_event']['eligible']}/"
                  f"{rows['same_event']['total']}, "
                  f"descriptive precision={rows['precision_descriptive']}",
                  flush=True)
            for label in [*BPM_BINS,"above_1920","above_3000"]:
                item=rows[label]
                print(f"    {label}: {item['eligible']}/{item['total']} "
                      f"{metric(item['timing'])} reasons={item['status']}",
                      flush=True)
            for item in rows["switch"]["worst"][:8]:
                print(f"    outlier anchor={item['anchor']:02d} "
                      f"frame={item['frame']} BPM={item['bpm']:.3f} "
                      f"delta={item['unwrapped_step_deg']:+.3f}deg "
                      f"RP1={item['projective_observed_deg']:+.3f}deg "
                      f"error={item['error_ms']:+.6f}ms "
                      f"guard={item['original_status']}",flush=True)
            print(f"    switch guards={rows['switch']['status']}",flush=True)
            print(f"    same-event guards={rows['same_event']['status']}",flush=True)
    print("No adopted Geometry/Hybrid change, Motor training, Validation or FINAL. "
          "Original Stage A remains FAIL.",flush=True)


if __name__=="__main__":
    main()
