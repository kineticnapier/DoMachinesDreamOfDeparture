from __future__ import annotations

"""Explain outliers behind the Stage T RP1 projective floor-change MAE."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_projective_error_audit import (
    CANDIDATES,run_projective_error_audit,
)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/training/prestige1.toml")
    args=ap.parse_args()
    report=run_projective_error_audit(PrestigeConfig.load(args.config))
    print("=== Stage T RP1: frozen floor-change timing outlier attribution ===",flush=True)
    for split in ("train_fit","train_heldout"):
        data=report["results"][split]
        print(f"{split}: switches={data['switch_count']} "
              f"same-event pairs={data['same_event_pair_count']}",flush=True)
        print(f"  Noneligible RP1 status={data['noneligible_projective_speed_by_status']}",
              flush=True)
        print(f"  Noneligible RP1 original reasons="
              f"{data['noneligible_original_reasons']}",flush=True)
        for name in CANDIDATES:
            result=data["floor_details_by_candidate"][name]
            error=result["error"]
            print(
                f"  {name}: floor eligible={result['eligible']} "
                f"MAE={error['mean_abs']}ms bias={error['mean']}ms "
                f"P95={error['p95_abs']}ms",
                flush=True)
            print(f"    |err| threshold counts (above ms): "
                  f"{result['absolute_error_count_above_ms']}",flush=True)
            print(f"    BPM Tile change={result['by_tile_bpm_changed']} "
                  f"Real change={result['by_real_bpm_changed']}",flush=True)
            print("    Error concentration by anchor:",flush=True)
            for item in sorted(result["by_anchor"],
                               key=lambda v:v["total_absolute_error_ms"],
                               reverse=True)[:8]:
                print(f"      anchor={item['anchor']:02d} n={item['count']} "
                      f"sum|err|={item['total_absolute_error_ms']:.6f}ms "
                      f"max|err|={item['largest_abs_error_ms']:.6f}ms",flush=True)
            print("    Largest absolute errors:",flush=True)
            for item in result["largest_errors"][:12]:
                print(
                    f"      anchor={item['anchor']:02d} frame={item['frame']} "
                    f"err={item['error_ms']:+.6f}ms "
                    f"({item['error_in_control_frames']:+.4f} control frames), "
                    f"phase={item['phase_equivalent_error_deg']:+.4f}deg "
                    f"TileBPM={item['tile_bpm']:.4f} (changed={item['tile_bpm_changed']}), "
                    f"RealBPM={item['real_bpm']:.4f} (changed={item['real_bpm_changed']}), "
                    f"stepRatio={item['projective_speed_ratio']:.6f} "
                    f"angularResidual={item['projective_angular_consistency_deg']:.6f}deg "
                    f"truth={item['oracle_remaining_ms']:.5f}ms "
                    f"prediction={item['predicted_remaining_ms']:.5f}ms",
                    flush=True)
    print("Read-only. No Geometry/Hybrid/RP1 change, training, Motor, Validation or FINAL.",
          flush=True)


if __name__=="__main__":
    main()
