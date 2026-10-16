from __future__ import annotations

"""Stage T read-only RP1 double-angle orbit-motion and countdown experiment."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_projective_orbit_audit import (
    CANDIDATES, run_projective_orbit_audit,
)


def mae(row:dict)->str:
    v=row["mae_ms"]
    return "n/a" if v is None else f"{v:.6f}ms"


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/training/prestige1.toml")
    args=parser.parse_args()
    report=run_projective_orbit_audit(PrestigeConfig.load(args.config))
    print("=== Stage T RP1 double-angle orbit read-only counterfactual ===",flush=True)
    for split in ("train_fit","train_heldout"):
        result=report["results"][split]
        print(f"{split}: switch={result['pair_totals']['all_switch']}, "
              f"same-event={result['pair_totals']['same_event']} "
              f"original={result['original_geometry_reason_at_switch']}",flush=True)
        print(f"  Earlier halfturn explained floor events="
              f"{result['classified_floor_halfturn_explained']}; "
              f"unexplained floor events={result['classified_floor_halfturn_unexplained']}",
              flush=True)
        for name in CANDIDATES:
            b=result["double_angle_candidates"][name]
            print(f"  {name}: same-event false positives="
                  f"{b['same_event']['eligible']}/{b['same_event']['total_pairs']}, "
                  f"descriptive precision={b['descriptive_precision']}",
                  flush=True)
            for key in ("original_motion","original_floor","floor_halfturn_unexplained",
                        "floor_halfturn_explained","all_switch","same_event"):
                item=b[key]
                angle=item["double_angle_step_consistency"]
                print(
                    f"    {key}: eligible={item['eligible']}/{item['total_pairs']} "
                    f"Timing MAE={mae(item['candidate_timing'])} "
                    f"GRU on same={mae(item['matched_frozen_gru'])} "
                    f"Hybrid on same={mae(item['matched_frozen_hybrid'])} "
                    f"mod-pi angular residual mean={angle['mean_abs_deg']}deg "
                    f"<=1deg={angle['within_1deg']}/{angle['count']}",
                    flush=True)
                print(f"      status={item['statuses']}",flush=True)
                if key in ("original_motion","floor_halfturn_unexplained"):
                    for ex in item["largest_errors"][:5]:
                        print(f"      example anchor={ex['anchor']:02d} frame={ex['frame']} "
                              f"err={ex['error_ms']:.6f}ms "
                              f"GRU={ex['gru_error_ms']:.6f}ms "
                              f"projective residual={ex['angle_residual_deg']:.6f}deg",
                              flush=True)
        carried=result["prior_carried_previous_step"]["floor_halfturn_unexplained"]
        print(f"  OLD carried-step in earlier unexplained cohort: "
              f"eligible={carried['eligible']} MAE={mae(carried['error'])}",flush=True)
    print("Research only. Original Geometry/Hybrid/Stage A unchanged; "
          "no Motor training, Validation or FINAL.",flush=True)


if __name__=="__main__":
    main()
