from __future__ import annotations

"""Read-only causal Geometry first-frame floor-transition counterfactual."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_first_frame_geometry_audit import (
    CANDIDATES, run_first_frame_audit,
)


def fmt(value: float | None) -> str:
    return f"{value:.3f}ms" if value is not None else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_first_frame_audit(PrestigeConfig.load(args.config))
    print("=== Stage T HUD first-frame Geometry guard counterfactual ===", flush=True)
    for split in ("train_fit", "train_heldout"):
        result = report["results"][split]
        continuity = result["crossing_orbit_motion"]
        print(
            f"{split}: oracle switches={result['oracle_true_switches']} "
            f"floor-changed guard={result['original_floor_changed_at_switch']} "
            f"prior motion valid={continuity['with_valid_preceding_step']}",
            flush=True,
        )
        print(
            "  Frozen Geometry at switch: "
            +str(result["original_geometry_abstention_at_switch"]), flush=True,
        )
        print(
            "  Orbit step difference (crossing - previous): "
            "mean absolute difference="
            + (f"{continuity['crossing_step_minus_previous_step_deg']['mae_ms']:.3f}deg"
               if continuity['crossing_step_minus_previous_step_deg']['mae_ms'] is not None
               else "none")
            + "; "
            f"flipped direction={continuity['direction_flips']}",
            flush=True,
        )
        for name in CANDIDATES:
            row = result["candidates_on_same_oracle_switch_frames"][name]
            print(
                f"  {name}: eligible={row['eligible_switches']} "
                f"MAE={fmt(row['absolute_time_error']['mae_ms'])}, "
                f"P95={fmt(row['absolute_time_error']['p95_abs_ms'])}, "
                f"GRU on same frames={fmt(row['same_frame_frozen_gru']['mae_ms'])}, "
                f"next-frame countdown residual="
                f"{fmt(row['next_frame_actual_geometry_countdown_residual']['mae_ms'])} "
                f"(n={row['next_frame_pair_count']})",
                flush=True,
            )
            print(f"    statuses={row['statuses_at_true_switch']}", flush=True)
        print(
            "  No Geometry recovery in same true event — "
            f"first reasons={result['unrecovered_before_next_oracle_event']['first_abstention_reason']}, "
            f"all reasons={result['unrecovered_before_next_oracle_event']['all_frame_abstention_reasons']}",
            flush=True,
        )
        for case in result["largest_counterfactual_errors"][:5]:
            print(
                f"  outlier: anchor={case['anchor']:02d} "
                f"frame={case['frame']} {case['candidate']} "
                f"error={fmt(case['signed_error_ms'])}",
                flush=True,
            )
    print(
        "Only causal HUD counterfactuals; no detector/adoption/training; "
        "original Stage A FAIL and no Validation/FINAL.",
        flush=True,
    )


if __name__ == "__main__":
    main()
