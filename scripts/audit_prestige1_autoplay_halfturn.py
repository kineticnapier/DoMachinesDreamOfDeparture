from __future__ import annotations

"""Read-only: measure whether note-switch orbit discontinuities equal 180 degrees."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_halfturn_audit import run_halfturn_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_halfturn_audit(PrestigeConfig.load(args.config))
    print("=== Stage T: 180-degree orbit-switch hypothesis (HUD-only, read-only) ===",
          flush=True)
    for role in ("train_fit","train_heldout"):
        entry=report["results"][role]
        print(f"{role}: oracle-labelled switches={entry['total_labelled_switches_with_prior_step']}",
              flush=True)
        print(f"  existing Geometry: {entry['original_reason_counts']}",flush=True)
        for label,cohort in entry["cohorts"].items():
            motion=cohort["halfturn_minus_prior_abs_degrees"]
            floor=cohort["entry_exit_halfturn_abs_degrees"]
            ratio=cohort["halfturn_corrected_speed_ratio"]
            print(f"  {label}: n={cohort['events']}",flush=True)
            print(
                f"    |wrap(cross-prior-180deg)|: mean={motion['mean']}deg "
                f"median={motion['median']}deg P95={motion['p95']}deg",
                flush=True,
            )
            print(
                f"    Floor new-entry minus old-exit halfturn residual: "
                f"mean={floor['mean']}deg P95={floor['p95']}deg",
                flush=True,
            )
            print(
                f"    fixed 1deg halfturn residual <=1deg: "
                f"{cohort['halfturn_residual_threshold_counts']['1.0']}/{cohort['events']}; "
                f"corrected speed within original [0.5,1.5]: "
                f"{cohort['corrected_within_original_speed_band']}/{cohort['events']}; "
                f"current floor same 15D: "
                f"{cohort['current_floor_features_unchanged']}/{cohort['events']}; "
                f"Tile BPM changed={cohort['tile_bpm_changed']}",
                flush=True,
            )
            print(
                f"    corrected speed ratio mean={ratio['mean']} "
                f"raw crossing mean={cohort['uncorrected_crossing_speed_ratio']['mean']}",
                flush=True,
            )
            for ex in cohort["worst_halfturn_residual_examples"][:5]:
                print(
                    f"    example anchor={ex['anchor']:02d} frame={ex['frame']} "
                    f"prior={ex['prior_step_deg']:.3f}deg "
                    f"cross={ex['crossing_step_deg']:.3f}deg "
                    f"after-halfturn={ex['halfturn_corrected_step_deg']:.3f}deg "
                    f"residual={ex['halfturn_minus_prior_absolute_deg']:.5f}deg",
                    flush=True,
                )
    print(
        "No Geometry guard removed. No training/Stage A override/Motor/Validation/FINAL.",
        flush=True,
    )


if __name__ == "__main__":
    main()
