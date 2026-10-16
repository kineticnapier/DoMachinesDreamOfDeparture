from __future__ import annotations

"""Read-only Stage T first-frame special marker and motion abstention diagnosis."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_special_motion_audit import (
    run_special_motion_audit,
)


def _mae(row: dict) -> str:
    value = row["mae_ms"]
    return f"{value:.3f}" if value is not None else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    data = run_special_motion_audit(PrestigeConfig.load(args.config))
    print("=== Stage T frozen HUD special-marker / orbit-motion breakdown ===",flush=True)
    for split in ("train_fit","train_heldout"):
        result = data["results"][split]
        print(
            f"{split}: true switches={result['true_note_switches']} "
            f"original={result['original_geometry_first_rejection']} "
            f"carried={result['counterfactual_carried_first_rejection']}",
            flush=True,
        )
        special = result["carried_special_event"]
        markers = special["marker_breakdown"]
        print(
            f"  Carried 'special_event' (AFTER floor guard relaxation): "
            f"n={markers['count']}",
            flush=True,
        )
        print(f"    Current markers: {markers['marker_occurrences_by_slot']['current']}",flush=True)
        print(f"    Next markers: {markers['marker_occurrences_by_slot']['next']}",flush=True)
        print(f"    Exact cooccurrence patterns: {markers['marker_patterns_exclusive']}",flush=True)
        print(f"    Other structural flags: {markers['structural_flags']}",flush=True)
        for label in ("original_inconsistent_orbit_motion",
                      "floor_changed_then_carried_motion_inconsistent"):
            row = result[label]
            quantities = row["quantities"]
            print(
                f"  {label}: n={row['count']} "
                f"crossing={row['speed_classes']['crossing_speed_class']} "
                f"prior/current={row['speed_classes']['prior_current_speed_class']}",
                flush=True,
            )
            print(
                f"    crossing ratio mean={quantities['crossing_speed_ratio']['signed_mean']}, "
                f"prior/current ratio mean="
                f"{quantities['prior_to_current_speed_ratio']['signed_mean']}, "
                f"direction flips={row['direction_flips']}, "
                f"Tile BPM changed={row['tile_bpm_changed_count']}, "
                f"Real BPM changed={row['real_bpm_changed_count']}",
                flush=True,
            )
            for item in row["most_inconsistent_crossing_samples"][:5]:
                print(
                    f"    sample anchor={item['anchor']:02d} frame={item['frame']} "
                    f"cross_ratio={item['crossing_ratio']:.3f} "
                    f"prior_ratio={item['prior_current_ratio']} "
                    f"markers={item['markers']}",
                    flush=True,
                )
    print(
        "Descriptive only. No guard relaxation, retraining, Motor adoption, "
        "Validation or FINAL. Stage A remains FAIL.",
        flush=True,
    )


if __name__ == "__main__":
    main()
