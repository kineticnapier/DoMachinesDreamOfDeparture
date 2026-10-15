from __future__ import annotations

"""Read-only: what 263D HUD features change when the next note changes?"""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_switch_hud_audit import run_switch_hud_audit


def _format_ms(value: float | None) -> str:
    return f"{value:.3f}ms" if value is not None else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_switch_hud_audit(PrestigeConfig.load(args.config))
    print("=== Stage T HUD note-switch audit (read-only Train20) ===", flush=True)
    keys = (
        "current_floor_changed", "next_floor_changed",
        "any_floor_changed", "current_angles_changed",
        "current_position_changed", "current_event_flags_changed",
        "orbit_xy_changed", "tile_bpm_changed",
        "real_bpm_changed", "feedback_changed",
    )
    for split in ("train_fit", "train_heldout"):
        data = report["results"][split]
        cue = data["visible_cue_differences"]
        counts = cue["pair_counts"]
        post = data["post_switch"]
        recovery = post["first_eligible_geometry_same_event"]
        print(
            f"{split}: switches={counts['true_switch']} "
            f"same-event={counts['same_event']}",
            flush=True,
        )
        for name in keys:
            row = cue["cues"][name]
            print(
                f"  {name}: changed on switches={row['switch_change_rate']:.1%}, "
                f"on same-event pairs={row['same_event_change_rate']:.1%}; "
                f"precision if used as fixed cue="
                f"{row['descriptive_precision']:.3f}"
                if row["descriptive_precision"] is not None else
                f"  {name}: no positive cue frames",
                flush=True,
            )
        print(
            f"  First Geometry in same oracle event: "
            f"{recovery['reached']}/{post['scored_true_event_switches']}; "
            f"delay-MAE (absolute delay, not timing error)="
            f"{_format_ms(recovery['delay_ms']['mae_ms'])}; "
            f"first Geometry timing MAE="
            f"{_format_ms(recovery['absolute_timing_error_on_first_geometry']['mae_ms'])}",
            flush=True,
        )
        for offset, row in post["offset_from_switch"].items():
            print(
                f"  +{int(offset)*10}ms: n={row['frames_with_same_oracle_event_and_observable']}"
                f" modes={row['modes']} "
                f"Hybrid MAE={_format_ms(row['errors']['hybrid']['mae_ms'])} "
                f"GRU MAE={_format_ms(row['errors']['near_only_gru']['mae_ms'])}",
                flush=True,
            )
        print("  Largest first-frame Hybrid errors:", flush=True)
        for case in post["largest_first_frame_errors"]:
            print(
                f"    anchor={case['anchor']:02d} frame={case['frame_index']} "
                f"abs_error={abs(case['first_hybrid_error_ms']):.3f}ms "
                f"mode={case['first_mode']} "
                f"geometry_delay={_format_ms(case['first_geometry_delay_ms'])} "
                f"reason={case['first_abstain_reason']}",
                flush=True,
            )
    print("No training; label-only oracle; Stage A remains FAIL; no Validation/FINAL.",
          flush=True)


if __name__ == "__main__":
    main()
