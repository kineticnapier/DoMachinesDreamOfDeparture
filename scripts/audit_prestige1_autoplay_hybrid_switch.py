from __future__ import annotations

"""Read-only framewise Geometry/near-only GRU Hybrid Timing diagnosis."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_hybrid_audit import (
    run_hybrid_audit, TRANSITIONS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_hybrid_audit(PrestigeConfig.load(args.config))
    print("=== Stage T stateless HUD Hybrid (Train20 only, read-only) ===", flush=True)
    for split in ("train_fit", "train_heldout"):
        frame = report["scores"][split]
        switch = report["switching"][split]
        print(f"{split}: observable={frame['visible_frames']} near={frame['near_frames']} "
              f"geometry near coverage={frame['geometry_near_coverage']:.3%}", flush=True)
        for name in ("constant", "near_only_gru", "geometry", "hybrid"):
            stats = frame["matched_baselines"][name]
            print(
                f"  {name}: near-MAE={stats['near']['mae_ms']}ms "
                f"(n={stats['near']['count']}) "
                f"all-visible-MAE={stats['all_visible']['mae_ms']}ms "
                f"(n={stats['all_visible']['count']})",
                flush=True,
            )
        print(
            f"  oracle event switches={switch['event_switches_observed']} "
            f"positive-jump direction recall={switch['positive_jump_direction_recall']} "
            f"false upward step rate on SAME oracle event="
            f"{switch['incorrect_upward_step_rate']}",
            flush=True,
        )
        for name in TRANSITIONS:
            pair = switch["by_mode_transition_and_oracle_event"][name]
            for label in ("same_event", "event_changed"):
                stats = pair[label]
                print(
                    f"  {name} ({label}): pairs={stats['pairs']} "
                    f"hybrid step-residual MAE="
                    f"{stats['hybrid_countdown_or_jump_residual']['mae_ms']}ms "
                    f"GRU-only counterfactual={stats['gru_only_counterfactual_residual']['mae_ms']}ms "
                    f"arrival error MAE={stats['after_hybrid_absolute_error']['mae_ms']}ms "
                    f">20ms jumps={stats['spikes_over_20ms']}",
                    flush=True,
                )
    print("SVG traces: " + ", ".join(report["plots"]), flush=True)
    print("Stateless read-only experiment; no training, Stage A override, Validation or FINAL.",
          flush=True)


if __name__ == "__main__":
    main()
