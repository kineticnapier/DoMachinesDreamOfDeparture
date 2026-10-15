from __future__ import annotations

"""Read-only Stage T countdown / event-switch / HUD-geometry diagnostic."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_countdown import run_countdown_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_countdown_audit(PrestigeConfig.load(args.config))
    print("=== Exact Auto Timing: countdown diagnosis (read-only Train20) ===", flush=True)
    for split in ("train_fit", "train_heldout"):
        print(split, flush=True)
        for name in ("constant", "mlp", "gru"):
            aggregate = report["timing_dynamics"][split][name]["aggregate"]
            same = aggregate["same_event_near_countdown"]
            full = aggregate["same_event_countdown"]
            jump = aggregate["event_switch_residual"]
            print(
                f"  {name}: same-event MAE={full['mae_ms']}ms "
                f"near-countdown MAE={same['mae_ms']}ms "
                f"mean predicted step={aggregate['mean_predicted_step_ms']}ms "
                f"switch residual MAE={jump['mae_ms']}ms "
                f"switch-rise recall={aggregate['upward_switch_recall']}",
                flush=True,
            )
        for model_name in ("mlp", "gru"):
            loss = report["loss_contributions"][split][model_name]
            print(
                f"  {model_name} original-loss contributions: "
                f"time_total={loss['time_term_total']:.4f} "
                f"bce_total={loss['event_bce_total']:.4f}", flush=True,
            )
            for label, contribution in loss["by_distance"].items():
                print(
                    f"    {label}: timing-term share="
                    f"{contribution['time_term_fraction']} "
                    f"BCE share={contribution['bce_fraction']}", flush=True,
                )
        geometry = report["geometry_baseline"][split]
        print(
            f"  Geometry eligible={geometry['eligible_frames']} "
            f"of {geometry['visible_candidate_frames']} HUD frames",
            flush=True,
        )
        for name, score in geometry["same_eligible_frame_comparison"].items():
            print(
                f"    {name}: geometry-subset near MAE="
                f"{score['near_200ms']['mae_ms']}ms "
                f"(n={score['near_200ms']['count']})",
                flush=True,
            )
    print(f"SVG traces: {', '.join(report['plots'])}", flush=True)
    print("Stage A remains FAIL; no new model training / Validation / FINAL.", flush=True)


if __name__ == "__main__":
    main()
