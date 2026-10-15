from __future__ import annotations

"""Compare saved old vs near-only GRU countdown (Train20 only, no training)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_near_countdown_audit import run_near_countdown_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = run_near_countdown_audit(PrestigeConfig.load(args.config))
    print("=== Stage T saved GRU countdown A/B (read-only) ===", flush=True)
    for split in ("train_fit", "train_heldout"):
        print(split, flush=True)
        for name in ("constant", "mlp", "old_gru", "near_only_gru"):
            group = result["metrics"][split][name]
            acc, time = group["prediction"], group["countdown"]
            print(
                f"  {name}: near-MAE={acc['near_mae_ms']:.3f}ms "
                f"all-visible-MAE={acc['all_visible_mae_ms']:.3f}ms "
                f"countdown MAE={time['near_countdown_mae_ms']:.3f}ms "
                f"near-predicted-step={time['mean_predicted_near_step_ms']:.3f}ms/10ms "
                f"switch-MAE={time['switch_mae_ms']:.3f}ms "
                f"switch-rise={time['switch_rise_detected']}/{time['switch_rise_count']}",
                flush=True,
            )
        geometry = result["geometry_near_subset"][split]
        print(f"  Geometry-matched near frames: {geometry['eligible_near_frames']}", flush=True)
        for name, metrics in geometry["error"].items():
            print(f"    {name}: near-MAE={metrics['mae_ms']}ms", flush=True)
    print("SVG overlay files: " + ", ".join(result["plots"]), flush=True)
    print("No training. Original Stage A FAIL; Motor still blocked; no Validation/FINAL.", flush=True)


if __name__ == "__main__":
    main()
