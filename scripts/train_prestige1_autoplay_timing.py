from __future__ import annotations

"""Train/read out exact-Autoplay Timing (Train20 chart-disjoint only)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_readout import run_autoplay_timing_readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    config = PrestigeConfig.load(args.config)
    result = run_autoplay_timing_readout(config)
    print("=== Exact Autoplay Timing: chart-disjoint Train-only ===", flush=True)
    for name in ("constant", "mlp", "gru"):
        metrics = result["train_heldout"][name]["aggregate"]
        print(
            f"{name}: near-MAE={metrics['near_mae_ms']:.3f}ms "
            f"all-visible-MAE={metrics['all_visible_mae_ms']:.3f}ms "
            f"near-p95={metrics['near_p95_abs_ms']:.3f}ms "
            f"balanced-accuracy={metrics['balanced_accuracy']:.3f} "
            f"FPR={metrics['false_positive_rate']:.3f} "
            f"positive={metrics['near_positive_frames']}",
            flush=True,
        )
    gate = result["research_gate"]
    print(
        "Research observability evidence="
        f"{gate['observability_evidence_passed']} reasons={gate['reasons']}",
        flush=True,
    )
    print(
        "Frozen Stage A remains FAIL; motor_student_training_permitted=False; "
        "Validation/FINAL not evaluated",
        flush=True,
    )


if __name__ == "__main__":
    main()
