from __future__ import annotations

"""Exactly one controlled near-only regression ablation, Train20 only."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_near_ablation import run_near_only_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = run_near_only_ablation(PrestigeConfig.load(args.config))
    print("=== Stage T near-only regression loss ablation ===", flush=True)
    for model in ("constant", "mlp", "gru"):
        original = result["baseline"]["train_heldout"][model]
        print(
            f"Frozen {model}: near-MAE={original['near_mae_ms']:.3f}ms "
            f"all-visible-MAE={original['all_visible_mae_ms']:.3f}ms",
            flush=True,
        )
    alternative = result["near_only_gru"]["train_heldout"]["aggregate"]
    print(
        f"New near-only GRU: near-MAE={alternative['near_mae_ms']:.3f}ms "
        f"all-visible-MAE={alternative['all_visible_mae_ms']:.3f}ms "
        f"near-p95={alternative['near_p95_abs_ms']:.3f}ms "
        f"FPR={alternative['false_positive_rate']:.3f} "
        f"balanced-accuracy={alternative['balanced_accuracy']:.3f}",
        flush=True,
    )
    print(
        f"Research-only evidence={result['new_research_gate']['observability_evidence_passed']}; "
        "Stage A remains FAIL; Motor Student remains blocked; "
        "Validation and FINAL untouched",
        flush=True,
    )


if __name__ == "__main__":
    main()
