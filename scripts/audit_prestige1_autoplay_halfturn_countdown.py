from __future__ import annotations

"""Read-only 180-degree corrected Geometry countdown and same-event false-positive audit."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_halfturn_countdown_audit import (
    CANDIDATES, run_halfturn_countdown_audit,
)


def fmt(row: dict) -> str:
    return "none" if row["mae_ms"] is None else f"{row['mae_ms']:.6f}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_halfturn_countdown_audit(PrestigeConfig.load(args.config))
    print("=== Stage T: 180-degree corrected Geometry countdown (read-only) ===",
          flush=True)
    for split in ("train_fit", "train_heldout"):
        result = report["results"][split]
        totals = result["label_only_pair_counts"]
        print(f"{split}: oracle switches={totals['switch']} "
              f"same-event pairs={totals['same']}",flush=True)
        print(
            f"  Original Geometry switch statuses: "
            f"{result['original_frozen_geometry_abstention_reasons']['switch']}",
            flush=True)
        for name in CANDIDATES:
            row=result["candidate_results"][name]
            switch=row["switch"]
            same=row["same_event"]
            print(
                f"  {name}: switch eligible={switch['eligible']}/{totals['switch']} "
                f"conditional MAE={fmt(switch['errors']['candidate'])}, "
                f"frozen GRU on same={fmt(switch['errors']['gru'])}, "
                f"frozen Hybrid on same={fmt(switch['errors']['hybrid'])}",
                flush=True,
            )
            print(
                f"    same-event false positives={same['eligible']}/{totals['same']} "
                f"(rate={same['false_positive_rate']:.6f}), "
                f"conditional MAE={fmt(same['errors']['candidate'])}; "
                f"descriptive precision={row['descriptive_precision_on_scored_pairs']}",
                flush=True,
            )
            print(f"    switch eligibility/rejections: {switch['reasons']}",
                  flush=True)
            print(f"    same-event eligibility/rejections: {same['reasons']}",
                  flush=True)
            for example in switch["largest_candidate_errors"][:5]:
                print(
                    f"    switch example anchor={example['anchor']:02d} "
                    f"frame={example['frame']} "
                    f"error={example['candidate_error_ms']:.4f}ms "
                    f"GRU={example['gru_error_ms']:.4f}ms",
                    flush=True,
                )
    print(
        "Counterfactual only: no Geometry/Hybrid change or adoption, no training,"
        " Stage A FAIL, no Motor/Validation/FINAL.",
        flush=True,
    )


if __name__ == "__main__":
    main()
