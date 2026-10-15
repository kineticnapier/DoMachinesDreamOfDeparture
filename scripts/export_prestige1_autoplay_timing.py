from __future__ import annotations

"""Export exact Autoplay timing truth (frozen Train20 ONLY, no training)."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_export import export_train_autoplay_timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = export_train_autoplay_timing(PrestigeConfig.load(args.config))
    print(
        f"Auto Timing: {result['total_targets']} targets; "
        f"{result['total_distinct_onsets']} exact onsets; "
        f"max physical-truth delta="
        f"{result['max_cross_source_error_ms']:.9f}ms; "
        f"Stage A overridden={result['stage_a_gate_overridden']}; "
        f"Student trained={not result['no_training']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
