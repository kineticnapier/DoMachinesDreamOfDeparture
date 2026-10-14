from __future__ import annotations

"""Prestige I Stage C v2.5 — Train20 Teacher timing/label quality audit."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_teacher_quality import run_train_teacher_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    report = run_train_teacher_quality(PrestigeConfig.load(args.config))
    before = report["timing_quality_baseline"]
    after = report["timing_quality_candidate"]
    counts = report["alignment_aggregate"]
    print(
        "Stage C v2.5 Train20: "
        f"Hit-only MAE baseline={before['mae_ms']:.3f}ms "
        f"candidate={after['mae_ms']:.3f}ms; "
        f"candidate perfect-family={after['perfect_family_rate']:.2%}; "
        f"mismatch={counts['mismatch_frames']}/{counts['frames']}; "
        f"protected={report['protected_inputs_unchanged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
