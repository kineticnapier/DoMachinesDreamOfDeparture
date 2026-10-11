from __future__ import annotations

"""Read-only Student timing representation / physical motor diagnostic."""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.timing_motor_probe import run_timing_motor_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only diagnostic linear readouts; never update Student."
    )
    parser.add_argument(
        "--config",
        default="configs/training/human_visible_timing_motor_probe.toml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("diagnostics/timing_motor_school8h"),
        help="Directory for JSON, CSV and compressed full-frame JSONL.",
    )
    parser.add_argument("--readout-steps", type=int, default=250)
    parser.add_argument("--skip-teacher-takeover", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    if config.mode != "human_visible_curriculum_dagger":
        raise SystemExit("requires human_visible_curriculum_dagger")
    prepared = _prepare(config)
    run_timing_motor_probe(
        prepared,
        output_dir=args.output_dir,
        readout_steps=args.readout_steps,
        with_teacher_takeover=not args.skip_teacher_takeover,
    )


if __name__ == "__main__":
    main()
