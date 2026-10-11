from __future__ import annotations

"""Offline Train/Validation readout audit + replay of only initial 200ms.

Usage: uv run python -u scripts/probe_timing_motor_v2.py
Results never overwrite the existing run2 diagnostic directory.
"""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.runner import _prepare
from dmdod.training.timing_motor_probe_v2 import run_probe_v2


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing",type=Path,
                        default=Path("diagnostics/timing_motor_school8h_run2"))
    parser.add_argument("--output-dir",type=Path,
                        default=Path("diagnostics/timing_motor_school8h_v2"))
    parser.add_argument("--config",default="configs/training/human_visible_timing_motor_probe.toml")
    parser.add_argument("--readout-steps",type=int,default=250)
    parser.add_argument("--offline-only",action="store_true",
                        help="No model loading or replay. Readout diagnosis from existing logs only.")
    args=parser.parse_args()
    # Offline readouts operate entirely on pre-existing Train/Validation logs.
    prepared=None
    if not args.offline_only:
        config=load_training_config(args.config)
        if config.mode!="human_visible_curriculum_dagger":
            raise SystemExit("v2 requires human_visible_curriculum_dagger")
        prepared=_prepare(config)
        if (prepared.key_count,prepared.input_dim)!=(8,263):
            raise SystemExit("v2 requires 8 keys / 263D input")
        if len(prepared.anchors)!=20 or len(prepared.validation)!=20:
            raise SystemExit("v2 requires fixed 20 Train / 20 Validation anchors")
    run_probe_v2(args.existing,args.output_dir,prepared=prepared,
                 steps=args.readout_steps,skip_replay=args.offline_only)


if __name__=="__main__":
    main()
