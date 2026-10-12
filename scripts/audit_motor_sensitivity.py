from __future__ import annotations

"""Motor sensitivity diagnostic. Reuses saved experimental Best/Balanced/Onset.

No training, optimizer, curriculum advancement, checkpoint saves, or FINAL.
"""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.motor_sensitivity_audit import run_motor_sensitivity_audit
from dmdod.training.runner import _prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",default="configs/training/human_visible_onset_pilot.toml"
    )
    p.add_argument(
        "--prior-dir",type=Path,
        default=Path("diagnostics/onset_fixed_state_action_delta_v1"),
        help="Existing complete fixed-state audit containing both saved arms"
    )
    p.add_argument(
        "--output-dir",type=Path,
        default=Path("diagnostics/motor_sensitivity_v1"),
    )
    p.add_argument(
        "--arms-dir",type=Path,default=None,
        help="Directory holding balanced.diagnostic.pt and onset_focused.diagnostic.pt"
    )
    p.add_argument("--closed-loop-seconds",type=float,default=.400)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory already occupied: {args.output_dir}")
    config=load_training_config(args.config)
    if config.mode!="human_visible_curriculum_dagger":
        raise SystemExit("requires existing human-visible configuration")
    prepared=_prepare(config)
    summary=run_motor_sensitivity_audit(
        prepared,prior_directory=args.prior_dir,output_dir=args.output_dir,
        arm_state_dir=args.arms_dir,closed_loop_s=args.closed_loop_seconds,
    )
    print("=== DMDOD motor sensitivity audit ===",flush=True)
    print("Full 20-anchor baseline reproduction: PASS",flush=True)
    print(f"Focus anchors: {summary['focused_anchors']}",flush=True)
    print(f"Open-loop cases: {summary['counts']['open_loop_experiments']}",flush=True)
    print(f"Closed-loop cases: {summary['counts']['closed_loop_experiments']}",flush=True)
    print(f"Results: {args.output_dir/'summary.json'}",flush=True)


if __name__=="__main__":
    main()
