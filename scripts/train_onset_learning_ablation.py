from __future__ import annotations

"""10-update A/B pilot and an independent zero-variance-masked timing audit.

NO existing Best/Progress checkpoint is modified or promoted.
"""

import argparse
from pathlib import Path

import torch

from dmdod.training.config import load_training_config
from dmdod.training.onset_ablation import run_onset_learning_ablation
from dmdod.training.onset_readout_audit import run_masked_readout_audit
from dmdod.training.runner import _prepare


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/training/human_visible_onset_pilot.toml")
    parser.add_argument("--report",type=Path,
                        default=Path("reports/human_visible_onset_pilot.json"))
    parser.add_argument("--existing-logs",type=Path,
                        default=Path("diagnostics/timing_motor_school8h_run2"))
    parser.add_argument("--readout-report",type=Path,
                        default=Path("reports/human_visible_onset_readout.json"))
    parser.add_argument("--readout-steps",type=int,default=250)
    parser.add_argument("--skip-readout",action="store_true")
    args=parser.parse_args()

    # Fail before any optimizer step if any requested result would overwrite data.
    protected=[args.report,args.report.with_suffix(".csv")]
    if not args.skip_readout:
        protected.extend([args.readout_report,args.readout_report.with_suffix(".csv")])
    if len({p.resolve() for p in protected})!=len(protected):
        raise ValueError("Report paths conflict")
    if any(p.exists() for p in protected):
        raise FileExistsError("One or more onset pilot output reports already exist")

    config=load_training_config(args.config)
    prepared=_prepare(config)
    report=run_onset_learning_ablation(
        prepared,config,report_path=args.report
    )
    print(f"Pilot comparison saved: {args.report}",flush=True)
    if not args.skip_readout:
        audit=run_masked_readout_audit(
            args.existing_logs,args.readout_report,
            device=torch.device("cpu"),steps=args.readout_steps,
        )
        print(
            "Visible readout zero-variance mask: "
            f"removed={audit['mask']['removed']}/263 "
            f"Validation MAE raw={audit['validation_aggregate_raw']['all']} "
            f"masked={audit['validation_aggregate_masked']['all']} "
            f"ridge={audit['validation_aggregate_ridge']['all']}",
            flush=True,
        )
        print(f"Readout diagnostic saved: {args.readout_report}",flush=True)


if __name__=="__main__":
    main()
