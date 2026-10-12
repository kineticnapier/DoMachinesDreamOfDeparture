from __future__ import annotations

"""Replay prior 10-update onset pilot and audit three frozen policies.

Only diagnostic-candidate checkpoints are emitted, never selected-best.
"""

import argparse
from pathlib import Path

from dmdod.training.config import load_training_config
from dmdod.training.fixed_state_action_audit import (
    ActionDeltaAudit, assert_state_equal, file_sha256,
)
from dmdod.training.onset_ablation import run_onset_learning_ablation
from dmdod.training.real_chart import clone_model_state
from dmdod.training.runner import _prepare


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",default="configs/training/human_visible_onset_pilot.toml"
    )
    parser.add_argument(
        "--output-dir",type=Path,
        default=Path("diagnostics/onset_fixed_state_action_delta_v1")
    )
    args=parser.parse_args()
    directory=args.output_dir
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"diagnostic output directory is not empty: {directory}")
    config=load_training_config(args.config)
    source=Path(config.run.checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"selected-best source does not exist: {source}")
    if source.resolve()==Path(config.run.output).resolve():
        raise ValueError("diagnostic output may not equal selected-best checkpoint")
    source_sha=file_sha256(source)
    prepared=_prepare(config)
    before=clone_model_state(prepared.model)
    directory.mkdir(parents=True,exist_ok=True)
    audit=ActionDeltaAudit(
        prepared,output_dir=directory,source_sha256=source_sha,
        seed=config.human_visible.seed,
    )
    report=directory/"pilot_reproduction.json"
    print(f"=== DMDOD fixed-state action delta audit ===\n"
          f"source={source} sha256={source_sha}\n"
          f"output={directory} (candidate checkpoints are diagnostic-only)",
          flush=True)
    result=run_onset_learning_ablation(
        prepared,config,report_path=report,audit=audit
    )
    audit_result=audit.complete(pilot_report=result)
    assert_state_equal(clone_model_state(prepared.model),before,
                       context="final Student original state")
    if file_sha256(source)!=source_sha:
        raise RuntimeError("selected-best source checkpoint sha256 changed")
    print("=== reproduction ===",flush=True)
    check=audit_result["reference_reproduction"]
    print("PASS" if check["match_all"] else "MISMATCH: inspect summary.json",
          flush=True)
    for arm,summary in audit_result["parameter_summaries"].items():
        print(f"{arm}: updated={summary['changed_parameter_elements']}/"
              f"{summary['trainable_parameter_elements']} "
              f"frozen/buffers={summary['frozen_and_buffer_state_entries_unchanged']} "
              "verified unchanged",flush=True)
    print(f"completed audit: {directory/'summary.json'}",flush=True)


if __name__=="__main__":
    main()
