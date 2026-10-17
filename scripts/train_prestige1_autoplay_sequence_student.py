from __future__ import annotations

"""Train the independent multi-event Timing Student on frozen Train20 only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from dmdod.training import prestige1 as frozen
from dmdod.training import prestige1_alignment_audit as alignment
from dmdod.training import prestige1_autoplay_readout as old_readout
from dmdod.training import prestige1_autoplay_sequence_student as sequence
from dmdod.training import prestige1_transfer as transfer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, checkpoint: Path, report_path: Path, *, device: str) -> dict:
    config = frozen.PrestigeConfig.load(config_path)
    output_root = transfer.output_dir(config)
    old_report_path = output_root / old_readout.REPORT
    if not old_report_path.is_file():
        raise FileNotFoundError("saved exact-Autoplay readout is required")
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old_report.get("format") != old_readout.FORMAT:
        raise RuntimeError("saved MLP/GRU report format changed")
    old_checkpoints = {
        "mlp": output_root / old_readout.MLP_CHECKPOINT,
        "gru": output_root / old_readout.GRU_CHECKPOINT,
    }
    if old_report.get("experimental_checkpoint_sha256") != {
        name: sha256(path) for name, path in old_checkpoints.items()
    }:
        raise RuntimeError("saved MLP/GRU checkpoint hash mismatch")
    selected, manifest = alignment._train_manifest(config)
    if manifest != old_report.get("train_manifest"):
        raise RuntimeError("Train20 manifest changed")
    partition = old_report.get("train_partition", {})
    fit_ids = set(partition.get("fit_anchor_ids", []))
    heldout_ids = set(partition.get("heldout_anchor_ids", []))
    if fit_ids & heldout_ids or fit_ids | heldout_ids != set(range(1, 21)):
        raise RuntimeError("invalid frozen Train20 partition")
    captured = []
    for index, named in enumerate(selected, 1):
        item = sequence.capture_sequence(named, config, device="cpu")
        captured.append(item)
        print(
            f"sequence capture {index:02d}/20 frames={len(item.visible)} "
            f"events={len(item.exact_event_times_s)} multi10="
            f"{int((item.present[:, 1] & (item.offsets_s[:, 1] <= .010)).sum())}",
            flush=True,
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    fit = [sequence.sequence_to(captured[index - 1], target_device) for index in sorted(fit_ids)]
    heldout = [sequence.sequence_to(captured[index - 1], target_device) for index in sorted(heldout_ids)]
    torch.manual_seed(config.seed)
    model = sequence.MultiEventTimingStudent(hidden=config.temporal_hidden).to(target_device)
    history = sequence.train_student(model, fit, seed=config.seed)
    threshold, calibration = sequence.select_threshold(model, heldout)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        torch.save({
            "format": sequence.FORMAT,
            "model_state_dict": model.state_dict(),
            "hidden": config.temporal_hidden,
            "input_dim": frozen.INPUT_DIM,
            "horizon_s": sequence.HORIZON_S,
            "max_events": sequence.MAX_EVENTS,
            "confidence_threshold": threshold,
            "train_partition": partition,
            "train_manifest": manifest,
            "old_baseline_checkpoint_sha256": old_report["experimental_checkpoint_sha256"],
            "stage_a_gate_overridden": False,
            "motor_student_training_permitted": False,
        }, stream)
    report = {
        "format": sequence.FORMAT,
        "role": "frozen Train20 fit15 / prior Train-heldout5 calibration only",
        "teacher": "compiled Autoplay exact event timestamps; no grid rounding",
        "student_input": "causal current/past 263D visible HUD only",
        "horizon_ms": sequence.HORIZON_S * 1000,
        "max_events": sequence.MAX_EVENTS,
        "epochs": sequence.TRAIN_EPOCHS,
        "updates": len(history),
        "learning_rate": sequence.LEARNING_RATE,
        "loss_first": history[0],
        "loss_last": history[-1],
        "loss_min": min(history),
        "selected_confidence_threshold": threshold,
        "threshold_selection": "maximum slot F1, then fewer FP, then higher threshold; prior Train-heldout5 only",
        "calibration_candidates": calibration,
        "train_partition": partition,
        "train_manifest": manifest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "old_baseline_checkpoint_sha256": old_report["experimental_checkpoint_sha256"],
        "validation_evaluated": False,
        "final_evaluated": False,
        "development_evaluated": False,
        "qualification_evaluated": False,
        "reserve_evaluated": False,
        "stage_a_gate_overridden": False,
        "motor_student_training_permitted": False,
        "limitations": [
            "Expert-driven capture can expose a behavioral cue through visible motor-state channels.",
            "Train-heldout5 selected confidence and is not an unknown-chart claim.",
            "Event-level de-duplicated comparison is a separate locked evaluation.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/prestige1.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for path in (args.checkpoint, args.output, args.status_file):
        if path.exists():
            parser.error("refusing to overwrite output: " + str(path))
    started = datetime.now(timezone.utc)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.config, args.checkpoint, args.output, device=args.device)
        status = {
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "report": str(args.output),
            "checkpoint": str(args.checkpoint),
        }
        with args.status_file.open("x", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    except BaseException as error:
        if not args.status_file.exists():
            with args.status_file.open("x", encoding="utf-8") as stream:
                json.dump({
                    "pid": os.getpid(), "started_at_utc": started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": 1, "report": str(args.output),
                    "checkpoint": str(args.checkpoint),
                    "error_type": type(error).__name__, "error": str(error),
                }, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
