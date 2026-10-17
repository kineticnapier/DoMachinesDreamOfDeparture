from __future__ import annotations

"""Locked unknown-chart comparison for causal Autoplay Timing Students."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from dmdod.data.multichart import discover_multichart_dataset
from dmdod.evaluation.autoplay_sequence_v1 import (
    aggregate_method_rows, load_models, produce_predictions, score_predictions,
)
from dmdod.evaluation.jrp_unseen_split_v2 import select_unseen_charts
from dmdod.training import prestige1 as frozen
from dmdod.training import prestige1_autoplay_readout as old_readout
from dmdod.training import prestige1_autoplay_sequence_student as sequence
from dmdod.training import prestige1_transfer as transfer
from dmdod.training.real_chart import compile_role, build_validation_segments


VERSION = "prestige1-autoplay-sequence-unknown-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualification_guard(path: Path | None, checkpoint: Path) -> None:
    if path is None or not path.is_file():
        raise RuntimeError("Qualification requires an explicit completed Development report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("version") != VERSION
        or report.get("partition") != "development"
        or report.get("qualification_permitted") is not True
        or report.get("new_checkpoint_sha256") != sha256(checkpoint)
    ):
        raise RuntimeError("Development did not license Qualification")


def audit(
    config_path: Path,
    checkpoint: Path,
    *,
    device: str,
    partition: str,
    development_report: Path | None,
) -> dict:
    if partition == "qualification":
        _qualification_guard(development_report, checkpoint)
    config = frozen.PrestigeConfig.load(config_path)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_root = transfer.output_dir(config)
    old_paths = {
        "mlp": output_root / old_readout.MLP_CHECKPOINT,
        "gru": output_root / old_readout.GRU_CHECKPOINT,
    }
    models, threshold, payload = load_models(
        checkpoint, old_paths["mlp"], old_paths["gru"], device=target_device
    )
    trained_hashes = {row["sha256"] for row in payload["train_manifest"]}
    dataset = discover_multichart_dataset(config.dataset)
    selected = select_unseen_charts(dataset.train, trained_hashes, partition)
    segments = build_validation_segments(
        compile_role(selected), window_s=config.validation_window_s
    )
    method_rows = {name: [] for name in ("timing_v1_4", "mlp", "gru", "new_student")}
    chart_rows = []
    for index, named in enumerate(segments, 1):
        captured = sequence.capture_sequence(named, config, device="cpu")
        accelerated = sequence.sequence_to(captured, target_device)
        legacy = sequence.legacy_comparator_sequence(accelerated)
        predictors = {
            "timing_v1_4": (None, captured, "v1.4", 1.0),
            "mlp": (models["mlp"], legacy, "old", 0.5),
            "gru": (models["gru"], legacy, "old", 0.5),
            "new_student": (models["new"], accelerated, "new", threshold),
        }
        chart_result = {
            "index": index, "chart": named.chart_name,
            "chart_sha256": named.chart_sha256,
            "window_s": [named.start_s, named.end_s],
            "frames": len(captured.visible),
            "exact_events": len(captured.exact_event_times_s),
            "multi_event_10ms_frames": int(
                (captured.present[:, 1] & (captured.offsets_s[:, 1] <= .010)).sum()
            ),
            "label_overflow_frames": int(captured.overflow.sum()),
            "methods": {},
        }
        for name, (model, source, kind, cutoff) in predictors.items():
            predictions, frame_metrics = produce_predictions(
                source, model, kind=kind, threshold=cutoff
            )
            score = score_predictions(captured, predictions)
            value = {"frame_metrics": frame_metrics, "score": score}
            chart_result["methods"][name] = value
            method_rows[name].append(value)
        chart_rows.append(chart_result)
        primary = {name: chart_result["methods"][name]["score"]
                   ["practical_10ms_tolerance_10_80ms_lead"] for name in method_rows}
        print(
            f"{partition} {index:02d}/20 {named.chart_name}: "
            + " ".join(f"{name}={value['matches']}/{value['targets']} fp={value['false_predictions']}"
                       for name, value in primary.items()),
            flush=True,
        )
    aggregate = {name: aggregate_method_rows(rows) for name, rows in method_rows.items()}
    baseline = aggregate["timing_v1_4"]["practical_10ms_tolerance_10_80ms_lead"]
    candidate = aggregate["new_student"]["practical_10ms_tolerance_10_80ms_lead"]
    reasons = []
    if candidate["f1"] < baseline["f1"] + 0.02:
        reasons.append("practical_f1_not_0.02_better_than_timing_v1_4")
    if candidate["false_predictions_per_target"] > baseline["false_predictions_per_target"]:
        reasons.append("false_predictions_per_target_regressed")
    if candidate["mae_ms"] is None or candidate["mae_ms"] > 10.0:
        reasons.append("practical_mae_above_10ms")
    if candidate["median_matched_lead_ms"] is None or candidate["median_matched_lead_ms"] < 20.0:
        reasons.append("median_matched_lead_below_20ms")
    passed = not reasons
    return {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": partition,
        "status": "PASS" if passed else "FAIL",
        "qualification_permitted": partition == "development" and passed,
        "reserve_opened": False,
        "selection": {
            "source_role": "Train", "checkpoint_train_excluded": True,
            "ordering": "content_sha256", "partition_sizes": [20, 20, 20],
            "window": "middle 8 seconds", "no_metric_based_chart_selection": True,
            "final10_is_not_holdout": True,
        },
        "student_contract": {
            "input": "causal current/past 263D visible HUD v2 only; legacy encoding only for frozen old baselines",
            "output": "up to 16 ordered future events, confidence, abstention",
            "horizon_ms": 80.0, "control_ms": 10.0, "physics_ms": 1.0,
            "truth_used_posthoc_only": True,
        },
        "primary_endpoint": (
            "unique-event F1 at 10ms tolerance and 10-80ms first-prediction lead; "
            ">=0.02 above Timing v1.4, no false/target regression, MAE<=10ms, median lead>=20ms"
        ),
        "failure_reasons": reasons,
        "new_checkpoint": str(checkpoint),
        "new_checkpoint_sha256": sha256(checkpoint),
        "old_checkpoint_sha256": {name: sha256(path) for name, path in old_paths.items()},
        "selected_confidence_threshold": threshold,
        "aggregate": aggregate,
        "charts": chart_rows,
        "stage_a_gate_overridden": False,
        "motor_student_training_permitted": False,
        "validation_role_evaluated": False,
        "final_role_evaluated": False,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/prestige1.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("development", "qualification"), required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output/status")
    started = datetime.now(timezone.utc)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(
            args.config, args.checkpoint, device=args.device,
            partition=args.partition, development_report=args.development_report,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        code = 0 if report["status"] == "PASS" else 1
        with args.status_file.open("x", encoding="utf-8") as stream:
            json.dump({
                "pid": os.getpid(), "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": code, "report": str(args.output),
            }, stream, indent=2)
        print(json.dumps({
            "status": report["status"], "partition": report["partition"],
            "failure_reasons": report["failure_reasons"],
            "aggregate": report["aggregate"],
        }, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(code)
    except SystemExit:
        raise
    except BaseException as error:
        if not args.status_file.exists():
            with args.status_file.open("x", encoding="utf-8") as stream:
                json.dump({
                    "pid": os.getpid(), "started_at_utc": started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": 1, "report": str(args.output),
                    "error_type": type(error).__name__, "error": str(error),
                }, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
