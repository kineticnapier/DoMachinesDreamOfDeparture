from __future__ import annotations

"""High-density and special-event stress audit for the frozen Timing Student."""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.evaluation.autoplay_sequence_v1 import (
    load_models, score_disjoint_frame_sequences,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import forecast_visible_window
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as simulator
from dmdod.training import prestige1_autoplay_readout as old_readout
from dmdod.training import prestige1_autoplay_sequence_student as sequence
from dmdod.training import prestige1 as frozen
from dmdod.training import prestige1_transfer as transfer


VERSION = "prestige1-autoplay-sequence-stress-v1"
PHASE_OFFSETS_MS = (0, 1, 2, 5)


def _special_fixture(kind: str):
    angles = [0.0] * 90
    actions = []
    if kind == "Pause":
        actions = [{"floor": 10, "eventType": "Pause", "duration": 2}]
    elif kind == "SetSpeed":
        actions = [{"floor": 10, "eventType": "SetSpeed", "speedType": "Bpm",
                    "beatsPerMinute": 8000}]
    elif kind == "Twirl":
        actions = [{"floor": 10, "eventType": "Twirl"}]
    elif kind == "FullTurn":
        angles = ([0.0, 0.0, 90.0, 90.0, 180.0, 180.0, 270.0, 270.0] * 14)[:90]
    else:
        raise ValueError("unknown special fixture")
    chart = compile_adofai(parse_adofai_text(json.dumps({
        "angleData": angles,
        "settings": {"bpm": 4000.0, "pitch": 100.0, "countdownTicks": 0},
        "actions": actions,
    })))
    return build_playable_segment(chart)


def _truth_rows(samples, target_times):
    rows = []
    for sample in samples:
        start = bisect_right(target_times, sample.seconds + 1e-9)
        end = bisect_right(target_times, sample.seconds + .010 + 1e-9)
        rows.append(tuple(value - sample.seconds for value in target_times[start:end]))
    return rows


def _evaluate_samples(samples, segment, model, threshold, device):
    target_times = [float(target.episode_time_s) for target in segment.targets]
    truth = _truth_rows(samples, target_times)
    visible = torch.tensor([row.hud for row in samples], dtype=torch.float32, device=device)
    with torch.no_grad():
        offsets, logits = model(visible)
    neural_rows = []
    neural_eligible = []
    for index in range(len(samples)):
        forecast = sequence.decode_forecast(
            offsets[index], logits[index], confidence_threshold=threshold
        )
        neural_rows.append(tuple(value for value in forecast.offsets_s if value <= .010 + 1e-7))
        neural_eligible.append(not forecast.abstained)
    baseline_rows = []
    baseline_eligible = []
    previous = None
    for sample in samples:
        forecast = forecast_visible_window(
            sample.hud, horizon_s=.010, previous_hud=previous
        )
        previous = sample.hud
        baseline_rows.append(tuple(forecast.times_s))
        baseline_eligible.append(forecast.eligible)
    return {
        "timing_v1_4": score_disjoint_frame_sequences(
            baseline_rows, truth, baseline_eligible),
        "new_student": score_disjoint_frame_sequences(
            neural_rows, truth, neural_eligible),
    }


def run(config_path: Path, checkpoint: Path, *, device: str) -> dict:
    config = frozen.PrestigeConfig.load(config_path)
    target_device = torch.device(device)
    output_root = transfer.output_dir(config)
    models, threshold, payload = load_models(
        checkpoint, output_root / old_readout.MLP_CHECKPOINT,
        output_root / old_readout.GRU_CHECKPOINT, device=target_device,
    )
    cases = []
    for bpm in (8000.0, 12000.0, 20000.0):
        for reverse in (False, True):
            segment, chart_sha = simulator._fixture(bpm, reverse=reverse)
            trace, _ = simulator._capture(segment)
            phases = {}
            for phase in PHASE_OFFSETS_MS:
                samples = trace[phase::10]
                phases[str(phase)] = _evaluate_samples(
                    samples, segment, models["new"], threshold, target_device
                )
            cases.append({"kind": "HighBPM", "bpm": bpm,
                          "direction": "ccw" if reverse else "cw",
                          "chart_sha256": chart_sha, "phases": phases})
            print(f"stress HighBPM {bpm:.0f} {'ccw' if reverse else 'cw'}", flush=True)
    for kind in ("Pause", "SetSpeed", "Twirl", "FullTurn"):
        segment = _special_fixture(kind)
        trace, _ = simulator._capture(segment)
        phases = {
            str(phase): _evaluate_samples(
                trace[phase::10], segment, models["new"], threshold, target_device
            ) for phase in PHASE_OFFSETS_MS
        }
        cases.append({"kind": kind, "bpm": 4000.0, "phases": phases})
        print("stress " + kind, flush=True)
    return {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_format": payload["format"],
        "confidence_threshold": threshold,
        "role": "deterministic simulator stress; not holdout generalization",
        "control_ms": 10.0, "physics_ms": 1.0,
        "phase_offsets_ms": PHASE_OFFSETS_MS,
        "cases": cases,
        "qualification_evaluated": False, "reserve_evaluated": False,
        "stage_a_gate_overridden": False,
    }


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
    if args.output.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output/status")
    started = datetime.now(timezone.utc)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.config, args.checkpoint, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        with args.status_file.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at_utc": started.isoformat(),
                       "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                       "exit_code": 0, "report": str(args.output)}, stream, indent=2)
    except BaseException as error:
        if not args.status_file.exists():
            with args.status_file.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at_utc": started.isoformat(),
                           "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                           "exit_code": 1, "report": str(args.output),
                           "error_type": type(error).__name__, "error": str(error)}, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
