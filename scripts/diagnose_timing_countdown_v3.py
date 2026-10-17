from __future__ import annotations

"""Diagnose v3 same-event countdown consistency on simple BPM fixtures."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.evaluation.timing_countdown_v3 import (
    evaluate_countdown, infer_neural, infer_v14,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.training import prestige1 as prestige
from dmdod.training import prestige1_autoplay_sequence_student as student


VERSION = "dmdod-neural-timing-countdown-diagnostics-v1"
BPMS = (120.0, 240.0, 480.0)
FLOOR_COUNT = 12
CONTROL_DT_S = 0.010
PHYSICS_DT_S = 0.001


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simple_fixture(bpm: float):
    if bpm not in BPMS:
        raise ValueError("only the fixed 120/240/480 BPM fixtures are permitted")
    chart_text = json.dumps({
        "angleData": [0.0] * FLOOR_COUNT,
        "settings": {
            "bpm": bpm, "pitch": 100.0, "countdownTicks": 0,
            "separateCountdownTime": False,
        },
        "actions": [],
    }, sort_keys=True, separators=(",", ":"))
    chart = compile_adofai(parse_adofai_text(chart_text))
    segment = build_playable_segment(chart, start_s=0.0, end_s=chart.duration_s)
    if len(segment.targets) != FLOOR_COUNT:
        raise RuntimeError("simple fixture target count drift")
    return segment, sha256(chart_text.encode("utf-8")).hexdigest()


def capture_hud(segment):
    """Capture genuine 10ms/1ms HUD frames with no truth-driven motor action."""

    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=prestige.KEY_COUNT,
        control_dt_s=CONTROL_DT_S, physics_dt_s=PHYSICS_DT_S,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=0.040,
    )
    observation = env.reset()
    idle = NKeyAction((0.0,) * prestige.KEY_COUNT)
    hud_rows = []
    frame_times = []
    bound = int((segment.duration_s + 0.100) / CONTROL_DT_S) + 20
    for _ in range(bound):
        now_s = float(env.privileged_episode_time_s())
        hud = tuple(float(value) for value in encode_n_key_hud_real_chart_observation(observation))
        if len(hud) != prestige.INPUT_DIM:
            raise RuntimeError("renderer did not produce a 263D HUD")
        frame_times.append(now_s)
        hud_rows.append(hud)
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    else:
        raise RuntimeError("simple fixture exceeded bounded runtime")
    return tuple(hud_rows), tuple(frame_times)


def load_v3(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format") != student.FORMAT:
        raise RuntimeError("Timing v3 checkpoint format mismatch")
    model = student.MultiEventTimingStudent(hidden=int(payload["hidden"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), float(payload["confidence_threshold"]), payload


def run(checkpoint: Path, *, device: str) -> dict:
    target_device = torch.device(device)
    model, threshold, payload = load_v3(checkpoint, target_device)
    cases = []
    for bpm in BPMS:
        segment, chart_sha = simple_fixture(bpm)
        hud_rows, frame_times = capture_hud(segment)

        # Inference finishes before exact event timestamps are read below.
        visible = torch.tensor(hud_rows, dtype=torch.float32, device=target_device)
        neural_forecasts = infer_neural(
            visible, frame_times, model, threshold=threshold
        )
        baseline_forecasts = infer_v14(hud_rows, frame_times)

        # EVALUATOR-ONLY boundary: truth is materialized only after both
        # HUD-only inference passes and their causal tracker traces complete.
        exact_event_times = tuple(float(target.episode_time_s) for target in segment.targets)
        cases.append({
            "bpm": bpm,
            "chart_sha256": chart_sha,
            "frames": len(frame_times),
            "targets": len(exact_event_times),
            "student_v3": evaluate_countdown(neural_forecasts, exact_event_times),
            "timing_v1_4": evaluate_countdown(baseline_forecasts, exact_event_times),
        })
        print(f"countdown {bpm:.0f} BPM: {len(frame_times)} frames", flush=True)

    methods = {}
    for method in ("student_v3", "timing_v1_4"):
        aggregates = [case[method]["aggregate"] for case in cases]
        total_frames = sum(row["frames"] for row in aggregates)
        methods[method] = {
            "events": sum(row["events"] for row in aggregates),
            "frames": total_frames,
            "presence_coverage": sum(
                row["presence_coverage"] * row["frames"] for row in aggregates
            ) / total_frames,
            "raw_due_mae_ms": sum(
                row["raw_due_mae_ms"] * row["events"] for row in aggregates
            ) / sum(row["events"] for row in aggregates),
            "raw_countdown_step_mae_ms": sum(
                row["raw_countdown_step_mae_ms"] * row["events"] for row in aggregates
            ) / sum(row["events"] for row in aggregates),
            "disappearances": sum(row["disappearances"] for row in aggregates),
            "reappearances": sum(row["reappearances"] for row in aggregates),
            "monotonic_violations": sum(row["monotonic_violations"] for row in aggregates),
            "tracker_id_switches_within_20ms": sum(
                row["tracker_id_switches_within_20ms"] for row in aggregates
            ),
            "events_with_multiple_tracker_ids_within_20ms": sum(
                row["events_with_multiple_tracker_ids_within_20ms"] for row in aggregates
            ),
        }
    v3 = methods["student_v3"]
    time_failure = (
        v3["raw_countdown_step_mae_ms"] > 2.0 or v3["raw_due_mae_ms"] > 10.0
    )
    reassociation_observed = v3["events_with_multiple_tracker_ids_within_20ms"] > 0
    methods["student_v3"]["failure_attribution"] = {
        "event_existence": v3["presence_coverage"] < 0.80,
        "time_regression": time_failure,
        "forecast_reassociation_observed": reassociation_observed,
        "forecast_reassociation_independent": reassociation_observed and not time_failure,
    }
    return {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_format": payload["format"],
        "confidence_threshold": threshold,
        "fixture_bpms": BPMS,
        "control_ms": CONTROL_DT_S * 1000.0,
        "physics_ms": PHYSICS_DT_S * 1000.0,
        "hud_dimension": prestige.INPUT_DIM,
        "capture_motor_policy": "all-zero idle; no truth-driven motor cue",
        "causality_contract": {
            "student_inputs": "current/past 263D HUD only",
            "truth_materialization": "after neural and v1.4 inference and tracker traces",
            "truth_use": "posthoc evaluator event association and error metrics only",
            "future_or_chart_internal_input_to_student": False,
        },
        "large_retraining_performed": False,
        "qualification_evaluated": False,
        "reserve_evaluated": False,
        "stage_a_gate_overridden": False,
        "methods": methods,
        "cases": cases,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output/status")
    started = datetime.now(timezone.utc)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.checkpoint, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        status = {
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "report": str(args.output),
        }
        with args.status_file.open("x", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2)
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
