from __future__ import annotations

"""DMDOGO v0.5 causal Continuity-v0.2 plus Motor compensation comparison."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_fingering_replay_viewer_v01 as v01
import build_fingering_replay_viewer_v03 as viewer
import run_dmdogo_v04_comparison as v04
import run_fingering_continuity_comparison_v03 as v03_compare
import run_full_chart_fingering_replay_v02 as baseline

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.evaluation.fingering_continuity_v03 import common_fast_metrics
from dmdod.planning.fingering_continuity_v02 import continuity_metrics
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-continuity-compensated-comparison-v0.5"
POLICY_ORDER = ("continuity", "latency_only", "integrated_v04",
                "continuity_compensated")
LABELS = {
    "continuity": "Continuity v0.2",
    "latency_only": "Latency-only",
    "integrated_v04": "Integrated v0.4",
    "continuity_compensated": "Continuity + Compensation v0.5",
}


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def run_v02_reference(segment, model, *, stop_after_s: float | None,
                      wall_timeout_s: float) -> dict:
    return baseline.run_network(
        segment, model, stop_after_s=stop_after_s,
        wall_timeout_s=wall_timeout_s, selector_mode="continuity",
        policy_name="continuity")


def run_continuity_arm(segment, model, *, compensation: bool,
                       stop_after_s: float | None, wall_timeout_s: float) -> dict:
    if not compensation:
        return run_v02_reference(
            segment, model, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s)
    return v04.run_research_arm(
        segment, model, policy_name="continuity_compensated",
        planning=False, compensation=True, selector_mode="continuity",
        stop_after_s=stop_after_s, wall_timeout_s=wall_timeout_s)


def _decision_view(row: dict | None) -> dict | None:
    return v04._decision_view(row)


def _payload(chart_path: Path, checkpoint: Path, compiled, segment,
             results: dict[str, dict]) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_policy = {name: {int(row["event_index"]): row for row in result["decisions"]}
                 for name, result in results.items()}
    replay_end = max(float(result["replay_end_s"]) for result in results.values())
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + .050:
            break
        rows = {name: by_policy[name].get(event.event_index) for name in POLICY_ORDER}
        choices = {row.get("chosen") for row in rows.values() if row is not None}
        events.append({
            "event_index": event.event_index,
            "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices),
            "chord_size": event.chord_size,
            "different": len(choices) > 1,
            "decisions": {name: _decision_view(rows[name]) for name in POLICY_ORDER},
        })
    return {
        "format": FORMAT, "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "default_policy": "continuity_compensated",
        "timebase": {
            "episode_duration_s": replay_end,
            "full_chart_duration_s": segment.duration_s,
            "segment_start_chart_s": segment.start_chart_s,
            "segment_end_chart_s": segment.end_chart_s,
            "pitch_percent": compiled.pitch_percent,
            "pitch_ratio": segment.pitch_ratio,
            "offset_ms": compiled.offset_ms,
            "separate_countdown_chart_s": compiled._separate_countdown_chart_seconds(),
        },
        "floors": viewer.rotation_profiles(load_adofai(chart_path), compiled, segment),
        "events": events,
        "policies": {
            name: {"label": LABELS[name], "style_alpha": 0.0,
                   "trace": results[name]["trace"],
                   "keydowns": results[name]["keydowns"],
                   "saved_metrics": results[name]["stats"]}
            for name in POLICY_ORDER
        },
        "key_overlay": {
            "mapping": [{"label": label, "contact": contact}
                        for label, contact in viewer.KEY_DISPLAY_MAP],
            "mapping_semantics": "Display aliases, not verified OS-key bindings.",
        },
        "limitations": [
            "Independent frozen-Motor episodes use exact AutoPlay times.",
            "v0.5 selects causally from current Motor state and committed continuity history.",
            "No future physical state, KeyDown, judgement, or Teacher rollout is read.",
        ],
    }


def run(chart_path: Path, checkpoint: Path, output_dir: Path, *,
        stop_after_s: float | None, wall_timeout_s: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    model = baseline.load_model(checkpoint)
    started = time.perf_counter()
    results = {
        "continuity": run_v02_reference(
            segment, model, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s),
        "latency_only": v04.run_research_arm(
            segment, model, policy_name="latency_only", planning=False,
            compensation=True, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s),
        "integrated_v04": v04.run_research_arm(
            segment, model, policy_name="integrated_v04", planning=True,
            compensation=True, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s),
        "continuity_compensated": run_continuity_arm(
            segment, model, compensation=True, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s),
    }
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_times = {event.event_index: event.episode_time_s for event in oracle.events}
    summaries = {}
    for name, result in results.items():
        summary = v03_compare.policy_summary(result)
        summary["compensation"] = v04._telemetry_summary(
            result.get("compensation_telemetry", []))
        summary["runtime"] = result.get("planner", {})
        summary["continuity"] = continuity_metrics(result["decisions"])
        summaries[name] = summary
    report = {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "events": len(oracle.events), "duration_s": segment.duration_s,
        "runtime_s": round(time.perf_counter() - started, 3),
        "policies": summaries,
        "pairwise_from_continuity": {
            name: v03_compare._pairwise(results["continuity"], results[name])
            for name in POLICY_ORDER[1:]
        },
        "common_fast": common_fast_metrics(
            {name: result["decisions"] for name, result in results.items()}, event_times),
        "frozen": {"network_checkpoint": True, "oracle_due_times": True,
                   "motor_constants": True, "scorer": True,
                   "existing_policy_semantics": True},
        "training_performed": False, "hyperparameter_search": False,
        "teacher_shadow_rollout_used": False,
    }
    payload = _payload(chart_path, checkpoint, compiled, segment, results)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison-v05.json", json.dumps(report, indent=2) + "\n")
    _write_once(output_dir / "replay-v05.json", json.dumps(payload, indent=2) + "\n")
    _write_once(output_dir / "replay-v05.html", viewer.render_html(payload))
    _write_once(output_dir / "telemetry-v05.json", json.dumps({
        name: {"decisions": result["decisions"],
               "compensation": result.get("compensation_telemetry", []),
               "runtime": result.get("planner", {})}
        for name, result in results.items()
    }, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        report = run(args.chart, args.checkpoint, args.output_dir,
                     stop_after_s=args.stop_after_s,
                     wall_timeout_s=args.wall_timeout_s)
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "exit_code": 0, "output_dir": str(args.output_dir),
                  "events": report["events"],
                  "policies": {name: row["termination_reason"]
                               for name, row in report["policies"].items()}}
    except BaseException as error:
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "exit_code": 1, "output_dir": str(args.output_dir),
                  "error_type": type(error).__name__, "error": str(error)}
        _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
        raise
    _write_once(args.status_file, json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
