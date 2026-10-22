from __future__ import annotations

"""Fixed three-arm Free/v0.2/v0.3 full-chart fingering comparison."""

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
import run_fingering_continuity_comparison_v02 as v02_compare
import run_full_chart_fingering_replay_v02 as runner

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.evaluation.fingering_continuity_v03 import common_fast_metrics
from dmdod.planning.fingering_continuity_v02 import continuity_metrics
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-fingering-continuity-comparison-v0.3"
POLICY_ORDER = ("free", "continuity", "continuity_v03")


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def policy_summary(result: dict) -> dict:
    errors = [float(row["actual_error_ms"]) for row in result["decisions"]
              if row.get("actual_error_ms") is not None]
    stats = dict(result["stats"])
    signed = stats.get("signed_timing_error_ms")
    if signed is None and errors:
        signed = sum(errors) / len(errors)
    return {
        "stats": stats,
        "signed_timing_error_ms": signed,
        "mean_arrival_headroom_ms": (-sum(errors) / len(errors) if errors else None),
        "minimum_arrival_headroom_ms": (-max(errors) if errors else None),
        "fast_network_guard_count": sum(
            bool(row.get("fast_network_guard_applied")) for row in result["decisions"]
        ),
        "continuity": continuity_metrics(result["decisions"]),
        "termination_reason": result["termination_reason"],
        "replay_end_s": result["replay_end_s"],
    }


def _pairwise(left: dict, right: dict) -> dict:
    left_by = {int(row["event_index"]): row for row in left["decisions"]}
    right_by = {int(row["event_index"]): row for row in right["decisions"]}
    common = sorted(left_by.keys() & right_by.keys())
    different = [index for index in common
                 if left_by[index].get("chosen") != right_by[index].get("chosen")]
    lost = [index for index in common
            if left_by[index].get("outcome") == "Hit"
            and right_by[index].get("outcome") != "Hit"]
    recovered = [index for index in common
                 if left_by[index].get("outcome") != "Hit"
                 and right_by[index].get("outcome") == "Hit"]
    return {
        "common_event_count": len(common),
        "different_event_count": len(different),
        "different_event_rate": len(different) / len(common) if common else 0.0,
        "lost_left_hit_events": lost,
        "recovered_hit_events": recovered,
        "hit_non_regression": not lost,
    }


def build_payload(chart_path: Path, checkpoint: Path, compiled, segment,
                  results: dict[str, dict]) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_policy = {
        name: {int(row["event_index"]): row for row in result["decisions"]}
        for name, result in results.items()
    }
    replay_end = max(float(result["replay_end_s"]) for result in results.values())
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + .035:
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
            "decisions": {
                name: v02_compare._decision_view(rows[name], None)
                for name in POLICY_ORDER
            },
        })
    labels = {"free": "Free", "continuity": "Continuity v0.2",
              "continuity_v03": "Continuity v0.3"}
    return {
        "format": FORMAT,
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "default_policy": "continuity_v03",
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
            name: {"label": labels[name], "style_alpha": 0.0,
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
            "All three policies use independent Motor episodes and the same frozen checkpoint.",
            "Continuity v0.3 is an untrained causal selector with a fixed fast-cadence Network guard.",
            "No Teacher rollout, training, Timing change, or physical-constant change is used.",
        ],
    }


def run(chart_path: Path, checkpoint: Path, output_dir: Path, *,
        stop_after_s: float | None, wall_timeout_s: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if chart_path.suffix.lower() != ".adofai":
        raise ValueError("chart must be a .adofai file")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    model = runner.load_model(checkpoint)
    started = time.perf_counter()
    results = {
        name: runner.run_network(
            segment, model, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s, selector_mode=name, policy_name=name,
        )
        for name in POLICY_ORDER
    }
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_times = {event.event_index: event.episode_time_s for event in oracle.events}
    decisions = {name: result["decisions"] for name, result in results.items()}
    summaries = {name: policy_summary(result) for name, result in results.items()}
    report = {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "duration_s": segment.duration_s,
        "events": len(oracle.events),
        "runtime_s": round(time.perf_counter() - started, 3),
        "policies": summaries,
        "pairwise": {
            "free_to_v02": _pairwise(results["free"], results["continuity"]),
            "free_to_v03": _pairwise(results["free"], results["continuity_v03"]),
            "v02_to_v03": _pairwise(results["continuity"], results["continuity_v03"]),
        },
        "common_fast": common_fast_metrics(decisions, event_times),
        "training_performed": False,
        "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False,
        "timing_source": "AutoPlay Oracle exact event times",
        "hyperparameter_search": False,
        "v03_fixed_rule": "At <=35 ms cadence, continuity selection is restricted to the frozen Network's top two eligible candidates.",
    }
    payload = build_payload(chart_path, checkpoint, compiled, segment, results)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison-v03.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v03.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v03.html", viewer.render_html(payload))
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
        status = {
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "output_dir": str(args.output_dir),
            "terminations": {name: report["policies"][name]["termination_reason"]
                             for name in POLICY_ORDER},
        }
    except BaseException as error:
        status = {
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 1, "error_type": type(error).__name__,
            "error": str(error), "output_dir": str(args.output_dir),
        }
        _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
        raise
    _write_once(args.status_file, json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
