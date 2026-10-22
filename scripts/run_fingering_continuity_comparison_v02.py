from __future__ import annotations

"""Run a fixed Free/Continuity A/B with the same Network and Motor inputs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_fingering_replay_viewer_v01 as v01
import build_fingering_replay_viewer_v03 as v03
import run_full_chart_fingering_replay_v02 as runner

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.fingering_continuity_v02 import continuity_metrics
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-fingering-continuity-comparison-v0.2"


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _decision_view(row: dict | None, other: dict | None) -> dict | None:
    result = v01._decision_view(row, other)
    if result is not None and row is not None:
        result.update({
            "selector_mode": row.get("selector_mode", "free"),
            "reason": row.get("selection_reason", "network_choice_retained"),
            "free_chosen": row.get("free_chosen"),
            "style_class": row.get("style_class"),
        })
    return result


def compare_results(free: dict, continuity: dict) -> dict:
    free_by = {int(row["event_index"]): row for row in free["decisions"]}
    continuity_by = {int(row["event_index"]): row for row in continuity["decisions"]}
    common = sorted(free_by.keys() & continuity_by.keys())
    different = [index for index in common
                 if free_by[index].get("chosen") != continuity_by[index].get("chosen")]
    lost = [index for index in common
            if free_by[index].get("outcome") == "Hit"
            and continuity_by[index].get("outcome") != "Hit"]
    recovered = [index for index in common
                 if free_by[index].get("outcome") != "Hit"
                 and continuity_by[index].get("outcome") == "Hit"]
    return {
        "common_event_count": len(common),
        "different_events": different,
        "different_event_count": len(different),
        "changed_event_rate": len(different) / len(common) if common else 0.0,
        "lost_baseline_hit_events": lost,
        "lost_baseline_hit_count": len(lost),
        "recovered_hit_events": recovered,
        "recovered_hit_count": len(recovered),
        "hit_non_regression": not lost,
        "free_continuity": continuity_metrics(free["decisions"]),
        "continuity_continuity": continuity_metrics(continuity["decisions"]),
    }


def build_payload(chart_path: Path, checkpoint: Path, compiled, segment,
                  free: dict, continuity: dict) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    free_by = {int(row["event_index"]): row for row in free["decisions"]}
    continuity_by = {int(row["event_index"]): row for row in continuity["decisions"]}
    replay_end = max(float(free["replay_end_s"]), float(continuity["replay_end_s"]))
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + .035:
            break
        left = free_by.get(event.event_index)
        right = continuity_by.get(event.event_index)
        events.append({
            "event_index": event.event_index,
            "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices),
            "chord_size": event.chord_size,
            "different": bool(left and right and left.get("chosen") != right.get("chosen")),
            "decisions": {
                "free": _decision_view(left, right),
                "continuity": _decision_view(right, left),
            },
        })
    return {
        "format": FORMAT,
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "default_policy": "continuity",
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
        "floors": v03.rotation_profiles(load_adofai(chart_path), compiled, segment),
        "events": events,
        "policies": {
            "free": {"label": "Free", "style_alpha": 0.0,
                     "trace": free["trace"], "keydowns": free["keydowns"],
                     "saved_metrics": free["stats"]},
            "continuity": {"label": "Continuity", "style_alpha": 0.0,
                           "trace": continuity["trace"],
                           "keydowns": continuity["keydowns"],
                           "saved_metrics": continuity["stats"]},
        },
        "key_overlay": {
            "mapping": [{"label": label, "contact": contact}
                        for label, contact in v03.KEY_DISPLAY_MAP],
            "mapping_semantics": "Display aliases, not verified OS-key bindings.",
        },
        "limitations": [
            "Continuity is a causal post-network selector, not a retrained model or human demonstration model.",
            "Only current eligibility and previously committed choices are used; no Teacher rollout is used.",
            "Axis names are Motor proxies and do not establish anatomical or OS-key equivalence.",
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
    free = runner.run_network(
        segment, model, stop_after_s=stop_after_s,
        wall_timeout_s=wall_timeout_s, selector_mode="free", policy_name="free")
    continuity = runner.run_network(
        segment, model, stop_after_s=stop_after_s,
        wall_timeout_s=wall_timeout_s, selector_mode="continuity",
        policy_name="continuity")
    comparison = compare_results(free, continuity)
    payload = build_payload(chart_path, checkpoint, compiled, segment, free, continuity)
    report = {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "duration_s": segment.duration_s,
        "events": len(AutoplayTimingTeacher.from_segment(segment).events),
        "runtime_s": round(time.perf_counter() - started, 3),
        "free": {"stats": free["stats"],
                 "termination_reason": free["termination_reason"],
                 "replay_end_s": free["replay_end_s"]},
        "continuity": {"stats": continuity["stats"],
                       "termination_reason": continuity["termination_reason"],
                       "replay_end_s": continuity["replay_end_s"]},
        "comparison": comparison,
        "training_performed": False,
        "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False,
        "timing_source": "AutoPlay Oracle exact event times",
        "hyperparameter_search": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v02.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v02.html", v03.render_html(payload))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=600.0)
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
            "free_termination": report["free"]["termination_reason"],
            "continuity_termination": report["continuity"]["termination_reason"],
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
