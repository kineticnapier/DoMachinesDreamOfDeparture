from __future__ import annotations

"""Run Fingering Network v0.1 on a selected ADOFAI chart and export replay data.

AutoPlay supplies event times only.  Network selection never calls the v0.2
Teacher counterfactual rollout/ranker.  Outputs are create-once artifacts.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training import fingering_network_v01 as network
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdod-full-chart-fingering-replay-v0.2"


def _write_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def load_model(checkpoint: Path) -> network.FingeringRanker:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != network.FORMAT:
        raise ValueError("unexpected Fingering Network checkpoint format")
    model = network.FingeringRanker(hidden=int(payload["hidden"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _trace_frame(env: CoupledContactRealChartEnv) -> list:
    frame = viewer._trace_frame(env)
    stats = env.stats
    frame.append({
        "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses,
        "overloaded": bool(stats.overloaded),
    })
    return frame


def _summarize(env: CoupledContactRealChartEnv) -> dict:
    stats = env.stats
    return {
        "targets": stats.targets, "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses, "overloaded": bool(stats.overloaded),
        "overload_counter": stats.overload_counter,
        "x_accuracy_percent": stats.x_accuracy_percent,
        "timing_mae_ms": stats.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
    }


def run_network(segment, model: network.FingeringRanker, *, stop_after_s: float | None = None,
                wall_timeout_s: float | None = None, style_alpha: float = 0.0,
                policy_name: str = "network_v0_1_free") -> dict:
    """Execute the causal Network policy; no Teacher rollout exists here."""

    env = CoupledContactRealChartEnv(segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    decisions: list[dict] = []
    keydowns: list[dict] = []
    frames = [_trace_frame(env)]
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    wall_started = time.monotonic()
    step_done = False
    for _ in range(maximum):
        if wall_timeout_s is not None and time.monotonic() - wall_started > wall_timeout_s:
            raise TimeoutError(f"physical execution exceeded {wall_timeout_s:.1f}s wall timeout")
        now = env.privileged_episode_time_s()
        committed = {
            index: item for index, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[o] for o in events[index].target_ordinals)
        }
        upcoming = tuple(event for event in events if event.event_index not in selected
                         and event.episode_time_s + teacher.PREVIEW_TAIL >= now)
        due = tuple(event for event in upcoming
                    if event.episode_time_s <= now + teacher.LEAD + 1e-9)
        for event in due:
            if event.event_index in selected:
                continue
            occupied = {axis for item in committed.values() for axis in item.axes}
            look = tuple(value for value in upcoming if value.event_index >= event.event_index)
            allocations = teacher.finger_options(event.chord_size)
            eligible = tuple(teacher.eligible(env.motor, occupied, value)[0]
                             for value in allocations)
            context = network.encode_context(env.motor, look, now, occupied)
            selected_index, scores, blended = network.choose_candidate_styled(
                model, context, allocations, eligible, alpha=style_alpha)
            free_indices = [index for index, value in enumerate(scores) if value is not None]
            free_index = min(free_indices, key=lambda index: (scores[index], index)) if free_indices else None
            allocation = allocations[selected_index] if selected_index is not None else ()
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
            decisions.append({
                "event_index": event.event_index,
                "ordinals": list(event.target_ordinals),
                "due_ms": round(event.episode_time_s * 1000.0, 3),
                "decision_ms": round(now * 1000.0, 3),
                "chosen": item.label if allocation else None,
                "axes": [finger.axis for finger in allocation],
                "network_scores": [value for value in scores],
                "blended_rank_scores": [value for value in blended],
                "eligible": list(eligible),
                "inference_inputs": "oracle_relative_times_and_current_motor_only",
                "policy": policy_name,
                "style_alpha": style_alpha,
                "free_chosen": (teacher.PlanItem(event.event_index, event.episode_time_s,
                                                  allocations[free_index]).label
                                if free_index is not None else None),
                "different_from_free": selected_index != free_index,
                "style_class": ("index_proxy" if allocation and all(f.axis.endswith("_1") for f in allocation)
                                else "ordinary_proxy" if allocation and all(int(f.axis.split("_")[1]) <= 4 for f in allocation)
                                else "thumb_or_side" if allocation else "unassigned"),
            })
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                keydowns.append({
                    "episode_s": round(event.time_s, 6),
                    "time_ms": round(event.time_s * 1000.0, 3),
                    "axis": viewer._physical_axis(event.key),
                    "contact": event.key,
                })
        frames.append(_trace_frame(env))
        step_done = step.done
        if step.done or (stop_after_s is not None and env.privileged_episode_time_s() >= stop_after_s):
            break
    else:
        raise RuntimeError("full-chart Network execution exceeded step budget")

    for row in decisions:
        used = [env._used[index] for index in row["ordinals"]]
        missed = [env._missed[index] for index in row["ordinals"]]
        row["outcome"] = "Hit" if all(used) else "Miss" if any(missed) else "Unresolved"
        nearby = [value for value in keydowns if value["axis"] in row["axes"]
                  and abs(value["time_ms"] - row["due_ms"]) <= 35.0]
        row["keydowns"] = [
            min((value for value in nearby if value["axis"] == axis),
                key=lambda value: abs(value["time_ms"] - row["due_ms"]))
            for axis in row["axes"] if any(value["axis"] == axis for value in nearby)
        ]
        errors = [value["time_ms"] - row["due_ms"] for value in row["keydowns"]]
        row["actual_error_ms"] = round(min(errors, key=abs), 3) if errors else None
        early = [trace for trace in env.judgement_trace
                 if trace["target_index"] in row["ordinals"] and trace["margin"] is None
                 and trace["time_s"] * 1000.0 < row["due_ms"]]
        row["too_early_keydowns"] = len(early)
        row["judgement"] = ("TooEarly + " + row["outcome"]) if early else row["outcome"]

    stats = _summarize(env)
    end_s = env.privileged_episode_time_s()
    if stop_after_s is not None and not step_done:
        reason = "diagnostic_prefix_complete"
    elif stats["overloaded"]:
        reason = "overload"
    elif stats["misses"] and end_s + teacher.CTRL < segment.duration_s:
        reason = "miss"
    else:
        reason = "chart_complete"
    return {
        "stats": stats, "decisions": decisions, "keydowns": keydowns,
        "trace": frames, "replay_end_s": round(end_s, 6),
        "termination_reason": reason,
        "failure_continuation": False,
        "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False,
        "policy": policy_name, "style_alpha": style_alpha,
    }


def build_payload(chart_path: Path, difficulty: str, checkpoint: Path,
                  compiled, segment, result: dict) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_event = {int(row["event_index"]): row for row in result["decisions"]}
    replay_end = float(result["replay_end_s"])
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + teacher.PREVIEW_TAIL:
            break
        row = by_event.get(event.event_index)
        events.append({
            "event_index": event.event_index, "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices), "chord_size": event.chord_size,
            "teacher": None, "network": viewer._decision_view(row, None),
        })
    pitch_ratio = max(1e-6, compiled.pitch_percent * .01)
    return {
        "format": FORMAT, "chart": chart_path.name,
        "chart_sha256": viewer.file_sha256(chart_path), "difficulty": difficulty,
        "checkpoint": str(checkpoint), "checkpoint_sha256": viewer.file_sha256(checkpoint),
        "timebase": {
            "episode_duration_s": replay_end,
            "full_chart_duration_s": segment.duration_s,
            "segment_start_chart_s": segment.start_chart_s,
            "segment_end_chart_s": segment.end_chart_s,
            "pitch_percent": compiled.pitch_percent, "pitch_ratio": pitch_ratio,
            "offset_ms": compiled.offset_ms, "initial_bpm": compiled.initial_bpm,
            "countdown_ticks": compiled.countdown_ticks,
            "separate_countdown_time": compiled.separate_countdown_time,
            "separate_countdown_chart_s": compiled._separate_countdown_chart_seconds(),
            "formula": {"chart": "segment_start_chart_s + episode_s * pitch_ratio",
                        "audio": "(chart_s + offset_ms/1000 - separate_countdown_chart_s) / pitch_ratio"},
        },
        "floors": viewer._floor_rows(compiled, segment), "events": events,
        "policies": {"network": {"trace": result["trace"],
                                    "keydowns": result["keydowns"],
                                    "saved_metrics": result["stats"]}},
        "termination_reason": result["termination_reason"],
        "rendering_limitations": [
            "Gameplay floor geometry/timing only; decorations and camera events are not rendered.",
            "Planet travel is diagnostic interpolation, not Unity rendering.",
            "Pause and special-event presentation is diagnostic, not Unity-identical.",
            "Physical metrics are saved from the run and are not recomputed by HTML.",
        ],
    }


def summarize_choices(decisions: list[dict]) -> dict:
    axes = Counter(axis for row in decisions for axis in row["axes"])
    changes = []
    for left, right in zip(decisions, decisions[1:]):
        if left["axes"] and right["axes"] and set(left["axes"]).isdisjoint(right["axes"]):
            gap = float(right["due_ms"]) - float(left["due_ms"])
            if gap <= 35.0:
                changes.append({"from_event": left["event_index"], "to_event": right["event_index"],
                                "gap_ms": round(gap, 3), "from": left["chosen"], "to": right["chosen"]})
    return {
        "axis_use": dict(sorted(axes.items())),
        "thumb_uses": sum(value for axis, value in axes.items() if axis.endswith("_5")),
        "side_uses": sum(value for axis, value in axes.items() if axis.endswith("_6")),
        "extreme_switches_le_35ms": changes,
    }


def run(chart_path: Path, difficulty: str, checkpoint: Path, output_dir: Path,
        *, stop_after_s: float | None = None, wall_timeout_s: float | None = None) -> dict:
    if chart_path.suffix.lower() != ".adofai":
        raise ValueError("chart must be a .adofai file")
    chart_sha256 = viewer.file_sha256(chart_path)
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    model = load_model(checkpoint)
    started = time.perf_counter()
    result = run_network(segment, model, stop_after_s=stop_after_s,
                         wall_timeout_s=wall_timeout_s)
    elapsed = time.perf_counter() - started
    payload = build_payload(chart_path, difficulty, checkpoint, compiled, segment, result)
    summary = {
        "format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name, "chart_sha256": chart_sha256,
        "difficulty": difficulty, "full_chart_duration_s": segment.duration_s,
        "total_events": len(AutoplayTimingTeacher.from_segment(segment).events),
        "replay_end_s": result["replay_end_s"], "termination_reason": result["termination_reason"],
        "runtime_s": round(elapsed, 3), "control_ms": 10.0, "physics_ms": 1.0,
        "stats": result["stats"], "choice_statistics": summarize_choices(result["decisions"]),
        "decisions_recorded": len(result["decisions"]),
        "teacher_shadow_rollout_used": False, "teacher_candidate_scores_used": False,
        "timing_source": "AutoPlay Oracle exact event times", "training_performed": False,
        "failure_continuation": False, "checkpoint_sha256": viewer.file_sha256(checkpoint),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "full-chart-replay.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "full-chart-replay.html", viewer.render_html(payload))
    _write_once(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _numbered_path(path: Path, number: int, *, status: bool = False) -> Path:
    if number == 1:
        return path
    if status and path.name.endswith(".status.json"):
        name = path.name[:-len(".status.json")]
        return path.with_name(f"{name}-{number:03d}.status.json")
    return path.with_name(f"{path.stem}-{number:03d}{path.suffix}") if status else path.with_name(
        f"{path.name}-{number:03d}")


def choose_output_paths(chart_path: Path, output_dir: Path | None,
                        status_file: Path | None) -> tuple[Path, Path]:
    """Pick a free output pair without overwriting earlier runs."""
    base_dir = output_dir or Path("artifacts/dmdogo") / chart_path.stem
    base_status = status_file or base_dir.with_name(base_dir.name + ".status.json")
    for number in range(1, 100000):
        directory = _numbered_path(base_dir, number)
        status = _numbered_path(base_status, number, status=True)
        if not directory.exists() and not status.exists():
            return directory, status
    raise RuntimeError("could not find a free DMDOGO output path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--difficulty", default="unrated",
                        help="Optional descriptive label, not an inferred difficulty rating")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt"))
    parser.add_argument("--output-dir", type=Path,
                        help="Default: artifacts/dmdogo/<chart-name>; auto-numbered if occupied")
    parser.add_argument("--status-file", type=Path,
                        help="Default: <output-dir>.status.json; auto-numbered if occupied")
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    output_dir, status_file = choose_output_paths(args.chart, args.output_dir, args.status_file)
    print(f"DMDOGO output: {output_dir}", flush=True)
    print(f"DMDOGO status: {status_file}", flush=True)
    started = datetime.now(timezone.utc)
    try:
        summary = run(args.chart, args.difficulty, args.checkpoint, output_dir,
                      stop_after_s=args.stop_after_s, wall_timeout_s=args.wall_timeout_s)
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
                  "output_dir": str(output_dir), "termination_reason": summary["termination_reason"]}
    except BaseException as error:
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 1,
                  "error_type": type(error).__name__, "error": str(error),
                  "output_dir": str(output_dir)}
        # Never replace a pre-existing status even if another process raced us.
        if not status_file.exists():
            _write_once(status_file, json.dumps(status, indent=2) + "\n")
        raise
    _write_once(status_file, json.dumps(status, indent=2) + "\n")
    print(f"DMDOGO replay: {output_dir / 'full-chart-replay.html'}", flush=True)


if __name__ == "__main__":
    main()
