from __future__ import annotations

"""Fixed one-shot Free versus Human-Proxy Fingering Network v0.1.1 run."""

import argparse
from collections import Counter
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
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdod-fingering-network-v0.1.1-human-proxy-comparison"
FIXED_HUMAN_ALPHA = 0.5


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _decision_view(row: dict | None) -> dict | None:
    result = v01._decision_view(row, None)
    if result is not None and row is not None:
        result.update({
            "style_alpha": row["style_alpha"], "style_class": row["style_class"],
            "free_chosen": row["free_chosen"],
            "reason": ("style_blend_changed_free_choice" if row["different_from_free"]
                       else "network_choice_retained"),
        })
    return result


def policy_statistics(result: dict, event_times: dict[int, float]) -> dict:
    decisions = result["decisions"]
    classes = Counter(row["style_class"] for row in decisions)
    dense = []
    for index, row in enumerate(decisions):
        due = event_times[row["event_index"]]
        neighbors = []
        if index:
            neighbors.append((due - event_times[decisions[index - 1]["event_index"]]) * 1000)
        if index + 1 < len(decisions):
            neighbors.append((event_times[decisions[index + 1]["event_index"]] - due) * 1000)
        if neighbors and min(neighbors) <= 35.0 and row["style_class"] == "thumb_or_side":
            dense.append(row["event_index"])
    total = max(1, len(decisions))
    return {
        "decisions": len(decisions), "index_proxy_uses": classes["index_proxy"],
        "ordinary_proxy_uses": classes["index_proxy"] + classes["ordinary_proxy"],
        "thumb_or_side_uses": classes["thumb_or_side"],
        "index_proxy_rate": classes["index_proxy"] / total,
        "ordinary_proxy_rate": (classes["index_proxy"] + classes["ordinary_proxy"]) / total,
        "thumb_or_side_rate": classes["thumb_or_side"] / total,
        "dense_le_35ms_special_events": dense,
    }


def compare_results(free: dict, human: dict, event_times: dict[int, float]) -> dict:
    free_by = {row["event_index"]: row for row in free["decisions"]}
    human_by = {row["event_index"]: row for row in human["decisions"]}
    common = sorted(free_by.keys() & human_by.keys())
    different = [index for index in common if free_by[index]["chosen"] != human_by[index]["chosen"]]
    worsened = []
    for index in common:
        left, right = free_by[index], human_by[index]
        left_error, right_error = left.get("actual_error_ms"), right.get("actual_error_ms")
        reason = None
        if left["outcome"] == "Hit" and right["outcome"] != "Hit":
            reason = "free_hit_human_not_hit"
        elif left_error is not None and right_error is not None and abs(right_error) > abs(left_error) + 1e-6:
            reason = "larger_absolute_timing_error"
        if reason:
            worsened.append({"event_index": index, "reason": reason,
                             "free_error_ms": left_error, "human_error_ms": right_error,
                             "free": left["chosen"], "human_proxy": right["chosen"]})
    return {
        "different_events": different, "different_event_count": len(different),
        "human_proxy_worsened_events": worsened,
        "human_proxy_worsened_event_count": len(worsened),
        "free_style": policy_statistics(free, event_times),
        "human_proxy_style": policy_statistics(human, event_times),
    }


def build_payload(chart_path: Path, checkpoint: Path, compiled, segment,
                  free: dict, human: dict) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    free_by = {row["event_index"]: row for row in free["decisions"]}
    human_by = {row["event_index"]: row for row in human["decisions"]}
    replay_end = max(float(free["replay_end_s"]), float(human["replay_end_s"]))
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + .035:
            break
        free_row, human_row = free_by.get(event.event_index), human_by.get(event.event_index)
        events.append({"event_index": event.event_index, "episode_s": round(event.episode_time_s, 9),
                       "chart_s": round(event.chart_time_s, 9),
                       "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
                       "floor_indices": list(event.floor_indices), "chord_size": event.chord_size,
                       "different": bool(free_row and human_row and free_row["chosen"] != human_row["chosen"]),
                       "decisions": {"free": _decision_view(free_row),
                                     "human_proxy": _decision_view(human_row)}})
    return {
        "format": FORMAT, "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": v01.file_sha256(checkpoint),
        "default_policy": "human_proxy",
        "timebase": {"episode_duration_s": replay_end,
                     "full_chart_duration_s": segment.duration_s,
                     "segment_start_chart_s": segment.start_chart_s,
                     "segment_end_chart_s": segment.end_chart_s,
                     "pitch_percent": compiled.pitch_percent,
                     "pitch_ratio": segment.pitch_ratio, "offset_ms": compiled.offset_ms,
                     "separate_countdown_chart_s": compiled._separate_countdown_chart_seconds()},
        "floors": v03.rotation_profiles(load_adofai(chart_path), compiled, segment),
        "events": events,
        "policies": {
            "free": {"label": "Free", "style_alpha": 0.0, "trace": free["trace"],
                     "keydowns": free["keydowns"], "saved_metrics": free["stats"]},
            "human_proxy": {"label": "Human-Proxy", "style_alpha": FIXED_HUMAN_ALPHA,
                            "trace": human["trace"], "keydowns": human["keydowns"],
                            "saved_metrics": human["stats"]},
        },
        "key_overlay": {"mapping": [{"label": label, "contact": contact}
                                      for label, contact in v03.KEY_DISPLAY_MAP],
                        "mapping_semantics": "Display aliases, not verified OS-key bindings."},
        "limitations": [
            "Human-Proxy is an untrained ordinal style prior, not human demonstration learning.",
            "Axes 1/2/3/4 are index/other ordinary finger proxies only; anatomy is not calibrated.",
            "Lower-row display aliases are logical contacts and do not define physical keyboard layout.",
        ],
    }


def run(chart_path: Path, checkpoint: Path, output_dir: Path, *,
        stop_after_s: float | None, wall_timeout_s: float) -> dict:
    if v01.file_sha256(chart_path) != runner.SELECTED_SHA256:
        raise RuntimeError("P16 chart SHA256 mismatch")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    model = runner.load_model(checkpoint)
    started = time.perf_counter()
    free = runner.run_network(segment, model, stop_after_s=stop_after_s,
                              wall_timeout_s=wall_timeout_s, style_alpha=0.0,
                              policy_name="free")
    human = runner.run_network(segment, model, stop_after_s=stop_after_s,
                               wall_timeout_s=wall_timeout_s, style_alpha=FIXED_HUMAN_ALPHA,
                               policy_name="human_proxy")
    oracle = AutoplayTimingTeacher.from_segment(segment)
    times = {event.event_index: event.episode_time_s for event in oracle.events}
    comparison = compare_results(free, human, times)
    payload = build_payload(chart_path, checkpoint, compiled, segment, free, human)
    report = {
        "format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name, "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "duration_s": segment.duration_s, "events": len(oracle.events),
        "runtime_s": round(time.perf_counter() - started, 3),
        "fixed_human_proxy_alpha": FIXED_HUMAN_ALPHA,
        "alpha_selected_before_full_run": True, "hyperparameter_search": False,
        "free": {"stats": free["stats"], "termination_reason": free["termination_reason"],
                 "replay_end_s": free["replay_end_s"]},
        "human_proxy": {"stats": human["stats"], "termination_reason": human["termination_reason"],
                        "replay_end_s": human["replay_end_s"]},
        "comparison": comparison, "training_performed": False,
        "teacher_shadow_rollout_used": False, "failure_continuation": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.html", v03.render_html(payload))
    return report


def run_single_policy(chart_path: Path, checkpoint: Path, output_dir: Path, *,
                      policy: str, stop_after_s: float | None,
                      wall_timeout_s: float) -> dict:
    if policy not in {"free", "human_proxy"}:
        raise ValueError("single policy must be free or human_proxy")
    if v01.file_sha256(chart_path) != runner.SELECTED_SHA256:
        raise RuntimeError("P16 chart SHA256 mismatch")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    model = runner.load_model(checkpoint)
    alpha = 0.0 if policy == "free" else FIXED_HUMAN_ALPHA
    started = time.perf_counter()
    result = runner.run_network(segment, model, stop_after_s=stop_after_s,
                                wall_timeout_s=wall_timeout_s, style_alpha=alpha,
                                policy_name=policy)
    artifact = {"format": FORMAT + "-single-policy", "policy": policy,
                "style_alpha": alpha, "chart_sha256": v01.file_sha256(chart_path),
                "checkpoint_sha256": v01.file_sha256(checkpoint),
                "runtime_s": round(time.perf_counter() - started, 3), "result": result}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "policy-result.json", json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    return artifact


def assemble_saved_policies(chart_path: Path, checkpoint: Path, output_dir: Path,
                            free_path: Path, human_path: Path) -> dict:
    free_artifact = json.loads(free_path.read_text(encoding="utf-8"))
    human_artifact = json.loads(human_path.read_text(encoding="utf-8"))
    if free_artifact["policy"] != "free" or human_artifact["policy"] != "human_proxy":
        raise ValueError("saved policy roles are reversed or invalid")
    expected_chart = v01.file_sha256(chart_path)
    expected_checkpoint = v01.file_sha256(checkpoint)
    for artifact in (free_artifact, human_artifact):
        if artifact["chart_sha256"] != expected_chart or artifact["checkpoint_sha256"] != expected_checkpoint:
            raise RuntimeError("saved policy provenance mismatch")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    free, human = free_artifact["result"], human_artifact["result"]
    oracle = AutoplayTimingTeacher.from_segment(segment)
    times = {event.event_index: event.episode_time_s for event in oracle.events}
    comparison = compare_results(free, human, times)
    payload = build_payload(chart_path, checkpoint, compiled, segment, free, human)
    report = {"format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "chart": chart_path.name, "chart_sha256": expected_chart,
              "checkpoint_sha256": expected_checkpoint, "duration_s": segment.duration_s,
              "events": len(oracle.events),
              "runtime_s": free_artifact["runtime_s"] + human_artifact["runtime_s"],
              "fixed_human_proxy_alpha": FIXED_HUMAN_ALPHA,
              "alpha_selected_before_full_run": True, "hyperparameter_search": False,
              "free": {"stats": free["stats"], "termination_reason": free["termination_reason"],
                       "replay_end_s": free["replay_end_s"]},
              "human_proxy": {"stats": human["stats"],
                              "termination_reason": human["termination_reason"],
                              "replay_end_s": human["replay_end_s"]},
              "comparison": comparison, "training_performed": False,
              "teacher_shadow_rollout_used": False, "failure_continuation": False,
              "assembled_without_inference": True,
              "source_policy_results": {"free": str(free_path), "human_proxy": str(human_path)}}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.html", v03.render_html(payload))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=300.0)
    parser.add_argument("--mode", choices=("both", "single", "assemble"), default="both")
    parser.add_argument("--policy", choices=("free", "human_proxy"))
    parser.add_argument("--free-result", type=Path)
    parser.add_argument("--human-result", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        if args.mode == "single":
            if args.policy is None:
                parser.error("--policy is required for single mode")
            artifact = run_single_policy(args.chart, args.checkpoint, args.output_dir,
                                         policy=args.policy, stop_after_s=args.stop_after_s,
                                         wall_timeout_s=args.wall_timeout_s)
            report = {args.policy: {"termination_reason": artifact["result"]["termination_reason"]}}
        elif args.mode == "assemble":
            if args.free_result is None or args.human_result is None:
                parser.error("--free-result and --human-result are required for assemble mode")
            report = assemble_saved_policies(args.chart, args.checkpoint, args.output_dir,
                                             args.free_result, args.human_result)
        else:
            report = run(args.chart, args.checkpoint, args.output_dir,
                         stop_after_s=args.stop_after_s, wall_timeout_s=args.wall_timeout_s)
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
                  "output_dir": str(args.output_dir),
                  "mode": args.mode,
                  "free_termination": report.get("free", {}).get("termination_reason"),
                  "human_proxy_termination": report.get("human_proxy", {}).get("termination_reason")}
    except BaseException as error:
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 1,
                  "error_type": type(error).__name__, "error": str(error)}
        _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
        raise
    _write_once(args.status_file, json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
