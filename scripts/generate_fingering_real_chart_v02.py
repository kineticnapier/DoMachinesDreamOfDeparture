from __future__ import annotations

"""Generate a bounded Train-only real-chart fingering pilot dataset."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import train_fingering_network_v01 as v01

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training import fingering_network_v01 as network
from dmdod.training.fingering_real_chart_v02 import select_dataset_windows
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdod-fingering-real-chart-dataset-v0.2"


def _json_sha(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()


def _write_once(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(content, bytes) else "x"
    kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with path.open(mode, **kwargs) as stream:
        stream.write(content)


def _bounded_segment(row: dict, *, max_focus_events: int, prefix_events: int):
    compiled = compile_adofai(load_adofai(row["path"]))
    full = build_playable_segment(compiled)
    events = AutoplayTimingTeacher.from_segment(full).events
    first = int(row["first_event"])
    last = min(int(row["last_event"]), first + max_focus_events - 1)
    prefix_index = max(0, first - prefix_events)
    start_episode = max(0.0, events[prefix_index].episode_time_s - 0.050)
    focus_episode = max(start_episode, float(row["start_s"]))
    end_episode = min(full.duration_s, events[last].episode_time_s + 0.060)
    start_chart = full.start_chart_s + start_episode * full.pitch_ratio
    end_chart = full.start_chart_s + end_episode * full.pitch_ratio
    segment = build_playable_segment(compiled, start_s=start_chart, end_s=end_chart)
    return segment, focus_episode - start_episode, {
        "source_first_event": first, "source_last_event": last,
        "source_start_episode_s": start_episode, "source_end_episode_s": end_episode,
        "prefix_events": first - prefix_index,
    }


def _collect_teacher(segment, *, focus_from_s: float):
    """Commit v0.2 Teacher choices and capture pre-label physical contexts."""

    env = CoupledContactRealChartEnv(segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    pending_examples = {}
    rows = []
    all_downs = []
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    for _ in range(maximum):
        now = env.privileged_episode_time_s()
        committed = {index: item for index, item in committed.items()
                     if item.due_s + teacher.PREVIEW_TAIL >= now
                     and not all(env._used[o] for o in events[index].target_ordinals)}
        upcoming = tuple(event for event in events if event.event_index not in selected
                         and event.episode_time_s + teacher.PREVIEW_TAIL >= now)
        due = tuple(event for event in upcoming
                    if event.episode_time_s <= now + teacher.LEAD + 1e-9)
        for event in due:
            if event.event_index in selected:
                continue
            occupied = {axis for item in committed.values() for axis in item.axes}
            look = tuple(value for value in upcoming if value.event_index >= event.event_index)
            if now < focus_from_s:
                choices = [allocation for allocation in teacher.finger_options(event.chord_size)
                           if teacher.eligible(env.motor, occupied, allocation)[0]]
                allocation = min(choices, key=lambda value: (
                    sum(int(finger.axis.split("_")[1]) for finger in value),
                    tuple(finger.label for finger in value),
                )) if choices else ()
                candidates = []
                context = None
                reason = "fixed_center_first_prefix"
            else:
                context = network.encode_context(env.motor, look, now, occupied)
                ranked = teacher.rank_first_choices(
                    env.motor, tuple(committed.values()), look, max_depth=3,
                )
                viable = [value for value in ranked if value["eligible"]]
                chosen = min(viable, key=lambda value: (value["score"], value["finger"])) if viable else None
                allocation = chosen["allocation"] if chosen else ()
                candidates = [{key: value for key, value in candidate.items() if key != "allocation"}
                              for candidate in ranked]
                reason = "physical_lookahead" if chosen else "no_feasible_axis"
                if viable:
                    candidate_tensors = torch.stack([
                        network.encode_candidate(value) for value in teacher.finger_options(event.chord_size)
                    ])
                    costs = torch.tensor([
                        float(value["score"]) if value["eligible"] else float("inf")
                        for value in ranked
                    ], dtype=torch.float32)
                    eligible = torch.tensor([bool(value["eligible"]) for value in ranked], dtype=torch.bool)
                    pending_examples[event.event_index] = network.RankingExample(
                        "", "0" * 64, event.event_index, context, candidate_tensors, costs, eligible,
                    )
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
            if now >= focus_from_s:
                rows.append({
                    "event_index": event.event_index, "ordinals": list(event.target_ordinals),
                    "due_ms": round(event.episode_time_s * 1000.0, 3),
                    "decision_ms": round(now * 1000.0, 3),
                    "chosen": item.label if allocation else None,
                    "fingers": [{"axis": finger.axis, "mode": finger.mode,
                                 "contacts": list(finger.logical_keys)} for finger in allocation],
                    "reason": reason, "candidates": candidates,
                    "depth": min((value.get("lookahead_depth", 0) for value in candidates
                                  if value["finger"] == item.label), default=0),
                    "continuation": next((value.get("best_continuation", []) for value in candidates
                                          if value["finger"] == item.label), []),
                })
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                all_downs.append({"time_ms": round(event.time_s * 1000.0, 3), "contact": event.key})
        if step.done:
            break
    else:
        raise RuntimeError("real-chart Teacher exceeded step budget")

    examples = []
    for row in rows:
        used = [env._used[index] for index in row["ordinals"]]
        missed = [env._missed[index] for index in row["ordinals"]]
        row["outcome"] = "hit" if all(used) else "miss" if any(missed) else "unresolved"
        contacts = [contact for finger in row["fingers"] for contact in finger["contacts"]]
        matches = [down for down in all_downs if down["contact"] in contacts
                   and abs(down["time_ms"] - row["due_ms"]) < 35]
        row["keydowns"] = matches
        row["actual_error_ms"] = (round(min((down["time_ms"] - row["due_ms"] for down in matches),
                                             key=abs), 3) if matches else None)
        valid_label = row["outcome"] == "hit" and row["event_index"] in pending_examples
        row["training_example_saved"] = valid_label
        if valid_label:
            old = pending_examples[row["event_index"]]
            examples.append(network.RankingExample(
                "pending", "0" * 64, old.event_index, old.context,
                old.candidates, old.teacher_costs, old.eligible,
            ))
    stats = {
        "targets": env.stats.targets, "hits": env.stats.hits, "misses": env.stats.misses,
        "too_early": env.stats.too_early_presses, "overloaded": env.stats.overloaded,
        "x_accuracy_percent": env.stats.x_accuracy_percent,
        "timing_mae_ms": env.stats.mean_abs_error_ms,
    }
    return {
        "stats": stats, "focus_from_s": focus_from_s,
        "focus_hits": sum(row["outcome"] == "hit" for row in rows),
        "focus_events": len(rows), "decisions": rows, "keydown_count": len(all_downs),
    }, examples


def _replace_example_identity(example, *, window_id: str, chart_sha256: str):
    return network.RankingExample(
        window_id, chart_sha256, example.event_index, example.context,
        example.candidates, example.teacher_costs, example.eligible,
    )


def _load_model(path: Path | None):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != network.FORMAT:
        raise ValueError("checkpoint is not Fingering Network v0.1")
    model = network.FingeringRanker(hidden=int(payload["hidden"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _aggregate(rows: list[dict]) -> dict:
    targets = sum(row["targets"] for row in rows)
    hits = sum(row["hits"] for row in rows)
    return {
        "windows": len(rows), "targets": targets, "hits": hits,
        "misses": sum(row["misses"] for row in rows),
        "too_early": sum(row["too_early"] for row in rows),
        "overloaded_windows": sum(bool(row["overloaded"]) for row in rows),
        "hit_rate": hits / targets if targets else None,
        "timing_mae_ms": (sum(row["timing_mae_ms"] * row["hits"] for row in rows
                               if row["timing_mae_ms"] is not None) / hits if hits else None),
    }


def _index_html(cases: list[dict]) -> str:
    rows = "".join(
        f'<tr><td>{escape(case["split"])}</td><td>{escape(case["chart"])}</td>'
        f'<td>{escape(case["difficulty"] or "unknown")}</td><td>{case["teacher"]["focus_hits"]}/{case["teacher"]["focus_events"]}</td>'
        f'<td>{case["center_first"]["hits"]}/{case["center_first"]["targets"]}</td>'
        f'<td><a href="{escape(case["viewer"])}">12軸タイムライン</a></td></tr>'
        for case in cases
    )
    return """<!doctype html><html lang="ja"><meta charset="utf-8"><title>Real-chart Fingering v0.2</title>
<style>body{font:14px/1.5 system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:auto;padding:16px}a{color:#93c5fd}table{width:100%;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #374151;text-align:left}</style>
<h1>Fingering Network v0.2 — Real-chart pilot</h1><p>Train-only。各リンクで12軸、採用指、全候補の物理score、失敗理由、実Hit/Missを確認できる。</p>
<table><tr><th>split</th><th>譜面</th><th>難易度</th><th>Teacher focus</th><th>CenterFirst whole segment</th><th>詳細</th></tr>""" + rows + "</table></html>"


def run(args) -> dict:
    started = time.perf_counter()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selected = select_dataset_windows(inventory, max_windows=args.max_windows)
    if not selected:
        raise RuntimeError("no eligible Train windows")
    model = _load_model(args.checkpoint)
    cases = []
    examples = {"fit": [], "quality": []}
    physical = {name: [] for name in ("center_first", "teacher_v0_2", "network_v0_1")}
    args.viewer_dir.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(selected, 1):
        segment, focus_from_s, bounds = _bounded_segment(
            row, max_focus_events=args.max_focus_events, prefix_events=args.prefix_events,
        )
        teacher_result, collected = _collect_teacher(segment, focus_from_s=focus_from_s)
        window_id = f'{row["sha256"][:12]}-{row["first_event"]}-{row["last_event"]}'
        examples[row["split"]].extend(
            _replace_example_identity(value, window_id=window_id, chart_sha256=row["sha256"])
            for value in collected
        )
        center = v01.run_policy(segment, "center_first")
        physical["center_first"].append(center)
        physical["teacher_v0_2"].append(teacher_result["stats"])
        network_result = v01.run_policy(segment, "network_v0_1", model) if model else None
        if network_result:
            physical["network_v0_1"].append(network_result)
        candidate_reasons = Counter(candidate["reason"] for decision in teacher_result["decisions"]
                                    for candidate in decision["candidates"])
        equivalent = sum(
            sum(candidate["eligible"] and candidate["score"] <= min(
                value["score"] for value in decision["candidates"] if value["eligible"]
            ) + network.GOOD_COST_TOLERANCE for candidate in decision["candidates"]) > 1
            for decision in teacher_result["decisions"] if any(
                value["eligible"] for value in decision["candidates"])
        )
        viewer_name = f"window-{index:02d}-{row['sha256'][:10]}.html"
        viewer_report = {
            "chart": row["chart"], "segment_end_s": segment.duration_s,
            "teacher": teacher_result,
        }
        _write_once(args.viewer_dir / viewer_name, teacher.render_html(viewer_report))
        cases.append({
            **row, **bounds, "segment_duration_s": segment.duration_s,
            "teacher": teacher_result, "center_first": center,
            "network_v0_1": network_result,
            "candidate_failure_reasons": dict(candidate_reasons),
            "events_with_multiple_equivalent_good": equivalent,
            "saved_examples": len(collected), "viewer": viewer_name,
        })
        print(f"window {index}/{len(selected)} {row['chart']}: "
              f"focus Hit {teacher_result['focus_hits']}/{teacher_result['focus_events']}", flush=True)

    fit_hashes = {case["sha256"] for case in cases if case["split"] == "fit"}
    quality_hashes = {case["sha256"] for case in cases if case["split"] == "quality"}
    failed_labeled = sum(
        decision["training_example_saved"] and decision["outcome"] != "hit"
        for case in cases for decision in case["teacher"]["decisions"]
    )
    overlap = bool(fit_hashes & quality_hashes)
    teacher_focus_events = sum(case["teacher"]["focus_events"] for case in cases)
    teacher_focus_hits = sum(case["teacher"]["focus_hits"] for case in cases)
    center_chosen = sum(
        teacher_decision["chosen"] == center_decision["chosen"]
        for case in cases
        for teacher_decision, center_decision in zip(
            case["teacher"]["decisions"],
            case["center_first"]["decisions"][-len(case["teacher"]["decisions"]):],
        )
    )
    total_examples = sum(len(value) for value in examples.values())
    quality_gate = {
        "verified_g10_or_harder_chart_present": any(case["verified_g10_or_harder"] for case in cases),
        "teacher_all_focus_events_hit": teacher_focus_hits == teacher_focus_events,
        "fit_quality_chart_disjoint": not overlap and bool(fit_hashes) and bool(quality_hashes),
        "failed_teacher_result_labeled_success": failed_labeled == 0,
        "nonempty_success_labels": total_examples > 0,
    }
    report = {
        "format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "Train only", "inventory": str(args.inventory),
        "limits": {"max_windows": args.max_windows, "max_focus_events": args.max_focus_events,
                   "prefix_events": args.prefix_events},
        "student_input_contract": [
            "current 12-axis position/velocity/pressed/availability",
            "current thumb/side topology", "up to three Timing-Oracle relative events",
            "candidate physical-axis allocation",
        ],
        "forbidden_student_inputs": [
            "candidate physical score", "chosen Teacher label", "future physical outcome",
            "chart difficulty", "chart identity",
        ],
        "split": {"unit": "chart SHA256", "fit": sorted(fit_hashes),
                  "quality": sorted(quality_hashes), "overlap": overlap},
        "cases": cases,
        "aggregate": {
            "teacher_focus_events": teacher_focus_events,
            "teacher_focus_hits": teacher_focus_hits,
            "saved_examples": total_examples,
            "center_first_agreement_events": center_chosen,
            "center_first_agreement_rate": center_chosen / teacher_focus_events if teacher_focus_events else None,
            "center_first": _aggregate(physical["center_first"]),
            "teacher_v0_2": _aggregate(physical["teacher_v0_2"]),
            "network_v0_1": _aggregate(physical["network_v0_1"]) if model else None,
        },
        "quality_gate": quality_gate,
        "ready_for_training": all(quality_gate.values()),
        "training_performed": False,
        "holdout_roles_opened": [],
        "elapsed_s": time.perf_counter() - started,
    }
    payload = {
        "format": FORMAT, "report_sha256": _json_sha(report),
        "split_unit": "chart SHA256", "examples": examples,
        "failed_teacher_results_included": False,
    }
    torch.save(payload, args.dataset)
    _write_once(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(args.viewer_dir / "index.html", _index_html(cases))
    return report


def main() -> None:
    # Windows shells may still expose cp932 even when report files are UTF-8.
    # A chart title must never abort an otherwise deterministic physics run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--viewer-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--max-focus-events", type=int, default=24)
    parser.add_argument("--prefix-events", type=int, default=6)
    args = parser.parse_args()
    if any(path.exists() for path in (args.dataset, args.report, args.viewer_dir, args.status_file)):
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        report = run(args)
        _write_once(args.status_file, json.dumps({
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "report": str(args.report), "dataset": str(args.dataset),
        }, indent=2) + "\n")
        print(json.dumps({"aggregate": report["aggregate"],
                          "quality_gate": report["quality_gate"],
                          "ready_for_training": report["ready_for_training"]}, indent=2), flush=True)
    except BaseException as error:
        if not args.status_file.exists():
            _write_once(args.status_file, json.dumps({
                "pid": os.getpid(), "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1, "error_type": type(error).__name__, "error": str(error),
            }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
