from __future__ import annotations

"""One fixed small train/evaluation run for Fingering Network v0.1."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import Coupled16ContactMotor
from dmdod.motor.keyboard import KeyEvent
from dmdod.training import fingering_network_v01 as network
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


VERSION = "dmdod-fingering-network-v0.1-fixed-experiment"
SEED = 4101
EPOCHS = 80
LEARNING_RATE = 5e-4
EVAL_BPMS = (900.0, 1600.0, 2800.0)
EVAL_FLOORS = 16


TRAIN_PATTERNS = (
    network.PatternSpec("train-alternating-90", (90.0,) * 9, (1,) * 10, "alternating"),
    network.PatternSpec("train-rolling-55", (55.0,) * 11, (1,) * 12, "rolling"),
    network.PatternSpec("train-short-repeat", (85.0, 30.0, 30.0, 85.0) * 2, (1,) * 9, "short_repeat"),
    network.PatternSpec("train-dense-22", (22.0,) * 11, (1,) * 12, "high_density"),
    network.PatternSpec("train-accelerando", (90.0, 75.0, 60.0, 45.0, 35.0, 28.0, 24.0, 22.0), (1,) * 9, "rolling"),
    network.PatternSpec("train-contact-pairs", (52.0, 48.0, 44.0), (2, 1, 2, 1), "contact_pair"),
)
VALIDATION_PATTERNS = (
    network.PatternSpec("validation-alternating-70", (70.0,) * 7, (1,) * 8, "alternating"),
    network.PatternSpec("validation-burst-32", (80.0, 32.0, 32.0, 75.0, 32.0, 32.0), (1,) * 7, "short_repeat"),
)
TEST_PATTERNS = (
    network.PatternSpec("test-rolling-43", (43.0,) * 8, (1,) * 9, "rolling"),
    network.PatternSpec("test-dense-18", (18.0,) * 9, (1,) * 10, "high_density"),
)


@dataclass(frozen=True)
class SyntheticEvent:
    event_index: int
    episode_time_s: float
    chord_size: int


def _events(pattern: network.PatternSpec) -> tuple[SyntheticEvent, ...]:
    return tuple(SyntheticEvent(index, due, pattern.chord_sizes[index])
                 for index, due in enumerate(pattern.event_times_s()))


def collect_pattern(pattern: network.PatternSpec) -> list[network.RankingExample]:
    """Collect labels from teacher shadows; features are captured beforehand."""

    motor = Coupled16ContactMotor(control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    events = _events(pattern)
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    examples = []
    bound = int((events[-1].episode_time_s + 0.150) / teacher.CTRL) + 20
    for _ in range(bound):
        now = motor.motor.time_s
        committed = {index: item for index, item in committed.items()
                     if item.due_s + teacher.PREVIEW_TAIL >= now}
        upcoming = tuple(event for event in events
                         if event.event_index not in selected
                         and event.episode_time_s + teacher.PREVIEW_TAIL >= now)
        due = tuple(event for event in upcoming
                    if event.episode_time_s <= now + teacher.LEAD + 1e-9)
        for event in due:
            if event.event_index in selected:
                continue
            occupied = {axis for item in committed.values() for axis in item.axes}
            look = tuple(value for value in upcoming if value.event_index >= event.event_index)
            context = network.encode_context(motor, look, now, occupied)
            allocations = teacher.finger_options(event.chord_size)
            rows = teacher.rank_first_choices(
                motor, tuple(committed.values()), look, max_depth=3
            )
            examples.append(network.make_example(
                pattern=pattern, event_index=event.event_index, context=context,
                allocations=allocations, candidate_rows=rows,
            ))
            viable = [row for row in rows if row["eligible"]]
            chosen = min(viable, key=lambda row: (row["score"], row["finger"])) if viable else None
            allocation = chosen["allocation"] if chosen else ()
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
        action = teacher.action_for(motor, tuple(committed.values()), now)
        motor.step(action)
        if len(selected) == len(events) and now > events[-1].episode_time_s + teacher.PREVIEW_TAIL:
            break
    if len(examples) != len(events):
        raise RuntimeError(f"incomplete pattern collection: {pattern.pattern_id}")
    return examples


def collect_split(patterns) -> list[network.RankingExample]:
    return [example for pattern in patterns for example in collect_pattern(pattern)]


def _fixture(bpm: float):
    chart_text = json.dumps({
        "angleData": [0.0] * EVAL_FLOORS,
        "settings": {"bpm": bpm, "pitch": 100.0, "countdownTicks": 0},
        "actions": [],
    }, sort_keys=True, separators=(",", ":"))
    chart = compile_adofai(parse_adofai_text(chart_text))
    return build_playable_segment(chart), sha256(chart_text.encode("utf-8")).hexdigest()


def _center_first(motor, occupied, chord_size):
    choices = [allocation for allocation in teacher.finger_options(chord_size)
               if teacher.eligible(motor, occupied, allocation)[0]]
    return min(choices, key=lambda allocation: (
        sum(int(finger.axis.split("_")[1]) for finger in allocation),
        tuple(finger.label for finger in allocation),
    )) if choices else ()


def _summarize_env(env: CoupledContactRealChartEnv) -> dict:
    stats = env.stats
    return {
        "targets": stats.targets, "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses, "overloaded": stats.overloaded,
        "overload_counter": stats.overload_counter,
        "x_accuracy_percent": stats.x_accuracy_percent,
        "timing_mae_ms": stats.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
    }


def run_policy(segment, policy: str, model: network.FingeringRanker | None = None) -> dict:
    if policy not in ("center_first", "teacher_v0_2", "network_v0_1"):
        raise ValueError("unknown fingering policy")
    if (policy == "network_v0_1") != (model is not None):
        raise ValueError("network model/policy mismatch")
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS
    )
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    decisions = []
    all_downs = []
    max_steps = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    for _ in range(max_steps):
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
            allocations = teacher.finger_options(event.chord_size)
            teacher_rows = None
            network_scores = None
            if policy == "center_first":
                allocation = _center_first(env.motor, occupied, event.chord_size)
                # Evaluator-only shadow, computed from the same pre-decision
                # state.  It never participates in the CenterFirst choice.
                teacher_rows = teacher.rank_first_choices(
                    env.motor, tuple(committed.values()), look, max_depth=3
                )
            elif policy == "teacher_v0_2":
                teacher_rows = teacher.rank_first_choices(
                    env.motor, tuple(committed.values()), look, max_depth=3
                )
                viable = [row for row in teacher_rows if row["eligible"]]
                chosen = min(viable, key=lambda row: (row["score"], row["finger"])) if viable else None
                allocation = chosen["allocation"] if chosen else ()
            else:
                valid = tuple(teacher.eligible(env.motor, occupied, value)[0]
                              for value in allocations)
                context = network.encode_context(env.motor, look, now, occupied)
                selected_index, network_scores = network.choose_candidate(
                    model, context, allocations, valid
                )
                allocation = allocations[selected_index] if selected_index is not None else ()
                # Evaluator-only teacher shadow, computed after Network inference.
                teacher_rows = teacher.rank_first_choices(
                    env.motor, tuple(committed.values()), look, max_depth=3
                )
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
            row = {
                "event_index": event.event_index,
                "ordinals": list(event.target_ordinals),
                "due_ms": round(event.episode_time_s * 1000.0, 3),
                "decision_ms": round(now * 1000.0, 3),
                "chosen": item.label if allocation else None,
                "axes": [finger.axis for finger in allocation],
            }
            if teacher_rows is not None:
                viable = [value for value in teacher_rows if value["eligible"]]
                teacher_chosen = min(viable, key=lambda value: (value["score"], value["finger"])) if viable else None
                row["teacher_chosen"] = teacher_chosen["finger"] if teacher_chosen else None
                row["different_from_teacher"] = row["chosen"] != row["teacher_chosen"]
                row["candidates"] = [{
                    "finger": candidate["finger"],
                    "network_score": (network_scores[index] if network_scores is not None else None),
                    "teacher_score": candidate.get("score"),
                    "teacher_eligible": candidate["eligible"],
                    "teacher_reason": candidate["reason"],
                    "teacher_rank": candidate.get("rank"),
                } for index, candidate in enumerate(teacher_rows)]
            decisions.append(row)
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                all_downs.append({"time_ms": round(event.time_s * 1000.0, 3),
                                  "physical_axis": _logical_to_axis(event.key)})
        if step.done:
            break
    else:
        raise RuntimeError("physical evaluation exceeded step budget")

    for row in decisions:
        used = [env._used[index] for index in row["ordinals"]]
        missed = [env._missed[index] for index in row["ordinals"]]
        row["outcome"] = "Hit" if all(used) else "Miss" if any(missed) else "Unresolved"
        nearby = [value for value in all_downs
                  if value["physical_axis"] in row["axes"]
                  and abs(value["time_ms"] - row["due_ms"]) <= 35.0]
        row["keydowns"] = [
            min((value for value in nearby if value["physical_axis"] == axis),
                key=lambda value: abs(value["time_ms"] - row["due_ms"]))
            for axis in row["axes"]
            if any(value["physical_axis"] == axis for value in nearby)
        ]
        early = [trace for trace in env.judgement_trace
                 if trace["target_index"] in row["ordinals"]
                 and trace["margin"] is None
                 and trace["time_s"] * 1000.0 < row["due_ms"]]
        row["too_early_keydowns"] = len(early)
        row["judgement"] = (
            "TooEarly + " + row["outcome"] if early else row["outcome"]
        )
    different_success = sum(
        row.get("different_from_teacher", False) and row["outcome"] == "Hit"
        for row in decisions
    )
    result = _summarize_env(env)
    result.update({
        "policy": policy, "decisions": decisions,
        "different_from_teacher": sum(row.get("different_from_teacher", False) for row in decisions),
        "different_from_teacher_but_hit": different_success,
    })
    return result


def _logical_to_axis(contact: str) -> str:
    hand, depth_text = contact.split("_")
    depth = int(depth_text)
    return f"{hand}_{6 if depth in (7, 8) else 5 if depth in (5, 6) else depth}"


def _aggregate(rows: list[dict]) -> dict:
    total_hits = sum(row["hits"] for row in rows)
    total_targets = sum(row["targets"] for row in rows)
    weighted_errors = [row["timing_mae_ms"] for row in rows for _ in range(row["hits"])
                       if row["timing_mae_ms"] is not None]
    x_weight = sum(row["hits"] for row in rows)
    return {
        "charts": len(rows), "targets": total_targets, "hits": total_hits,
        "misses": sum(row["misses"] for row in rows),
        "too_early": sum(row["too_early"] for row in rows),
        "overloaded_charts": sum(row["overloaded"] for row in rows),
        "physical_keydowns": sum(row["physical_keydowns"] for row in rows),
        "hit_rate": total_hits / total_targets if total_targets else None,
        "timing_mae_ms": sum(weighted_errors) / len(weighted_errors) if weighted_errors else None,
        "x_accuracy_percent": (
            sum(row["x_accuracy_percent"] * row["hits"] for row in rows) / x_weight
            if x_weight else None
        ),
        "different_from_teacher": sum(row["different_from_teacher"] for row in rows),
        "different_from_teacher_but_hit": sum(row["different_from_teacher_but_hit"] for row in rows),
    }


def physical_axis_label(axis: str) -> str:
    hand, depth = axis.split("_")
    side = "左手" if hand == "left" else "右手"
    return f"{side}・{'親指' if depth == '5' else '側面' if depth == '6' else '指' + depth}"


def allocation_label(value: str | None) -> str:
    if not value:
        return "未割当"
    parts = []
    for token in value.split(" + "):
        axis, *mode = token.split("/")
        suffix = {"tip": "先端", "flat": "平面", "edge": "接触"}.get(mode[0], "") if mode else ""
        parts.append(physical_axis_label(axis) + (f"({suffix})" if suffix else ""))
    return " + ".join(parts)


def render_viewer(report: dict) -> str:
    sections = []
    for case in report["physical_evaluation"]["cases"]:
        result = case["network_v0_1"]
        rows = result["decisions"]
        end_ms = max((row["due_ms"] for row in rows), default=1.0)
        lanes = []
        for axis in teacher.AXES:
            dots = []
            for row in rows:
                if axis in row["axes"]:
                    left = row["due_ms"] / end_ms * 100.0
                    css = "dot diff" if row.get("different_from_teacher") else "dot"
                    dots.append(f'<a class="{css}" style="left:{left:.3f}%" href="#c{case["bpm"]}e{row["event_index"]}"></a>')
            lanes.append(f'<div class="lane"><span>{escape(physical_axis_label(axis))}</span><div class="track">{"".join(dots)}</div></div>')
        detail = []
        for row in rows:
            candidates = "".join(
                f'<tr class="{"chosen" if c["finger"] == row["chosen"] else ""}"><td>{escape(allocation_label(c["finger"]))}</td>'
                f'<td>{"—" if c["network_score"] is None else format(c["network_score"], ".4f")}</td>'
                f'<td>{"—" if c["teacher_score"] is None else c["teacher_score"]}</td><td>{escape(c["teacher_reason"])}</td></tr>'
                for c in row.get("candidates", [])
            )
            keydowns = ", ".join(f'{value["time_ms"]}ms {physical_axis_label(value["physical_axis"])}'
                                  for value in row["keydowns"]) or "—"
            detail.append(
                f'<tr id="c{case["bpm"]}e{row["event_index"]}" class="{"different" if row.get("different_from_teacher") else ""}">'
                f'<td>#{row["event_index"]}</td><td>{row["due_ms"]}</td><td>{escape(allocation_label(row["chosen"]))}</td>'
                f'<td>{escape(allocation_label(row.get("teacher_chosen")))}</td><td>{escape(keydowns)}</td><td>{row.get("judgement", row["outcome"])}</td>'
                f'<td><details><summary>{len(row.get("candidates", []))}候補</summary><table><tr><th>物理部位</th><th>Network</th><th>Teacher</th><th>評価</th></tr>{candidates}</table></details></td></tr>'
            )
        sections.append(
            f'<section><h2>{case["bpm"]:.0f} BPM</h2><p>Network: {result["hits"]}/{result["targets"]} Hit, '
            f'Miss {result["misses"]}, TooEarly {result["too_early"]}, Overload {result["overloaded"]}</p>'
            + "".join(lanes)
            + '<table><tr><th>Event</th><th>due</th><th>Network</th><th>Teacher</th><th>実KeyDown</th><th>判定</th><th>候補比較</th></tr>'
            + "".join(detail) + "</table></section>"
        )
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Fingering Network v0.1</title>
<style>body{font:14px/1.5 system-ui;background:#111827;color:#e5e7eb;max-width:1400px;margin:auto;padding:16px}table{width:100%;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #374151;text-align:left}.lane{display:grid;grid-template-columns:110px 1fr;gap:8px}.track{height:18px;background:#374151;position:relative}.dot{position:absolute;width:7px;height:13px;top:2px;background:#60a5fa}.dot.diff,.different{background:#5b3b46}.chosen{outline:1px solid #fbbf24}details{max-width:650px}</style>
<h1>DMDOD Fingering Network v0.1 — 12物理軸</h1><p>橙枠はNetwork採用候補、赤背景はTeacherと異なる判断。論理キーボード名ではなく物理部位を表示。</p>''' + "".join(sections) + "</html>"


def _write_once(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(content, bytes) else "x"
    kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with path.open(mode, **kwargs) as stream:
        stream.write(content)


def run(checkpoint: Path, dataset: Path, report_path: Path, viewer: Path) -> dict:
    torch.manual_seed(SEED)
    split_patterns = {
        "train": TRAIN_PATTERNS, "validation": VALIDATION_PATTERNS, "test": TEST_PATTERNS,
    }
    split_examples = {name: collect_split(patterns) for name, patterns in split_patterns.items()}
    hashes = {name: {pattern.sha256 for pattern in patterns}
              for name, patterns in split_patterns.items()}
    if any(hashes[a] & hashes[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("pattern-series split leakage")

    model = network.FingeringRanker()
    initial_train = network.ranking_metrics(model, split_examples["train"])
    losses = network.train_fixed(
        model, split_examples["train"], epochs=EPOCHS,
        learning_rate=LEARNING_RATE, seed=SEED,
    )
    model.eval()
    split_metrics = {name: network.ranking_metrics(model, values)
                     for name, values in split_examples.items()}

    checkpoint_payload = {
        "format": network.FORMAT, "model_state_dict": model.state_dict(),
        "hidden": model.hidden, "seed": SEED, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_pattern_sha256": sorted(hashes["train"]),
        "validation_used_for_selection": False,
        "test_used_for_selection": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(checkpoint_payload, checkpoint)
    dataset_payload = {
        "format": network.FORMAT + "-dataset",
        "splits": {name: [{
            "pattern_id": pattern.pattern_id, "sha256": pattern.sha256,
            "family": pattern.family, "events": len(pattern.chord_sizes),
        } for pattern in patterns] for name, patterns in split_patterns.items()},
        "examples": {name: values for name, values in split_examples.items()},
    }
    dataset.parent.mkdir(parents=True, exist_ok=True)
    if dataset.exists():
        raise FileExistsError(dataset)
    torch.save(dataset_payload, dataset)

    cases = []
    by_policy = {name: [] for name in ("center_first", "teacher_v0_2", "network_v0_1")}
    for bpm in EVAL_BPMS:
        segment, chart_sha = _fixture(bpm)
        case = {"bpm": bpm, "chart_sha256": chart_sha}
        for policy in by_policy:
            result = run_policy(segment, policy, model if policy == "network_v0_1" else None)
            case[policy] = result
            by_policy[policy].append(result)
        cases.append(case)
        print(f"physical {bpm:.0f} BPM complete", flush=True)

    report = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "oracle_timing": True, "timing_student_used": False,
        "student_inputs": [
            "up to 3 oracle-relative event times/chord sizes/presence",
            "current 12-axis position/velocity/pressed/availability",
            "current thumb and hand-side topology",
            "candidate physical-axis allocation",
        ],
        "forbidden_student_inputs": [
            "teacher score", "teacher selected result", "counterfactual future physical result",
            "logical keyboard binding name",
        ],
        "dataset": {
            "split_unit": "whole pattern series SHA256",
            "splits": dataset_payload["splits"],
            "example_counts": {name: len(values) for name, values in split_examples.items()},
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
            "qualification_used": False, "reserve_used": False, "final_used": False,
        },
        "training": {
            "initial_train_metrics": initial_train,
            "split_metrics": split_metrics,
            "first_loss": losses[0], "last_loss": losses[-1],
            "runs": 1, "hyperparameter_search": False,
        },
        "physical_evaluation": {
            "conditions": {
                "bpms": EVAL_BPMS, "floors_per_chart": EVAL_FLOORS,
                "control_ms": 10.0, "physics_ms": 1.0,
                "same_oracle_event_times": True,
            },
            "aggregate": {policy: _aggregate(rows) for policy, rows in by_policy.items()},
            "cases": cases,
        },
        "checkpoint": str(checkpoint), "dataset_artifact": str(dataset),
        "stage_a_gate_overridden": False, "existing_checkpoint_modified": False,
    }
    _write_once(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_once(viewer, render_viewer(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.checkpoint, args.dataset, args.report, args.viewer, args.status_file)
    if any(path.exists() for path in paths):
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        run(args.checkpoint, args.dataset, args.report, args.viewer)
        status = {"pid": os.getpid(), "started_at_utc": started.isoformat(),
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "exit_code": 0, "report": str(args.report)}
        _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
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
