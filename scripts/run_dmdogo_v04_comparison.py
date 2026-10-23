from __future__ import annotations

"""Fixed six-arm DMDOGO v0.4 Motor-latency and short-horizon comparison."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as v01
import build_fingering_replay_viewer_v03 as viewer
import run_fingering_continuity_comparison_v02 as v02_compare
import run_fingering_continuity_comparison_v03 as v03_compare
import run_full_chart_fingering_replay_v02 as baseline

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.evaluation.fingering_continuity_v03 import common_fast_metrics
from dmdod.motor.contact_16k import ContactAction, PHYSICAL_AXES, ThumbMode
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.fingering_continuity_v02 import ContinuitySelector, continuity_metrics
from dmdod.planning.fingering_planner_v04 import (
    CandidateNode, EventCandidates, ShortHorizonFingeringPlanner,
)
from dmdod.planning.motor_lead_compensation_v04 import MotorLeadCompensator
from dmdod.training import fingering_network_v01 as network
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-motor-planning-comparison-v0.4"
POLICY_ORDER = (
    "free", "continuity", "continuity_v03",
    "latency_only", "planning_only", "integrated_v04",
)
LABELS = {
    "free": "Free",
    "continuity": "Continuity v0.2",
    "continuity_v03": "Continuity v0.3",
    "latency_only": "Latency-only",
    "planning_only": "Planning-only",
    "integrated_v04": "Integrated v0.4",
}


def _write_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _allocation_id(allocation: tuple[teacher.Finger, ...]) -> str:
    return " + ".join(finger.label for finger in allocation)


def _ranked_nodes(model, motor, event, look, now: float, occupied: set[str],
                  *, live: bool) -> tuple[tuple[CandidateNode, ...], tuple, tuple]:
    allocations = teacher.finger_options(event.chord_size)
    eligibility = tuple(teacher.eligible(motor, occupied, value)[0]
                        for value in allocations)
    context = network.encode_context(motor, look, now, occupied)
    _, scores = network.choose_candidate(model, context, allocations, eligibility)
    ranked = sorted((index for index, value in enumerate(scores) if value is not None),
                    key=lambda index: (scores[index], index))
    rank_by = {index: rank for rank, index in enumerate(ranked)}
    nodes = tuple(
        CandidateNode(event.event_index, event.episode_time_s * 1000.0,
                      _allocation_id(allocation),
                      tuple(finger.axis for finger in allocation),
                      rank_by.get(index, len(allocations)), bool(eligibility[index]))
        for index, allocation in enumerate(allocations)
        if index in rank_by
    )
    return nodes, allocations, scores


def _planner_choice(planner: ShortHorizonFingeringPlanner, model, motor,
                    event, look, now: float, occupied: set[str]) -> tuple:
    groups = []
    current_allocations = current_scores = None
    for offset, future in enumerate(look[:planner.horizon]):
        future_look = look[offset:]
        nodes, allocations, scores = _ranked_nodes(
            model, motor, future, future_look, now, occupied, live=offset == 0)
        groups.append(EventCandidates(future.event_index,
                                      future.episode_time_s * 1000.0, nodes))
        if offset == 0:
            current_allocations, current_scores = allocations, scores
    current_nodes = groups[0].candidates
    cached = planner.cached_choice(event.event_index, current_nodes)
    if cached is not None:
        selected_id = cached.allocation_id
        preview = [selected_id]
        reason = "cached_short_horizon_plan"
        evaluated = 0
        peak = 0
        objective = None
    else:
        plan = planner.plan(tuple(groups))
        planner.commit(plan)
        if not plan.allocations:
            return None, current_allocations, current_scores, (), {
                "reason": "no_bounded_plan", "preview": [],
                "evaluated": plan.evaluated_candidates, "peak": plan.peak_beam,
                "objective": list(plan.objective),
            }
        selected_id = plan.allocations[0].allocation_id
        preview = [node.allocation_id for node in plan.allocations]
        reason = "bounded_beam_plan"
        evaluated = plan.evaluated_candidates
        peak = plan.peak_beam
        objective = list(plan.objective)
    selected = next((index for index, allocation in enumerate(current_allocations)
                     if _allocation_id(allocation) == selected_id), None)
    return selected, current_allocations, current_scores, current_nodes, {
        "reason": reason, "preview": preview, "evaluated": evaluated,
        "peak": peak, "objective": objective,
    }


def _compensated_action(motor, plans: tuple[teacher.PlanItem, ...], now: float,
                        compensator: MotorLeadCompensator,
                        armed: dict[tuple[int, str], dict]) -> tuple[ContactAction, list[dict]]:
    observation = motor.motor.observe()
    commands = {axis: -1.0 if observation.pressed(axis) else 0.0
                for axis in PHYSICAL_AXES}
    modes = list(motor._modes)
    edges = list(motor._edge)
    started = []
    for item in plans:
        if now > item.due_s + teacher.PREVIEW_TAIL:
            continue
        for finger in item.fingers:
            key = (item.event_index, finger.axis)
            if observation.pressed(finger.axis):
                continue
            if key not in armed:
                prediction = compensator.predict(motor, axis=finger.axis, mode=finger.mode)
                decision = compensator.decide(now_s=now, due_s=item.due_s,
                                               prediction=prediction)
                if not decision.start:
                    continue
                if not teacher._safe_topology(motor, finger):
                    continue
                record = {
                    "event_index": item.event_index, "axis": finger.axis,
                    "mode": finger.mode, "due_s": item.due_s,
                    "command_start_s": now,
                    "predicted_latency_s": prediction.earliest_arrival_s,
                    "predicted_keydown_s": decision.predicted_keydown_s,
                    "prediction_valid": prediction.valid,
                    "fallback": decision.fallback, "reason": decision.reason,
                    "contacts": list(prediction.contacts),
                }
                armed[key] = record
                started.append(record)
            hand, depth_text = finger.axis.split("_")
            hand_index = 0 if hand == "left" else 1
            depth = int(depth_text)
            if depth == 5:
                modes[hand_index] = (ThumbMode.FLAT if finger.mode == "flat"
                                     else ThumbMode.TIP)
            elif depth == 6:
                edges[hand_index] = True
            commands[finger.axis] = 1.0
    return ContactAction(tuple(commands[axis] for axis in PHYSICAL_AXES),
                         tuple(modes), tuple(edges)), started


def run_research_arm(segment, model, *, policy_name: str,
                     planning: bool, compensation: bool,
                     stop_after_s: float | None, wall_timeout_s: float,
                     fixed_choices: dict[int, str] | None = None,
                     selector_mode: str = "free") -> dict:
    if selector_mode not in {"free", "continuity", "continuity_v03"}:
        raise ValueError("unsupported causal selector mode")
    env = CoupledContactRealChartEnv(segment, control_dt_s=teacher.CTRL,
                                     physics_dt_s=teacher.PHYS)
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    decisions = []
    keydowns = []
    frames = [baseline._trace_frame(env)]
    planner = ShortHorizonFingeringPlanner()
    continuity_selector = ContinuitySelector(mode=selector_mode)
    compensator = MotorLeadCompensator()
    armed: dict[tuple[int, str], dict] = {}
    compensation_rows = []
    planner_evaluated = planner_peak = 0
    inference_seconds = 0.0
    max_inference_seconds = 0.0
    inference_calls = 0
    controller_seconds = 0.0
    max_controller_seconds = 0.0
    controller_calls = 0
    horizon = compensator.max_prediction_s if compensation else teacher.LEAD
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    wall_started = time.monotonic()
    step_done = False
    for _ in range(maximum):
        if time.monotonic() - wall_started > wall_timeout_s:
            raise TimeoutError(f"physical execution exceeded {wall_timeout_s:.1f}s")
        now = env.privileged_episode_time_s()
        committed = {
            index: item for index, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[ordinal] for ordinal in events[index].target_ordinals)
        }
        live_keys = {(index, axis) for index, item in committed.items() for axis in item.axes}
        armed = {key: value for key, value in armed.items() if key in live_keys}
        upcoming = tuple(event for event in events if event.event_index not in selected
                         and event.episode_time_s + teacher.PREVIEW_TAIL >= now)
        due = tuple(event for event in upcoming
                    if event.episode_time_s <= now + horizon + 1e-9)
        for event in due:
            occupied = {axis for item in committed.values() for axis in item.axes}
            look = tuple(value for value in upcoming if value.event_index >= event.event_index)
            started_inference = time.perf_counter()
            allocations = teacher.finger_options(event.chord_size)
            eligibility = tuple(teacher.eligible(env.motor, occupied, value)[0]
                                for value in allocations)
            planning_info = {"reason": "free_network_argmin", "preview": [],
                             "evaluated": 0, "peak": 0, "objective": None}
            if fixed_choices is not None:
                wanted = fixed_choices[event.event_index]
                selected_index = next((index for index, allocation in enumerate(allocations)
                                       if _allocation_id(allocation) == wanted), None)
                if selected_index is None or not eligibility[selected_index]:
                    raise RuntimeError(
                        f"fixed allocation is not live-eligible at event {event.event_index}")
                free_index = selected_index
                scores = tuple(0.0 if index == selected_index else None
                               for index in range(len(allocations)))
                planning_info["reason"] = "fixed_free_allocation_posthoc_ablation"
                continuity_costs = tuple(None for _ in allocations)
                combined_rank_scores = tuple(None for _ in allocations)
            else:
                context = network.encode_context(env.motor, look, now, occupied)
                free_index, scores = network.choose_candidate(
                    model, context, allocations, eligibility)
                selected_index = free_index
                if selector_mode != "free":
                    selection = continuity_selector.select(
                        scores, allocations, eligibility,
                        event.event_index, event.episode_time_s)
                    selected_index = selection.selected_index
                    planning_info["reason"] = selection.reason
                    continuity_costs = selection.continuity_costs
                    combined_rank_scores = selection.combined_rank_scores
                else:
                    continuity_costs = tuple(None for _ in allocations)
                    combined_rank_scores = tuple(None for _ in allocations)
            if planning and fixed_choices is None:
                selected_index, allocations, scores, _, planning_info = _planner_choice(
                    planner, model, env.motor, event, look, now, occupied)
                planner_evaluated += planning_info["evaluated"]
                planner_peak = max(planner_peak, planning_info["peak"])
            inference_elapsed = time.perf_counter() - started_inference
            inference_seconds += inference_elapsed
            max_inference_seconds = max(max_inference_seconds, inference_elapsed)
            inference_calls += 1
            allocation = allocations[selected_index] if selected_index is not None else ()
            if selected_index is not None and not eligibility[selected_index]:
                raise RuntimeError("research policy selected an ineligible live allocation")
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
            decisions.append({
                "event_index": event.event_index, "ordinals": list(event.target_ordinals),
                "due_ms": round(event.episode_time_s * 1000.0, 3),
                "decision_ms": round(now * 1000.0, 3),
                "chosen": item.label if allocation else None,
                "axes": [finger.axis for finger in allocation],
                "network_scores": list(scores), "eligible": list(eligibility),
                "continuity_costs": list(continuity_costs),
                "combined_rank_scores": list(combined_rank_scores),
                "selected_candidate_index": selected_index,
                "free_chosen": (teacher.PlanItem(event.event_index, event.episode_time_s,
                                                   allocations[free_index]).label
                                if free_index is not None else None),
                "different_from_free": selected_index != free_index,
                "selector_mode": policy_name, "policy": policy_name,
                "selection_reason": planning_info["reason"],
                "plan_preview": planning_info["preview"],
                "planner_objective": planning_info["objective"],
                "planner_candidates_evaluated": planning_info["evaluated"],
                "planner_peak_beam": planning_info["peak"],
                "compensation_enabled": compensation,
                "inference_inputs": "oracle_future_times_current_motor_no_future_outcomes",
                "style_class": ("index_proxy" if allocation and all(f.axis.endswith("_1") for f in allocation)
                                else "ordinary_proxy" if allocation and all(int(f.axis.split("_")[1]) <= 4 for f in allocation)
                                else "thumb_or_side" if allocation else "unassigned"),
            })
        if compensation:
            controller_started = time.perf_counter()
            action, starts = _compensated_action(
                env.motor, tuple(committed.values()), now, compensator, armed)
            controller_elapsed = time.perf_counter() - controller_started
            controller_seconds += controller_elapsed
            max_controller_seconds = max(max_controller_seconds, controller_elapsed)
            controller_calls += 1
            compensation_rows.extend(starts)
        else:
            action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for physical in env.last_events:
            if physical.event is not KeyEvent.DOWN:
                continue
            axis = v01._physical_axis(physical.key)
            keydowns.append({"episode_s": round(physical.time_s, 6),
                             "time_ms": round(physical.time_s * 1000.0, 3),
                             "axis": axis, "contact": physical.key})
            matching = sorted((key for key in armed if key[1] == axis),
                              key=lambda key: armed[key]["due_s"])
            if matching:
                record = armed.pop(matching[0])
                record["actual_keydown_s"] = physical.time_s
                record["actual_latency_s"] = physical.time_s - record["command_start_s"]
                if record["predicted_keydown_s"] is not None:
                    record["prediction_error_ms"] = 1000.0 * (
                        physical.time_s - record["predicted_keydown_s"])
        frames.append(baseline._trace_frame(env))
        step_done = step.done
        if step.done or (stop_after_s is not None
                         and env.privileged_episode_time_s() >= stop_after_s):
            break
    else:
        raise RuntimeError("v0.4 execution exceeded step budget")

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
        row["too_early_keydowns"] = sum(
            trace["target_index"] in row["ordinals"] and trace["margin"] is None
            and trace["time_s"] * 1000.0 < row["due_ms"]
            for trace in env.judgement_trace
        )
        row["judgement"] = (("TooEarly + " if row["too_early_keydowns"] else "")
                            + row["outcome"])
    stats = baseline._summarize(env)
    end_s = env.privileged_episode_time_s()
    reason = ("diagnostic_prefix_complete" if stop_after_s is not None and not step_done
              else "overload" if stats["overloaded"]
              else "miss" if stats["misses"] and end_s + teacher.CTRL < segment.duration_s
              else "chart_complete")
    return {
        "stats": stats, "decisions": decisions, "keydowns": keydowns,
        "trace": frames, "replay_end_s": round(end_s, 6),
        "termination_reason": reason, "failure_continuation": False,
        "teacher_shadow_rollout_used": False, "teacher_candidate_scores_used": False,
        "policy": policy_name, "style_alpha": 0.0, "selector_mode": policy_name,
        "causal_selector_mode": selector_mode,
        "compensation_telemetry": compensation_rows,
        "planner": {"enabled": planning, "replans": planner.replan_count,
                    "discards": planner.discard_count,
                    "evaluated_candidates": planner_evaluated,
                    "peak_beam": planner_peak,
                    "inference_seconds": inference_seconds,
                    "inference_calls": inference_calls,
                    "mean_inference_ms": (1000.0 * inference_seconds / inference_calls
                                          if inference_calls else 0.0),
                    "max_inference_ms": 1000.0 * max_inference_seconds,
                    "controller_seconds": controller_seconds,
                    "controller_calls": controller_calls,
                    "mean_controller_ms": (1000.0 * controller_seconds / controller_calls
                                           if controller_calls else 0.0),
                    "max_controller_ms": 1000.0 * max_controller_seconds},
        "fixed_choices_posthoc": fixed_choices is not None,
    }


def _decision_view(row: dict | None) -> dict | None:
    result = v02_compare._decision_view(row, None)
    if result is not None and row is not None:
        result["plan_preview"] = row.get("plan_preview", [])
        result["compensation_enabled"] = row.get("compensation_enabled", False)
        result["reason"] = row.get("selection_reason")
    return result


def build_payload(chart_path: Path, checkpoint: Path, compiled, segment,
                  results: dict[str, dict]) -> dict:
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_policy = {name: {int(row["event_index"]): row for row in result["decisions"]}
                 for name, result in results.items()}
    replay_end = max(float(result["replay_end_s"]) for result in results.values())
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + teacher.PREVIEW_TAIL:
            break
        rows = {name: by_policy[name].get(event.event_index) for name in POLICY_ORDER}
        choices = {row.get("chosen") for row in rows.values() if row is not None}
        events.append({
            "event_index": event.event_index, "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices), "chord_size": event.chord_size,
            "different": len(choices) > 1,
            "decisions": {name: _decision_view(rows[name]) for name in POLICY_ORDER},
        })
    return {
        "format": FORMAT, "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": v01.file_sha256(checkpoint),
        "default_policy": "integrated_v04",
        "timebase": {"episode_duration_s": replay_end,
                     "full_chart_duration_s": segment.duration_s,
                     "segment_start_chart_s": segment.start_chart_s,
                     "segment_end_chart_s": segment.end_chart_s,
                     "pitch_percent": compiled.pitch_percent,
                     "pitch_ratio": segment.pitch_ratio, "offset_ms": compiled.offset_ms,
                     "separate_countdown_chart_s": compiled._separate_countdown_chart_seconds()},
        "floors": viewer.rotation_profiles(load_adofai(chart_path), compiled, segment),
        "events": events,
        "policies": {name: {"label": LABELS[name], "style_alpha": 0.0,
                            "trace": results[name]["trace"],
                            "keydowns": results[name]["keydowns"],
                            "saved_metrics": results[name]["stats"]}
                     for name in POLICY_ORDER},
        "key_overlay": {"mapping": [{"label": label, "contact": contact}
                                     for label, contact in viewer.KEY_DISPLAY_MAP],
                        "mapping_semantics": "Display aliases, not verified OS-key bindings."},
        "limitations": [
            "Six independent Motor episodes use one frozen checkpoint and exact AutoPlay times.",
            "Latency prediction is a current-state isolated Motor forward model, not future observation.",
            "Future planner eligibility is provisional; live eligibility is rechecked before commit.",
        ],
    }


def _telemetry_summary(rows: list[dict]) -> dict:
    complete = [row for row in rows if "actual_latency_s" in row]
    errors = [row["prediction_error_ms"] for row in complete
              if "prediction_error_ms" in row]
    return {
        "command_starts": len(rows), "completed": len(complete),
        "valid_predictions": sum(row["prediction_valid"] for row in rows),
        "fallbacks": sum(row["fallback"] for row in rows),
        "prediction_error_mae_ms": (sum(abs(value) for value in errors) / len(errors)
                                    if errors else None),
        "prediction_error_signed_ms": (sum(errors) / len(errors) if errors else None),
        "reasons": {reason: sum(row["reason"] == reason for row in rows)
                    for reason in sorted({row["reason"] for row in rows})},
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
        "free": baseline.run_network(segment, model, stop_after_s=stop_after_s,
                                     wall_timeout_s=wall_timeout_s,
                                     selector_mode="free", policy_name="free"),
        "continuity": baseline.run_network(segment, model, stop_after_s=stop_after_s,
                                           wall_timeout_s=wall_timeout_s,
                                           selector_mode="continuity", policy_name="continuity"),
        "continuity_v03": baseline.run_network(segment, model, stop_after_s=stop_after_s,
                                               wall_timeout_s=wall_timeout_s,
                                               selector_mode="continuity_v03",
                                               policy_name="continuity_v03"),
        "latency_only": run_research_arm(segment, model, policy_name="latency_only",
                                         planning=False, compensation=True,
                                         stop_after_s=stop_after_s,
                                         wall_timeout_s=wall_timeout_s),
        "planning_only": run_research_arm(segment, model, policy_name="planning_only",
                                          planning=True, compensation=False,
                                          stop_after_s=stop_after_s,
                                          wall_timeout_s=wall_timeout_s),
        "integrated_v04": run_research_arm(segment, model, policy_name="integrated_v04",
                                           planning=True, compensation=True,
                                           stop_after_s=stop_after_s,
                                           wall_timeout_s=wall_timeout_s),
    }
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_times = {event.event_index: event.episode_time_s for event in oracle.events}
    summaries = {}
    for name, result in results.items():
        summary = v03_compare.policy_summary(result)
        summary["compensation"] = _telemetry_summary(result.get("compensation_telemetry", []))
        summary["planner"] = result.get("planner", {"enabled": False})
        summaries[name] = summary
    report = {
        "format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name, "chart_sha256": v01.file_sha256(chart_path),
        "checkpoint_sha256": v01.file_sha256(checkpoint),
        "events": len(oracle.events), "duration_s": segment.duration_s,
        "runtime_s": round(time.perf_counter() - started, 3),
        "policies": summaries,
        "pairwise_from_free": {name: v03_compare._pairwise(results["free"], results[name])
                               for name in POLICY_ORDER[1:]},
        "common_fast": common_fast_metrics(
            {name: result["decisions"] for name, result in results.items()}, event_times),
        "frozen": {"network_checkpoint": True, "oracle_due_times": True,
                   "motor_constants": True, "scorer": True,
                   "baseline_paths": True},
        "training_performed": False, "hyperparameter_search": False,
        "teacher_shadow_rollout_used": False,
    }
    payload = build_payload(chart_path, checkpoint, compiled, segment, results)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "comparison-v04.json", json.dumps(report, indent=2) + "\n")
    _write_once(output_dir / "replay-v04.json", json.dumps(payload, indent=2) + "\n")
    _write_once(output_dir / "replay-v04.html", viewer.render_html(payload))
    _write_once(output_dir / "telemetry-v04.json", json.dumps(
        {name: {"compensation": result.get("compensation_telemetry", []),
                "planner": result.get("planner", {})}
         for name, result in results.items()}, indent=2) + "\n")
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
                  "terminations": {name: report["policies"][name]["termination_reason"]
                                   for name in POLICY_ORDER}}
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
