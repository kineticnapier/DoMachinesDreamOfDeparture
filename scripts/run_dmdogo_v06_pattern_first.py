from __future__ import annotations

"""Research-only Pattern-first scheduler, physical Motor and self-contained replay."""

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
import run_dmdogo_v04_comparison as v04
import run_fingering_continuity_comparison_v03 as v03_compare
import run_full_chart_fingering_replay_v02 as baseline

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.pattern_motor_adapter_v06 import PatternMotorMapping
from dmdod.planning.pattern_scheduler_v06 import (
    PatternEvent, PatternFirstScheduler, schedule_metrics,
)
from dmdod.planning.motor_lead_compensation_v04 import MotorLeadCompensator
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-pattern-first-physical-v0.6"


def _write_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _schedule(events, minimum_reuse_s: float):
    scheduler = PatternFirstScheduler(
        tuple(f"key_{index}" for index in range(8)),
        minimum_reuse_s=minimum_reuse_s,
        minimum_cycle_keys=2,
    )
    return scheduler.schedule(tuple(
        PatternEvent(event.event_index, event.episode_time_s, event.chord_size)
        for event in events
    ))


def run_pattern_arm(segment, *, minimum_reuse_s: float = .080,
                    stop_after_s: float | None = None,
                    wall_timeout_s: float = 900.0) -> dict:
    """Execute exactly one independent Motor episode without the Network."""
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    schedule = _schedule(events, minimum_reuse_s)
    if not schedule.feasible:
        raise RuntimeError(f"abstract schedule failed: {schedule.failure_reason}")
    assignment_by = {
        event.event_index: schedule.assignments[index]
        for index, event in enumerate(events)
    }
    mapping = PatternMotorMapping.default_eight()
    committed: dict[int, teacher.PlanItem] = {}
    selected: set[int] = set()
    decisions: list[dict] = []
    keydowns: list[dict] = []
    frames = [baseline._trace_frame(env)]
    compensator = MotorLeadCompensator()
    armed: dict[tuple[int, str], dict] = {}
    compensation_rows: list[dict] = []
    deferrals: dict[int, int] = {}
    deferral_reasons: dict[str, int] = {}
    unresolved_reasons: dict[int, str] = {}
    selection_seconds = controller_seconds = 0.0
    selection_peak = controller_peak = 0.0
    selection_calls = controller_calls = 0
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
            and not all(env._used[o] for o in events[index].target_ordinals)
        }
        live_keys = {
            (index, axis) for index, item in committed.items()
            for axis in item.axes
        }
        armed = {key: value for key, value in armed.items() if key in live_keys}
        upcoming = tuple(
            event for event in events
            if event.event_index not in selected
            and event.episode_time_s + teacher.PREVIEW_TAIL >= now
        )
        due = tuple(
            event for event in upcoming
            if event.episode_time_s <= now + compensator.max_prediction_s + 1e-9
        )
        for event in due:
            start = time.perf_counter()
            occupied = {axis for item in committed.values() for axis in item.axes}
            abstract_keys = assignment_by[event.event_index]
            desired_axes = mapping.desired_axes(abstract_keys)

            def live_eligibility(axes: tuple[str, ...]) -> tuple[bool, str]:
                allocation = tuple(teacher.Finger(axis) for axis in axes)
                return teacher.eligible(env.motor, occupied, allocation)

            choice = mapping.choose(
                abstract_keys, remaining_s=event.episode_time_s - now,
                legacy_lead_s=compensator.legacy_lead_s,
                eligible=live_eligibility,
            )
            elapsed = time.perf_counter() - start
            selection_seconds += elapsed
            selection_peak = max(selection_peak, elapsed)
            selection_calls += 1
            if choice.axes is None:
                index = event.event_index
                deferrals[index] = deferrals.get(index, 0) + 1
                deferral_reasons[choice.desired_reason] = (
                    deferral_reasons.get(choice.desired_reason, 0) + 1)
                unresolved_reasons[index] = choice.disposition
                continue

            chosen = tuple(teacher.Finger(axis) for axis in choice.axes)
            selected.add(event.event_index)
            unresolved_reasons.pop(event.event_index, None)
            item = teacher.PlanItem(event.event_index, event.episode_time_s, chosen)
            committed[event.event_index] = item
            local_replan = choice.disposition == "local_replan"
            decisions.append({
                "event_index": event.event_index,
                "ordinals": list(event.target_ordinals),
                "due_ms": round(event.episode_time_s * 1000, 3),
                "decision_ms": round(now * 1000, 3),
                "chosen": item.label,
                "axes": [finger.axis for finger in chosen],
                "abstract_keys": list(abstract_keys),
                "desired_axes": list(desired_axes),
                "physical_eligible": True,
                "local_replan": local_replan,
                "desired_ineligibility_reason": (
                    choice.desired_reason if local_replan else None),
                "physical_choice_disposition": choice.disposition,
                "physical_candidates_checked": choice.attempted,
                "deferral_frames": deferrals.get(event.event_index, 0),
                "selector_mode": "pattern_first_v06",
                "policy": "pattern_first_v06",
                "selection_reason": (
                    "physical_local_replan" if local_replan
                    else "abstract_pattern_mapping"),
                "plan_preview": [],
                "compensation_enabled": True,
                "free_chosen": None,
                "different_from_free": False,
                "style_class": "pattern_first_abstract",
                "inference_inputs": (
                    "oracle_schedule_and_live_motor_without_network_or_future_outcomes"),
            })

        start = time.perf_counter()
        action, starts = v04._compensated_action(
            env.motor, tuple(committed.values()), now, compensator, armed)
        elapsed = time.perf_counter() - start
        controller_seconds += elapsed
        controller_peak = max(controller_peak, elapsed)
        controller_calls += 1
        compensation_rows.extend(starts)
        step = env.step(action)
        for physical in env.last_events:
            if physical.event is not KeyEvent.DOWN:
                continue
            axis = v01._physical_axis(physical.key)
            keydowns.append({
                "episode_s": round(physical.time_s, 6),
                "time_ms": round(physical.time_s * 1000, 3),
                "axis": axis,
                "contact": physical.key,
            })
            matching = sorted(
                (key for key in armed if key[1] == axis),
                key=lambda key: armed[key]["due_s"])
            if matching:
                record = armed.pop(matching[0])
                record["actual_keydown_s"] = physical.time_s
                record["actual_latency_s"] = (
                    physical.time_s - record["command_start_s"])
                if record["predicted_keydown_s"] is not None:
                    record["prediction_error_ms"] = 1000.0 * (
                        physical.time_s - record["predicted_keydown_s"])
        frames.append(baseline._trace_frame(env))
        step_done = step.done
        if step.done or (stop_after_s is not None
                         and env.privileged_episode_time_s() >= stop_after_s):
            break
    else:
        raise RuntimeError("v0.6 execution exceeded step budget")

    for row in decisions:
        used = [env._used[index] for index in row["ordinals"]]
        missed = [env._missed[index] for index in row["ordinals"]]
        row["outcome"] = (
            "Hit" if all(used)
            else "Miss" if any(missed) else "Unresolved")
        # Diagnostic matching only: these are not canonical scorer associations.
        nearby = [
            value for value in keydowns if value["axis"] in row["axes"]
            and abs(value["time_ms"] - row["due_ms"]) <= 35.0
        ]
        row["keydowns"] = [
            min((value for value in nearby if value["axis"] == axis),
                key=lambda value: abs(value["time_ms"] - row["due_ms"]))
            for axis in row["axes"]
            if any(value["axis"] == axis for value in nearby)
        ]
        errors = [value["time_ms"] - row["due_ms"] for value in row["keydowns"]]
        row["actual_error_ms"] = (
            round(min(errors, key=abs), 3) if errors else None)
        early = [
            trace for trace in env.judgement_trace
            if trace["target_index"] in row["ordinals"]
            and trace["margin"] is None
            and trace["time_s"] * 1000.0 < row["due_ms"]
        ]
        row["too_early_keydowns"] = len(early)
        row["judgement"] = (
            ("TooEarly + " if early else "") + row["outcome"])

    stats = baseline._summarize(env)
    end_s = env.privileged_episode_time_s()
    reason = (
        "diagnostic_prefix_complete" if stop_after_s is not None and not step_done
        else "overload" if stats["overloaded"]
        else "miss" if stats["misses"] and end_s + teacher.CTRL < segment.duration_s
        else "chart_complete"
    )
    physical = {
        "local_replans": sum(row["local_replan"] for row in decisions),
        "deferred_events": sum(row["deferral_frames"] > 0 for row in decisions),
        "deferral_frames": sum(deferrals.values()),
        "deferral_reasons": dict(sorted(deferral_reasons.items())),
        "unassigned_events": len(events) - len(decisions),
        "unassigned_event_indices": [
            event.event_index for event in events
            if event.event_index not in selected
        ],
        "unresolved_reasons": {
            str(index): value for index, value in unresolved_reasons.items()
        },
        "selection_calls": selection_calls,
        "mean_selection_ms": (
            1000.0 * selection_seconds / selection_calls
            if selection_calls else None),
        "max_selection_ms": 1000.0 * selection_peak,
        "controller_calls": controller_calls,
        "mean_controller_ms": (
            1000.0 * controller_seconds / controller_calls
            if controller_calls else None),
        "max_controller_ms": 1000.0 * controller_peak,
    }
    return {
        "stats": stats,
        "decisions": decisions,
        "keydowns": keydowns,
        "trace": frames,
        "replay_end_s": round(end_s, 6),
        "termination_reason": reason,
        "policy": "pattern_first_v06",
        "selector_mode": "pattern_first_v06",
        "network_used": False,
        "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False,
        "training_performed": False,
        "compensation_telemetry": compensation_rows,
        "abstract_schedule": {
            "feasible": schedule.feasible,
            "metrics": schedule_metrics(schedule),
            "patterns": [
                {
                    "start_event_index": span.start_event_index,
                    "end_event_index": span.end_event_index,
                    "width": span.width,
                    "cycle": list(span.cycle),
                    "event_count": span.event_count,
                }
                for span in schedule.patterns
            ],
        },
        "physical_mapping": dict(zip(
            mapping.logical_keys, mapping.physical_axes)),
        "physical": physical,
    }
