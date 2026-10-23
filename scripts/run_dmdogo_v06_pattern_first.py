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
from dmdod.planning.human_phrase_planner_v07 import HumanPhrasePlannerV07, PhraseEvent
from dmdod.planning.pattern_scheduler_v06 import (
    PatternEvent, PatternFirstScheduler, schedule_metrics,
)
from dmdod.planning.motor_lead_compensation_v04 import MotorLeadCompensator
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdogo-pattern-first-physical-v0.6"
FORMAT_V07 = "dmdogo-human-phrase-physical-v0.7"


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
                    wall_timeout_s: float = 900.0,
                    planning_mode: str = "pattern_v06") -> dict:
    """Execute exactly one independent Motor episode without the Network."""
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    if planning_mode not in {"pattern_v06", "phrase_v07"}:
        raise ValueError("planning_mode must be pattern_v06 or phrase_v07")
    mapping = PatternMotorMapping.default_eight()
    phrase_by_event = {}
    if planning_mode == "pattern_v06":
        schedule = _schedule(events, minimum_reuse_s)
        if not schedule.feasible:
            raise RuntimeError(f"abstract schedule failed: {schedule.failure_reason}")
        assignment_by = {
            event.event_index: schedule.assignments[index]
            for index, event in enumerate(events)
        }
        abstract_plan = {
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
        }
    else:
        phrase_plan = HumanPhrasePlannerV07(
            minimum_reuse_s=minimum_reuse_s,
        ).plan(tuple(
            PhraseEvent(event.event_index, event.episode_time_s, event.chord_size)
            for event in events
        ))
        if not phrase_plan.feasible:
            raise RuntimeError(f"v0.7 offline phrase plan failed: {phrase_plan.failure_reason}")
        axis_to_key = dict(zip(mapping.physical_axes, mapping.logical_keys))
        assignment_by = {
            event.event_index: tuple(axis_to_key[axis] for axis in phrase_plan.assignments[index])
            for index, event in enumerate(events)
        }
        for phrase in phrase_plan.phrases:
            for index in range(phrase.start_event_index, phrase.end_event_index + 1):
                phrase_by_event[index] = phrase
        abstract_plan = {
            "feasible": phrase_plan.feasible,
            "metrics": phrase_plan.metrics,
            "model_contract": phrase_plan.model_contract,
            "phrases": [
                {
                    "start_event_index": p.start_event_index,
                    "end_event_index": p.end_event_index,
                    "event_count": p.event_count,
                    "rhythm_class": p.rhythm_class,
                    "gap_motif_s": list(p.gap_motif_s),
                    "boundary_reason": p.boundary_reason,
                    "gesture_template": p.template,
                    "mirrored": p.mirrored,
                    "phase": p.phase,
                }
                for p in phrase_plan.phrases
            ],
        }
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
                "selector_mode": (
                    "human_phrase_v07" if planning_mode == "phrase_v07"
                    else "pattern_first_v06"),
                "policy": (
                    "human_phrase_v07" if planning_mode == "phrase_v07"
                    else "pattern_first_v06"),
                "selection_reason": (
                    "physical_local_replan" if local_replan
                    else "rhythm_phrase_template" if planning_mode == "phrase_v07"
                    else "abstract_pattern_mapping"),
                "rhythm_class": (
                    phrase_by_event[event.event_index].rhythm_class
                    if event.event_index in phrase_by_event else None),
                "gesture_template": (
                    phrase_by_event[event.event_index].template
                    if event.event_index in phrase_by_event else None),
                "plan_preview": [],
                "compensation_enabled": True,
                "free_chosen": None,
                "different_from_free": False,
                "style_class": (
                    "untrained_rhythm_phrase_hypothesis" if planning_mode == "phrase_v07"
                    else "pattern_first_abstract"),
                "inference_inputs": (
                    "offline_oracle_rhythm_phrase_and_live_motor_no_future_outcomes"
                    if planning_mode == "phrase_v07"
                    else "oracle_schedule_and_live_motor_without_network_or_future_outcomes"),
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
        "policy": ("human_phrase_v07" if planning_mode == "phrase_v07"
                   else "pattern_first_v06"),
        "selector_mode": ("human_phrase_v07" if planning_mode == "phrase_v07"
                          else "pattern_first_v06"),
        "planning_mode": planning_mode,
        "network_used": False,
        "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False,
        "training_performed": False,
        "compensation_telemetry": compensation_rows,
        "abstract_schedule": abstract_plan,
        "physical_mapping": dict(zip(
            mapping.logical_keys, mapping.physical_axes)),
        "physical": physical,
    }


def build_payload(chart_path: Path, compiled, segment, result: dict) -> dict:
    """A self-contained single-policy replay; no Network was used."""
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_event = {int(row["event_index"]): row for row in result["decisions"]}
    is_v07 = result.get("planning_mode") == "phrase_v07"
    policy_key = "human_phrase_v07" if is_v07 else "pattern_first_v06"
    replay_end = float(result["replay_end_s"])
    events = []
    for event in oracle.events:
        if event.episode_time_s > replay_end + teacher.PREVIEW_TAIL:
            break
        events.append({
            "event_index": event.event_index,
            "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices),
            "chord_size": event.chord_size,
            "different": False,
            "decisions": {
                policy_key: v03_compare.v02_compare._decision_view(
                    by_event.get(event.event_index), None),
            },
        })
    return {
        "format": FORMAT_V07 if is_v07 else FORMAT,
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "default_policy": policy_key,
        "timebase": {
            "episode_duration_s": replay_end,
            "full_chart_duration_s": segment.duration_s,
            "segment_start_chart_s": segment.start_chart_s,
            "segment_end_chart_s": segment.end_chart_s,
            "pitch_percent": compiled.pitch_percent,
            "pitch_ratio": segment.pitch_ratio,
            "offset_ms": compiled.offset_ms,
            "separate_countdown_chart_s": (
                compiled._separate_countdown_chart_seconds()),
        },
        "floors": viewer.rotation_profiles(
            load_adofai(chart_path), compiled, segment),
        "events": events,
        "policies": {
            policy_key: {
                "label": ("Rhythm phrase v0.7 (untrained physical)" if is_v07
                          else "Pattern-first v0.6 (physical)"),
                "style_alpha": 0.0,
                "trace": result["trace"],
                "keydowns": result["keydowns"],
                "saved_metrics": result["stats"],
            },
        },
        "key_overlay": {
            "mapping": [
                {"label": label, "contact": contact}
                for label, contact in viewer.KEY_DISPLAY_MAP
            ],
            "mapping_semantics": (
                "Display aliases, not verified OS-key bindings."),
        },
        "limitations": [
            "v0.7 rhythm gesture templates are untrained hypotheses, not human demonstrations.",
            "Offline Oracle timing is privileged and not HUD-only Student perception.",
            "Pattern-first uses no learned Network and only ordinary physical axes.",
            "Abstract periodic schedules do not establish physical Hit feasibility.",
            "Physical local alternatives may break the abstract key cycle.",
            "Nearest-axis diagnostic KeyDowns do not prove event correspondence.",
        ],
    }


def run(chart_path: Path, output_dir: Path, *,
        minimum_reuse_s: float = .080,
        stop_after_s: float | None = None,
        wall_timeout_s: float = 900.0,
        planning_mode: str = "pattern_v06") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if chart_path.suffix.lower() != ".adofai":
        raise ValueError("chart must be a .adofai file")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    started = time.perf_counter()
    result = run_pattern_arm(
        segment, minimum_reuse_s=minimum_reuse_s,
        stop_after_s=stop_after_s, wall_timeout_s=wall_timeout_s,
        planning_mode=planning_mode)
    oracle = AutoplayTimingTeacher.from_segment(segment)
    payload = build_payload(chart_path, compiled, segment, result)
    report = {
        "format": FORMAT_V07 if planning_mode == "phrase_v07" else FORMAT,
        "planning_mode": planning_mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name,
        "chart_sha256": v01.file_sha256(chart_path),
        "total_events": len(oracle.events),
        "duration_s": segment.duration_s,
        "runtime_s": round(time.perf_counter() - started, 3),
        "termination_reason": result["termination_reason"],
        "stats": result["stats"],
        "physical": result["physical"],
        "abstract_schedule": result["abstract_schedule"],
        "network_used": False,
        "training_performed": False,
        "teacher_candidate_scores_used": False,
        "timing_source": "AutoPlay Oracle exact event times",
        "minimum_reuse_s": minimum_reuse_s,
        "partial_run": result["termination_reason"] == "diagnostic_prefix_complete",
    }
    rendered = viewer.render_html(payload)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "summary.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    suffix = "v07" if planning_mode == "phrase_v07" else "v06"
    _write_once(output_dir / f"replay-{suffix}.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / f"replay-{suffix}.html", rendered)
    _write_once(output_dir / f"telemetry-{suffix}.json",
                json.dumps({
                    "compensation": result["compensation_telemetry"],
                    "physical": result["physical"],
                    "decisions": result["decisions"],
                }, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status-file", required=True, type=Path)
    parser.add_argument("--minimum-reuse-ms", type=float, default=80.0)
    parser.add_argument("--planning-mode", choices=("pattern_v06", "phrase_v07"),
                        default="pattern_v06")
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.status_file.exists():
        parser.error("refusing to overwrite existing output or status")
    started = datetime.now(timezone.utc)
    try:
        report = run(
            args.chart, args.output_dir,
            minimum_reuse_s=args.minimum_reuse_ms * .001,
            stop_after_s=args.stop_after_s,
            wall_timeout_s=args.wall_timeout_s,
            planning_mode=args.planning_mode)
        status = {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "exit_code": 0,
            "output_dir": str(args.output_dir),
            "termination_reason": report["termination_reason"],
        }
    except BaseException as error:
        status = {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "exit_code": 1,
            "output_dir": str(args.output_dir),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if not args.status_file.exists():
            _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
        raise
    _write_once(args.status_file, json.dumps(status, indent=2) + "\n")
    suffix = "v07" if args.planning_mode == "phrase_v07" else "v06"
    print(f"DMDOGO {suffix}: {args.output_dir / f'replay-{suffix}.html'}")


if __name__ == "__main__":
    main()
