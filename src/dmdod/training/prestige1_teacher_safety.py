from __future__ import annotations

"""Prestige I v2.3 Train20 Expert/Teacher physical safety audit, no learning."""

from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.capacity import CenterFirstNKeyTeacher

from . import prestige1 as p
from . import prestige1_alignment_audit as audit
from . import prestige1_transfer as transfer
from .n_key import _teacher_action
from .prestige1_target_contract import contract_from_env

FORMAT = "prestige1-stage-c-v23-train-only-teacher-safety"
REPORT = "stage_c_v23_teacher_safety.json"
SOURCE_REPORT = audit.REPORT
LAST_FRAMES = 12
LAST_EVENT_WINDOW_S = .150


def _run_teacher(named, config, *, anchor):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment, key_count=p.KEY_COUNT, capture_overload_trace=True,
        control_dt_s=p.CONTROL_DT, physics_dt_s=p.PHYSICS_DT,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    teacher = CenterFirstNKeyTeacher(p.KEY_COUNT)
    obs = env.reset()
    last_frames = deque(maxlen=LAST_FRAMES)
    phase_counts = Counter()
    positive_command_counts = Counter()
    positive_frames = 0
    max_steps = math.ceil((named.segment.duration_s + 2.0)/p.CONTROL_DT)+200
    for i in range(max_steps):
        visible = encode_n_key_hud_real_chart_observation(obs)
        if len(visible) != p.INPUT_DIM:
            raise RuntimeError("263D HUD changed")
        now_s = float(env.privileged_episode_time_s())
        contract = contract_from_env(env, now_s)
        phase = contract.phase if contract is not None else "all_resolved"
        phase_counts[phase] += 1
        command = _teacher_action(env, teacher, obs, lead_s=config.lead_s)
        pressed_commands = [k for k, value in zip(env.key_names, command.values)
                            if value > 0]
        if pressed_commands:
            positive_frames += 1
            positive_command_counts[phase] += 1
        step = env.step(command)
        last_frames.append({
            "frame": i, "command_time_s": now_s,
            "target_before_command": (contract.to_json() if contract is not None else None),
            "teacher_action_8k": list(command.values),
            "positive_command_keys": pressed_commands,
            "motor_positions_at_start_m": list(obs.motor.positions_m),
            "motor_velocities_at_start_m_s": list(obs.motor.velocities_m_s),
            "motor_pressed_at_start": list(obs.motor.pressed_flags),
        })
        obs = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("Expert Train20 replay exceeded frame budget")
    trace = env.overload_trace
    stats = env.stats
    if trace is None:
        raise RuntimeError("Teacher diagnostic trace absent")
    if (len(trace.action_frames) != i+1
            or len(trace.physical_keydowns) != env.physical_keydowns
            or len(trace.too_early_events) != stats.too_early_presses):
        raise RuntimeError("Teacher frame/keydown/too-early trace desynchronized")
    if bool(stats.overloaded) != (trace.termination.reason == "Overload"):
        raise RuntimeError("Overload reason does not agree with stats")
    # Independent full-episode consistency check: every physical KeyDown
    # MUST have been interpreted under the same next-target sequence.
    if len(trace.keydown_times_s) != len(trace.physical_keydowns):
        raise RuntimeError("Physical KeyDown timestamp count diverges")
    terminal_s = float(trace.termination.time_s)
    def in_tail(time):
        return terminal_s-LAST_EVENT_WINDOW_S-1e-8 <= time <= terminal_s+1e-8
    keydowns = [asdict(x) for x in trace.physical_keydowns]
    early = [asdict(x) for x in trace.too_early_events]
    failed = [x for x in early if x["fail_overload"]]
    if bool(failed) != bool(stats.overloaded):
        raise RuntimeError("TooEarly threshold event differs from physics overload")
    for physical, stamped in zip(trace.physical_keydowns, trace.keydown_times_s):
        if abs(physical.time_s - stamped) > 1e-8:
            raise RuntimeError("Physical KeyDown clocks not synchronized")
    resolved = int(stats.hits)+int(stats.misses)
    return {
        "anchor": anchor, "chart": named.chart_name,
        "chart_sha256": named.chart_sha256,
        "stats": {
            "hits": int(stats.hits), "misses": int(stats.misses),
            "targets": int(stats.targets), "unjudged": int(stats.targets)-resolved,
            "too_early": int(stats.too_early_presses),
            "keydowns": int(env.physical_keydowns),
            "overload": bool(stats.overloaded),
        },
        "termination": asdict(trace.termination),
        "phase_frame_counts": dict(sorted(phase_counts.items())),
        "positive_teacher_command_frames": positive_frames,
        "positive_command_frames_by_phase": dict(sorted(positive_command_counts.items())),
        "last_12_control_frames": list(last_frames),
        "all_physical_keydowns": keydowns,
        "all_too_early_events": early,
        "physical_keydowns_last_150ms": [x for x in keydowns if in_tail(x["time_s"])],
        "too_early_last_150ms": [x for x in early if in_tail(x["time_s"])],
        "failure_event": failed[-1] if failed else None,
        "trace_context": (
            "Commands are sampled at 10ms; KeyDowns are 1ms motor events. "
            "A KeyDown may follow an earlier command due to actuator inertia. "
            "The phase at command time is evaluator-only, not given to Teacher/Student."),
    }


def _same_as_v22(row, previous):
    fields = ("hits", "misses", "targets", "too_early", "overload", "keydowns", "unjudged")
    if row["anchor"] != previous["anchor"] or row["chart"] != previous["chart"]:
        raise RuntimeError("Train v2.2 and v2.3 chart identity mismatch")
    for field in fields:
        if row["stats"][field] != previous["physics"][field]:
            raise RuntimeError(
                f"Train v2.3 replay divergence #{row['anchor']:02d} "
                f"{field}: v2.2={previous['physics'][field]} "
                f"v2.3={row['stats'][field]}")


def run_train_teacher_safety(config):
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Refuse to replace existing Train safety report")
    before_best = p._protect_old_best(config)
    stage_a, stage_c = audit._provenance(config)
    segments, manifest = audit._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Train20 manifest changed since Stage A")
    old_report_file = transfer.output_dir(config) / SOURCE_REPORT
    previous = json.loads(old_report_file.read_text(encoding="utf-8"))
    if (previous.get("format") != audit.FORMAT
            or previous.get("role") != "Train20 only"
            or previous.get("train_manifest") != manifest
            or previous.get("validation_evaluated") is not False
            or previous.get("final_evaluated") is not False
            or previous.get("protected_inputs_unchanged") is not True
            or previous.get("config_sha256") != config.signature()
            or len(previous.get("per_anchor", [])) != 20):
        raise RuntimeError("Original v2.2 Train audit provenance mismatch")
    originals = audit._protected_paths(config, stage_c)
    originals["stage_c_v22_train_audit"] = old_report_file
    before = {k: p.sha256(v) for k, v in originals.items()}
    rows = []
    for i, named in enumerate(segments, 1):
        row = _run_teacher(named, config, anchor=i)
        _same_as_v22(row, previous["per_anchor"][i-1])
        rows.append(row)
        print(f"Prestige Train Teacher Safety {i:02d}/20: "
              f"H={row['stats']['hits']} "
              f"early={row['stats']['too_early']} "
              f"overload={row['stats']['overload']}", flush=True)
    if ({k: p.sha256(v) for k, v in originals.items()} != before
            or before_best != p._protect_old_best(config)):
        raise RuntimeError("Protected artifacts changed during Teacher safety replay")
    unsafe = [row["anchor"] for row in rows if row["stats"]["overload"]]
    totals = {
        "safe": len(rows)-len(unsafe), "overloaded": len(unsafe),
        "unsafe_anchors": unsafe,
        "hits": sum(row["stats"]["hits"] for row in rows),
        "misses": sum(row["stats"]["misses"] for row in rows),
        "too_early": sum(row["stats"]["too_early"] for row in rows),
        "unjudged": sum(row["stats"]["unjudged"] for row in rows),
        "keydowns": sum(row["stats"]["keydowns"] for row in rows),
    }
    report = {
        "format": FORMAT, "stage": "C-v2.3", "role": "Train20 only",
        "read_only": True, "no_training": True,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "original_stage_a_gate": stage_a["gate"],
        "stage_a_gate_overridden": False,
        "target_contract": (
            "Evaluator-only next unresolved target, timing classified with "
            "actual simulator judgement windows including late pass."),
        "control_dt_s": p.CONTROL_DT,
        "physics_dt_s": p.PHYSICS_DT,
        "config_sha256": config.signature(),
        "train_manifest": manifest, "aggregate": totals,
        "replay_matches_stage_c_v22": True,
        "per_anchor": rows,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report
