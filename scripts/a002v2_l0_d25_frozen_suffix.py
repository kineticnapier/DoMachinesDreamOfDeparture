#!/usr/bin/env python3
"""D25: separate a single step362 Motor command from later policy responses.

Uses only frozen D22 action streams. No policy inference or reward evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from dmdod.motor.env import MotorEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS
import a002v2_l0_d23_archived_motor_state as d23
import a002v2_l0_d24_1ms_physical_replay as d24

SCHEMA = "research-a-v2-l0-d25-frozen-suffix-v1"
PROTOCOL = "docs/research-a-v2-l0-d25-frozen-suffix-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
STEP = 362
CONTROLS = 480
TICKS_PER_CONTROL = 10
DT = 0.001
TIME_TOL = 1e-8
MAX_TICKS = 38400
WINDOW_START, WINDOW_END = 3.620, 3.730
require = d23.require

BASES = ("before", "after")
ARCHIVED = ("before", "after", "before_to_left", "after_to_none")
NEGATIVES = ("before_negative", "after_negative")
NEW = ("before_fixed_suffix_left", "after_fixed_suffix_none")
ALL = ARCHIVED + NEW + NEGATIVES


def actions(trace):
    return [step["action"] for step in trace["steps"]]


def build_plans(traces):
    require(set(traces) == set(d23.SERIES), "unexpected D22 trace names")
    plans = {name: actions(traces[name]) for name in ARCHIVED}
    for name in ARCHIVED:
        require(len(plans[name]) == CONTROLS and
                all(type(a) is int and 0 <= a < 4 for a in plans[name]),
                name + ": malformed action stream")
    require(plans["before"][STEP] == 0 and plans["after"][STEP] == 2 and
            plans["before_to_left"][STEP] == 2 and
            plans["after_to_none"][STEP] == 0,
            "step362 commands differ from preregistration")
    plans["before_fixed_suffix_left"] = list(plans["before"])
    plans["before_fixed_suffix_left"][STEP] = 2
    plans["after_fixed_suffix_none"] = list(plans["after"])
    plans["after_fixed_suffix_none"][STEP] = 0
    plans["before_negative"] = list(plans["before"])
    plans["after_negative"] = list(plans["after"])
    require(tuple(plans) == ALL, "plan ordering mismatch")
    for base, changed in (("before", "before_fixed_suffix_left"),
                          ("after", "after_fixed_suffix_none")):
        require([i for i, (a, b) in enumerate(zip(plans[base], plans[changed]))
                 if a != b] == [STEP], changed + ": changed more than step362")
    for base, changed in (("before", "before_to_left"),
                          ("after", "after_to_none")):
        require(plans[base][:STEP] == plans[changed][:STEP],
                changed + ": archived prefix mismatch")
    return plans


def replay_plan(plan, *, name):
    require(0 < len(plan) <= CONTROLS, name + ": invalid plan length")
    env = MotorEnv(same_hand=True, control_dt_s=0.010)
    env.reset()
    require(env.physics_substeps == TICKS_PER_CONTROL and
            abs(env.sim.config.dt_s - DT) < 1e-15 and
            env.sim.keyboard.config.actuation_m == d23.ACTUATION_M,
            name + ": Motor configuration mismatch")
    states = []
    window = []
    keydowns = []
    for index, action in enumerate(plan):
        require(type(action) is int and 0 <= action < 4,
                name + ": invalid action at " + str(index))
        require(abs(env.sim.time_s - index * 0.010) <= TIME_TOL,
                name + ": noncontinuous control clock")
        desired = ACTION_COMMANDS[action]
        for substep in range(TICKS_PER_CONTROL):
            previous = d24.snapshot(env)
            outcome = env.sim.step(desired.left, -1.0)
            current = d24.snapshot(env)
            require(abs(current["time_s"] -
                        (index * TICKS_PER_CONTROL + substep + 1) * DT) <= TIME_TOL,
                    name + ": noncontinuous physical clock")
            for field, value in current.items():
                if type(value) in (float, int):
                    require(math.isfinite(value),
                            name + ": nonfinite state " + field)
            for key, event in outcome.events:
                if event is not KeyEvent.DOWN:
                    continue
                require(key == "left", name + ": A0 right KeyDown")
                if not keydowns:
                    require(previous["left_pressed"] is False and
                            current["left_pressed"] is True and
                            previous["left_position_m"] < d23.ACTUATION_M and
                            current["left_position_m"] >= d23.ACTUATION_M,
                            name + ": first KeyDown did not cross threshold")
                keydowns.append({"time_s": outcome.time_s, "key": key,
                                 "source_step": index, "substep": substep + 1})
            states.append(current)
            if WINDOW_START - TIME_TOL <= current["time_s"] <= WINDOW_END + TIME_TOL:
                window.append(current)
    require(len(states) == len(plan) * TICKS_PER_CONTROL,
            name + ": wrong physical tick count")
    return {"ticks": len(states), "keydowns": keydowns,
            "first_keydown_s": keydowns[0]["time_s"] if keydowns else None,
            "keydown_count": len(keydowns)}, window, states


def check_archived_replay(name, replay, trace):
    original = trace["row"]["physical_trace"]
    require(len(original) == replay["keydown_count"],
            name + ": archived KeyDown count mismatch")
    for index, (new, old) in enumerate(zip(replay["keydowns"], original)):
        require(new["key"] == old["key"] and
                abs(new["time_s"] - old["time_s"]) <= TIME_TOL,
                name + ": archived KeyDown mismatch at " + str(index))
    require(abs(replay["first_keydown_s"] -
                d23.EXPECTED_FIRST[name][0]) <= TIME_TOL,
            name + ": archived first KeyDown mismatch")


def compare_pair(plans, series, base, closed, fixed):
    closed_actions = plans[closed]
    base_actions = plans[base]
    suffix_diff = [i for i in range(STEP + 1, CONTROLS)
                   if closed_actions[i] != base_actions[i]]
    physical_diff = [i for i in range(STEP + 1, CONTROLS)
                     if ACTION_COMMANDS[closed_actions[i]].left !=
                        ACTION_COMMANDS[base_actions[i]].left]
    first_closed = series[closed]["first_keydown_s"]
    first_fixed = series[fixed]["first_keydown_s"]
    first_base = series[base]["first_keydown_s"]
    require(first_closed is not None and first_fixed is not None and
            first_base is not None, base + ": missing first KeyDown")
    first_event_step = int(math.floor(
        min(first_closed, first_fixed) / 0.010 + 1e-8))
    diff_before_first = [i for i in physical_diff if i <= first_event_step]
    return {
        "baseline": base, "closed_loop": closed, "frozen_suffix": fixed,
        "suffix_action_differences": len(suffix_diff),
        "first_suffix_action_difference_step":
            suffix_diff[0] if suffix_diff else None,
        "suffix_effective_left_command_differences": len(physical_diff),
        "first_effective_difference_step":
            physical_diff[0] if physical_diff else None,
        "effective_differences_before_first_keydown": len(diff_before_first),
        "first_keydown_s": {"baseline": first_base,
                            "closed_loop": first_closed, "frozen_suffix": first_fixed},
        "fixed_minus_baseline_ms": (first_fixed - first_base) * 1000,
        "closed_minus_baseline_ms": (first_closed - first_base) * 1000,
        "fixed_minus_closed_ms": (first_fixed - first_closed) * 1000,
        "first_keydown_equal_within_1ms":
            abs(first_fixed - first_closed) < DT / 2,
        "keydown_count": {"baseline": series[base]["keydown_count"],
                          "closed_loop": series[closed]["keydown_count"],
                          "frozen_suffix": series[fixed]["keydown_count"]},
        "physical_keydown_sequence_equal":
            series[closed]["keydowns"] == series[fixed]["keydowns"],
    }


def analyze(source, traces):
    d23.validate_sources(source, traces)
    plans = build_plans(traces)
    series, windows, states = {}, {}, {}
    for name in ALL:
        series[name], windows[name], states[name] = replay_plan(plans[name], name=name)
        require(series[name]["ticks"] == CONTROLS * TICKS_PER_CONTROL,
                name + ": incomplete replay")
    require(sum(x["ticks"] for x in series.values()) == MAX_TICKS,
            "physical tick limit mismatch")
    for name in ARCHIVED:
        check_archived_replay(name, series[name], traces[name])
    for base, negative in (("before", "before_negative"),
                           ("after", "after_negative")):
        require(states[base] == states[negative] and
                series[base] == series[negative],
                base + ": negative control mismatch")
    for base, changed in (("before", "before_to_left"),
                          ("after", "after_to_none"),
                          ("before", "before_fixed_suffix_left"),
                          ("after", "after_fixed_suffix_none")):
        require(states[base][:STEP * TICKS_PER_CONTROL] ==
                states[changed][:STEP * TICKS_PER_CONTROL],
                changed + ": pre-intervention physical mismatch")
    pairs = {
        "before_to_left": compare_pair(
            plans, series, "before", "before_to_left", "before_fixed_suffix_left"),
        "after_to_none": compare_pair(
            plans, series, "after", "after_to_none", "after_fixed_suffix_none"),
    }
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_frozen_suffix_comparison",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "step": STEP, "control_dt_s": 0.010, "physical_dt_s": DT,
        "series": series, "pairs": pairs,
        "total_physical_ticks": MAX_TICKS,
        "total_physical_keydowns": sum(x["keydown_count"] for x in series.values()),
        "archived_event_parity": True,
        "negative_controls_exact": True,
        "pre_intervention_state_exact": True,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_policy_inferences": 0,
        "new_motor_replays": len(series), "new_hit_extra_evaluations": 0,
        "d19_archived_nonintervention_sampled": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50,
        },
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": "A0 single seed, archived open-loop commands; new replays are unscored. "
                      "No learning gain or unique body-state mediation.",
    }, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.window_out.exists():
        raise FileExistsError("refusing to overwrite existing D25 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result, windows = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            json.JSONDecodeError) as exc:
        result = {"schema": SCHEMA, "protocol": PROTOCOL,
                  "classification": "invalid_frozen_suffix_comparison",
                  "error": str(exc), "new_training_episodes": 0,
                  "new_ppo_updates": 0, "new_policy_inferences": 0}
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print("D25_FAILURE", json.dumps(result, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    args.window_out.write_text(json.dumps(windows, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    print("D25_RESULT", json.dumps({
        "classification": result["classification"],
        "pairs": result["pairs"],
        "total_physical_ticks": result["total_physical_ticks"],
        "total_physical_keydowns": result["total_physical_keydowns"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
