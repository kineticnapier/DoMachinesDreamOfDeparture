#!/usr/bin/env python3
"""D24: replay archived D22 action sequences using the ordinary 1ms Motor path.

No policy, reward, PPO, or Motor implementation is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from dmdod.motor.env import MotorEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS
import a002v2_l0_d23_archived_motor_state as d23

SCHEMA = "research-a-v2-l0-d24-1ms-physical-replay-v1"
PROTOCOL = "docs/research-a-v2-l0-d24-1ms-physical-replay-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
DT = 0.001
SUBSTEPS = 10
CONTROLS = 480
FLOAT_TOL = 1e-9
TIME_TOL = 1e-8
WINDOW_START = 3.620
WINDOW_END = 3.730
MOTOR_KEYS = (
    "left_position_m", "right_position_m", "left_velocity_m_s",
    "right_velocity_m_s", "left_pressed", "right_pressed",
)
DIAG_KEYS = (
    "time_s", "left_activation", "right_activation", "left_fatigue",
    "right_fatigue", "hand_fatigue", "hand_coordination",
    "bilateral_coordination",
)
require = d23.require


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value),
            name + ": nonfinite or nonnumeric")
    return float(value)


def snapshot(env):
    motor = asdict(env.observe())
    diag = asdict(env.diagnostics())
    return {
        "time_s": env.sim.time_s,
        "left_position_m": motor["left_position_m"],
        "left_velocity_m_s": motor["left_velocity_m_s"],
        "left_pressed": motor["left_pressed"],
        "left_activation": diag["left_activation"],
        "left_fatigue": diag["left_fatigue"],
        "hand_fatigue": diag["hand_fatigue"],
        "hand_coordination": diag["hand_coordination"],
        "actuation_gap_m": env.sim.keyboard.config.actuation_m -
                           motor["left_position_m"],
    }


def compare_fields(actual, archived, keys, name, maximum):
    for key in keys:
        require(key in archived and key in actual,
                name + ": missing " + key)
        a, b = actual[key], archived[key]
        if type(b) is bool:
            require(type(a) is bool and a is b,
                    name + "/" + key + ": bool mismatch")
        else:
            difference = abs(finite(a, name + "/" + key) -
                             finite(b, name + "/" + key))
            maximum[key] = max(maximum.get(key, 0.0), difference)
            tol = TIME_TOL if key == "time_s" else FLOAT_TOL
            require(difference <= tol,
                    name + "/" + key + ": difference " +
                    format(difference, ".12g") + " exceeds " + str(tol))


def replay_series(trace, *, name):
    """Run exactly the archived commands, independent of the saved policy."""
    steps = trace["steps"]
    env = MotorEnv(same_hand=True, control_dt_s=0.010)
    env.reset()
    require(env.physics_substeps == SUBSTEPS and
            abs(env.sim.config.dt_s - DT) < 1e-15 and
            env.sim.keyboard.config.actuation_m == d23.ACTUATION_M,
            name + ": Motor configuration mismatch")
    all_ticks = []
    window = []
    down = []
    first_down = None
    maximum = {}
    for index, step in enumerate(steps):
        require(step["step"] == index and
                type(step["action"]) is int and
                0 <= step["action"] < 4, name + ": invalid action index")
        require(abs(finite(step["time_s"], "step time") -
                    env.sim.time_s) <= TIME_TOL,
                name + ": control time mismatch at " + str(index))
        pre_motor = asdict(env.observe())
        pre_diag = asdict(env.diagnostics())
        compare_fields(pre_motor, step["pre_motor"], MOTOR_KEYS,
                       name + "/control" + str(index), maximum)
        compare_fields(pre_diag, step["pre_diagnostics"], DIAG_KEYS,
                       name + "/diagnostic" + str(index), maximum)
        desired = ACTION_COMMANDS[step["action"]]
        # A0: right finger is physically disabled, as in the original D22.
        for tick in range(SUBSTEPS):
            previous = snapshot(env)
            outcome = env.sim.step(desired.left, -1.0)
            current = snapshot(env)
            require(abs(current["time_s"] -
                        (index * SUBSTEPS + tick + 1) * DT) <= TIME_TOL,
                    name + ": 1ms tick clock mismatch")
            for key, event in outcome.events:
                if event is not KeyEvent.DOWN:
                    continue
                entry = {"time_s": outcome.time_s, "key": key,
                         "source_step": index, "substep": tick + 1}
                down.append(entry)
                if first_down is None:
                    require(key == "left" and
                            previous["left_pressed"] is False and
                            current["left_pressed"] is True and
                            previous["left_position_m"] < d23.ACTUATION_M and
                            current["left_position_m"] >= d23.ACTUATION_M,
                            name + ": first KeyDown did not cross threshold")
                    first_down = {
                        **entry, "before_tick": previous,
                        "after_tick": current,
                    }
            all_ticks.append(current)
            if WINDOW_START - TIME_TOL <= current["time_s"] <= WINDOW_END + TIME_TOL:
                window.append(current)
    require(len(all_ticks) == len(steps) * SUBSTEPS,
            name + ": physical tick count mismatch")
    require(abs(env.sim.time_s - len(steps) * SUBSTEPS * DT) <= TIME_TOL,
            name + ": final time mismatch")
    original = trace["row"]["physical_trace"]
    require(len(down) == len(original), name + ": KeyDown count mismatch")
    for index, (new, old) in enumerate(zip(down, original)):
        require(new["key"] == old["key"] and
                abs(new["time_s"] - old["time_s"]) <= TIME_TOL,
                name + ": KeyDown mismatch at event " + str(index))
    require(first_down is not None, name + ": no physical KeyDown")
    return {
        "ticks": len(all_ticks),
        "keydown_count": len(down),
        "first_keydown": first_down,
        "keydowns": down,
        "max_abs_control_state_difference": maximum,
        "archived_hits": trace["row"]["hits"],
        "archived_extras": trace["row"]["extra_keydowns"],
    }, window, all_ticks


def analyze(source, traces):
    d23.validate_sources(source, traces)
    metrics = {}
    windows = {}
    full_ticks = {}
    for name in d23.SERIES:
        metrics[name], windows[name], full_ticks[name] = replay_series(
            traces[name], name=name)
        require(metrics[name]["ticks"] == CONTROLS * SUBSTEPS,
                name + ": unexpected number of ticks")
        require(abs(metrics[name]["first_keydown"]["time_s"] -
                    d23.EXPECTED_FIRST[name][0]) <= TIME_TOL,
                name + ": unexpected first KeyDown time")
    require(full_ticks["before"] == full_ticks["before_negative"] and
            full_ticks["after"] == full_ticks["after_negative"],
            "1ms negative control full-tick mismatch")
    prefix = d23.INTERVENTION_STEP * SUBSTEPS
    for baseline, changed in (
        ("before", "before_to_left"), ("after", "after_to_none")
    ):
        require(full_ticks[baseline][:prefix] ==
                full_ticks[changed][:prefix],
                baseline + ": 1ms pre-intervention divergence")
    first_times = {
        name: metrics[name]["first_keydown"]["time_s"]
        for name in d23.SERIES
    }
    shifts = {
        "before_to_left_minus_before_ms":
            (first_times["before_to_left"] - first_times["before"]) * 1000,
        "after_to_none_minus_after_ms":
            (first_times["after_to_none"] - first_times["after"]) * 1000,
    }
    require(abs(shifts["before_to_left_minus_before_ms"] + 29.0) < 1e-6 and
            abs(shifts["after_to_none_minus_after_ms"] - 29.0) < 1e-6,
            "D22 first KeyDown shift mismatch")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_1ms_physical_replay",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "tick_dt_s": DT, "control_dt_s": SUBSTEPS * DT,
        "series": metrics, "first_keydown_shift_ms": shifts,
        "total_replayed_physical_ticks": sum(m["ticks"] for m in metrics.values()),
        "total_replayed_keydowns": sum(m["keydown_count"] for m in metrics.values()),
        "negative_controls_exact": True,
        "pre_intervention_tick_histories_exact": True,
        "d19_sampled_archived": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_keydowns": 49, "after_keydowns": 50,
        },
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_policy_inferences": 0,
        "new_motor_replays": len(metrics),
        "new_simulated_keydowns": sum(m["keydown_count"] for m in metrics.values()),
        "new_hit_extra_evaluations": 0,
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("Replayed archived commands, not independently sampled "
                       "policy responses. No unique state mediation or learning gain."),
    }, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.window_out.exists():
        raise FileExistsError("refusing to overwrite D24 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result, windows = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_1ms_physical_replay",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_policy_inferences": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D24_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    args.window_out.write_text(json.dumps(windows, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print("D24_RESULT", json.dumps({
        "classification": result["classification"],
        "first_keydown_shift_ms": result["first_keydown_shift_ms"],
        "total_replayed_physical_ticks": result["total_replayed_physical_ticks"],
        "total_replayed_keydowns": result["total_replayed_keydowns"],
        "first_crossings": {
            name: {"before": metric["first_keydown"]["before_tick"],
                   "after": metric["first_keydown"]["after_tick"]}
            for name, metric in result["series"].items()
        },
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
