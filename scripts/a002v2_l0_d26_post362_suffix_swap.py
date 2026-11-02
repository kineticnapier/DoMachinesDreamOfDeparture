#!/usr/bin/env python3
"""D26: isolate residual KeyDown by swapping only archived post-362 A0 commands.

No policy inference, training, PPO, reward or Hit/Extra re-scoring.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import a002v2_l0_d23_archived_motor_state as d23
import a002v2_l0_d25_frozen_suffix as d25
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS

SCHEMA = "research-a-v2-l0-d26-post362-suffix-swap-v1"
PROTOCOL = "docs/research-a-v2-l0-d26-post362-suffix-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
STEP = 362
CONTROLS = 480
SUBSTEPS = 10
MAX_TICKS = 28800
TIME_TOL = 1e-8
ORIGINALS = ("before", "after_to_none")
SWAPPED = ("before_with_after_suffix", "after_with_before_suffix")
NEGATIVES = ("before_negative_d26", "after_negative_d26")
ALL = ORIGINALS + SWAPPED + NEGATIVES
require = d23.require


def left_command(action):
    require(type(action) is int and 0 <= action < 4, "invalid action")
    return ACTION_COMMANDS[action].left


def action_differences(a, b, start, end):
    require(len(a) == len(b) == CONTROLS, "invalid control length")
    raw = [i for i in range(start, end) if a[i] != b[i]]
    effective = [i for i in range(start, end)
                 if left_command(a[i]) != left_command(b[i])]
    return {
        "action_differences": len(raw),
        "first_action_difference_step": raw[0] if raw else None,
        "effective_left_differences": len(effective),
        "first_effective_difference_step":
            effective[0] if effective else None,
        "effective_difference_steps": effective,
    }


def build_plans(traces):
    # Reuse the D25 stream-shape and step362 intervention checks.
    archived = d25.build_plans(traces)
    before = archived["before"]
    after = archived["after_to_none"]
    require(len(before) == len(after) == CONTROLS, "invalid source length")
    require(before[STEP] == after[STEP] == 0,
            "step362 must be none in both comparison streams")
    # The prefix is allowed to differ by action index, not physical left command.
    prefix = action_differences(before, after, 0, STEP + 1)
    require(prefix["effective_left_differences"] == 0,
            "effective left-command prefix differs: invalid causal isolation")
    plans = {
        "before": list(before),
        "after_to_none": list(after),
        "before_with_after_suffix": list(before[:STEP + 1] + after[STEP + 1:]),
        "after_with_before_suffix": list(after[:STEP + 1] + before[STEP + 1:]),
        "before_negative_d26": list(before),
        "after_negative_d26": list(after),
    }
    require(tuple(plans) == ALL, "unexpected replay series")
    for original, swapped, donor in (
        ("before", "before_with_after_suffix", "after_to_none"),
        ("after_to_none", "after_with_before_suffix", "before"),
    ):
        require(plans[swapped][:STEP + 1] == plans[original][:STEP + 1],
                swapped + ": changed prefix")
        require(plans[swapped][STEP + 1:] == plans[donor][STEP + 1:],
                swapped + ": suffix does not match donor")
    return plans, prefix, action_differences(before, after, STEP + 1, CONTROLS)


def event_sequence(replay):
    return [(e["key"], e["time_s"]) for e in replay["keydowns"]]


def event_match(left, right):
    a, b = event_sequence(left), event_sequence(right)
    return len(a) == len(b) and all(
        x[0] == y[0] and abs(x[1] - y[1]) <= TIME_TOL
        for x, y in zip(a, b)
    )


def analyze(source, traces):
    d23.validate_sources(source, traces)
    plans, prefix_diff, suffix_diff = build_plans(traces)
    require(len(plans) == 6 and len(plans) * CONTROLS * SUBSTEPS == MAX_TICKS,
            "pre-registered physical tick budget exceeded")
    series, states = {}, {}
    for name in ALL:
        series[name], _, states[name] = d25.replay_plan(plans[name], name=name)
        require(series[name]["ticks"] == CONTROLS * SUBSTEPS,
                name + ": wrong physical tick count")
        require(all(math.isfinite(e["time_s"]) for e in series[name]["keydowns"]),
                name + ": nonfinite KeyDown time")
    for name in ORIGINALS:
        d25.check_archived_replay(name, series[name], traces[name])
    for original, negative in (
        ("before", "before_negative_d26"),
        ("after_to_none", "after_negative_d26"),
    ):
        require(states[original] == states[negative] and
                series[original] == series[negative],
                original + ": negative control mismatch")
    prefix_ticks = (STEP + 1) * SUBSTEPS
    for original, swapped in (
        ("before", "before_with_after_suffix"),
        ("after_to_none", "after_with_before_suffix"),
    ):
        require(states[original][:prefix_ticks] == states[swapped][:prefix_ticks],
                swapped + ": pre-swap physical state changed")
    require(sum(x["ticks"] for x in series.values()) == MAX_TICKS,
            "physical tick budget mismatch")
    # These comparisons measure an intervention on archived commands, not
    # closed-loop policy behavior or new Hit/Extra scoring.
    comparisons = {
        "before_with_after_suffix": {
            "matches_before_events": event_match(
                series["before_with_after_suffix"], series["before"]),
            "matches_after_to_none_events": event_match(
                series["before_with_after_suffix"], series["after_to_none"]),
        },
        "after_with_before_suffix": {
            "matches_before_events": event_match(
                series["after_with_before_suffix"], series["before"]),
            "matches_after_to_none_events": event_match(
                series["after_with_before_suffix"], series["after_to_none"]),
        },
    }
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "classification": "valid_post362_suffix_swap",
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "intervention_step": STEP,
        "suffix_start_step": STEP + 1,
        "arm": "a0",
        "seed": 320403,
        "target_count": 16,
        "prefix_action_comparison": prefix_diff,
        "suffix_action_comparison": suffix_diff,
        "series": series,
        "comparisons": comparisons,
        "archived_scored_originals": {
            "before": {"hits": 1, "extras": 1, "physical_keydowns": 2},
            "after_to_none": {"hits": 1, "extras": 2, "physical_keydowns": 3},
        },
        "new_series_hit_extra_scoring": "unscored",
        "total_physical_ticks": MAX_TICKS,
        "total_physical_keydowns": sum(x["keydown_count"] for x in series.values()),
        "negative_controls_exact": True,
        "pre_swap_states_exact": True,
        "new_training_episodes": 0,
        "new_ppo_updates": 0,
        "new_adam_steps": 0,
        "new_policy_inferences": 0,
        "new_motor_replays": len(series),
        "new_hit_extra_evaluations": 0,
        "d19_archived_sampled": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50,
        },
        "a1_evaluated": False,
        "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("One A0 seed, open-loop archived commands. "
                       "Swapped series are unscored. No learning improvement, "
                       "generalization, or unique single-step attribution."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite existing D26 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_post362_suffix_swap",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_policy_inferences": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D26_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D26_RESULT", json.dumps({
        "classification": result["classification"],
        "prefix": result["prefix_action_comparison"],
        "suffix": result["suffix_action_comparison"],
        "series": {k: {"keydowns": v["keydowns"],
                        "keydown_count": v["keydown_count"]}
                   for k, v in result["series"].items()},
        "comparisons": result["comparisons"],
        "total_physical_ticks": result["total_physical_ticks"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
