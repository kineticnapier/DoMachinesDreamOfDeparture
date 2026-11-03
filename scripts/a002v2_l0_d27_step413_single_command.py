#!/usr/bin/env python3
"""D27: intervene on exactly one archived A0 left command at step413.

No new policy inference, learning, PPO updates or Hit/Extra re-scoring.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import a002v2_l0_d23_archived_motor_state as d23
import a002v2_l0_d25_frozen_suffix as d25
import a002v2_l0_d26_post362_suffix_swap as d26

SCHEMA = "research-a-v2-l0-d27-step413-single-command-v1"
PROTOCOL = "docs/research-a-v2-l0-d27-step413-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
STEP = 413
LATER_STEP = 453
CONTROLS = 480
SUBSTEPS = 10
MAX_TICKS = 28800
TIME_TOL = 1e-8
EXTRA_TIME_S = 4.327
ORIGINALS = ("before", "after_to_none")
INTERVENTIONS = ("before_step413_from_after", "after_step413_from_before")
NEGATIVES = ("before_negative_d27", "after_negative_d27")
ALL = ORIGINALS + INTERVENTIONS + NEGATIVES
require = d23.require


def build_plans(traces):
    # D26 validates the D22 step362 provenance and physical-equivalent prefix.
    d26_plans, prefix, suffix = d26.build_plans(traces)
    before = d26_plans["before"]
    after = d26_plans["after_to_none"]
    require(prefix["effective_left_differences"] == 0,
            "pre-step363 effective left commands differ")
    require(suffix["effective_difference_steps"] == [STEP, LATER_STEP],
            "D26 two-step effective left-command difference changed")
    require(d26.left_command(before[STEP]) != d26.left_command(after[STEP]),
            "step413 does not differ physically")
    require(d26.action_differences(before, after, 0, STEP)[
        "effective_left_differences"] == 0,
        "pre-step413 effective left commands differ")
    plans = {
        "before": list(before),
        "after_to_none": list(after),
        "before_step413_from_after": list(before),
        "after_step413_from_before": list(after),
        "before_negative_d27": list(before),
        "after_negative_d27": list(after),
    }
    plans["before_step413_from_after"][STEP] = after[STEP]
    plans["after_step413_from_before"][STEP] = before[STEP]
    require(tuple(plans) == ALL, "series ordering mismatch")
    for base, altered, donor in (
        ("before", "before_step413_from_after", "after_to_none"),
        ("after_to_none", "after_step413_from_before", "before"),
    ):
        differences = [i for i, (a, b) in enumerate(zip(
            plans[base], plans[altered])) if a != b]
        require(differences == [STEP], altered + ": not a one-step intervention")
        require(plans[altered][STEP] == plans[donor][STEP],
                altered + ": donor command mismatch")
        require(plans[altered][LATER_STEP] == plans[base][LATER_STEP],
                altered + ": step453 was changed")
    return plans, prefix, suffix


def extra_at_432(replay):
    return any(
        event["key"] == "left"
        and abs(event["time_s"] - EXTRA_TIME_S) <= TIME_TOL
        and event["source_step"] == 432
        for event in replay["keydowns"]
    )


def first_state_divergence(original, altered):
    require(len(original) == len(altered) == CONTROLS * SUBSTEPS,
            "wrong state sequence length")
    tick = next((i for i, (a, b) in enumerate(zip(original, altered))
                 if a != b), None)
    return {
        "tick_index": tick,
        "time_s": original[tick]["time_s"] if tick is not None else None,
        "after_step413_start": tick is None or tick >= STEP * SUBSTEPS,
    }


def analyze(source, traces):
    d23.validate_sources(source, traces)
    plans, prefix, suffix = build_plans(traces)
    require(len(plans) == 6 and len(plans) * CONTROLS * SUBSTEPS == MAX_TICKS,
            "physical tick budget exceeded")
    series, states = {}, {}
    for name in ALL:
        series[name], _, states[name] = d25.replay_plan(plans[name], name=name)
        require(series[name]["ticks"] == CONTROLS * SUBSTEPS,
                name + ": incomplete physical replay")
        require(all(math.isfinite(e["time_s"]) for e in series[name]["keydowns"]),
                name + ": nonfinite KeyDown")
    for name in ORIGINALS:
        d25.check_archived_replay(name, series[name], traces[name])
    for original, negative in (
        ("before", "before_negative_d27"),
        ("after_to_none", "after_negative_d27"),
    ):
        require(states[original] == states[negative]
                and series[original] == series[negative],
                original + ": negative control mismatch")
    comparisons = {}
    for original, altered in (
        ("before", "before_step413_from_after"),
        ("after_to_none", "after_step413_from_before"),
    ):
        require(states[original][:STEP * SUBSTEPS] ==
                states[altered][:STEP * SUBSTEPS],
                altered + ": physical state differs before step413")
        divergence = first_state_divergence(states[original], states[altered])
        require(divergence["after_step413_start"],
                altered + ": state divergence before intervention")
        comparisons[altered] = {
            "baseline": original,
            "first_state_divergence": divergence,
            "keydowns_equal_to_before": d26.event_match(
                series[altered], series["before"]),
            "keydowns_equal_to_after_to_none": d26.event_match(
                series[altered], series["after_to_none"]),
            "has_extra_keydown_at_4_327s": extra_at_432(series[altered]),
            "baseline_has_extra_keydown_at_4_327s": extra_at_432(
                series[original]),
            "physical_keydown_count_delta": (
                series[altered]["keydown_count"] -
                series[original]["keydown_count"]),
        }
    require(series["before"]["keydown_count"] == 2 and
            series["after_to_none"]["keydown_count"] == 3 and
            not extra_at_432(series["before"]) and
            extra_at_432(series["after_to_none"]),
            "archived original event expectations mismatch")
    require(sum(s["ticks"] for s in series.values()) == MAX_TICKS,
            "wrong total physical tick count")
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "classification": "valid_step413_single_command_intervention",
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "arm": "a0",
        "seed": 320403,
        "target_count": 16,
        "intervention_step": STEP,
        "later_unchanged_step": LATER_STEP,
        "step413_commands": {
            "before_action": plans["before"][STEP],
            "after_to_none_action": plans["after_to_none"][STEP],
            "before_left": d26.left_command(plans["before"][STEP]),
            "after_to_none_left": d26.left_command(
                plans["after_to_none"][STEP]),
        },
        "prefix_action_comparison": prefix,
        "suffix_action_comparison": suffix,
        "series": series,
        "comparisons": comparisons,
        "archived_scored_originals": {
            "before": {"hits": 1, "extras": 1, "physical_keydowns": 2},
            "after_to_none": {"hits": 1, "extras": 2, "physical_keydowns": 3},
        },
        "new_series_hit_extra_scoring": "unscored",
        "total_physical_ticks": MAX_TICKS,
        "total_physical_keydowns": sum(
            x["keydown_count"] for x in series.values()),
        "negative_controls_exact": True,
        "pre_step413_states_exact": True,
        "new_training_episodes": 0,
        "new_ppo_updates": 0,
        "new_adam_steps": 0,
        "new_policy_inferences": 0,
        "new_motor_replays": len(series),
        "new_hit_extra_evaluations": 0,
        "d19_archived_nonintervention_sampled": {
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
        "limitation": ("Single A0 seed and archived open-loop commands. "
                       "New series are unscored; no new learning, generalization, "
                       "or uniquely proven full causal mechanism."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", required=True, type=Path)
    parser.add_argument("--d22-traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D27 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "classification": "invalid_step413_single_command_intervention",
            "error": str(exc),
            "new_training_episodes": 0,
            "new_ppo_updates": 0,
            "new_policy_inferences": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D27_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D27_RESULT", json.dumps({
        "classification": result["classification"],
        "step413_commands": result["step413_commands"],
        "series": {k: {"keydowns": v["keydowns"],
                        "keydown_count": v["keydown_count"],
                        "has_4_327": extra_at_432(v)}
                   for k, v in result["series"].items()},
        "comparisons": result["comparisons"],
        "total_physical_ticks": result["total_physical_ticks"],
        "total_physical_keydowns": result["total_physical_keydowns"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
