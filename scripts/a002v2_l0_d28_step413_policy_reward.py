#!/usr/bin/env python3
"""D28: audit archived step413 policy outputs and reward, without new inference.

Source is the frozen D22 result/steps archive. No learning, Motor, or re-scoring.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import a002v2_l0_d23_archived_motor_state as d23
import a002v2_l0_d26_post362_suffix_swap as d26

SCHEMA = "research-a-v2-l0-d28-step413-policy-reward-v1"
PROTOCOL = "docs/research-a-v2-l0-d28-step413-policy-reward-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
STEP = 413
EVENT_STEP = 432
CONTROLS = 480
EXTRA_TIME_S = 4.327
TOL = 1e-8
SERIES = ("before", "after_to_none")
require = d23.require


def probabilities(step):
    p = step["action_probabilities"]
    require(type(p) is list and len(p) == 4, "four probabilities required")
    require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0
                for x in p), "invalid probability")
    require(abs(sum(p) - 1.0) <= 1e-6, "probabilities do not sum to one")
    return [float(x) for x in p]


def snapshot(step):
    p = probabilities(step)
    require(type(step["sampled_action"]) is int and
            type(step["action"]) is int and
            0 <= step["sampled_action"] < 4 and
            0 <= step["action"] < 4, "invalid action")
    require(not step["intervention_applied"] and
            step["action"] == step["sampled_action"],
            "unexpected step413 intervention")
    obs = step["observation"]
    require(type(obs) is list and len(obs) == 8 and
            all(type(x) in (int, float) and math.isfinite(x) for x in obs),
            "invalid observation")
    require(math.isfinite(step["reward"]), "nonfinite reward")
    return {
        "step": step["step"],
        "time_s": step["time_s"],
        "observation": obs,
        "action_probabilities": p,
        "argmax_action": max(range(4), key=lambda i: p[i]),
        "sampled_action": step["sampled_action"],
        "executed_action": step["action"],
        "left_probability": p[2] + p[3],
        "reward": step["reward"],
        "pre_motor": step["pre_motor"],
        "pre_diagnostics": step["pre_diagnostics"],
    }


def event_at_extra(trace):
    matches = [entry for entry in trace["physical_keydowns"]
               if abs(entry["event"]["time_s"] - EXTRA_TIME_S) <= TOL
               and entry["event"]["key"] == "left"]
    require(len(matches) <= 1, "duplicate 4.327s KeyDown")
    if not matches:
        return None
    item = matches[0]
    require(item["source_step"] == EVENT_STEP and
            item["event"]["hit"] is False,
            "4.327s event provenance or judgement changed")
    reward = item["event"]["immediate_reward"]
    require(type(reward) in (int, float) and math.isfinite(reward),
            "nonfinite immediate event reward")
    return {
        "time_s": item["event"]["time_s"],
        "source_step": item["source_step"],
        "hit": item["event"]["hit"],
        "closest_target_ordinal": item["event"]["closest_target_ordinal"],
        "offset_ms": item["event"]["offset_ms"],
        "immediate_reward": reward,
    }


def analyze(source, traces):
    d23.validate_sources(source, traces)
    require(all(name in traces for name in
                (*SERIES, "before_negative", "after_negative")),
            "missing source series")
    require(traces["before"] == traces["before_negative"] and
            traces["after"] == traces["after_negative"],
            "D22 negative controls differ")
    b, a = traces["before"], traces["after_to_none"]
    require(len(b["steps"]) == len(a["steps"]) == CONTROLS,
            "unexpected control count")
    # Verify same physical left command throughout the preceding prefix.
    require(all(d26.left_command(b["steps"][i]["action"]) ==
                d26.left_command(a["steps"][i]["action"])
                for i in range(STEP)), "physical left-command prefix differs")
    snaps = {}
    for name, trace in (("before", b), ("after_to_none", a)):
        steps = trace["steps"]
        for index in (STEP, EVENT_STEP):
            require(steps[index]["step"] == index and
                    abs(steps[index]["time_s"] - index / 100) <= TOL,
                    name + ": invalid step/time")
        require(all(math.isfinite(s["reward"])
                    for s in steps[STEP:EVENT_STEP + 1]),
                name + ": nonfinite reward window")
        snaps[name] = snapshot(steps[STEP])
    require(snaps["before"]["executed_action"] == 0 and
            snaps["after_to_none"]["executed_action"] == 2,
            "D27 step413 source action mismatch")
    p, q = snaps["before"]["action_probabilities"], snaps["after_to_none"]["action_probabilities"]
    obs1, obs2 = snaps["before"]["observation"], snaps["after_to_none"]["observation"]
    before_extra, after_extra = event_at_extra(b), event_at_extra(a)
    require(before_extra is None and after_extra is not None,
            "D27 archived extra event mismatch")
    require(b["row"]["physical_keydowns"] == len(b["physical_keydowns"]) == 2 and
            a["row"]["physical_keydowns"] == len(a["physical_keydowns"]) == 3,
            "archived KeyDown counts mismatch")
    require(b["row"]["hits"] == 1 and b["row"]["extra_keydowns"] == 1 and
            a["row"]["hits"] == 1 and a["row"]["extra_keydowns"] == 2,
            "archived Hit/Extra counts mismatch")
    reward_window = {}
    for name, trace in (("before", b), ("after_to_none", a)):
        reward_window[name] = {
            "step413_reward": trace["steps"][STEP]["reward"],
            "step432_reward": trace["steps"][EVENT_STEP]["reward"],
            "step413_through_432_reward_sum": sum(
                float(s["reward"]) for s in
                trace["steps"][STEP:EVENT_STEP + 1]),
            "step413_through_432_nonzero_reward_steps": [
                {"step": s["step"], "reward": s["reward"]}
                for s in trace["steps"][STEP:EVENT_STEP + 1]
                if s["reward"] != 0],
        }
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "classification": "valid_step413_archived_policy_reward_audit",
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "arm": "a0",
        "seed": 320403,
        "targets": 16,
        "control_steps_read": 6 * CONTROLS,
        "step413": snaps,
        "step413_comparison": {
            "observation_exact": obs1 == obs2,
            "observation_max_abs_difference": max(
                abs(x-y) for x, y in zip(obs1, obs2)),
            "pre_motor_exact": snaps["before"]["pre_motor"] ==
                               snaps["after_to_none"]["pre_motor"],
            "pre_diagnostics_exact": snaps["before"]["pre_diagnostics"] ==
                                     snaps["after_to_none"]["pre_diagnostics"],
            "probability_delta_after_minus_before": [y-x for x, y in zip(p, q)],
            "probability_l1": sum(abs(y-x) for x, y in zip(p, q)),
            "left_command_probability_delta": (
                snaps["after_to_none"]["left_probability"] -
                snaps["before"]["left_probability"]),
            "same_argmax": snaps["before"]["argmax_action"] ==
                           snaps["after_to_none"]["argmax_action"],
            "sampled_action_differs": True,
        },
        "extra_at_4_327": {
            "before": before_extra, "after_to_none": after_extra,
        },
        "reward_window": reward_window,
        "archived_scored": {
            "before": {"hits": 1, "extras": 1, "physical_keydowns": 2},
            "after_to_none": {"hits": 1, "extras": 2, "physical_keydowns": 3},
        },
        "d19_archived_nonintervention_sampled": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50,
        },
        "negative_controls_exact": True,
        "new_training_episodes": 0,
        "new_ppo_updates": 0,
        "new_adam_steps": 0,
        "new_policy_inferences": 0,
        "new_motor_replays": 0,
        "new_physical_keydowns": 0,
        "new_hit_extra_evaluations": 0,
        "a1_evaluated": False,
        "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("Single archived A0 seed. Observational policy/reward "
                       "audit only; cannot isolate stochastic sampling, "
                       "PPO credit assignment or general learning."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D28 output")
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
            "classification": "invalid_step413_archived_policy_reward_audit",
            "error": str(exc),
            "new_training_episodes": 0,
            "new_ppo_updates": 0,
            "new_policy_inferences": 0,
            "new_motor_replays": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D28_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    brief = {
        "classification": result["classification"],
        "step413": {k: {"probabilities": v["action_probabilities"],
                        "sampled_action": v["sampled_action"],
                        "argmax_action": v["argmax_action"],
                        "reward": v["reward"]}
                    for k, v in result["step413"].items()},
        "comparison": result["step413_comparison"],
        "extra_at_4_327": result["extra_at_4_327"],
        "reward_window": result["reward_window"],
    }
    print("D28_RESULT", json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
