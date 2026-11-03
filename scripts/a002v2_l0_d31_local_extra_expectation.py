#!/usr/bin/env python3
"""D31: conditional local A0 KeyDown expectation from archived D27/D30 evidence."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

from dmdod.research.a002_timing_only_env import ACTION_COMMANDS, TimingOnlyTwoFingerEnv

SCHEMA = "research-a-v2-l0-d31-local-a0-keydown-expectation-v1"
PROTOCOL = "docs/research-a-v2-l0-d31-local-extra-expectation-protocol.md"
SOURCE_D27_RUN = 41556553776
SOURCE_D30_RUN = 41574924385


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_a0_mapping():
    # Source-level contract: in A0 the right motor is always commanded to -1.
    source = inspect.getsource(TimingOnlyTwoFingerEnv.step)
    require('self.arm == "a1"' in source and
            "MotorAction(desired.left, -1.0)" in source,
            "A0 physical-right disable contract changed")
    require(len(ACTION_COMMANDS) == 4, "action count changed")
    left = [float(x.left) for x in ACTION_COMMANDS]
    require(left == [-1.0, -1.0, 1.0, 1.0],
            "none/right and left/both no longer physically equivalent in A0")
    return left


def validate_d27(d27):
    require(d27["schema"] == "research-a-v2-l0-d27-step413-single-command-v1",
            "D27 schema mismatch")
    require(d27["classification"] == "valid_step413_single_command_intervention",
            "D27 source invalid")
    require(d27["source_run"] == 41515190995 and
            d27["source_artifact"] == 13175252296, "D27 provenance mismatch")
    require(d27["arm"] == "a0" and d27["seed"] == 320403 and
            d27["intervention_step"] == 413 and
            d27["later_unchanged_step"] == 453,
            "D27 arm/seed/step mismatch")
    require(d27["negative_controls_exact"] is True and
            d27["pre_step413_states_exact"] is True, "D27 controls failed")
    require(d27["step413_commands"]["before_action"] == 0 and
            d27["step413_commands"]["after_to_none_action"] == 2,
            "D27 actions differ from preregistered none/left")
    series = d27["series"]
    require(series["before"]["keydown_count"] == 2 and
            series["after_to_none"]["keydown_count"] == 3,
            "D27 original KeyDown count mismatch")
    pairs = [
        ("before", "before_step413_from_after", False, True, 1),
        ("after_to_none", "after_step413_from_before", True, False, -1),
    ]
    for base, altered, base_event, altered_event, delta in pairs:
        c = d27["comparisons"][altered]
        require(c["baseline"] == base and
                c["baseline_has_extra_keydown_at_4_327s"] is base_event and
                c["has_extra_keydown_at_4_327s"] is altered_event and
                c["physical_keydown_count_delta"] == delta,
                "D27 intervention reversal mismatch: " + altered)
        require(series[altered]["keydown_count"] ==
                series[base]["keydown_count"] + delta,
                "D27 series count inconsistent: " + altered)
    require(d27["new_series_hit_extra_scoring"] == "unscored",
            "D27 counterfactual scoring status changed")
    return {"before_suffix": [0, 1], "after_suffix": [0, 1]}


def validate_probs(p):
    require(type(p) is list and len(p) == 4, "expected four action probabilities")
    require(all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1
                for x in p), "invalid action probability")
    require(abs(sum(p) - 1.0) <= 1e-6, "action probabilities not normalized")


def analyze(d27, d30):
    mapping = validate_a0_mapping()
    effects = validate_d27(d27)
    require(d30["schema"] == "research-a-v2-l0-d30-common-rng-seed-sensitivity-v1"
            and d30["classification"] == "valid_seed_sensitivity_audit",
            "D30 source invalid")
    require(d30["source_run"] == 41515190995 and
            d30["source_artifact"] == 13175252296 and
            d30["arm"] == "a0" and d30["source_seed"] == 320403 and
            d30["step"] == 413, "D30 provenance mismatch")
    replay = d30["d29_replay"]
    require(replay["before"] == replay["after_to_none"] == 480 and
            replay["pivot_rng_state_exact"] is True and
            replay["archived_step413_cross_draw"] == [0, 2],
            "D29 positive control mismatch")
    p, q = d30["p_before"], d30["p_after"]
    validate_probs(p)
    validate_probs(q)
    require(d30["sampling"]["seed_count"] == 4096 and
            d30["sampling"]["disagreement_count"] == 6,
            "D30 archived 4096-seed provenance mismatch")
    left_actions = [i for i, value in enumerate(mapping) if value > 0]
    require(left_actions == [2, 3], "effective-left action indices changed")
    before = sum(p[i] for i in left_actions)
    after = sum(q[i] for i in left_actions)
    delta = after - before
    require(all(math.isfinite(x) for x in (before, after, delta)) and before > 0,
            "nonfinite local expectation")
    per_suffix = {}
    for suffix, indicator in effects.items():
        require(indicator == [0, 1], "D27 physical effect does not map to binary command")
        # E[event] = P(left effective)*1 + P(no-left effective)*0.
        per_suffix[suffix] = {
            "before_event_probability": before,
            "after_event_probability": after,
            "difference_probability": delta,
        }
    require(per_suffix["before_suffix"] == per_suffix["after_suffix"],
            "suffix conditional expectations differ")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_local_a0_keydown_expectation",
        "source_d27_run": SOURCE_D27_RUN, "source_d30_run": SOURCE_D30_RUN,
        "arm": "a0", "source_seed": 320403, "step": 413,
        "event_time_s": 4.327,
        "action_names": ["none", "right", "left", "both"],
        "effective_left_action_indices": left_actions,
        "p_before": p, "p_after": q,
        "conditional_expectation_by_frozen_suffix": per_suffix,
        "before_probability": before, "after_probability": after,
        "difference_probability": delta,
        "difference_percentage_points": 100 * delta,
        "ratio_after_to_before": after / before,
        "expected_additional_physical_keydowns_per_one_fixed_step": delta,
        "counterfactual_hit_extra_scoring": "unscored",
        "d19_archived_nonintervention_sampled": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50},
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_policy_inferences": 0,
        "new_motor_replays": 0, "new_physical_keydowns": 0,
        "new_hit_extra_evaluations": 0,
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("One fixed A0 observation and archived open-loop suffix. "
                       "Expected physical KeyDown at 4.327s, not scored Extra, "
                       "not full rollout expectation or PPO credit assignment."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d27-result", type=Path, required=True)
    ap.add_argument("--d30-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D31 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        d27 = json.loads(args.d27_result.read_text(encoding="utf-8"))
        d30 = json.loads(args.d30_result.read_text(encoding="utf-8"))
        result = analyze(d27, d30)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_local_a0_keydown_expectation",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_motor_replays": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D31_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D31_RESULT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
