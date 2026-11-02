#!/usr/bin/env python3
"""D14: audit float32 ULP distances in the archived D12 pre-Adam gradients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import a002v2_l0_d13_frozen_gradient_stage as d13

PROTOCOL = "docs/research-a-v2-l0-d14-gradient-ulp-protocol.md"
SOURCE_RUN = d13.SOURCE_RUN
SOURCE_ARTIFACT = d13.SOURCE_ARTIFACT
BINS = ("1", "2", "3-4", "5-16", "17+")


def ordered_float32_bits(bits):
    """Map finite IEEE-754 float32 bits into monotonically ordered integers."""
    d13.decode_bits(bits)
    unsigned = bits & 0xFFFFFFFF
    if unsigned & 0x80000000:
        return (~unsigned) & 0xFFFFFFFF
    return unsigned | 0x80000000


def ulp_distance(left, right):
    return abs(ordered_float32_bits(left) - ordered_float32_bits(right))


def bucket(distance):
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if distance <= 4:
        return "3-4"
    if distance <= 16:
        return "5-16"
    return "17+"


def analyze_stage(native, default):
    summary = d13.compare_parameters(native, default)
    hist = {key: 0 for key in BINS}
    max_ulp = 0
    multi_ulp = 0
    sign_reversals = 0
    zero_transitions = 0
    signed_zero_only = 0
    per_parameter = {}
    first = None
    for name, a in native.items():
        b = default[name]
        local_count = 0
        local_max = 0
        for index, (x, y) in enumerate(zip(a, b)):
            if x == y:
                continue
            xf, yf = d13.decode_bits(x), d13.decode_bits(y)
            distance = ulp_distance(x, y)
            if distance <= 0:
                raise ValueError("different float32 bit patterns with zero ULP distance")
            hist[bucket(distance)] += 1
            local_count += 1
            local_max = max(local_max, distance)
            max_ulp = max(max_ulp, distance)
            if distance > 1:
                multi_ulp += 1
            if (xf < 0 < yf) or (yf < 0 < xf):
                sign_reversals += 1
            if (xf == 0) != (yf == 0):
                zero_transitions += 1
            if xf == 0 and yf == 0:
                signed_zero_only += 1
            if first is None:
                first = {"name": name, "index": index, "native_bits": x,
                         "default_bits": y, "ulp_distance": distance}
        if local_count:
            per_parameter[name] = {"different_elements": local_count,
                                   "max_ulp_distance": local_max}
    if sum(hist.values()) != summary["different_elements"]:
        raise ValueError("ULP histogram does not match D13 gradient differences")
    if sum(v["different_elements"] for v in per_parameter.values()) != summary["different_elements"]:
        raise ValueError("ULP per-parameter counts do not match D13")
    if first is not None and (
        first["name"] != summary["first_difference"]["name"]
        or first["index"] != summary["first_difference"]["index"]
        or first["native_bits"] != summary["first_difference"]["native_bits"]
        or first["default_bits"] != summary["first_difference"]["default_bits"]
    ):
        raise ValueError("ULP first difference does not match D13")
    return {"total_elements": summary["total_elements"],
            "different_elements": summary["different_elements"],
            "max_abs_difference": summary["max_abs_difference"],
            "max_ulp_distance": max_ulp, "more_than_one_ulp": multi_ulp,
            "ulp_histogram": hist, "sign_reversals": sign_reversals,
            "zero_to_nonzero_transitions": zero_transitions,
            "signed_zero_only_differences": signed_zero_only,
            "first_difference": first, "by_parameter": per_parameter}


def analyze(source):
    prior = d13.analyze(source)  # Includes the frozen SHA, runtime and D12/D13 count checks.
    if prior["classification"] != "gradient_difference_before_first_adam_step":
        raise ValueError("D13 historical first-gradient classification mismatch")
    native = source["native"]["actor_and_critic_gradients"]
    default = source["default"]["actor_and_critic_gradients"]
    stages = [analyze_stage(n, d) for n, d in zip(native, default)]
    if (len(stages) != 2 or
        stages[0]["different_elements"] != 292 or
        stages[1]["different_elements"] != 407):
        raise ValueError("D13 archived gradient counts changed")
    if all(stage["different_elements"] == 0 for stage in stages):
        raise ValueError("D13 archived nonzero gradient differences disappeared")
    classification = ("multi_ulp_difference_observed"
                      if any(stage["more_than_one_ulp"] for stage in stages)
                      else "all_differences_within_one_ulp")
    return {"schema": "research-a-v2-l0-d14-gradient-ulp-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "source_artifact": SOURCE_ARTIFACT,
            "classification": classification,
            "initial_state_sha256": prior["initial_state_sha256"],
            "batch_sha256": prior["batch_sha256"],
            "cpu_capabilities": prior["cpu_capabilities"],
            "before_adam_step_1": stages[0],
            "before_adam_step_2": stages[1],
            "targets": prior["targets"], "hits": prior["hits"],
            "extras": prior["extras"], "physical_keydowns": prior["physical_keydowns"],
            "new_training_episodes": 0, "new_ppo_updates": 0,
            "new_motor_steps": 0, "new_physical_keydowns": 0,
            "historical_r2_gradients_verified": False,
            "historical_d3_d5_cause_verified": False,
            "teacher_free_fingering_acquired": False,
            "human_equivalence_verified": False,
            "single_finger_physical_impossibility_verified": False,
            "formal_study_002_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D14 output")
    try:
        source = {name: json.loads((args.source_dir / (name + ".json")).read_text(
            encoding="utf-8")) for name in d13.REQUIRED}
        result = analyze(source)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        result = {"schema": "research-a-v2-l0-d14-gradient-ulp-v1",
                  "protocol": PROTOCOL, "source_run": SOURCE_RUN,
                  "source_artifact": SOURCE_ARTIFACT,
                  "classification": "invalid_archived_input",
                  "error": str(exc), "new_training_episodes": 0,
                  "new_ppo_updates": 0, "new_motor_steps": 0,
                  "new_physical_keydowns": 0}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("classification", "before_adam_step_1", "before_adam_step_2",
                       "hits", "extras", "physical_keydowns")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
