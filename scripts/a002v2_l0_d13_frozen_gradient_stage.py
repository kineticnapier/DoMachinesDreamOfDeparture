#!/usr/bin/env python3
"""D13: audit the first CPU-dependent gradient stage from archived D12 JSON only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

SOURCE_RUN = 41424435292
SOURCE_ARTIFACT = 13130809065
PROTOCOL = "docs/research-a-v2-l0-d13-frozen-gradient-stage-protocol.md"
REQUIRED = ("capture", "native", "default", "comparison")
FLOAT32_MIN = -(1 << 31)
FLOAT32_MAX = (1 << 31) - 1


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode_bits(bits):
    require(type(bits) is int and FLOAT32_MIN <= bits <= FLOAT32_MAX,
            "invalid float32 bit representation")
    value = struct.unpack("<f", struct.pack("<i", bits))[0]
    require(math.isfinite(value), "nonfinite gradient value")
    return value


def compare_parameters(native, default):
    require(isinstance(native, dict) and isinstance(default, dict),
            "gradient record must be a parameter map")
    require(native.keys() == default.keys() and bool(native),
            "gradient parameter names mismatch")
    changed = 0
    changed_names = 0
    total = 0
    maximum = 0.0
    first = None
    by_parameter = {}
    for name in native:
        a, b = native[name], default[name]
        require(isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and len(a) > 0, "gradient tensor shape mismatch: " + name)
        local = 0
        for index, (x, y) in enumerate(zip(a, b)):
            xf, yf = decode_bits(x), decode_bits(y)
            total += 1
            if x != y:
                changed += 1
                local += 1
                delta = abs(xf - yf)
                maximum = max(maximum, delta)
                if first is None:
                    first = {"name": name, "index": index,
                             "native_bits": x, "default_bits": y,
                             "native_value": xf, "default_value": yf,
                             "absolute_difference": delta}
        if local:
            changed_names += 1
            by_parameter[name] = local
    return {"total_elements": total, "different_elements": changed,
            "different_parameters": changed_names,
            "max_abs_difference": maximum,
            "first_difference": first, "differences_by_parameter": by_parameter}


def validate_archived(source):
    require(set(source) == set(REQUIRED), "archived file set incomplete")
    c, n, d, previous = (source[k] for k in REQUIRED)
    require(c.get("schema") == "research-a-v2-l0-d12-capture-v1",
            "unexpected D12 capture schema")
    require(n.get("schema") == "research-a-v2-l0-d12-update-v1"
            and d.get("schema") == "research-a-v2-l0-d12-update-v1",
            "unexpected D12 update schema")
    require(previous.get("schema") == "research-a-v2-l0-d12-comparison-v1",
            "unexpected D12 comparison schema")
    require(previous.get("classification") == "ppo_update_difference_observed",
            "D12 source result differs")
    require(n.get("condition") == "native" and d.get("condition") == "default",
            "D12 CPU condition mismatch")
    require(all(row.get("initial_state_sha256") == c.get("initial_state_sha256")
                and row.get("batch_sha256") == c.get("batch_sha256")
                for row in (n, d, previous)), "D12 frozen identity mismatch")
    require(all(isinstance(c.get(k), str) and len(c[k]) == 64
                for k in ("initial_state_sha256", "batch_sha256")),
            "missing D12 SHA256")
    require(c.get("episodes") == 8 and c.get("targets") == 128
            and c.get("hits") == 0 and c.get("extras") == 9
            and c.get("physical_keydowns") == 9,
            "D12 physical event totals mismatch")
    for row in (n, d):
        require(row.get("episodes") == 8 and row.get("targets") == 128
                and row.get("hits") == 0 and row.get("extras") == 9
                and row.get("physical_keydowns") == 9
                and row.get("ppo_updates") == 1 and row.get("adam_steps") == 2
                and row.get("new_training_episodes") == 0
                and row.get("motor_steps") == 0
                and row.get("new_physical_keydowns") == 0,
                "D12 update/physical event contract mismatch")
    nr, dr = n["runtime"], d["runtime"]
    require(all(nr.get(k) == dr.get(k) for k in
                ("python", "torch", "platform", "cpu_model", "torch_threads")),
            "D12 runtime mismatch")
    require(nr.get("torch_threads") == 1
            and nr.get("cpu_capability") != dr.get("cpu_capability"),
            "D12 CPU dispatch not distinct")
    require(previous.get("same_environment") is True
            and previous.get("distinct_dispatch") is True,
            "D12 environment attestation mismatch")
    require(previous.get("weights", {}).get("different_elements") == 10
            and previous.get("adam", {}).get("exp_avg", {}).get("different_elements") == 572
            and previous.get("adam", {}).get("exp_avg_sq", {}).get("different_elements") == 739,
            "D12 post-update evidence mismatch")
    require(previous.get("targets") == 128 and previous.get("hits") == 0
            and previous.get("extras") == 9
            and previous.get("physical_keydowns") == 9,
            "D12 comparison physical counts mismatch")
    require(set(n["losses"]) == set(d["losses"]) and len(n["losses"]) == 4,
            "D12 loss fields mismatch")
    require(all(type(v) in (int, float) and math.isfinite(v)
                for row in (n, d) for v in row["losses"].values()),
            "nonfinite D12 losses")
    require(all(n["losses"][k] == d["losses"][k]
                and previous["loss_delta_default_minus_native"][k] == 0
                for k in n["losses"]), "D12 loss delta differs")
    return c, n, d, previous


def analyze(source):
    capture, native, default, previous = validate_archived(source)
    ng = native.get("actor_and_critic_gradients")
    dg = default.get("actor_and_critic_gradients")
    require(isinstance(ng, list) and isinstance(dg, list)
            and len(ng) == len(dg) == 2, "D12 Adam gradient count mismatch")
    stages = [compare_parameters(a, b) for a, b in zip(ng, dg)]
    require(all(stage["total_elements"] == 1509 for stage in stages),
            "D12 gradient parameter count mismatch")
    archived_stages = previous.get("gradients_by_epoch")
    require(isinstance(archived_stages, list) and len(archived_stages) == 2,
            "D12 archived gradient comparison missing")
    for i, stage in enumerate(stages):
        require(stage["different_elements"] == archived_stages[i]["different_elements"]
                and stage["max_abs_difference"] == archived_stages[i]["max_abs_difference"]
                and stage["first_difference"] == archived_stages[i]["first_difference"],
                "D12 archived gradient comparison differs at step " + str(i + 1))
    if stages[0]["different_elements"]:
        classification = "gradient_difference_before_first_adam_step"
    elif stages[1]["different_elements"]:
        classification = "gradient_difference_only_before_second_adam_step"
    else:
        classification = "no_gradient_difference_in_two_steps"
    return {"schema": "research-a-v2-l0-d13-gradient-stage-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "source_artifact": SOURCE_ARTIFACT,
            "classification": classification,
            "initial_state_sha256": capture["initial_state_sha256"],
            "batch_sha256": capture["batch_sha256"],
            "cpu_capabilities": {"native": native["runtime"]["cpu_capability"],
                                 "default": default["runtime"]["cpu_capability"]},
            "gradient_before_adam_step_1": stages[0],
            "gradient_before_adam_step_2": stages[1],
            "post_update_weight_difference_elements": 10,
            "post_update_adam_exp_avg_difference_elements": 572,
            "post_update_adam_exp_avg_sq_difference_elements": 739,
            "loss_differences": 0, "targets": 128, "hits": 0,
            "extras": 9, "physical_keydowns": 9,
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
        raise FileExistsError("refusing to overwrite D13 output")
    try:
        source = {name: json.loads((args.source_dir / (name + ".json")).read_text(
            encoding="utf-8")) for name in REQUIRED}
        result = analyze(source)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        result = {"schema": "research-a-v2-l0-d13-gradient-stage-v1",
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
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("classification", "source_run", "hits", "extras",
                               "physical_keydowns", "gradient_before_adam_step_1",
                               "gradient_before_adam_step_2")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
