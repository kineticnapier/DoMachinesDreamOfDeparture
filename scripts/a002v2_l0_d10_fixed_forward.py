#!/usr/bin/env python3
"""D10: bit-exact fixed-input forward-stage comparison, with no learning/Motor steps."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d8_cpu_dispatch_entropy import initial_digest, cpu_name
from a002v2_l0_d5_r1_prefix_order import save_new_json
from dmdod.research.a002_timing_only_env import OBS_DIM

PROTOCOL = "docs/research-a-v2-l0-d10-fixed-forward-protocol.md"
SOURCE_RUN = 41397065687
STAGES = ("input", "shared.linear1", "shared.tanh1", "shared.linear2",
          "shared.tanh2", "actor.logits", "critic.value", "softmax", "entropy")
INTERVALS = (210, 240, 250)
STARTS = (.835, 1.075, 1.290)
SYNTHETIC = (
    ("all_zero", [0.] * 8),
    ("all_one", [1.] * 8),
    ("mixed_positive", [-1., -.5, 0., .125, .25, .5, .75, 1.]),
    ("alternating", [1., -1., .75, -.75, .5, -.5, .25, -.25]),
)


def bits(tensor):
    t = tensor.detach().to(dtype=torch.float32).contiguous().reshape(-1)
    if not bool(torch.isfinite(t).all()):
        raise ValueError("nonfinite forward tensor")
    return t.view(torch.int32).tolist()


def record(tensor):
    return {"bits": bits(tensor),
            "values": tensor.detach().to(dtype=torch.float32).reshape(-1).tolist()}


def inputs():
    if OBS_DIM != 8:
        raise ValueError("D10 preregistered inputs require OBS_DIM=8")
    result = []
    for period in INTERVALS:
        for start in STARTS:
            env = l0.LoggedLowSpeedEnv(arm="a0", interval_ms=period, start_s=start)
            observation = env.reset()  # No step, physical KeyDown or reward.
            x = torch.tensor(observation, dtype=torch.float32)
            if x.shape != (8,):
                raise ValueError("reset observation is not 8-dimensional")
            result.append((f"reset_{period}_{start:.3f}", x))
    for name, values in SYNTHETIC:
        result.append((name, torch.tensor(values, dtype=torch.float32)))
    if len(result) != 13:
        raise AssertionError("expected 13 fixed probes")
    return result


def forward_stages(model, x):
    with torch.no_grad():
        a = model.shared[0](x)
        b = model.shared[1](a)
        c = model.shared[2](b)
        d = model.shared[3](c)
        logits = model.actor(d)
        value = model.critic(d).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        actual_logits, actual_value = model(x)
        if not torch.equal(logits, actual_logits) or not torch.equal(value, actual_value):
            raise AssertionError("stage decomposition differs from original ActorCritic.forward")
    return dict(zip(STAGES, (x, a, b, c, d, logits, value, probs, entropy)))


def validate_d8(native, default):
    if native.get("condition") != "native" or default.get("condition") != "default":
        raise ValueError("D8 condition labels differ")
    sha = native.get("initial_state_sha256")
    if not sha or sha != default.get("initial_state_sha256"):
        raise ValueError("D8 archived initial weights not identical")
    if not default.get("loaded_original_native_weights"):
        raise ValueError("D8 generic path did not use original native weights")
    if native.get("runtime", {}).get("torch") != default.get("runtime", {}).get("torch"):
        raise ValueError("D8 PyTorch versions differ")
    return sha


def run(condition, weights_path, native_path, default_path):
    if condition not in ("native", "default"):
        raise ValueError("invalid condition")
    configured = os.environ.get("ATEN_CPU_CAPABILITY")
    if (condition == "native" and configured is not None or
            condition == "default" and configured != "default"):
        raise ValueError("ATEN_CPU_CAPABILITY must be set before Python startup")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    default = json.loads(default_path.read_text(encoding="utf-8"))
    expected = validate_d8(native, default)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = l0.ActorCritic()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True),
                          strict=True)
    model.eval()
    digest = initial_digest(model)
    if digest != expected:
        raise ValueError("D8 shared weights SHA mismatch")
    probes = []
    for label, x in inputs():
        stages = {key: record(value) for key, value in forward_stages(model, x).items()}
        probes.append({"label": label, "stages": stages})
    input_bits = [p["stages"]["input"]["bits"] for p in probes]
    input_sha = sha256(json.dumps(input_bits, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "research-a-v2-l0-d10-forward-v1",
        "protocol": PROTOCOL, "source_run": SOURCE_RUN,
        "condition": condition, "initial_state_sha256": digest,
        "input_bits_sha256": input_sha, "probe_count": len(probes),
        "stages": list(STAGES), "probes": probes,
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "platform": platform.platform(), "cpu_model": cpu_name(),
                    "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                    "torch_threads": torch.get_num_threads(),
                    "ATEN_CPU_CAPABILITY": configured},
        "new_training_episodes": 0, "ppo_updates": 0,
        "motor_steps": 0, "new_physical_keydowns": 0,
    }


def compare(native, default):
    if native.get("condition") != "native" or default.get("condition") != "default":
        raise ValueError("D10 condition labels differ")
    if native.get("initial_state_sha256") != default.get("initial_state_sha256"):
        raise ValueError("D10 initial weights differ")
    if native.get("input_bits_sha256") != default.get("input_bits_sha256"):
        raise ValueError("D10 input bits differ")
    if native.get("stages") != list(STAGES) or default.get("stages") != list(STAGES):
        raise ValueError("D10 stage list differs")
    if native.get("probe_count") != 13 or default.get("probe_count") != 13:
        raise ValueError("D10 probe count differs")
    for data in (native, default):
        if any(data.get(k) != 0 for k in
               ("new_training_episodes", "ppo_updates", "motor_steps", "new_physical_keydowns")):
            raise ValueError("D10 must not execute training or Motor steps")
    na, de = native["runtime"], default["runtime"]
    same_environment = all(na[k] == de[k] for k in
                           ("torch", "platform", "cpu_model", "torch_threads"))
    distinct = na["cpu_capability"] != de["cpu_capability"]
    if na["torch_threads"] != 1 or not same_environment:
        classification = "invalid_diagnostic"
    elif not distinct:
        classification = "dispatch_not_distinct"
    else:
        classification = None
    first = None
    stage_counts = {name: 0 for name in STAGES}
    stage_max_diffs = {name: 0. for name in STAGES}
    for i, (p, q) in enumerate(zip(native["probes"], default["probes"])):
        if p["label"] != q["label"]:
            raise ValueError("probe labels differ")
        for stage in STAGES:
            a, b = p["stages"][stage], q["stages"][stage]
            if len(a["bits"]) != len(b["bits"]) or len(a["values"]) != len(b["values"]):
                raise ValueError("stage tensor lengths differ")
            for j, (ab, bb) in enumerate(zip(a["bits"], b["bits"])):
                if ab != bb:
                    stage_counts[stage] += 1
                    diff = abs(a["values"][j] - b["values"][j])
                    stage_max_diffs[stage] = max(stage_max_diffs[stage], diff)
                    if first is None:
                        first = {"stage": stage, "probe": p["label"],
                                 "probe_index": i, "element_index": j,
                                 "native_bits": ab, "default_bits": bb,
                                 "native_value": a["values"][j],
                                 "default_value": b["values"][j],
                                 "absolute_difference": diff}
    # The first differing stage is defined by the fixed stage order,
    # not the first probe iteration in the JSON.
    if first is not None:
        earliest_stage = next(s for s in STAGES if stage_counts[s])
        if first["stage"] != earliest_stage:
            for i, (p, q) in enumerate(zip(native["probes"], default["probes"])):
                a, b = p["stages"][earliest_stage], q["stages"][earliest_stage]
                for j, (ab, bb) in enumerate(zip(a["bits"], b["bits"])):
                    if ab != bb:
                        first = {"stage": earliest_stage, "probe": p["label"],
                                 "probe_index": i, "element_index": j,
                                 "native_bits": ab, "default_bits": bb,
                                 "native_value": a["values"][j],
                                 "default_value": b["values"][j],
                                 "absolute_difference": abs(a["values"][j] - b["values"][j])}
                        break
                if first["stage"] == earliest_stage:
                    break
    if classification is None:
        classification = ("stage_divergence_observed" if first is not None
                          else "no_stage_divergence_on_fixed_probes")
    return {"schema": "research-a-v2-l0-d10-comparison-v1",
            "protocol": PROTOCOL, "classification": classification,
            "same_environment": same_environment, "distinct_dispatch": distinct,
            "initial_state_sha256": native["initial_state_sha256"],
            "input_bits_sha256": native["input_bits_sha256"],
            "cpu_capabilities": {"native": na["cpu_capability"],
                                 "default": de["cpu_capability"]},
            "first_difference": first, "different_elements_by_stage": stage_counts,
            "max_abs_difference_by_stage": stage_max_diffs,
            "probes": native["probe_count"], "new_training_episodes": 0,
            "ppo_updates": 0, "motor_steps": 0, "new_physical_keydowns": 0,
            "historical_d3_d5_cause_verified": False,
            "historical_r2_gradients_verified": False,
            "formal_study_002_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("evaluate", "compare"))
    parser.add_argument("--condition", choices=("native", "default"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--d8-native", type=Path)
    parser.add_argument("--d8-default", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite D10 output")
    if args.mode == "evaluate":
        if not (args.condition and args.weights and args.d8_native and args.d8_default):
            parser.error("evaluate requires condition, weights and D8 archives")
        result = run(args.condition, args.weights, args.d8_native, args.d8_default)
    else:
        if not (args.native and args.default):
            parser.error("compare requires native and default records")
        result = compare(json.loads(args.native.read_text()),
                         json.loads(args.default.read_text()))
    save_new_json(args.out, result)
    print("D10_RESULT", json.dumps({
        "classification": result.get("classification"),
        "first_difference": result.get("first_difference"),
        "different_elements_by_stage": result.get("different_elements_by_stage"),
        "cpu_capabilities": result.get("cpu_capabilities"),
    }), flush=True)
    if result.get("classification") == "invalid_diagnostic":
        raise AssertionError("D10 invalid comparison, preserve artifact")


if __name__ == "__main__":
    main()
