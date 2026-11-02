#!/usr/bin/env python3
"""D12: compare one original PPO update on an identical frozen physical rollout."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import sys

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d8_cpu_dispatch_entropy import initial_digest, cpu_name
from a002v2_l0_d10_fixed_forward import validate_d8
from a002v2_l0_d5_r1_prefix_order import save_new_json

PROTOCOL = "docs/research-a-v2-l0-d12-frozen-ppo-cpu-protocol.md"
SOURCE_RUN = 41397065687
FIELDS = ("observations", "actions", "old_log_probs", "old_values", "returns")


def runtime():
    return {"python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "cpu_model": cpu_name(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
            "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY")}


def require_dispatch(condition):
    configured = os.environ.get("ATEN_CPU_CAPABILITY")
    if condition == "native" and configured is not None:
        raise ValueError("native must run without ATEN_CPU_CAPABILITY")
    if condition == "default" and configured != "default":
        raise ValueError("default must be set before Python startup")
    if condition not in ("native", "default"):
        raise ValueError("unknown CPU condition")


def weights_digest(weights, native_path, default_path):
    native = json.loads(native_path.read_text(encoding="utf-8"))
    default = json.loads(default_path.read_text(encoding="utf-8"))
    expected = validate_d8(native, default)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = l0.ActorCritic()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True),
                          strict=True)
    if initial_digest(model) != expected:
        raise ValueError("frozen D8 initial weights SHA mismatch")
    return model, expected


def tensor_bits(value):
    value = value.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise ValueError("expected float32 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite tensor")
    return value.reshape(-1).view(torch.int32).tolist()


def batch_digest(batch):
    if len(batch) != 8:
        raise ValueError("expected eight frozen episodes")
    digest = sha256()
    for ep in batch:
        if set(FIELDS) - set(ep):
            raise ValueError("incomplete frozen PPO episode")
        n = ep["actions"].numel()
        if not n or ep["observations"].shape != (n, 8):
            raise ValueError("invalid frozen observation shape")
        for key in FIELDS:
            tensor = ep[key].detach().cpu().contiguous()
            if tensor.shape[0] != n or not bool(torch.isfinite(tensor).all()):
                raise ValueError("nonfinite or malformed PPO tensor")
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        digest.update(json.dumps({key: ep[key] for key in
                                  ("score", "interval_ms", "start_s")},
                                 sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


class CapturingEnv(l0.LoggedLowSpeedEnv):
    traces = []

    def step(self, action_index):
        observation, reward, finished = super().step(action_index)
        if finished:
            result = self.result()
            if len(self.down_trace) != result.physical_keydowns:
                raise ValueError("physical KeyDown trace accounting differs")
            if result.hits + result.extra_keydowns != result.physical_keydowns:
                raise ValueError("Hit + Extra differs from physical KeyDown")
            type(self).traces.append({
                "score": asdict(result),
                "events": list(self.down_trace),
            })
        return observation, reward, finished


def capture(weights, native_path, default_path, batch_out):
    require_dispatch("native")
    if batch_out.exists():
        raise FileExistsError("refusing to overwrite frozen PPO batch")
    model, expected = weights_digest(weights, native_path, default_path)
    rng = random.Random(0)
    original_env = l0.LoggedLowSpeedEnv
    CapturingEnv.traces = []
    l0.LoggedLowSpeedEnv = CapturingEnv
    try:
        batch = [l0.train_episode(model, arm="a0", rng=rng) for _ in range(8)]
    finally:
        l0.LoggedLowSpeedEnv = original_env
    if len(CapturingEnv.traces) != 8:
        raise ValueError("not all physical traces captured")
    for ep, trace in zip(batch, CapturingEnv.traces):
        if ep["score"] != trace["score"]:
            raise ValueError("PPO episode score differs from physical trace")
    targets = sum(ep["score"]["targets"] for ep in batch)
    hits = sum(ep["score"]["hits"] for ep in batch)
    extras = sum(ep["score"]["extra_keydowns"] for ep in batch)
    keydowns = sum(ep["score"]["physical_keydowns"] for ep in batch)
    if targets != 128 or hits + extras != keydowns:
        raise ValueError("physical training event accounting invalid")
    batch_sha = batch_digest(batch)
    torch.save(batch, batch_out)
    if batch_digest(torch.load(batch_out, map_location="cpu", weights_only=True)) != batch_sha:
        raise ValueError("frozen batch serialization roundtrip failed")
    return {"schema": "research-a-v2-l0-d12-capture-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "initial_state_sha256": expected, "batch_sha256": batch_sha,
            "runtime": runtime(), "arm": "a0", "seed": 0,
            "episodes": 8, "targets": targets, "hits": hits,
            "extras": extras, "physical_keydowns": keydowns,
            "physical_traces": CapturingEnv.traces,
            "new_training_episodes": 8, "ppo_updates": 0,
            "adam_steps": 0}


def update(condition, weights, native_path, default_path, batch_path, capture_path):
    require_dispatch(condition)
    source = json.loads(capture_path.read_text(encoding="utf-8"))
    model, expected = weights_digest(weights, native_path, default_path)
    batch = torch.load(batch_path, map_location="cpu", weights_only=True)
    batch_sha = batch_digest(batch)
    if (source["schema"] != "research-a-v2-l0-d12-capture-v1" or
            source["initial_state_sha256"] != expected or
            source["batch_sha256"] != batch_sha or
            source["episodes"] != 8 or source["targets"] != 128):
        raise ValueError("frozen PPO batch does not match capture record")
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    gradients = []
    original_step = optimizer.step

    def record_step(*args, **kwargs):
        gradients.append({name: tensor_bits(parameter.grad)
                          for name, parameter in model.named_parameters()})
        return original_step(*args, **kwargs)

    optimizer.step = record_step
    try:
        # The unchanged original PPO function executes two epochs / Adam steps.
        losses = l0.update_ppo(model, optimizer, batch)
    finally:
        optimizer.step = original_step
    if len(gradients) != 2:
        raise ValueError("original PPO did not run two Adam steps")
    weights_after = {name: tensor_bits(value)
                     for name, value in model.named_parameters()}
    adam = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        if int(state["step"].item()) != 2:
            raise ValueError("Adam step count differs")
        adam[name] = {"exp_avg": tensor_bits(state["exp_avg"]),
                      "exp_avg_sq": tensor_bits(state["exp_avg_sq"])}
    if any(not math.isfinite(float(v)) for v in losses.values()):
        raise ValueError("nonfinite original PPO loss")
    return {"schema": "research-a-v2-l0-d12-update-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "condition": condition, "initial_state_sha256": expected,
            "batch_sha256": batch_sha, "runtime": runtime(),
            "losses": losses, "actor_and_critic_gradients": gradients,
            "weights_after": weights_after, "adam_after": adam,
            "episodes": 8, "targets": source["targets"],
            "hits": source["hits"], "extras": source["extras"],
            "physical_keydowns": source["physical_keydowns"],
            "new_training_episodes": 0, "ppo_updates": 1,
            "adam_steps": 2, "motor_steps": 0, "new_physical_keydowns": 0}


def bits_float(value):
    return struct.unpack("<f", struct.pack("<i", value))[0]


def compare_bits(left, right):
    if left.keys() != right.keys():
        raise ValueError("PPO parameter names differ")
    count, maximum, first = 0, 0.0, None
    for name in left:
        a, b = left[name], right[name]
        if len(a) != len(b):
            raise ValueError("PPO tensor shapes differ")
        for index, (x, y) in enumerate(zip(a, b)):
            if x != y:
                count += 1
                delta = abs(bits_float(x) - bits_float(y))
                maximum = max(maximum, delta)
                if first is None:
                    first = {"name": name, "index": index, "native_bits": x,
                             "default_bits": y, "native_value": bits_float(x),
                             "default_value": bits_float(y),
                             "absolute_difference": delta}
    return {"different_elements": count, "max_abs_difference": maximum,
            "first_difference": first}


def compare(native, default, capture_record):
    if (native["condition"] != "native" or default["condition"] != "default" or
            native["initial_state_sha256"] != default["initial_state_sha256"] or
            native["batch_sha256"] != default["batch_sha256"] or
            native["batch_sha256"] != capture_record["batch_sha256"] or
            native["initial_state_sha256"] != capture_record["initial_state_sha256"]):
        raise ValueError("D12 frozen model/batch identity mismatch")
    for row in (native, default):
        if (row["episodes"] != 8 or row["targets"] != 128 or
                row["ppo_updates"] != 1 or row["adam_steps"] != 2 or
                row["new_training_episodes"] != 0 or
                row["motor_steps"] != 0 or row["new_physical_keydowns"] != 0):
            raise ValueError("D12 one-update execution contract violated")
        for key in ("hits", "extras", "physical_keydowns"):
            if row[key] != capture_record[key]:
                raise ValueError("physical training record differs")
    a, b = native["runtime"], default["runtime"]
    same_environment = all(a[k] == b[k] for k in
                           ("python", "torch", "platform", "cpu_model", "torch_threads"))
    distinct = a["cpu_capability"] != b["cpu_capability"]
    valid = same_environment and distinct and a["torch_threads"] == 1
    weights = compare_bits(native["weights_after"], default["weights_after"])
    gradients = [compare_bits(n, d) for n, d in
                 zip(native["actor_and_critic_gradients"],
                     default["actor_and_critic_gradients"])]
    if len(gradients) != 2:
        raise ValueError("PPO gradient epochs differ")
    adam = {key: compare_bits(
        {n: state[key] for n, state in native["adam_after"].items()},
        {n: state[key] for n, state in default["adam_after"].items()})
        for key in ("exp_avg", "exp_avg_sq")}
    if native["losses"].keys() != default["losses"].keys():
        raise ValueError("PPO loss names differ")
    loss_delta = {k: default["losses"][k] - native["losses"][k]
                  for k in native["losses"]}
    changed = (weights["different_elements"] > 0 or
               any(v["different_elements"] > 0 for v in adam.values()) or
               any(v != 0 for v in loss_delta.values()))
    classification = ("invalid_diagnostic" if not same_environment or
                      a["torch_threads"] != 1 else
                      "dispatch_not_distinct" if not distinct else
                      "ppo_update_difference_observed" if changed else
                      "no_ppo_update_difference_observed")
    return {"schema": "research-a-v2-l0-d12-comparison-v1",
            "protocol": PROTOCOL, "classification": classification,
            "same_environment": same_environment, "distinct_dispatch": distinct,
            "cpu_capabilities": {"native": a["cpu_capability"],
                                 "default": b["cpu_capability"]},
            "initial_state_sha256": native["initial_state_sha256"],
            "batch_sha256": native["batch_sha256"],
            "weights": weights, "gradients_by_epoch": gradients,
            "adam": adam, "loss_delta_default_minus_native": loss_delta,
            "episodes_captured": capture_record["episodes"],
            "targets": capture_record["targets"], "hits": capture_record["hits"],
            "extras": capture_record["extras"],
            "physical_keydowns": capture_record["physical_keydowns"],
            "new_training_episodes": 8, "ppo_updates_per_condition": 1,
            "adam_steps_per_condition": 2,
            "historical_d3_d5_cause_verified": False,
            "historical_r2_gradients_verified": False,
            "formal_study_002_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "update", "compare"))
    parser.add_argument("--condition", choices=("native", "default"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--d8-native", type=Path)
    parser.add_argument("--d8-default", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D12 output")
    if args.mode == "capture":
        if not all((args.weights, args.d8_native, args.d8_default, args.batch)):
            parser.error("capture requires weights, D8 archives, batch path")
        result = capture(args.weights, args.d8_native, args.d8_default, args.batch)
    elif args.mode == "update":
        if not all((args.weights, args.d8_native, args.d8_default,
                    args.batch, args.capture, args.condition)):
            parser.error("update requires condition, weights, D8 archives, batch and capture")
        result = update(args.condition, args.weights, args.d8_native,
                        args.d8_default, args.batch, args.capture)
    else:
        if not all((args.capture, args.native, args.default)):
            parser.error("compare requires capture and both updates")
        result = compare(json.loads(args.native.read_text(encoding="utf-8")),
                         json.loads(args.default.read_text(encoding="utf-8")),
                         json.loads(args.capture.read_text(encoding="utf-8")))
    save_new_json(args.out, result)
    print("D12_RESULT", json.dumps({
        "mode": args.mode, "classification": result.get("classification"),
        "episodes": result.get("episodes", result.get("episodes_captured")),
        "hits": result.get("hits"), "extras": result.get("extras"),
        "physical_keydowns": result.get("physical_keydowns"),
        "weights": result.get("weights"), "adam": result.get("adam"),
        "loss_delta": result.get("loss_delta_default_minus_native"),
    }), flush=True)
    if result.get("classification") in ("invalid_diagnostic", "dispatch_not_distinct"):
        raise AssertionError("D12 preregistered stop condition")


if __name__ == "__main__":
    main()
