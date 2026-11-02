#!/usr/bin/env python3
"""D11: freeze D8 dynamic evaluation observations, then compare CPU forward stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import struct
import sys

import torch

import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d10_fixed_forward as d10
from a002v2_l0_d8_cpu_dispatch_entropy import initial_digest, cpu_name
from a002v2_l0_d5_r1_prefix_order import save_new_json

PROTOCOL = "docs/research-a-v2-l0-d11-dynamic-forward-protocol.md"
SOURCE_RUN = 41397065687
EXPECTED = {"argmax": (9, 0, 0, 0), "sampled": (36, 5, 44, 49)}
STAGES = d10.STAGES


def digest(value):
    return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def float_from_bits(value):
    return struct.unpack("<f", struct.pack("<i", value))[0]


def load_archives(native_path, default_path):
    native = json.loads(native_path.read_text(encoding="utf-8"))
    default = json.loads(default_path.read_text(encoding="utf-8"))
    expected_sha = d10.validate_d8(native, default)
    for mode, (episodes, hits, extras, keydowns) in EXPECTED.items():
        a, b = native["stage0"][mode], default["stage0"][mode]
        for item in (a, b):
            if (item["episodes"], item["hits"], item["extra_keydowns"],
                    item["physical_left"] + item["physical_right"]) != (
                    episodes, hits, extras, keydowns):
                raise ValueError("D8 archived physical totals differ")
        for x, y in zip(a["rows"], b["rows"]):
            if (x["physical_trace"] != y["physical_trace"] or
                    x["controls"] != y["controls"] or x["steps"] != y["steps"]):
                raise ValueError("D8 physical traces/controls/steps differ")
        if len(a["rows"]) != episodes or len(b["rows"]) != episodes:
            raise ValueError("D8 archive episode count differs")
    return native, default, expected_sha


def make_model(weights, expected_sha):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = l0.ActorCritic()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True),
                          strict=True)
    model.eval()
    if initial_digest(model) != expected_sha:
        raise ValueError("D8 initial state SHA mismatch")
    return model


def runtime():
    return {"python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "cpu_model": cpu_name(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
            "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY")}


def capture_episode(model, mode, interval, start, seed):
    env = l0.LoggedLowSpeedEnv(arm="a0", interval_ms=interval, start_s=start)
    observation = env.reset()
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)
    controls = [0] * 4
    entropy_total = 0.0
    observations = []
    steps = 0
    with torch.no_grad():
        while True:
            x = torch.tensor(observation, dtype=torch.float32)
            observations.append(d10.bits(x))
            logits, _value = model(x)
            probs = torch.softmax(logits, dim=-1)
            entropy_total += float(
                -(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
            action = (int(probs.argmax().item()) if mode == "argmax" else
                      int(torch.multinomial(probs, 1, generator=rng).item()))
            controls[action] += 1
            steps += 1
            observation, _reward, finished = env.step(action)
            if finished:
                break
    result = env.result()
    if len(env.down_trace) != result.physical_keydowns:
        raise ValueError("physical trace accounting differs")
    row = {"arm": "a0", "interval_ms": interval, "start_s": start,
           "mode": mode, "sampling_seed": seed, **asdict(result),
           "physical_left": sum(e["key"] == "left" for e in env.down_trace),
           "physical_right": sum(e["key"] == "right" for e in env.down_trace),
           "controls": controls, "steps": steps,
           "mean_action_entropy": entropy_total / steps,
           "physical_trace": env.down_trace,
           "matched_keys": "".join("L" if k == "left" else "R"
                                   for k in result.matched_keys)}
    return row, observations


def capture(weights, d8_native, d8_default):
    if os.environ.get("ATEN_CPU_CAPABILITY") is not None:
        raise ValueError("native capture requires unset CPU override")
    native, _default, expected_sha = load_archives(d8_native, d8_default)
    model = make_model(weights, expected_sha)
    captured = []
    for mode in ("argmax", "sampled"):
        archived_rows = native["stage0"][mode]["rows"]
        case = 0
        position = 0
        for interval in l0.EVAL_INTERVALS_MS:
            for start in l0.EVAL_STARTS_S:
                repetitions = range(l0.REPETITIONS) if mode == "sampled" else (0,)
                for rep in repetitions:
                    seed = l0.eval_seed(case, rep) if mode == "sampled" else None
                    row, observations = capture_episode(model, mode, interval, start, seed)
                    row = {"case_index": case, "repetition": rep, **row}
                    if row != archived_rows[position]:
                        raise ValueError(
                            f"D8 native evaluation mismatch at {mode} row {position}")
                    captured.append({"mode": mode, "case_index": case,
                                     "repetition": rep, "interval_ms": interval,
                                     "start_s": start, "sampling_seed": seed,
                                     "observations_bits": observations})
                    position += 1
                case += 1
        if position != EXPECTED[mode][0]:
            raise ValueError("unexpected capture episode count")
    total_steps = sum(len(x["observations_bits"]) for x in captured)
    return {"schema": "research-a-v2-l0-d11-capture-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "initial_state_sha256": expected_sha,
            "observations_sha256": digest(captured),
            "episodes": len(captured), "steps": total_steps,
            "native_physical_keydowns": 49, "native_hits": 5, "native_extras": 44,
            "runtime": runtime(), "rows": captured,
            "new_training_episodes": 0, "ppo_updates": 0,
            "motor_steps": total_steps, "new_physical_keydowns": 49}


def forward(condition, weights, capture_path, d8_native, d8_default):
    configured = os.environ.get("ATEN_CPU_CAPABILITY")
    if condition not in ("native", "default"):
        raise ValueError("invalid condition")
    if ((condition == "native" and configured is not None) or
            (condition == "default" and configured != "default")):
        raise ValueError("ATEN_CPU_CAPABILITY must be fixed before startup")
    _native, _default, expected_sha = load_archives(d8_native, d8_default)
    source = json.loads(capture_path.read_text(encoding="utf-8"))
    if (source["schema"] != "research-a-v2-l0-d11-capture-v1" or
            source["initial_state_sha256"] != expected_sha or
            source["observations_sha256"] != digest(source["rows"]) or
            source["episodes"] != 45):
        raise ValueError("frozen observation source invalid")
    model = make_model(weights, expected_sha)
    rows = []
    for row in source["rows"]:
        stages = []
        for values in row["observations_bits"]:
            if len(values) != 8:
                raise ValueError("observation dimension differs")
            x = torch.tensor(values, dtype=torch.int32).view(torch.float32)
            result = d10.forward_stages(model, x)
            stages.append([d10.bits(result[key]) for key in STAGES])
        rows.append({"mode": row["mode"], "case_index": row["case_index"],
                     "repetition": row["repetition"], "stages": stages})
    return {"schema": "research-a-v2-l0-d11-forward-v1",
            "protocol": PROTOCOL, "source_run": SOURCE_RUN,
            "condition": condition, "initial_state_sha256": expected_sha,
            "observations_sha256": source["observations_sha256"],
            "episodes": len(rows), "steps": sum(len(r["stages"]) for r in rows),
            "stages": list(STAGES), "runtime": runtime(), "rows": rows,
            "new_training_episodes": 0, "ppo_updates": 0,
            "motor_steps": 0, "new_physical_keydowns": 0}


def mean_entropies(data):
    out = {}
    for mode in ("argmax", "sampled"):
        rows = [r for r in data["rows"] if r["mode"] == mode]
        weighted = 0.0
        total_steps = 0
        for row in rows:
            n = len(row["stages"])
            if not n:
                raise ValueError("empty forward episode")
            entropies = [float_from_bits(s[STAGES.index("entropy")][0])
                         for s in row["stages"]]
            weighted += (sum(entropies) / n) * n
            total_steps += n
        out[mode] = weighted / total_steps
    return out


def compare(native, default, capture_record, d8_native, d8_default):
    archive_n, archive_d, expected_sha = load_archives(d8_native, d8_default)
    if (native["condition"] != "native" or default["condition"] != "default" or
            native["initial_state_sha256"] != expected_sha or
            default["initial_state_sha256"] != expected_sha or
            native["observations_sha256"] != default["observations_sha256"] or
            native["observations_sha256"] != capture_record["observations_sha256"]):
        raise ValueError("D11 frozen input/weight conditions differ")
    if (native["stages"] != list(STAGES) or default["stages"] != list(STAGES) or
            native["episodes"] != 45 or default["episodes"] != 45 or
            native["steps"] != default["steps"] or
            native["steps"] != capture_record["steps"]):
        raise ValueError("D11 stage count differs")
    for row in (native, default):
        if any(row[k] != 0 for k in
               ("new_training_episodes", "ppo_updates", "motor_steps",
                "new_physical_keydowns")):
            raise ValueError("forward pass must not step Motor or train")
    a, b = native["runtime"], default["runtime"]
    same_environment = all(a[k] == b[k] for k in
                           ("torch", "platform", "cpu_model", "torch_threads"))
    distinct = a["cpu_capability"] != b["cpu_capability"]
    valid = same_environment and distinct and a["torch_threads"] == 1
    counts = {key: 0 for key in STAGES}
    maximum = {key: 0.0 for key in STAGES}
    first_by_stage = {}
    for row_index, (nr, dr) in enumerate(zip(native["rows"], default["rows"])):
        for key in ("mode", "case_index", "repetition"):
            if nr[key] != dr[key]:
                raise ValueError("D11 row identity differs")
        if len(nr["stages"]) != len(dr["stages"]):
            raise ValueError("D11 row steps differ")
        for step, (ns, ds) in enumerate(zip(nr["stages"], dr["stages"])):
            if len(ns) != len(STAGES) or len(ds) != len(STAGES):
                raise ValueError("D11 stage width differs")
            for idx, name in enumerate(STAGES):
                if len(ns[idx]) != len(ds[idx]):
                    raise ValueError("D11 stage element count differs")
                for j, (x, y) in enumerate(zip(ns[idx], ds[idx])):
                    if x != y:
                        counts[name] += 1
                        delta = abs(float_from_bits(x) - float_from_bits(y))
                        maximum[name] = max(maximum[name], delta)
                        if name not in first_by_stage:
                            first_by_stage[name] = {
                                "stage": name, "row_index": row_index,
                                "mode": nr["mode"], "case_index": nr["case_index"],
                                "repetition": nr["repetition"], "step": step,
                                "element_index": j, "native_bits": x,
                                "default_bits": y,
                                "native_value": float_from_bits(x),
                                "default_value": float_from_bits(y),
                                "absolute_difference": delta}
    first = next((first_by_stage[s] for s in STAGES if s in first_by_stage), None)
    means = {"native": mean_entropies(native), "default": mean_entropies(default)}
    archived = {"native": {m: archive_n["stage0"][m]["mean_entropy"] for m in EXPECTED},
                "default": {m: archive_d["stage0"][m]["mean_entropy"] for m in EXPECTED}}
    exact = {c: {m: means[c][m] == archived[c][m] for m in EXPECTED}
             for c in ("native", "default")}
    if not valid:
        classification = "invalid_diagnostic"
    elif not all(all(v.values()) for v in exact.values()):
        classification = "archived_entropy_not_reproduced"
    else:
        classification = ("dynamic_stage_divergence_observed" if first
                          else "no_dynamic_stage_divergence")
    return {"schema": "research-a-v2-l0-d11-comparison-v1",
            "protocol": PROTOCOL, "classification": classification,
            "same_environment": same_environment, "distinct_dispatch": distinct,
            "cpu_capabilities": {"native": a["cpu_capability"],
                                 "default": b["cpu_capability"]},
            "initial_state_sha256": expected_sha,
            "observations_sha256": native["observations_sha256"],
            "episodes": native["episodes"], "steps": native["steps"],
            "different_elements_by_stage": counts,
            "max_abs_difference_by_stage": maximum,
            "first_difference": first,
            "mean_entropy": means, "archived_d8_mean_entropy": archived,
            "archived_entropy_exact": exact,
            "native_physical_keydowns": capture_record["native_physical_keydowns"],
            "native_hits": capture_record["native_hits"],
            "native_extras": capture_record["native_extras"],
            "new_training_episodes": 0, "ppo_updates": 0,
            "historical_d3_d5_cause_verified": False,
            "historical_r2_gradients_verified": False,
            "formal_study_002_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "forward", "compare"))
    parser.add_argument("--condition", choices=("native", "default"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--d8-native", type=Path, required=True)
    parser.add_argument("--d8-default", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D11 output")
    if args.mode == "capture":
        if not args.weights:
            parser.error("capture requires --weights")
        result = capture(args.weights, args.d8_native, args.d8_default)
    elif args.mode == "forward":
        if not (args.weights and args.capture and args.condition):
            parser.error("forward requires --weights, --capture and --condition")
        result = forward(args.condition, args.weights, args.capture,
                         args.d8_native, args.d8_default)
    else:
        if not (args.capture and args.native and args.default):
            parser.error("compare requires --capture, --native and --default")
        result = compare(
            json.loads(args.native.read_text(encoding="utf-8")),
            json.loads(args.default.read_text(encoding="utf-8")),
            json.loads(args.capture.read_text(encoding="utf-8")),
            args.d8_native, args.d8_default)
    save_new_json(args.out, result)
    print("D11_RESULT", json.dumps({
        "mode": args.mode, "classification": result.get("classification"),
        "episodes": result.get("episodes"), "steps": result.get("steps"),
        "first_difference": result.get("first_difference"),
        "mean_entropy": result.get("mean_entropy"),
        "archived_entropy_exact": result.get("archived_entropy_exact")}),
        flush=True)
    if result.get("classification") in (
            "invalid_diagnostic", "archived_entropy_not_reproduced"):
        raise AssertionError("D11 stop criterion; preserve negative artifact")


if __name__ == "__main__":
    main()
