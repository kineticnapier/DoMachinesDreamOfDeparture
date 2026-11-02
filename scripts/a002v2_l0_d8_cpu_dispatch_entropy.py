#!/usr/bin/env python3
"""D8: compare pretraining evaluation under two ATen CPU dispatch paths.

No training, no PPO updates, no change to Motor, reward, or evaluation.
Protocol: docs/research-a-v2-l0-d8-cpu-dispatch-protocol.md
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d2_onset_advantage import source_models
from a002v2_l0_d4_r2_first_update import first_difference
from a002v2_l0_d5_r1_prefix_order import save_new_json

PROTOCOL = "docs/research-a-v2-l0-d8-cpu-dispatch-protocol.md"
R1_RUN, D3_RUN, D7_RUN = 41309837474, 41369778494, 41391082538
D7_ENTROPY = 1.3561848983669778
ARM, SEED = "a0", 0
EXPECTED_EPISODES = {"argmax": 9, "sampled": 36}


def initial_digest(model):
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    return sha256(stream.getvalue()).hexdigest()


def validate_archives(r1_dir: Path, d3_file: Path, d7_file: Path):
    models, _ = source_models(r1_dir, "r1")
    original = [m for m in models if (m["arm"], m["seed"]) == (ARM, SEED)]
    d3 = json.loads(d3_file.read_text(encoding="utf-8"))
    replay = [m for m in d3["models"]
              if (m["mode"], m["arm"], m["seed"]) == ("r1", ARM, SEED)]
    if len(original) != 1 or len(replay) != 1:
        raise ValueError("missing or duplicate R1 A0 seed0")
    original, replay = original[0], replay[0]
    if not replay["original_reproduced"]:
        raise ValueError("D3 R1 replay not exact")
    if not (original["checkpoint_sha256"] ==
            replay["source_checkpoint_sha256"] ==
            replay["replay_checkpoint_sha256"]):
        raise ValueError("original/D3 checkpoint SHA mismatch")
    if original["initial_state_sha256"] != d3["models"][0].get(
            "source_initial_state_sha256", original["initial_state_sha256"]):
        raise ValueError("D3 initial digest field mismatch")
    if original["evaluations"] != replay["evaluations"]:
        raise ValueError("original/D3 evaluations differ")
    if original["training"] != replay["training_log"]:
        raise ValueError("original/D3 training differs")
    if len(original["evaluations"]) != 3 or len(replay["all_updates"]) != 32:
        raise ValueError("incomplete R1/D3 archive")
    stage0 = original["evaluations"][0]
    d7 = json.loads(d7_file.read_text(encoding="utf-8"))
    diff = d7.get("stage0_difference")
    if d7.get("classification") != "initial_evaluation_diverged":
        raise ValueError("D7 archival stop classification differs")
    if not isinstance(diff, dict) or diff.get("path") != "stage0.argmax.mean_entropy":
        raise ValueError("D7 archival initial entropy difference missing")
    if diff.get("expected") != stage0["argmax"]["mean_entropy"]:
        raise ValueError("D7 archival original entropy differs")
    if diff.get("actual") != D7_ENTROPY:
        raise ValueError("D7 archival observed entropy differs")
    if stage0["episode"] != 0:
        raise ValueError("archived stage0 not initial")
    for mode, count in EXPECTED_EPISODES.items():
        if stage0[mode]["episodes"] != count:
            raise ValueError("unexpected archived evaluation count")
    return {
        "schema": "research-a-v2-l0-d8-archive-validation-v1",
        "protocol": PROTOCOL,
        "original_initial_sha256": original["initial_state_sha256"],
        "original_checkpoint_sha256": original["checkpoint_sha256"],
        "original_stage0": stage0,
        "source_runs": {"r1": R1_RUN, "d3": D3_RUN, "d7": D7_RUN},
        "d3_r1_reproduced": True,
        "d7_initial_entropy_difference_verified": True,
    }


def cpu_name():
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return "unknown"


def evaluate(expected: dict, condition: str, weights_file=None, weights_out=None):
    if condition not in ("native", "default"):
        raise ValueError("unknown condition")
    configured = os.environ.get("ATEN_CPU_CAPABILITY")
    if condition == "native" and configured is not None:
        raise ValueError("native must have ATEN_CPU_CAPABILITY unset")
    if condition == "default" and configured != "default":
        raise ValueError("default must set ATEN_CPU_CAPABILITY before Python start")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = l0.ActorCritic()
    model.eval()
    seed_only_digest = initial_digest(model)
    if condition == "native":
        if weights_file is not None:
            raise ValueError("native must generate original weights, not load them")
        if weights_out is None or weights_out.exists():
            raise ValueError("native requires unused --weights-out path")
        if seed_only_digest != expected["original_initial_sha256"]:
            raise ValueError("native initial weights differ from original R1")
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_out)
    else:
        if weights_file is None or not weights_file.is_file():
            raise ValueError("default requires native initial weights file")
        model.load_state_dict(
            torch.load(weights_file, map_location="cpu", weights_only=True),
            strict=True)
    digest = initial_digest(model)
    if digest != expected["original_initial_sha256"]:
        raise ValueError("shared initial model weights differ from original R1")
    stage0 = {
        "episode": 0,
        "argmax": l0.evaluate_set(arm=ARM, model=model, mode="argmax"),
        "sampled": l0.evaluate_set(arm=ARM, model=model, mode="sampled"),
    }
    for mode, count in EXPECTED_EPISODES.items():
        if stage0[mode]["episodes"] != count:
            raise ValueError("evaluation episode count differs")
        if stage0[mode]["targets"] != count * 16:
            raise ValueError("evaluation target count differs")
        if (stage0[mode]["hits"] + stage0[mode]["extra_keydowns"] !=
                stage0[mode]["physical_left"] + stage0[mode]["physical_right"]):
            raise ValueError("physical KeyDown accounting differs")
    return {
        "schema": "research-a-v2-l0-d8-cpu-dispatch-stage0-v1",
        "protocol": PROTOCOL, "condition": condition,
        "initial_state_sha256": digest,
        "seed_only_initial_state_sha256": seed_only_digest,
        "loaded_original_native_weights": condition == "default",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_model": cpu_name(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "ATEN_CPU_CAPABILITY": configured,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "stage0": stage0,
        "archived_r1_first_difference": first_difference(
            expected["original_stage0"], stage0, "archived_r1.stage0"),
    }


def compare(expected: dict, native: dict, default: dict):
    for label, row in (("native", native), ("default", default)):
        if row["condition"] != label:
            raise ValueError("condition label mismatch")
        if row["initial_state_sha256"] != expected["original_initial_sha256"]:
            raise ValueError("initial model weights mismatch: " + label)
        if row["runtime"]["torch_threads"] != 1:
            raise ValueError("thread count mismatch")
    if native["loaded_original_native_weights"] or not default["loaded_original_native_weights"]:
        raise ValueError("CPU comparison did not reuse the exact native initial weights")
    a, b = native["runtime"], default["runtime"]
    same_cpu = a["cpu_model"] == b["cpu_model"]
    same_torch = a["torch"] == b["torch"]
    same_platform = a["platform"] == b["platform"]
    distinct = a["cpu_capability"] != b["cpu_capability"]
    difference = first_difference(native["stage0"], default["stage0"], "stage0")
    if not (same_cpu and same_torch and same_platform):
        classification = "invalid_diagnostic"
    elif not distinct:
        classification = "dispatch_not_distinct"
    elif difference is None:
        classification = "no_dispatch_effect_observed"
    else:
        classification = "dispatch_effect_observed"
    out = {
        "schema": "research-a-v2-l0-d8-cpu-dispatch-comparison-v1",
        "protocol": PROTOCOL,
        "classification": classification,
        "same_cpu": same_cpu, "same_torch": same_torch,
        "same_platform": same_platform, "distinct_dispatch": distinct,
        "cpu_capabilities": {"native": a["cpu_capability"],
                             "default": b["cpu_capability"]},
        "initial_state_sha256_match": True,
        "seed_only_initial_sha256": {
            "native": native["seed_only_initial_state_sha256"],
            "default": default["seed_only_initial_state_sha256"],
        },
        "seed_only_initial_weights_equal":
            native["seed_only_initial_state_sha256"] ==
            default["seed_only_initial_state_sha256"],
        "stage0_first_difference": difference,
        "argmax_entropy": {
            "archived_r1_d3": expected["original_stage0"]["argmax"]["mean_entropy"],
            "archived_d7": D7_ENTROPY,
            "native": native["stage0"]["argmax"]["mean_entropy"],
            "default": default["stage0"]["argmax"]["mean_entropy"],
        },
        "argmax_entropy_delta_default_minus_native":
            default["stage0"]["argmax"]["mean_entropy"] -
            native["stage0"]["argmax"]["mean_entropy"],
        "r1_full_stage0_exact": {
            "native": native["archived_r1_first_difference"] is None,
            "default": default["archived_r1_first_difference"] is None,
        },
        "matches_d7_known_entropy": {
            "native": native["stage0"]["argmax"]["mean_entropy"] == D7_ENTROPY,
            "default": default["stage0"]["argmax"]["mean_entropy"] == D7_ENTROPY,
        },
        "physical": {
            mode: {condition: {
                key: data["stage0"][mode][key] for key in
                ("hits", "extra_keydowns", "physical_left", "physical_right",
                 "targets", "episodes")}
                for condition, data in (("native", native), ("default", default))}
            for mode in EXPECTED_EPISODES
        },
        "physical_trace_exact": {
            mode: first_difference(
                [row["physical_trace"] for row in native["stage0"][mode]["rows"]],
                [row["physical_trace"] for row in default["stage0"][mode]["rows"]],
                mode + ".physical_trace") is None
            for mode in EXPECTED_EPISODES
        },
        "action_controls_exact": {
            mode: first_difference(
                [row["controls"] for row in native["stage0"][mode]["rows"]],
                [row["controls"] for row in default["stage0"][mode]["rows"]],
                mode + ".controls") is None
            for mode in EXPECTED_EPISODES
        },
        "new_training_episodes": 0, "ppo_updates": 0,
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "evaluate", "compare"),
                        required=True)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--d3", type=Path)
    parser.add_argument("--d7", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--condition", choices=("native", "default"))
    parser.add_argument("--native", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite D8 output")
    if args.mode == "validate":
        if not (args.r1 and args.d3 and args.d7):
            parser.error("validate requires --r1, --d3 and --d7")
        result = validate_archives(args.r1, args.d3, args.d7)
    elif args.mode == "evaluate":
        if not (args.expected and args.condition):
            parser.error("evaluate requires --expected and --condition")
        result = evaluate(json.loads(args.expected.read_text()), args.condition,
                          weights_file=args.weights, weights_out=args.weights_out)
    else:
        if not (args.expected and args.native and args.default):
            parser.error("compare requires --expected, --native and --default")
        result = compare(
            json.loads(args.expected.read_text()),
            json.loads(args.native.read_text()),
            json.loads(args.default.read_text()))
    save_new_json(args.out, result)
    print("D8_RESULT", json.dumps({
        "mode": args.mode, "classification": result.get("classification"),
        "cpu_capabilities": result.get("cpu_capabilities"),
        "argmax_entropy": result.get("argmax_entropy"),
        "stage0_first_difference": result.get("stage0_first_difference"),
    }, ensure_ascii=False), flush=True)
    if result.get("classification") == "invalid_diagnostic":
        raise AssertionError("D8 invalid diagnostic; preserve artifact")


if __name__ == "__main__":
    main()
