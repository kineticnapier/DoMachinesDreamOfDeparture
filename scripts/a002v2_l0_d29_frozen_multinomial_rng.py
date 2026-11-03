#!/usr/bin/env python3
"""D29: replay archived multinomial draws and swap step413 probabilities at fixed RNG state.

No policy inference, training, Motor replay, or new scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

import a002v2_l0_d28_step413_policy_reward as d28

SCHEMA = "research-a-v2-l0-d29-frozen-multinomial-rng-v1"
PROTOCOL = "docs/research-a-v2-l0-d29-frozen-multinomial-protocol.md"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
SEED = 320403
STEP = 413
CONTROLS = 480
require = d28.require


def validate_runtime():
    require(torch.__version__ == "2.5.1+cpu",
            "PyTorch runtime differs from frozen original: " + torch.__version__)
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "ATEN_CPU_CAPABILITY override forbidden")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "CPU thread count differs")


def tensor_probs(step):
    p = d28.probabilities(step)
    return torch.tensor(p, dtype=torch.float32, device="cpu")


def state_digest(state):
    require(state.dtype == torch.uint8 and state.device.type == "cpu",
            "invalid RNG state")
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def draw_at_state(probs, state):
    rng = torch.Generator(device="cpu")
    rng.set_state(state.clone())
    return int(torch.multinomial(probs, 1, generator=rng).item())


def replay_sequence(steps, seed=SEED, pivot=STEP):
    require(len(steps) > pivot, "missing pivot step")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    pivot_state = None
    matches = 0
    for i, step in enumerate(steps):
        require(type(step.get("step")) is int and step["step"] == i,
                "step index mismatch at " + str(i))
        require(type(step.get("sampled_action")) is int and
                0 <= step["sampled_action"] <= 3,
                "invalid archived sampled action at " + str(i))
        p = tensor_probs(step)
        if i == pivot:
            pivot_state = rng.get_state().clone()
        sampled = int(torch.multinomial(p, 1, generator=rng).item())
        require(sampled == step["sampled_action"],
                f"sample mismatch at step {i}: replay={sampled}, archived={step['sampled_action']}")
        matches += 1
    require(pivot_state is not None, "missing pivot state")
    return {"matched_steps": matches,
            "pivot_state_sha256": state_digest(pivot_state),
            "pivot_state": pivot_state}


def analyze(source, traces):
    # Reuse D28's complete D22 provenance and negative-control validation.
    d28.analyze(source, traces)
    validate_runtime()
    b = traces["before"]["steps"]
    a = traces["after_to_none"]["steps"]
    require(len(b) == len(a) == CONTROLS, "unexpected control count")
    with torch.no_grad():
        br = replay_sequence(b)
        ar = replay_sequence(a)
        pb, pa = tensor_probs(b[STEP]), tensor_probs(a[STEP])
        states = {"before": br["pivot_state"], "after_to_none": ar["pivot_state"]}
        probs = {"before": pb, "after_to_none": pa}
        cross = {state_name: {
            prob_name: draw_at_state(p, state)
            for prob_name, p in probs.items()}
            for state_name, state in states.items()}
    state_exact = torch.equal(br["pivot_state"], ar["pivot_state"])
    # Draws from each original RNG state must match the archived original action.
    require(cross["before"]["before"] == b[STEP]["sampled_action"] and
            cross["after_to_none"]["after_to_none"] == a[STEP]["sampled_action"],
            "pivot original draw mismatch")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_frozen_multinomial_replay",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "runtime": {"torch": torch.__version__, "threads": torch.get_num_threads(),
                    "ATEN_CPU_CAPABILITY": None},
        "arm": "a0", "seed": SEED, "step": STEP, "time_s": 4.13,
        "replay": {name: {"matched_steps": replay["matched_steps"],
                          "pivot_state_sha256": replay["pivot_state_sha256"]}
                   for name, replay in (("before", br), ("after_to_none", ar))},
        "pivot_rng_state_exact": state_exact,
        "pivot_probability_float32": {name: p.tolist() for name, p in probs.items()},
        "pivot_archived_actions": {"before": b[STEP]["sampled_action"],
                                   "after_to_none": a[STEP]["sampled_action"]},
        "pivot_cross_draws": cross,
        "pivot_swapping_probabilities_changes_draw": {
            name: draws["before"] != draws["after_to_none"]
            for name, draws in cross.items()},
        "same_rng_state_suffices_for_original_difference":
            (state_exact and cross["before"]["before"] !=
             cross["before"]["after_to_none"]),
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_policy_inferences": 0,
        "new_motor_replays": 0, "new_physical_keydowns": 0,
        "new_hit_extra_evaluations": 0,
        "archived_d19_nonintervention_sampled": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50},
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("One archived A0 seed and CPU torch.multinomial implementation; "
                       "replaying action sampling does not establish PPO learning, "
                       "other seeds, A1/A3, or physical performance improvement."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D29 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            RuntimeError, json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_frozen_multinomial_replay",
                   "error": str(exc), "new_training_episodes": 0,
                   "new_ppo_updates": 0, "new_policy_inferences": 0,
                   "new_motor_replays": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D29_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D29_RESULT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
