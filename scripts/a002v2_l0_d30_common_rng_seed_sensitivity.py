#!/usr/bin/env python3
"""D30: audit common-RNG sampling sensitivity for archived step413 probabilities.

No learning, policy inference, Motor replay or physical KeyDown generation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

import a002v2_l0_d29_frozen_multinomial_rng as d29

SCHEMA = "research-a-v2-l0-d30-common-rng-seed-sensitivity-v1"
PROTOCOL = "docs/research-a-v2-l0-d30-common-rng-seed-sensitivity-protocol.md"
SEEDS = 4096
require = d29.require


def sample_pair(p_before, p_after, seed):
    """Draw once per distribution from identical fresh CPU RNG states."""
    require(type(seed) is int and seed >= 0, "invalid seed")
    g_before = torch.Generator(device="cpu").manual_seed(seed)
    g_after = torch.Generator(device="cpu").manual_seed(seed)
    require(torch.equal(g_before.get_state(), g_after.get_state()),
            "initial RNG state differs")
    before = int(torch.multinomial(p_before, 1, generator=g_before).item())
    after = int(torch.multinomial(p_after, 1, generator=g_after).item())
    return before, after


def sensitivity(p_before, p_after, count=SEEDS):
    require(type(count) is int and count > 0, "invalid fixed seed count")
    require(p_before.shape == p_after.shape == (4,), "invalid 4-action probabilities")
    for p in (p_before, p_after):
        require(p.dtype == torch.float32 and p.device.type == "cpu", "invalid probability tensor")
        require(bool(torch.isfinite(p).all()) and bool((p >= 0).all()),
                "nonfinite or negative probability")
        require(abs(float(p.sum()) - 1.0) < 1e-5, "probabilities not normalized")
    counts = [[0 for _ in range(4)] for _ in range(4)]
    changed_seeds = []
    for seed in range(count):
        before, after = sample_pair(p_before, p_after, seed)
        counts[before][after] += 1
        if before != after:
            changed_seeds.append({"seed": seed, "before": before, "after": after})
    require(sum(sum(row) for row in counts) == count, "count mismatch")
    disagreement = len(changed_seeds)
    require(disagreement == sum(counts[i][j] for i in range(4)
                                for j in range(4) if i != j),
            "disagreement count mismatch")
    tv = 0.5 * float(torch.abs(p_after.double() - p_before.double()).sum())
    require(math.isfinite(tv) and 0 <= tv <= 1, "invalid total variation")
    return {
        "seed_first": 0, "seed_last": count - 1, "seed_count": count,
        "draws_per_seed": 2, "total_draws": count * 2,
        "transition_matrix_before_rows_after_columns": counts,
        "before_action_counts": [sum(row) for row in counts],
        "after_action_counts": [sum(counts[i][j] for i in range(4)) for j in range(4)],
        "disagreement_count": disagreement,
        "disagreement_fraction": disagreement / count,
        "none_to_left": counts[0][2], "left_to_none": counts[2][0],
        "changed_seeds": changed_seeds,
        "total_variation_distance": tv,
        "total_variation_is_not_a_bound_on_this_coupling": True,
    }


def analyze(source, traces):
    # D29 validates D22 provenance, controls, original 960 draws and pivot RNG.
    d29_result = d29.analyze(source, traces)
    require(d29_result["pivot_rng_state_exact"] is True, "D29 pivot RNG mismatch")
    require(d29_result["replay"]["before"]["matched_steps"] == 480 and
            d29_result["replay"]["after_to_none"]["matched_steps"] == 480,
            "D29 archived replay mismatch")
    require(d29_result["pivot_cross_draws"]["before"] == {
        "before": 0, "after_to_none": 2}, "D29 positive control mismatch")
    p_before = torch.tensor(d29_result["pivot_probability_float32"]["before"],
                            dtype=torch.float32, device="cpu")
    p_after = torch.tensor(d29_result["pivot_probability_float32"]["after_to_none"],
                           dtype=torch.float32, device="cpu")
    with torch.no_grad():
        result = sensitivity(p_before, p_after)
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_seed_sensitivity_audit",
        "source_run": d29.SOURCE_RUN, "source_artifact": d29.SOURCE_ARTIFACT,
        "d29_replay": {"before": 480, "after_to_none": 480,
                       "pivot_rng_state_exact": True,
                       "archived_step413_cross_draw": [0, 2]},
        "arm": "a0", "source_seed": d29.SEED, "step": d29.STEP,
        "runtime": {"torch": torch.__version__, "threads": torch.get_num_threads(),
                    "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY")},
        "p_before": p_before.tolist(), "p_after": p_after.tolist(),
        "sampling": result,
        "archived_d19_nonintervention_sampled": {
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
        "limitation": ("Fixed 4096 independent initial CPU seeds; no physical "
                       "Motor evaluation or claim about other RNG states, "
                       "A1/A3, PPO training or human equivalence."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", type=Path, required=True)
    parser.add_argument("--d22-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D30 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        result = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            RuntimeError, json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_seed_sensitivity_audit",
                   "error": str(exc), "new_training_episodes": 0,
                   "new_ppo_updates": 0, "new_motor_replays": 0}
        args.out.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print("D30_FAILURE", json.dumps(failure), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("D30_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
