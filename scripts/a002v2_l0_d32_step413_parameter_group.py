#!/usr/bin/env python3
"""D32: fixed A0 step413 parameter-group checkpoint swap; no optimizer or Motor."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch

import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d18_frozen_policy_probability as d18

PROTOCOL = "docs/research-a-v2-l0-d32-step413-parameter-group-protocol.md"
SCHEMA = "research-a-v2-l0-d32-step413-parameter-group-swap-v1"
SOURCE_D8_RUN = 41397065687
SOURCE_D12_RUN = 41424435292
SOURCE_D28_RUN = 41562899406
EXPECTED_D22_RUN = 41515190995
PROB_TOL = 5e-6
RECON_TOL = 1e-12
GROUPS = ("shared", "actor", "critic")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_d28(source):
    require(source.get("schema") == "research-a-v2-l0-d28-step413-policy-reward-v1"
            and source.get("classification") == "valid_step413_archived_policy_reward_audit",
            "D28 schema/classification mismatch")
    require(source.get("source_run") == EXPECTED_D22_RUN
            and source.get("source_artifact") == 13175252296
            and source.get("arm") == "a0"
            and source.get("seed") == 320403
            and source.get("targets") == 16,
            "D28 provenance mismatch")
    snaps = source["step413"]
    b, a = snaps["before"], snaps["after_to_none"]
    require(b["step"] == a["step"] == 413
            and abs(b["time_s"] - 4.13) <= 1e-8
            and abs(a["time_s"] - 4.13) <= 1e-8,
            "D28 pivot step mismatch")
    require(source["step413_comparison"]["observation_exact"] is True
            and source["step413_comparison"]["pre_motor_exact"] is True
            and source["step413_comparison"]["pre_diagnostics_exact"] is True,
            "D28 pre-step413 state mismatch")
    obs = b["observation"]
    require(type(obs) is list and len(obs) == 8
            and all(type(x) in (int, float) and math.isfinite(x) for x in obs)
            and obs == a["observation"], "D28 observation mismatch")
    for snap in (b, a):
        p = snap["action_probabilities"]
        require(type(p) is list and len(p) == 4
                and all(type(x) in (int, float) and math.isfinite(x)
                        and 0 <= x <= 1 for x in p)
                and abs(sum(p) - 1.0) <= 1e-6,
                "D28 action probabilities invalid")
    require(b["sampled_action"] == 0 and a["sampled_action"] == 2
            and b["argmax_action"] == a["argmax_action"] == 0,
            "D28 actions changed")
    require(source["extra_at_4_327"]["before"] is None
            and source["extra_at_4_327"]["after_to_none"]["immediate_reward"] == -0.05,
            "D28 physical KeyDown/reward provenance mismatch")
    return obs, b["action_probabilities"], a["action_probabilities"]


def group_for(name):
    group = name.split(".", 1)[0]
    require(group in GROUPS and "." in name,
            "unexpected ActorCritic parameter: " + name)
    return group


def group_changes(before, after):
    b, a = d18.weights_bits(before), d18.weights_bits(after)
    require(set(b) == set(a) and bool(b), "weight names mismatch")
    result = {g: {"parameters": 0, "elements": 0, "changed_elements": 0}
              for g in GROUPS}
    for name in b:
        group = group_for(name)
        require(len(b[name]) == len(a[name]), "weight shape mismatch")
        r = result[group]
        r["parameters"] += 1
        r["elements"] += len(b[name])
        r["changed_elements"] += sum(x != y for x, y in zip(b[name], a[name]))
    require(sum(r["elements"] for r in result.values()) == 1509,
            "ActorCritic parameter count mismatch")
    require(all(r["parameters"] > 0 for r in result.values()),
            "empty ActorCritic parameter group")
    return result


def mixed_model(before, after, *, shared, actor, critic=False):
    require(all(type(v) is bool for v in (shared, actor, critic)),
            "group flags must be bool")
    b = dict(before.named_parameters())
    a = dict(after.named_parameters())
    require(b.keys() == a.keys(), "parameter names mismatch")
    out = copy.deepcopy(before)
    with torch.no_grad():
        for name, param in out.named_parameters():
            group = group_for(name)
            use_after = {"shared": shared, "actor": actor, "critic": critic}[group]
            param.copy_(a[name] if use_after else b[name])
    expected = {name: d12.tensor_bits(a[name] if
                 {"shared": shared, "actor": actor, "critic": critic}[group_for(name)]
                 else b[name]) for name in b}
    require(d18.weights_bits(out) == expected,
            "hybrid parameter bit mismatch")
    return out


def infer(model, observation):
    require(type(observation) is list and len(observation) == 8,
            "observation must contain eight components")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(observation, dtype=torch.float32)
        logits, _ = model(x)
        p = torch.softmax(logits, dim=-1)
    require(logits.shape == p.shape == (4,)
            and bool(torch.isfinite(logits).all())
            and bool(torch.isfinite(p).all()),
            "invalid policy output")
    prob = [float(v) for v in p.tolist()]
    require(abs(sum(prob) - 1.0) <= 1e-6
            and all(0 <= x <= 1 for x in prob),
            "probability normalization failed")
    return logits, p, {
        "logits": [float(v) for v in logits.tolist()],
        "probabilities": prob,
        "effective_left_probability": float(prob[2] + prob[3]),
        "argmax_action": int(p.argmax().item()),
    }


def contributions(p00, p10, p01, p11):
    vals = (p00, p10, p01, p11)
    require(all(type(x) in (float, int) and math.isfinite(x)
                and 0 <= x <= 1 for x in vals),
            "invalid contribution input")
    delta = p11 - p00
    shared = 0.5 * ((p10 - p00) + (p11 - p01))
    actor = 0.5 * ((p01 - p00) + (p11 - p10))
    interaction = p11 - p10 - p01 + p00
    error = abs((shared + actor) - delta)
    require(error <= RECON_TOL, "contribution reconstruction failed")
    return {"before": p00, "after": p11, "delta": delta,
            "delta_percentage_points": 100 * delta,
            "shared_symmetric_contribution": shared,
            "actor_symmetric_contribution": actor,
            "interaction": interaction,
            "reconstruction_abs_error": error}


def run(d8_dir: Path, d12_dir: Path, d28_path: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "must use native CPU path")
    require(torch.__version__ == "2.5.1+cpu",
            "pinned PyTorch 2.5.1+cpu required")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "single CPU thread required")
    source = d15.load_source(d12_dir)
    capture, native, _default, _comparison = d13.validate_archived(source)
    initial, sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(sha == capture["initial_state_sha256"],
            "D8/D12 initial SHA mismatch")
    after = d18.restore_updated(initial, native["weights_after"])
    obs, archived_before, archived_after = validate_d28(
        json.loads(d28_path.read_text(encoding="utf-8")))
    original_before = d18.weights_bits(initial)
    original_after = d18.weights_bits(after)
    changes = group_changes(initial, after)
    outputs = {}
    raw = {}
    for label, shared, actor, critic in (
            ("00", False, False, False),
            ("10", True, False, False),
            ("01", False, True, False),
            ("11", True, True, False),
            ("111", True, True, True)):
        model = mixed_model(initial, after, shared=shared, actor=actor, critic=critic)
        logits, probs, values = infer(model, obs)
        raw[label] = (logits, probs)
        outputs[label] = values
    require(torch.equal(raw["11"][0], raw["111"][0])
            and torch.equal(raw["11"][1], raw["111"][1]),
            "critic-only negative control differs")
    require(d18.weights_bits(initial) == original_before
            and d18.weights_bits(after) == original_after,
            "source model weights mutated")
    before_diff = max(abs(x-y) for x, y in zip(
        outputs["00"]["probabilities"], archived_before))
    after_diff = max(abs(x-y) for x, y in zip(
        outputs["11"]["probabilities"], archived_after))
    require(before_diff <= PROB_TOL and after_diff <= PROB_TOL,
            "D28 archived probability mismatch")
    c = contributions(*(outputs[k]["effective_left_probability"]
                        for k in ("00", "10", "01", "11")))
    require(outputs["00"]["argmax_action"] == outputs["11"]["argmax_action"] == 0,
            "D28 argmax mismatch")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_step413_parameter_group_swap",
        "source_runs": {"d8": SOURCE_D8_RUN, "d12": SOURCE_D12_RUN,
                        "d28": SOURCE_D28_RUN},
        "initial_state_sha256": sha,
        "batch_sha256": capture["batch_sha256"],
        "arm": "a0", "seed": 320403, "step": 413,
        "time_s": 4.13, "target_event_time_s": 4.327,
        "group_changes": changes,
        "crossed_conditions": outputs,
        "critic_only_negative_control_bit_exact": True,
        "d28_probability_max_abs_diff": {
            "before": before_diff, "after": after_diff},
        "left_command_probability_contributions": c,
        "archived_d19_nonintervention": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50},
        "new_policy_inferences": 5,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_backward_calls": 0,
        "new_motor_replays": 0, "new_physical_keydowns": 0,
        "new_hit_extra_evaluations": 0,
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": "Fixed A0 observation and hybrid weight groups only; not PPO gradient/reward attribution or closed-loop Extra probability.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d8-dir", type=Path, required=True)
    ap.add_argument("--d12-dir", type=Path, required=True)
    ap.add_argument("--d28-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D32 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir, args.d28_result)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_step413_parameter_group_swap",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_motor_replays": 0,
            "new_physical_keydowns": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D32_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D32_RESULT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
