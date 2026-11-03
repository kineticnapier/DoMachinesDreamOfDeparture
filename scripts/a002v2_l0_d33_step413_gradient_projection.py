#!/usr/bin/env python3
"""D33: frozen first-PPO gradient directional projection at A0 step413; no updates."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.distributions import Categorical

import a002_exploratory_ppo as ppo
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d17_action_gradient as d17
import a002v2_l0_d32_step413_parameter_group as d32
import a002v2_l0_d18_frozen_policy_probability as d18

PROTOCOL = "docs/research-a-v2-l0-d33-step413-gradient-projection-protocol.md"
SCHEMA = "research-a-v2-l0-d33-step413-gradient-projection-v1"
RUNS = {"d8": 41397065687, "d12": 41424435292,
        "d17": 41472753103, "d28": 41562899406}
GROUPS = ("shared", "actor", "critic")
NAMES = ("none", "right", "left", "both")
COUNTS = (1204, 1170, 783, 643)
RECON_LIMIT = 1e-7


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dot_by_group(names, target_grad, loss_grad):
    require(len(names) == len(target_grad) == len(loss_grad) and bool(names),
            "gradient length mismatch")
    by = {g: 0.0 for g in GROUPS}
    for name, a, b in zip(names, target_grad, loss_grad):
        group = d32.group_for(name)
        require(a.shape == b.shape, "gradient tensor shape mismatch")
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                "nonfinite gradient")
        by[group] += -float(torch.dot(a.reshape(-1).double(),
                                      b.reshape(-1).double()).item())
    total = sum(by.values())
    require(math.isfinite(total), "nonfinite directional projection")
    return {"by_parameter_group": by, "total": total}


def validate_archived_d17(row, initial_sha, batch_sha, counts, groups, policy, full):
    require(row.get("schema") == "research-a-v2-l0-d17-action-gradient-v1"
            and row.get("classification") == "valid_action_gradient_decomposition",
            "D17 archive schema/classification mismatch")
    require(row.get("initial_state_sha256") == initial_sha
            and row.get("batch_sha256") == batch_sha
            and row.get("arm") == "a0" and row.get("seed") == 0
            and row.get("targets") == 128
            and row.get("hits") == 0
            and row.get("extras") == 9
            and row.get("physical_keydowns") == 9,
            "D17 archive provenance mismatch")
    require(row.get("counts") == list(counts) == list(COUNTS),
            "D17 action counts mismatch")
    for i, name in enumerate(NAMES):
        r = row["by_action"][name]
        require(r["index"] == i and r["count"] == counts[i],
                "D17 action metadata mismatch")
        norm = d17.gradient_metrics(groups[i], full)["l2_norm"]
        require(abs(norm - r["l2_norm"]) <= 1e-5,
                "D17 action gradient norm mismatch: " + name)
    require(abs(d17.gradient_metrics(policy, full)["l2_norm"]
                - row["policy_gradient_l2"]) <= 1e-5,
            "D17 policy gradient norm mismatch")
    require(abs(d17.gradient_metrics(full, full)["l2_norm"]
                - row["full_ppo_gradient_l2"]) <= 1e-5,
            "D17 full gradient norm mismatch")


def first_ppo_gradients(model, batch):
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    old_log_probs = torch.cat([ep["old_log_probs"] for ep in batch])
    returns = torch.cat([ep["returns"] for ep in batch])
    old_values = torch.cat([ep["old_values"] for ep in batch])
    require(observations.shape == (3800, 8)
            and actions.shape == (3800,) and actions.dtype == torch.long,
            "D17 batch shape mismatch")
    require(all(t.shape == (3800,) for t in
                (old_log_probs, returns, old_values)), "D17 vector shape mismatch")
    require(bool(((actions >= 0) & (actions < 4)).all()),
            "invalid frozen action")
    counts = [int((actions == i).sum().item()) for i in range(4)]
    require(tuple(counts) == COUNTS, "D17 action counts differ")
    params = list(model.parameters())
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-6)
    model.train()
    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    ratio = (distribution.log_prob(actions) - old_log_probs).exp()
    objective = torch.min(ratio * advantages,
                          ratio.clamp(1 - ppo.CLIP, 1 + ppo.CLIP) * advantages)
    policy_loss = -objective.mean()
    value_loss = (returns - values).pow(2).mean()
    entropy = distribution.entropy().mean()
    auxiliary = ppo.VALUE_COEF * value_loss - ppo.ENTROPY_COEF * entropy
    full_loss = policy_loss + auxiliary
    require(all(bool(torch.isfinite(v).all()) for v in
                (ratio, objective, policy_loss, value_loss, entropy, full_loss)),
            "nonfinite PPO intermediate")
    # Each action loss is divided by all 3800 rows, not the action count.
    groups = [d17.flat_grad(-(objective * (actions == i).to(
        objective.dtype)).sum() / 3800, params) for i in range(4)]
    policy = d17.flat_grad(policy_loss, params)
    aux = d17.flat_grad(auxiliary, params)
    full = d17.flat_grad(full_loss, params, retain_graph=False)
    policy_err = d17.max_difference(d17.sum_grads(groups), policy)
    full_err = d17.max_difference(d17.sum_grads(groups + [aux]), full)
    require(policy_err <= 1e-6 and full_err <= 1e-6,
            "D17 gradient reconstruction failed")
    return groups, policy, aux, full, counts, {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "full_loss": float(full_loss.item()),
        "policy_reconstruction_max_abs_error": policy_err,
        "full_reconstruction_max_abs_error": full_err,
    }


def validate_reconstruction(projected):
    for group in GROUPS:
        part = sum(projected[n]["by_parameter_group"][group] for n in NAMES)
        require(abs(part - projected["policy"]["by_parameter_group"][group])
                <= RECON_LIMIT, "action projection reconstruction: " + group)
        require(abs(part + projected["auxiliary"]["by_parameter_group"][group]
                    - projected["full"]["by_parameter_group"][group])
                <= RECON_LIMIT, "full projection reconstruction: " + group)
    policy_err = abs(sum(projected[n]["total"] for n in NAMES)
                     - projected["policy"]["total"])
    full_err = abs(sum(projected[n]["total"] for n in NAMES)
                   + projected["auxiliary"]["total"] - projected["full"]["total"])
    require(policy_err <= RECON_LIMIT and full_err <= RECON_LIMIT,
            "total projection reconstruction failed")
    return {"policy": policy_err, "full": full_err}


def run(d8_dir: Path, d12_dir: Path, d17_path: Path, d28_path: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "native CPU path required")
    require(torch.__version__ == "2.5.1+cpu",
            "pinned torch 2.5.1+cpu required")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "one CPU thread required")
    source = d15.load_source(d12_dir)
    capture, native, _default, _comparison = d13.validate_archived(source)
    model, sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(sha == capture["initial_state_sha256"],
            "D8/D12 initial weights mismatch")
    batch = torch.load(d12_dir / "frozen-batch.pt",
                       map_location="cpu", weights_only=True)
    batch_sha = d12.batch_digest(batch)
    require(batch_sha == capture["batch_sha256"], "D12 batch SHA mismatch")
    require(len(batch) == 8 and sum(ep["score"]["targets"] for ep in batch) == 128,
            "D12 frozen episode/target mismatch")
    for key, expected in (("hits", 0), ("extra_keydowns", 9),
                          ("physical_keydowns", 9)):
        require(sum(ep["score"][key] for ep in batch) == expected,
                "D12 physical record mismatch: " + key)
    observation, archived_pre, archived_post = d32.validate_d28(
        json.loads(d28_path.read_text(encoding="utf-8")))
    original_bits = d18.weights_bits(model)
    names = [n for n, _ in model.named_parameters()]
    params = list(model.parameters())
    require(sum(p.numel() for p in params) == 1509, "model parameter count mismatch")
    groups, policy, aux, full, counts, diagnostics = first_ppo_gradients(model, batch)
    d17_record = json.loads(d17_path.read_text(encoding="utf-8"))
    validate_archived_d17(d17_record, sha, batch_sha, counts, groups, policy, full)
    model.eval()
    x = torch.tensor(observation, dtype=torch.float32)
    logits, _ = model(x)
    probs = torch.softmax(logits, dim=-1)
    require(probs.shape == (4,) and bool(torch.isfinite(probs).all()),
            "step413 policy output invalid")
    max_diff = max(abs(float(probs[i].item()) - archived_pre[i])
                   for i in range(4))
    require(max_diff <= 5e-6, "D28 initial probabilities mismatch")
    p_left = probs[2] + probs[3]
    grad_p = d17.flat_grad(p_left, params, retain_graph=False)
    require(all(torch.count_nonzero(g).item() == 0
                for n, g in zip(names, grad_p) if n.startswith("critic.")),
            "critic-only probability Jacobian not zero")
    projected = {}
    for label, grad in zip(NAMES + ("policy", "auxiliary", "full"),
                           groups + [policy, aux, full]):
        projected[label] = dot_by_group(names, grad_p, grad)
    reconstruction = validate_reconstruction(projected)
    require(d18.weights_bits(model) == original_bits,
            "source model weights modified")
    require(all(param.grad is None for param in params),
            "unexpected populated parameter.grad")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_step413_gradient_projection",
        "source_runs": RUNS, "initial_state_sha256": sha,
        "batch_sha256": batch_sha, "arm": "a0",
        "frozen_batch_seed": 0, "step413_seed": 320403,
        "frozen_episodes": 8, "frozen_rows": 3800,
        "frozen_targets": 128, "frozen_hits": 0,
        "frozen_extras": 9, "frozen_physical_keydowns": 9,
        "step": 413, "time_s": 4.13,
        "action_counts": dict(zip(NAMES, counts)),
        "step413_left_probability_before": float(p_left.item()),
        "step413_archived_left_probability_after":
            float(archived_post[2] + archived_post[3]),
        "step413_archived_actual_delta":
            float(sum(archived_post[2:]) - sum(archived_pre[2:])),
        "step413_archived_probability_max_abs_error": max_diff,
        "first_ppo_gradients": diagnostics,
        "projected_negative_gradient_direction_per_unit_sgd_lr": projected,
        "projection_reconstruction_abs_errors": reconstruction,
        "critic_probability_jacobian_exact_zero": True,
        "weights_unchanged": True, "autograd_grad_calls": 8,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_replays": 0,
        "new_physical_keydowns": 0, "new_hit_extra_evaluations": 0,
        "archived_d19_nonintervention": {
            "targets": 576, "before_hits": 5, "after_hits": 4,
            "before_extras": 44, "after_extras": 46,
            "before_physical_keydowns": 49, "after_physical_keydowns": 50},
        "a1_evaluated": False, "a3_evaluated": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": "Infinitesimal SGD directional derivative at frozen initial model; not causal attribution of actual two-step Adam update or physical Hit/Extra."
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d8-dir", type=Path, required=True)
    ap.add_argument("--d12-dir", type=Path, required=True)
    ap.add_argument("--d17-result", type=Path, required=True)
    ap.add_argument("--d28-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D33 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir, args.d17_result, args.d28_result)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_step413_gradient_projection",
                   "error": str(exc), "new_training_episodes": 0,
                   "new_ppo_updates": 0, "new_adam_steps": 0,
                   "new_motor_replays": 0, "new_physical_keydowns": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D33_FAILURE", json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D33_RESULT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
