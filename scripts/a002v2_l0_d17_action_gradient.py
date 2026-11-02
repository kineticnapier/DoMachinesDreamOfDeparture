#!/usr/bin/env python3
"""D17: decompose frozen first PPO policy gradient by selected action; no updates."""
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
import a002v2_l0_d16_first_batch_action_signal as d16

PROTOCOL = "docs/research-a-v2-l0-d17-action-gradient-protocol.md"
SCHEMA = "research-a-v2-l0-d17-action-gradient-v1"
ACTION_NAMES = d16.ACTION_NAMES
RECONSTRUCTION_LIMIT = 1e-6
EXPECTED_COUNTS = (1204, 1170, 783, 643)
EXPECTED_POLICY_LOSS = 1.2360120926757645e-08

def require(condition, message):
    if not condition:
        raise ValueError(message)

def flat_grad(loss, params, *, retain_graph=True):
    raw = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    return [torch.zeros_like(p).detach() if g is None else g.detach().clone()
            for p, g in zip(params, raw)]

def max_difference(a, b):
    require(len(a) == len(b) and len(a) > 0, "gradient list mismatch")
    return max(float((x - y).abs().max().item()) for x, y in zip(a, b))

def sum_grads(groups):
    require(bool(groups) and all(len(g) == len(groups[0]) for g in groups),
            "gradient group mismatch")
    return [sum((g[i] for g in groups[1:]), groups[0][i].clone())
            for i in range(len(groups[0]))]

def gradient_metrics(gradient, full):
    a = torch.cat([g.reshape(-1).double() for g in gradient])
    b = torch.cat([g.reshape(-1).double() for g in full])
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            "nonfinite gradient")
    norm = float(torch.linalg.vector_norm(a).item())
    full_norm = float(torch.linalg.vector_norm(b).item())
    dot = float(torch.dot(a, b).item())
    cosine = dot / (norm * full_norm) if norm > 0 and full_norm > 0 else None
    return {"l2_norm": norm, "dot_with_full_ppo_gradient": dot,
            "cosine_with_full_ppo_gradient": cosine}

def decompose(model, batch):
    """Seven autograd.grad calls: 4 groups, policy, auxiliary, full PPO."""
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    old_log_probs = torch.cat([ep["old_log_probs"] for ep in batch])
    returns = torch.cat([ep["returns"] for ep in batch])
    old_values = torch.cat([ep["old_values"] for ep in batch])
    require(observations.ndim == 2 and observations.shape[1] == 8,
            "unexpected observation shape")
    require(actions.dtype == torch.long and actions.ndim == 1,
            "unexpected action dtype/shape")
    n = int(actions.numel())
    require(n > 0 and all(v.numel() == n for v in
                          (old_log_probs, returns, old_values)),
            "batch tensor shape mismatch")
    require(bool(((actions >= 0) & (actions < 4)).all()), "invalid action")
    require(all(bool(torch.isfinite(v).all()) for v in
                (observations, old_log_probs, returns, old_values)),
            "nonfinite input")
    model.train()
    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    require("actor.bias" in names and "critic.bias" in names,
            "unexpected ActorCritic structure")
    before = {name: d12.tensor_bits(param) for name, param in model.named_parameters()}
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-6)
    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    ratio = (log_probs - old_log_probs).exp()
    raw = ratio * advantages
    clipped = ratio.clamp(1 - ppo.CLIP, 1 + ppo.CLIP) * advantages
    objective = torch.min(raw, clipped)
    policy_loss = -objective.mean()
    value_loss = (returns - values).pow(2).mean()
    entropy = distribution.entropy().mean()
    auxiliary = ppo.VALUE_COEF * value_loss - ppo.ENTROPY_COEF * entropy
    full_loss = policy_loss + auxiliary
    require(all(bool(torch.isfinite(v).all()) for v in
                (ratio, advantages, objective, policy_loss, value_loss,
                 entropy, full_loss)), "nonfinite PPO value")
    counts = [int((actions == a).sum().item()) for a in range(4)]
    require(sum(counts) == n, "action counts do not partition batch")
    # Divide by ALL rows, never by the number of rows in each action group.
    groups = [flat_grad(-(objective * (actions == a).to(objective.dtype)).sum() / n,
                        params) for a in range(4)]
    policy_grad = flat_grad(policy_loss, params)
    aux_grad = flat_grad(auxiliary, params)
    full_grad = flat_grad(full_loss, params, retain_graph=False)
    policy_error = max_difference(sum_grads(groups), policy_grad)
    full_error = max_difference(sum_grads(groups + [aux_grad]), full_grad)
    require(policy_error <= RECONSTRUCTION_LIMIT,
            "policy gradient reconstruction error")
    require(full_error <= RECONSTRUCTION_LIMIT,
            "full PPO gradient reconstruction error")
    after = {name: d12.tensor_bits(param) for name, param in model.named_parameters()}
    require(before == after, "gradient analysis modified weights")
    require(all(p.grad is None for p in params),
            "autograd.grad unexpectedly populated parameter.grad")
    bias_index = names.index("actor.bias")
    critic_index = names.index("critic.bias")
    require(all(bool(torch.equal(g[critic_index],
                                 torch.zeros_like(g[critic_index]))) for g in groups),
            "policy group has critic-only gradient")
    by_action = {}
    for a, name in enumerate(ACTION_NAMES):
        by_action[name] = {
            "index": a, "count": counts[a],
            "actor_bias_gradient": [float(v) for v in groups[a][bias_index].tolist()],
            **gradient_metrics(groups[a], full_grad),
        }
    return {
        "steps": n, "counts": counts, "by_action": by_action,
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "full_loss": float(full_loss.item()),
        "policy_gradient_l2": gradient_metrics(policy_grad, full_grad)["l2_norm"],
        "auxiliary_gradient_l2": gradient_metrics(aux_grad, full_grad)["l2_norm"],
        "full_ppo_gradient_l2": gradient_metrics(full_grad, full_grad)["l2_norm"],
        "policy_reconstruction_max_abs_error": policy_error,
        "full_reconstruction_max_abs_error": full_error,
        "autograd_grad_calls": 7, "weights_unchanged": True,
    }

def run(d8_dir: Path, d12_dir: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "D17 must run on native CPU path")
    torch.set_num_threads(1)
    source = d15.load_source(d12_dir)
    capture, _, _, _ = d13.validate_archived(source)
    model, initial_sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(initial_sha == capture["initial_state_sha256"],
            "initial weight SHA mismatch")
    batch = torch.load(d12_dir / "frozen-batch.pt", map_location="cpu",
                       weights_only=True)
    require(d12.batch_digest(batch) == capture["batch_sha256"],
            "frozen batch SHA mismatch")
    require(len(batch) == 8, "unexpected episode count")
    require(sum(ep["score"]["targets"] for ep in batch) == 128,
            "batch target count mismatch")
    for key, expected in (("hits", 0), ("extra_keydowns", 9),
                          ("physical_keydowns", 9)):
        require(sum(ep["score"][key] for ep in batch) == expected,
                "batch physical score mismatch: " + key)
    prior = d16.frozen_forward(model, batch)
    require([prior["by_action"][name]["count"] for name in ACTION_NAMES]
            == list(EXPECTED_COUNTS), "D16 selected-action counts differ")
    require(math.isclose(prior["policy_loss"], EXPECTED_POLICY_LOSS,
                         rel_tol=0, abs_tol=1e-7),
            "D16 first policy loss differs")
    measured = decompose(model, batch)
    require(measured["counts"] == list(EXPECTED_COUNTS),
            "gradient action counts differ from D16")
    require(math.isclose(measured["policy_loss"], prior["policy_loss"],
                         rel_tol=0, abs_tol=1e-7),
            "gradient policy loss differs from D16")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_action_gradient_decomposition",
        "source_run": d13.SOURCE_RUN, "source_artifact": d13.SOURCE_ARTIFACT,
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(), "arm": "a0", "seed": 0,
        "episodes": 8, "targets": 128, "hits": 0, "extras": 9,
        "physical_keydowns": 9, "d16_forward_policy_loss": prior["policy_loss"],
        **measured,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_steps": 0,
        "new_physical_keydowns": 0,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-dir", type=Path, required=True)
    parser.add_argument("--d12-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D17 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_gradient_decomposition",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_adam_steps": 0,
            "new_motor_steps": 0, "new_physical_keydowns": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "classification", "counts", "by_action", "policy_gradient_l2",
        "full_ppo_gradient_l2", "policy_reconstruction_max_abs_error",
        "full_reconstruction_max_abs_error", "hits", "extras",
        "physical_keydowns")}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
