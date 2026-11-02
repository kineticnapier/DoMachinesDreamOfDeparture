#!/usr/bin/env python3
"""D16: frozen first A0 PPO batch, selected-action learning-signal audit (no updates)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.distributions import Categorical

import a002_exploratory_ppo as ppo
import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15

PROTOCOL = "docs/research-a-v2-l0-d16-first-batch-action-signal-protocol.md"
SCHEMA = "research-a-v2-l0-d16-action-signal-v1"
ACTION_NAMES = l0.ACTION_NAMES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(actions, advantages, ratios, objectives, raw_objectives, clipped_objectives):
    """Summarize original PPO selected-action terms without interpreting as gradients."""
    tensors = (actions, advantages, ratios, objectives, raw_objectives, clipped_objectives)
    require(all(t.ndim == 1 for t in tensors), "expected one-dimensional tensors")
    n = actions.numel()
    require(n > 0 and all(t.numel() == n for t in tensors), "PPO term shape mismatch")
    require(actions.dtype == torch.long, "actions must be int64")
    require(bool(((actions >= 0) & (actions < 4)).all()), "invalid action index")
    require(all(bool(torch.isfinite(t).all()) for t in tensors[1:]), "nonfinite PPO term")
    av = [float(v) for v in advantages.tolist()]
    rv = [float(v) for v in ratios.tolist()]
    ov = [float(v) for v in objectives.tolist()]
    raw = [float(v) for v in raw_objectives.tolist()]
    clipped = [float(v) for v in clipped_objectives.tolist()]
    labels = actions.tolist()
    require(all(math.isclose(ov[i], min(raw[i], clipped[i]), rel_tol=0, abs_tol=1e-7)
                for i in range(n)), "objective not original PPO minimum")
    by_action = {}
    for a, name in enumerate(ACTION_NAMES):
        indices = [i for i, x in enumerate(labels) if x == a]
        count = len(indices)
        selected_adv = [av[i] for i in indices]
        selected_obj = [ov[i] for i in indices]
        selected_ratios = [rv[i] for i in indices]
        by_action[name] = {
            "index": a, "count": count,
            "positive_advantage": sum(v > 0 for v in selected_adv),
            "negative_advantage": sum(v < 0 for v in selected_adv),
            "zero_advantage": sum(v == 0 for v in selected_adv),
            "advantage_sum": math.fsum(selected_adv),
            "advantage_mean": math.fsum(selected_adv) / count if count else None,
            "objective_sum": math.fsum(selected_obj),
            "objective_mean": math.fsum(selected_obj) / count if count else None,
            "ratio_min": min(selected_ratios) if count else None,
            "ratio_max": max(selected_ratios) if count else None,
            "clipped_side_selected": sum(clipped[i] < raw[i] for i in indices),
        }
    require(sum(row["count"] for row in by_action.values()) == n,
            "selected-action counts do not cover batch")
    require(all(row["count"] == row["positive_advantage"] +
                row["negative_advantage"] + row["zero_advantage"]
                for row in by_action.values()), "advantage sign counts differ")
    total_adv = math.fsum(av)
    total_obj = math.fsum(ov)
    require(math.isclose(math.fsum(row["advantage_sum"] for row in by_action.values()),
                         total_adv, rel_tol=0, abs_tol=1e-10),
            "advantage totals differ")
    require(math.isclose(math.fsum(row["objective_sum"] for row in by_action.values()),
                         total_obj, rel_tol=0, abs_tol=1e-10),
            "policy objective totals differ")
    rows = [{"index": i, "action": labels[i], "advantage": av[i],
             "ratio": rv[i], "objective": ov[i],
             "raw_objective": raw[i], "clipped_objective": clipped[i],
             "clipped_side_selected": clipped[i] < raw[i]} for i in range(n)]
    return {"steps": n, "advantage_sum": total_adv, "objective_sum": total_obj,
            "by_action": by_action, "steps_detail": rows}


def frozen_forward(model, batch):
    """Use the original update_ppo first-epoch formula, with no backward/Adam."""
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    old_log_probs = torch.cat([ep["old_log_probs"] for ep in batch])
    returns = torch.cat([ep["returns"] for ep in batch])
    old_values = torch.cat([ep["old_values"] for ep in batch])
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-6)
    model.train()
    with torch.no_grad():
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
    result = summarize(actions, advantages, ratio, objective, raw, clipped)
    result["policy_loss"] = float(policy_loss.item())
    result["value_loss"] = float(value_loss.item())
    result["entropy"] = float(entropy.item())
    result["first_epoch_loss"] = float((
        policy_loss + ppo.VALUE_COEF * value_loss -
        ppo.ENTROPY_COEF * entropy).item())
    require(all(math.isfinite(result[k]) for k in
                ("policy_loss", "value_loss", "entropy", "first_epoch_loss")),
            "nonfinite first epoch loss")
    require(math.isclose(result["policy_loss"],
                         -result["objective_sum"] / result["steps"],
                         rel_tol=0, abs_tol=1e-6),
            "policy loss and objective total differ")
    return result


def run(d8_dir: Path, d12_dir: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "D16 must run on native CPU path")
    torch.set_num_threads(1)
    source = d15.load_source(d12_dir)
    capture, _, _, _ = d13.validate_archived(source)
    model, initial_sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(initial_sha == capture["initial_state_sha256"], "initial weight SHA mismatch")
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
    before = {name: d12.tensor_bits(param) for name, param in model.named_parameters()}
    result = frozen_forward(model, batch)
    after = {name: d12.tensor_bits(param) for name, param in model.named_parameters()}
    require(before == after, "forward modified model weights")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_action_signal_audit",
        "source_run": d13.SOURCE_RUN, "source_artifact": d13.SOURCE_ARTIFACT,
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(), "arm": "a0", "seed": 0,
        "episodes": 8, "targets": 128, "hits": 0, "extras": 9,
        "physical_keydowns": 9, **result,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_backward_calls": 0, "new_adam_steps": 0,
        "new_motor_steps": 0, "new_physical_keydowns": 0,
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
        raise FileExistsError("refusing to overwrite D16 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_audit", "error": str(exc),
                   "new_training_episodes": 0, "new_ppo_updates": 0,
                   "new_backward_calls": 0, "new_adam_steps": 0,
                   "new_motor_steps": 0, "new_physical_keydowns": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("classification", "steps", "by_action",
                               "policy_loss", "hits", "extras",
                               "physical_keydowns")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
