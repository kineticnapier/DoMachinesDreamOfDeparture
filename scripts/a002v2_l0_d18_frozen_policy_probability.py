#!/usr/bin/env python3
"""D18: compare archived pre/post-PPO policy probabilities on frozen observations."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch
from torch.distributions import Categorical

import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d16_first_batch_action_signal as d16

PROTOCOL = "docs/research-a-v2-l0-d18-frozen-policy-probability-protocol.md"
SCHEMA = "research-a-v2-l0-d18-frozen-policy-probability-v1"
COUNTS = (1204, 1170, 783, 643)
ACTION_NAMES = ("none", "right", "left", "both")
N = 3800


def require(ok, message):
    if not ok:
        raise ValueError(message)


def weights_bits(model):
    return {name: d12.tensor_bits(param) for name, param in model.named_parameters()}


def restore_updated(initial_model, archived):
    """Restore archived signed int32 float bits; never train or mutate initial_model."""
    require(isinstance(archived, dict), "missing archived updated weights")
    model = copy.deepcopy(initial_model)
    named = dict(model.named_parameters())
    require(set(archived) == set(named), "updated weight parameter names mismatch")
    total = 0
    with torch.no_grad():
        for name, param in named.items():
            bits = archived[name]
            require(isinstance(bits, list) and len(bits) == param.numel(),
                    "updated weight shape mismatch: " + name)
            require(all(type(v) is int and -(1 << 31) <= v < (1 << 31)
                        for v in bits), "invalid int32 weight bits: " + name)
            restored = torch.tensor(bits, dtype=torch.int32).view(torch.float32)
            restored = restored.reshape(param.shape)
            require(bool(torch.isfinite(restored).all()),
                    "nonfinite updated weights: " + name)
            param.copy_(restored)
            total += param.numel()
    require(total == 1509, "unexpected ActorCritic parameter count")
    require(weights_bits(model) == archived, "updated weight bit roundtrip differs")
    return model


def policy_probabilities(model, observations):
    require(observations.ndim == 2 and observations.shape[1] == 8,
            "unexpected observation shape")
    require(bool(torch.isfinite(observations).all()), "nonfinite observation")
    model.eval()
    with torch.no_grad():
        logits, _ = model(observations)
        probs = Categorical(logits=logits).probs.detach().clone()
    require(probs.shape == (observations.shape[0], 4),
            "unexpected policy output shape")
    require(bool(torch.isfinite(probs).all()), "nonfinite policy probabilities")
    require(bool(((probs >= 0) & (probs <= 1)).all()),
            "policy probability outside [0,1]")
    require(float((probs.sum(dim=1) - 1).abs().max().item()) <= 1e-6,
            "policy probabilities do not sum to one")
    require(bool((probs > 0).all()), "KL undefined for zero probability")
    return probs


def compare_probabilities(before, after, actions):
    require(before.shape == after.shape == (N, 4), "probability tensor shape mismatch")
    require(actions.shape == (N,) and actions.dtype == torch.long,
            "selected-action shape mismatch")
    require(bool(((actions >= 0) & (actions < 4)).all()), "invalid selected action")
    # Compute differences in float64 after the original float32 forward passes.
    b, a = before.double(), after.double()
    delta = a - b
    l1 = delta.abs().sum(dim=1)
    # Float32 softmax row sums can differ from 1 by an ulp. Normalize only\n    # for the mathematical KL, without changing reported raw probabilities.\n    bn = b / b.sum(dim=1, keepdim=True)\n    an = a / a.sum(dim=1, keepdim=True)\n    kl = (bn * (bn.log() - an.log())).sum(dim=1)
    require(bool(torch.isfinite(delta).all()) and bool(torch.isfinite(kl).all()),
            "nonfinite probability difference")
    require(float(kl.min().item()) >= -1e-8, "KL unexpectedly negative")
    changes = before.argmax(dim=1) != after.argmax(dim=1)
    chosen = torch.arange(N)
    selected = delta[chosen, actions]
    by_action = {}
    for i, name in enumerate(ACTION_NAMES):
        d = delta[:, i]
        by_action[name] = {
            "before_mean": float(b[:, i].mean().item()),
            "after_mean": float(a[:, i].mean().item()),
            "mean_delta": float(d.mean().item()),
            "increase_rows": int((d > 0).sum().item()),
            "decrease_rows": int((d < 0).sum().item()),
            "unchanged_rows": int((d == 0).sum().item()),
            "max_abs_delta": float(d.abs().max().item()),
        }
        require(sum(by_action[name][k] for k in
                    ("increase_rows", "decrease_rows", "unchanged_rows")) == N,
                "action change accounting mismatch")
    return {
        "steps": N,
        "by_action": by_action,
        "selected_action_mean_delta": float(selected.mean().item()),
        "selected_action_increase_rows": int((selected > 0).sum().item()),
        "selected_action_decrease_rows": int((selected < 0).sum().item()),
        "selected_action_unchanged_rows": int((selected == 0).sum().item()),
        "argmax_changed_rows": int(changes.sum().item()),
        "argmax_unchanged_rows": int((~changes).sum().item()),
        "l1_mean": float(l1.mean().item()),
        "l1_max": float(l1.max().item()),
        "kl_before_after_mean": float(kl.mean().item()),
        "kl_before_after_max": float(kl.max().item()),
    }


def run(d8_dir: Path, d12_dir: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "D18 forward must run on native CPU path")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "D18 requires one CPU thread")
    require(torch.__version__ == "2.5.1+cpu",
            "D18 requires pinned torch 2.5.1+cpu")
    source = d15.load_source(d12_dir)
    capture, native, default, comparison = d13.validate_archived(source)
    model, initial_sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(initial_sha == capture["initial_state_sha256"],
            "initial state SHA mismatch")
    batch = torch.load(d12_dir / "frozen-batch.pt", map_location="cpu",
                       weights_only=True)
    require(d12.batch_digest(batch) == capture["batch_sha256"],
            "frozen batch SHA mismatch")
    require(len(batch) == 8 and sum(ep["score"]["targets"] for ep in batch) == 128,
            "frozen episode/target count mismatch")
    for key, expected in (("hits", 0), ("extra_keydowns", 9),
                          ("physical_keydowns", 9)):
        require(sum(ep["score"][key] for ep in batch) == expected,
                "frozen physical record mismatch: " + key)
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    require(observations.shape == (N, 8) and actions.shape == (N,),
            "unexpected frozen batch row count")
    counts = tuple(int((actions == i).sum().item()) for i in range(4))
    require(counts == COUNTS, "D16 action counts differ")
    initial_bits = weights_bits(model)
    require(sum(len(v) for v in initial_bits.values()) == 1509,
            "initial model structure differs")
    native_model = restore_updated(model, native["weights_after"])
    default_model = restore_updated(model, default["weights_after"])
    # The archived post-update difference count is a source invariant.
    post_difference = d13.compare_parameters(
        native["weights_after"], default["weights_after"])
    require(post_difference["different_elements"] ==
            comparison["weights"]["different_elements"] == 10,
            "D12 post-update difference differs")
    pre = policy_probabilities(model, observations)
    pre_again = policy_probabilities(model, observations)
    require(torch.equal(pre, pre_again), "negative control before/before mismatch")
    post_native = policy_probabilities(native_model, observations)
    post_default = policy_probabilities(default_model, observations)
    require(initial_bits == weights_bits(model), "initial weights mutated")
    require(weights_bits(native_model) == native["weights_after"] and
            weights_bits(default_model) == default["weights_after"],
            "updated weights mutated")
    native_metrics = compare_probabilities(pre, post_native, actions)
    default_metrics = compare_probabilities(pre, post_default, actions)
    cross_metrics = compare_probabilities(post_native, post_default, actions)
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_frozen_policy_probability_comparison",
        "source_run": d13.SOURCE_RUN, "source_artifact": d13.SOURCE_ARTIFACT,
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(),
        "arm": "a0", "seed": 0, "episodes": 8,
        "targets": 128, "hits": 0, "extras": 9, "physical_keydowns": 9,
        "selected_action_counts": dict(zip(ACTION_NAMES, counts)),
        "archived_adam_steps_per_condition": 2,
        "archived_post_weight_difference_elements": 10,
        "negative_control_bit_exact": True,
        "before_to_after_native": native_metrics,
        "before_to_after_default": default_metrics,
        "after_native_to_after_default": cross_metrics,
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
        raise FileExistsError("refusing to overwrite D18 result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_frozen_policy_probability_comparison",
            "error": str(exc), "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_backward_calls": 0,
            "new_adam_steps": 0, "new_motor_steps": 0,
            "new_physical_keydowns": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("classification", "targets", "hits", "extras",
                               "physical_keydowns", "before_to_after_native",
                               "before_to_after_default", "after_native_to_after_default")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
