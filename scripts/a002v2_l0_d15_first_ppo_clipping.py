#!/usr/bin/env python3
"""D15: determine whether original first PPO gradient norm clipping is active."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

import a002_exploratory_ppo as ppo
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13

PROTOCOL = "docs/research-a-v2-l0-d15-first-ppo-clipping-protocol.md"
SCHEMA = "research-a-v2-l0-d15-first-ppo-clipping-v1"
SOURCE_RUN = d13.SOURCE_RUN
SOURCE_ARTIFACT = d13.SOURCE_ARTIFACT
THRESHOLD = 1.0


def load_source(folder: Path):
    source = {name: json.loads((folder / (name + ".json")).read_text(
        encoding="utf-8")) for name in d13.REQUIRED}
    prior = d13.analyze(source)
    d13.require(prior["classification"] == "gradient_difference_before_first_adam_step",
                "D13 archived gradient stage differs")
    return source


def classify(norm: float, differences: int):
    if not math.isfinite(norm) or norm < 0:
        raise ValueError("invalid preclip norm")
    if norm > THRESHOLD:
        if differences <= 0:
            raise ValueError("active clipping did not change any gradient")
        return "clipping_active"
    if differences != 0:
        raise ValueError("inactive clipping changed gradients")
    return "clipping_inactive"


def original_first_backward(model, batch):
    """Reproduce the first epoch of the original PPO exactly, without Adam."""
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    old_log_probs = torch.cat([ep["old_log_probs"] for ep in batch])
    returns = torch.cat([ep["returns"] for ep in batch])
    old_values = torch.cat([ep["old_values"] for ep in batch])
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-6)
    model.train()
    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    ratio = (log_probs - old_log_probs).exp()
    objective = torch.min(
        ratio * advantages, ratio.clamp(1 - ppo.CLIP, 1 + ppo.CLIP) * advantages)
    policy_loss = -objective.mean()
    value_loss = (returns - values).pow(2).mean()
    entropy = distribution.entropy().mean()
    loss = policy_loss + ppo.VALUE_COEF * value_loss - ppo.ENTROPY_COEF * entropy
    model.zero_grad(set_to_none=True)
    loss.backward()
    before = {name: d12.tensor_bits(param.grad)
              for name, param in model.named_parameters()}
    norm = float(nn.utils.clip_grad_norm_(model.parameters(), THRESHOLD).item())
    after = {name: d12.tensor_bits(param.grad)
             for name, param in model.named_parameters()}
    return before, after, norm


def run_condition(condition, source, d8_dir, d12_dir):
    d12.require_dispatch(condition)
    torch.set_num_threads(1)
    c, native, default, _ = d13.validate_archived(source)
    row = {"native": native, "default": default}[condition]
    model, digest = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    d13.require(digest == c["initial_state_sha256"],
                "D12 initial weights SHA mismatch")
    batch = torch.load(d12_dir / "frozen-batch.pt",
                       map_location="cpu", weights_only=True)
    d13.require(d12.batch_digest(batch) == c["batch_sha256"],
                "D12 frozen PPO batch SHA mismatch")
    runtime = d12.runtime()
    for key in ("torch", "cpu_model", "cpu_capability", "torch_threads"):
        d13.require(runtime[key] == row["runtime"][key],
                    "D12 runtime mismatch: " + key)
    before, after, norm = original_first_backward(model, batch)
    archived = row["actor_and_critic_gradients"][0]
    d13.require(after == archived,
                "D12 first pre-Adam gradient bit-exact replay mismatch")
    delta = d13.compare_parameters(before, after)
    d13.require(delta["total_elements"] == 1509, "gradient shape changed")
    state = classify(norm, delta["different_elements"])
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "condition": condition, "classification": state,
        "initial_state_sha256": c["initial_state_sha256"],
        "batch_sha256": c["batch_sha256"], "runtime": runtime,
        "threshold": THRESHOLD, "preclip_norm": norm,
        "effective_clip_coefficient": min(1.0, THRESHOLD / (norm + 1e-6)),
        "preclip": before, "postclip": after,
        "pre_post_comparison": delta,
        "archived_first_gradient_exact_match": True,
        "targets": c["targets"], "hits": c["hits"],
        "extras": c["extras"], "physical_keydowns": c["physical_keydowns"],
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_steps": 0,
        "new_physical_keydowns": 0,
    }


def compare(source, native, default):
    c, n, d, prior = d13.validate_archived(source)
    d13.require(native["schema"] == default["schema"] == SCHEMA,
                "D15 schema mismatch")
    d13.require(native["condition"] == "native" and
                default["condition"] == "default", "D15 condition mismatch")
    for row, original in ((native, n), (default, d)):
        d13.require(row["initial_state_sha256"] == c["initial_state_sha256"]
                    and row["batch_sha256"] == c["batch_sha256"],
                    "D15 frozen identity mismatch")
        d13.require(row["runtime"]["cpu_capability"] ==
                    original["runtime"]["cpu_capability"],
                    "D15 CPU capability differs from archived D12")
        d13.require(row["archived_first_gradient_exact_match"] is True
                    and row["postclip"] == original["actor_and_critic_gradients"][0],
                    "D15 postclip replay differs from D12")
        d13.require(row["classification"] ==
                    classify(row["preclip_norm"],
                             d13.compare_parameters(
                                 row["preclip"], row["postclip"])["different_elements"]),
                    "D15 clipping classification mismatch")
    d13.require(native["runtime"]["cpu_capability"] !=
                default["runtime"]["cpu_capability"],
                "CPU capabilities not distinct")
    before = d13.compare_parameters(native["preclip"], default["preclip"])
    after = d13.compare_parameters(native["postclip"], default["postclip"])
    d13.require(after["different_elements"] ==
                prior["gradients_by_epoch"][0]["different_elements"] == 292
                and after["max_abs_difference"] ==
                prior["gradients_by_epoch"][0]["max_abs_difference"],
                "D13 first clipped-gradient comparison mismatch")
    statuses = (native["classification"], default["classification"])
    classification = (
        "clipping_active_in_both" if statuses == ("clipping_active",) * 2
        else "clipping_inactive_in_both" if statuses == ("clipping_inactive",) * 2
        else "clipping_mixed")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL, "classification": classification,
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "initial_state_sha256": c["initial_state_sha256"],
        "batch_sha256": c["batch_sha256"],
        "native_preclip_norm": native["preclip_norm"],
        "default_preclip_norm": default["preclip_norm"],
        "native_clip_coefficient": native["effective_clip_coefficient"],
        "default_clip_coefficient": default["effective_clip_coefficient"],
        "native_pre_post_comparison": native["pre_post_comparison"],
        "default_pre_post_comparison": default["pre_post_comparison"],
        "cpu_preclip_comparison": before, "cpu_postclip_comparison": after,
        "archived_first_gradient_exact_match": True,
        "targets": c["targets"], "hits": c["hits"],
        "extras": c["extras"], "physical_keydowns": c["physical_keydowns"],
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
    parser.add_argument("--mode", choices=("condition", "compare"), required=True)
    parser.add_argument("--condition", choices=("native", "default"))
    parser.add_argument("--d8-dir", type=Path)
    parser.add_argument("--d12-dir", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D15 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = load_source(args.d12_dir)
        if args.mode == "condition":
            if args.condition is None or args.d8_dir is None:
                raise ValueError("missing condition or D8 input")
            result = run_condition(args.condition, source, args.d8_dir, args.d12_dir)
        else:
            if args.native is None or args.default is None:
                raise ValueError("missing D15 condition outputs")
            result = compare(
                source, json.loads(args.native.read_text(encoding="utf-8")),
                json.loads(args.default.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError,
            RuntimeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_replay", "error": str(exc),
            "new_training_episodes": 0, "new_ppo_updates": 0,
            "new_adam_steps": 0, "new_motor_steps": 0,
            "new_physical_keydowns": 0,
        }
        args.out.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("classification", "condition", "preclip_norm",
                               "native_preclip_norm", "default_preclip_norm",
                               "cpu_preclip_comparison", "cpu_postclip_comparison",
                               "pre_post_comparison", "hits", "extras",
                               "physical_keydowns")}), flush=True)


if __name__ == "__main__":
    main()
