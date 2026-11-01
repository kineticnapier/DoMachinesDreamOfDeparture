#!/usr/bin/env python3
"""Research A002-P-S1: argmax vs action-sampling evaluation of saved R0/R1 policies.

No training or model selection. Protocol is committed before executing this file.
Original R0/R1 paired model artifact: Actions 41300077879 / artifact 13080738533.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch

from a002_exploratory_ppo import ActorCritic, EVAL_INTERVALS_MS, EVAL_STARTS_S
from dmdod.research.a002_timing_only_env import TimingOnlyTwoFingerEnv
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv

SOURCE_RUN = 41300077879
SOURCE_ARTIFACT = 13080738533
PROTOCOL = "docs/research-a-experiment-002p-s1-sampled-eval-protocol.md"
REPETITIONS = 16
ACTION_LABELS = ("none", "right", "left", "both")
ENV_FOR_REWARD = {
    "r0_original": TimingOnlyTwoFingerEnv,
    "r1_no_positive_near_miss": NoPositiveNearMissEnv,
}


def fixed_sampling_seed(eval_index: int, repetition: int) -> int:
    if not (0 <= eval_index < 9 and 0 <= repetition < REPETITIONS):
        raise ValueError("invalid evaluation index or repetition")
    return 100000 + 100*eval_index + repetition


def load_checkpoint(path: Path, sha256: str, arm: str, seed: int) -> ActorCritic:
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"original checkpoint hash mismatch: {path}")
    stored = torch.load(path, map_location="cpu", weights_only=True)
    if (stored["format"] != "research-a002p-pilot-v1"
            or stored["arm"] != arm or stored["seed"] != seed
            or stored["episodes"] != 96):
        raise ValueError("saved model metadata is inconsistent with source report")
    model = ActorCritic()
    model.load_state_dict(stored["policy"], strict=True)
    model.eval()
    return model


def run_episode(model: ActorCritic, *, env_cls: type, arm: str,
                interval_ms: int, start_s: float,
                action_mode: str, generator_seed: int | None = None) -> dict:
    if action_mode not in ("argmax", "sampled"):
        raise ValueError("unrecognized action mode")
    if action_mode == "sampled" and generator_seed is None:
        raise ValueError("stochastic mode requires a fixed generator seed")

    env = env_cls(arm=arm, interval_ms=interval_ms, start_s=start_s)
    observation = env.reset()
    generator = torch.Generator(device="cpu")
    if generator_seed is not None:
        generator.manual_seed(generator_seed)

    physical_keys: list[str] = []
    judge_original = env._judge_keydown

    # Instrumentation only: preserve the reward and its state transitions.
    def logged_keydown(event_time_s: float, key: str) -> float:
        physical_keys.append(key)
        return judge_original(event_time_s, key)

    env._judge_keydown = logged_keydown
    action_counts = [0] * 4
    probabilities = [0.0] * 4
    entropy_sum = 0.0
    steps = 0
    with torch.no_grad():
        while True:
            x = torch.tensor(observation, dtype=torch.float32)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)
            if action_mode == "argmax":
                action = int(torch.argmax(probs).item())
            else:
                action = int(torch.multinomial(probs, 1, generator=generator).item())
            action_counts[action] += 1
            for k in range(4):
                probabilities[k] += float(probs[k].item())
            entropy_sum += float(
                -(probs*torch.log(probs.clamp_min(1e-12))).sum().item()
            )
            steps += 1
            observation, _reward, done = env.step(action)
            if done:
                break

    score = env.result()
    if len(physical_keys) != score.physical_keydowns:
        raise AssertionError("physical KeyDown hook missed events")
    matched = ["L" if k == "left" else "R" for k in score.matched_keys]
    actual = ["L" if k == "left" else "R" for k in physical_keys]
    return {
        "interval_ms": interval_ms,
        "start_s": start_s,
        "action_mode": action_mode,
        "generator_seed": generator_seed if action_mode == "sampled" else None,
        "hits": score.hits,
        "targets": score.targets,
        "extra_keydowns": score.extra_keydowns,
        "physical_keydowns": score.physical_keydowns,
        "physical_left": actual.count("L"),
        "physical_right": actual.count("R"),
        "physical_both_used": "L" in actual and "R" in actual,
        "physical_switch_count": sum(a != b for a,b in zip(actual,actual[1:])),
        "physical_transition_count": max(0, len(actual)-1),
        "hit_switch_count": sum(a != b for a,b in zip(matched,matched[1:])),
        "hit_transition_count": max(0, len(matched)-1),
        "matched_keys": "".join(matched),
        "full_hit": score.full_hit,
        "steps": steps,
        "action_counts": action_counts,
        "probability_sums": probabilities,
        "entropy_sum": entropy_sum,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty evaluation group")
    total_steps = sum(r["steps"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    physical_transitions = sum(r["physical_transition_count"] for r in rows)
    hit_transitions = sum(r["hit_transition_count"] for r in rows)
    counts = [sum(r["action_counts"][i] for r in rows) for i in range(4)]
    softmax = [sum(r["probability_sums"][i] for r in rows)/total_steps for i in range(4)]
    return {
        "episodes": len(rows),
        "hits": hits, "targets": targets,
        "hit_rate": hits/targets,
        "extra_keydowns": sum(r["extra_keydowns"] for r in rows),
        "extra_keydowns_per_episode": sum(r["extra_keydowns"] for r in rows)/len(rows),
        "physical_keydowns": sum(r["physical_keydowns"] for r in rows),
        "physical_left": sum(r["physical_left"] for r in rows),
        "physical_right": sum(r["physical_right"] for r in rows),
        "episodes_with_both_physical_keys": sum(r["physical_both_used"] for r in rows),
        "physical_switch_fraction": (
            sum(r["physical_switch_count"] for r in rows)/physical_transitions
            if physical_transitions else None
        ),
        "matched_hit_switch_fraction": (
            sum(r["hit_switch_count"] for r in rows)/hit_transitions
            if hit_transitions else None
        ),
        "matched_hit_transitions": hit_transitions,
        "full_hit_episodes": sum(r["full_hit"] for r in rows),
        "action_selection_fraction": dict(zip(
            ACTION_LABELS, (v/total_steps for v in counts)
        )),
        "mean_action_probabilities": dict(zip(ACTION_LABELS, softmax)),
        "mean_entropy_nats": sum(r["entropy_sum"] for r in rows)/total_steps,
        "max_entropy_nats": math.log(4),
    }


def compare_original(row: dict, source: dict, index: int) -> None:
    for key in ("interval_ms", "start_s", "hits", "targets",
                "extra_keydowns", "physical_keydowns", "matched_keys", "full_hit"):
        if row[key] != source[key]:
            raise AssertionError(
                f"argmax original mismatch evaluation={index} field={key} "
                f"current={row[key]!r} source={source[key]!r}"
            )


def evaluate_saved_artifact(root: Path) -> dict:
    report = json.loads((root/"paired-results.json").read_text(encoding="utf-8"))
    if report["schema"] != "research-a002p-r1-paired-reward-ablation-v1":
        raise ValueError("unexpected reward ablation artifact schema")
    if not report["control_r0_matches_original_all_evaluations"]:
        raise ValueError("original reward control was not reproduced")
    source_rows = report["all_results"]
    expected = {(reward, arm, seed)
                for reward in ENV_FOR_REWARD
                for arm in ("a0","a1")
                for seed in (0,1)}
    actual = {(r["reward_condition"], r["arm"], r["seed"]) for r in source_rows}
    if actual != expected or len(source_rows) != 8:
        raise ValueError("artifact does not contain all eight independent models")

    results = []
    for entry in source_rows:
        reward = entry["reward_condition"]
        arm = entry["arm"]
        seed = entry["seed"]
        stored = entry["training_record"]
        checkpoint = (root/"checkpoints"/reward/stored["checkpoint_file"])
        model = load_checkpoint(checkpoint, stored["checkpoint_sha256"], arm, seed)
        env_cls = ENV_FOR_REWARD[reward]
        orig_evaluations = stored["evaluations"][-1]["evaluation"]["rows"]
        if len(orig_evaluations) != 9:
            raise ValueError("expected 9 original evaluation episodes")

        argmax_rows = []
        sampled_rows = []
        eval_index = 0
        for interval_ms in EVAL_INTERVALS_MS:
            for start_s in EVAL_STARTS_S:
                deterministic = run_episode(
                    model, env_cls=env_cls, arm=arm,
                    interval_ms=interval_ms, start_s=start_s, action_mode="argmax"
                )
                compare_original(deterministic, orig_evaluations[eval_index], eval_index)
                argmax_rows.append(deterministic)
                for repetition in range(REPETITIONS):
                    sampled = run_episode(
                        model, env_cls=env_cls, arm=arm,
                        interval_ms=interval_ms, start_s=start_s,
                        action_mode="sampled",
                        generator_seed=fixed_sampling_seed(eval_index, repetition),
                    )
                    sampled_rows.append({
                        "eval_index": eval_index, "repetition": repetition, **sampled
                    })
                eval_index += 1

        results.append({
            "reward_condition": reward,
            "arm": arm, "training_seed": seed,
            "original_checkpoint_sha256": stored["checkpoint_sha256"],
            "argmax": aggregate(argmax_rows),
            "sampled": aggregate(sampled_rows),
            "argmax_episodes": argmax_rows,
            "sampled_episodes": sampled_rows,
        })
        print("S1_EVALUATED_MODEL", json.dumps({
            "reward": reward, "arm": arm, "seed": seed,
            "greedy_hit_rate": results[-1]["argmax"]["hit_rate"],
            "sampled_hit_rate": results[-1]["sampled"]["hit_rate"],
            "greedy_physical_both": results[-1]["argmax"]["episodes_with_both_physical_keys"],
            "sampled_physical_both": results[-1]["sampled"]["episodes_with_both_physical_keys"],
            "sampled_physical_keys": {
                "left": results[-1]["sampled"]["physical_left"],
                "right": results[-1]["sampled"]["physical_right"],
            },
        }, ensure_ascii=False), flush=True)

    return {
        "schema": "research-a002p-s1-saved-model-action-eval-v1",
        "status": "post-hoc evaluation of original frozen models, no retraining",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "protocol": PROTOCOL, "repetitions_per_condition": REPETITIONS,
        "sampling_seeds_rule": "100000 + 100*eval_index + repetition",
        "argmax_original_scores_verified": True,
        "all_results": results,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    data=evaluate_saved_artifact(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("S1_FINAL_SUMMARY",json.dumps({
        "argmax_verified":data["argmax_original_scores_verified"],
        "rows":[{
            "reward":r["reward_condition"],"arm":r["arm"],
            "seed":r["training_seed"],
            "argmax":r["argmax"],"sampled":r["sampled"],
        } for r in data["all_results"]],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
