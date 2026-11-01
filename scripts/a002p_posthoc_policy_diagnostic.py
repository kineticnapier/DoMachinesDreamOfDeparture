#!/usr/bin/env python3
"""Post-hoc diagnostics on ORIGINAL A002-P checkpoints, without any training.

The original experiment was run in Actions 41296142433. This script loads its
archived (unchanged) checkpoints and evaluation split, verifies their hashes,
and compares its evaluation scoring to the original pilot JSON.

Only final checkpoints are available. Entropy/returns during training are read
from the original recorded blocks, never reconstructed as intermediate weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import torch

from a002_exploratory_ppo import ActorCritic, EVAL_INTERVALS_MS, EVAL_STARTS_S
from dmdod.research.a002_timing_only_env import TimingOnlyTwoFingerEnv

SOURCE_RUN = 41296142433
SOURCE_HEAD = "69a8e3b54e8d8c8fa28c1e46e522a897de9ac6e4"
SOURCE_ARTIFACT_ID = 13079276880
EXPECTED_MODEL_SHA256 = {
    ("a0", 0): "e34fa22737b6c045479aeb0c59e86d3fa84971443b43fe9d3371f0cc3ca2c70d",
    ("a0", 1): "737ee1d285ff37bef32ed4b9ceb20bcbc77c7865e33c74c8769b684b24900ce0",
    ("a1", 0): "c7fde81bd786df9affb0e25858ca7b2b1fa942e725cbe0556bd08beeb6e46c49",
    ("a1", 1): "e1b460f562c0184373130a801ad21f989cd03f506d2be127a30903a961910dc3",
}
ACTION_LABELS = ("none", "right", "left", "both")
BUCKET_MS = (5.0, 10.0, 20.0, 40.0, 80.0)


def checked_checkpoint(path: Path, arm: str, seed: int) -> ActorCritic:
    expected_sha = EXPECTED_MODEL_SHA256[(arm, seed)]
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"checkpoint SHA256 mismatch for {path.name}: {actual_sha}")
    # The archived checkpoint was produced by our own known training script.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["format"] != "research-a002p-pilot-v1" or data["arm"] != arm:
        raise ValueError("checkpoint arm/format is incompatible")
    if data["seed"] != seed or data["episodes"] != 96:
        raise ValueError("checkpoint seed/episode count is incompatible")
    network = ActorCritic()
    network.load_state_dict(data["policy"], strict=True)
    network.eval()
    return network


def bin_abs_ms(values: list[float]) -> dict[str, int]:
    names = ("0-5", "5-10", "10-20", "20-40", "40-80", "80+")
    counts = [0] * len(names)
    for value in values:
        x = abs(value)
        counts[next((i for i, bound in enumerate(BUCKET_MS) if x < bound), 5)] += 1
    return dict(zip(names, counts))


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": round(median(values), 4) if values else None,
        "mean_abs": round(mean(abs(x) for x in values), 4) if values else None,
        "abs_hist_ms": bin_abs_ms(values),
    }


def traced_episode(model: ActorCritic, *, arm: str, interval_ms: float,
                   start_s: float, force_single: bool = False) -> dict:
    physical_arm = "a0" if force_single else arm
    env = TimingOnlyTwoFingerEnv(
        interval_ms=interval_ms, start_s=start_s, arm=physical_arm
    )
    observation = env.reset()
    history: list[dict] = []
    original_judge = env._judge_keydown
    commands = Counter()
    probabilities = [0.0] * 4
    entropies = []
    remaining_clipped_high = 0
    remaining_clipped_low = 0
    total_steps = 0

    def trace_judge(time_s: float, key: str) -> float:
        pre_hits = env.hits
        pre_cursor = env.cursor
        due = env.targets[pre_cursor] if pre_cursor < len(env.targets) else None
        nearest_index = min(
            range(len(env.targets)),
            key=lambda i: abs(env.targets[i] - time_s),
        )
        nearest_error_ms = (time_s-env.targets[nearest_index])*1000
        score = original_judge(time_s, key)
        history.append({
            "time_s": round(time_s, 6), "key": key,
            "hit": env.hits > pre_hits,
            "cursor_before": pre_cursor,
            "cursor_after": env.cursor,
            "due_error_ms": round((time_s-due)*1000, 4) if due is not None else None,
            "nearest_error_ms": round(nearest_error_ms, 4),
            "nearest_target_ordinal": nearest_index,
            "reward": round(score, 6),
        })
        return score

    env._judge_keydown = trace_judge
    with torch.no_grad():
        while True:
            remaining_clipped_high += observation[0] >= 4.0
            remaining_clipped_low += observation[0] <= -2.0
            obs = torch.tensor(observation, dtype=torch.float32)
            logits, _ = model(obs)
            probs = logits.softmax(dim=-1)
            action = int(torch.argmax(probs).item())
            commands[action] += 1
            for i in range(4):
                probabilities[i] += float(probs[i].item())
            entropies.append(float(
                -(probs * probs.clamp_min(1e-12).log()).sum().item()
            ))
            total_steps += 1
            observation, _reward, done = env.step(action)
            if done:
                break
    score = env.result()
    hits = [x for x in history if x["hit"]]
    extras = [x for x in history if not x["hit"]]
    down_counts = dict(Counter(x["key"] for x in history))
    nearest_extra = [x["nearest_error_ms"] for x in extras]
    due_extra = [x["due_error_ms"] for x in extras if x["due_error_ms"] is not None]
    extra_positive = [x for x in extras if x["reward"] > 0]
    return {
        "interval_ms": interval_ms,
        "start_s": start_s,
        "arm": arm,
        "force_single": force_single,
        "hits": score.hits,
        "targets": score.targets,
        "extra_keydowns": score.extra_keydowns,
        "physical_keydowns": score.physical_keydowns,
        "matched_keys": "".join("L" if key == "left" else "R"
                               for key in score.matched_keys),
        "total_steps": total_steps,
        "selected_action_fraction": {
            label: commands[i] / total_steps for i, label in enumerate(ACTION_LABELS)
        },
        "average_action_probability": {
            label: probabilities[i] / total_steps for i, label in enumerate(ACTION_LABELS)
        },
        "mean_action_entropy": mean(entropies),
        "max_action_entropy": math.log(4.0),
        "remaining_clipped_fraction": {
            "high": remaining_clipped_high / total_steps,
            "low": remaining_clipped_low / total_steps,
        },
        "physical_down_by_key": down_counts,
        "extra_keydowns_with_positive_step_reward": len(extra_positive),
        "extra_keydowns_without_positive_step_reward": len(extras) - len(extra_positive),
        "sum_step_reward_for_extra_keydowns": round(sum(x["reward"] for x in extras), 6),
        "hit_error_ms": summarize([x["due_error_ms"] for x in hits]),
        "extra_nearest_error_ms": summarize(nearest_extra),
        "extra_due_error_ms": summarize(due_extra),
        "extras_near_an_existing_target_within_20ms": sum(
            abs(x) <= 20.0 + 1e-8 for x in nearest_extra
        ),
        "extras_near_an_existing_target_within_80ms": sum(
            abs(x) <= 80.0 + 1e-8 for x in nearest_extra
        ),
        "first_10_keydowns": history[:10],
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("episodes required")
    steps = sum(x["total_steps"] for x in rows)
    mean_probs = {
        label: sum(x["average_action_probability"][label] * x["total_steps"]
                   for x in rows) / steps
        for label in ACTION_LABELS
    }
    selected = {
        label: sum(x["selected_action_fraction"][label] * x["total_steps"]
                   for x in rows) / steps
        for label in ACTION_LABELS
    }
    return {
        "episodes": len(rows),
        "hits": sum(r["hits"] for r in rows),
        "targets": sum(r["targets"] for r in rows),
        "extra_keydowns": sum(r["extra_keydowns"] for r in rows),
        "physical_keydowns": sum(r["physical_keydowns"] for r in rows),
        "extra_keydowns_with_positive_step_reward": sum(r["extra_keydowns_with_positive_step_reward"] for r in rows),
        "extra_keydowns_without_positive_step_reward": sum(r["extra_keydowns_without_positive_step_reward"] for r in rows),
        "sum_step_reward_for_extra_keydowns": round(sum(r["sum_step_reward_for_extra_keydowns"] for r in rows), 6),
        "physical_down_by_key": dict(
            sum((Counter(r["physical_down_by_key"]) for r in rows), Counter())
        ),
        "action_selection_fraction": selected,
        "average_softmax_probability": mean_probs,
        "mean_actor_entropy_nats": sum(
            r["mean_action_entropy"]*r["total_steps"] for r in rows
        ) / steps,
        "entropy_max_nats": math.log(4.0),
        "extra_within_20ms_of_any_target": sum(
            r["extras_near_an_existing_target_within_20ms"] for r in rows
        ),
        "extra_within_80ms_of_any_target": sum(
            r["extras_near_an_existing_target_within_80ms"] for r in rows
        ),
    }


def diagnose(input_dir: Path) -> dict:
    original_path = input_dir / "pilot.json"
    raw = json.loads(original_path.read_text(encoding="utf-8"))
    if raw["schema"] != "research-a002p-pilot-ppo-v1" or raw["quick"]:
        raise ValueError("source pilot report must be full original experiment")
    if (tuple(raw["settings"]["eval_interval_ms"]) != EVAL_INTERVALS_MS
            or tuple(raw["settings"]["eval_starts_s"]) != EVAL_STARTS_S):
        raise ValueError("source split differs from original evaluation")
    results = []
    for result in raw["all_results"]:
        arm, seed = result["arm"], result["seed"]
        model_path = input_dir/"checkpoints"/result["checkpoint_file"]
        model = checked_checkpoint(model_path, arm, seed)
        rows = [
            traced_episode(model, arm=arm, interval_ms=period, start_s=start)
            for period in EVAL_INTERVALS_MS for start in EVAL_STARTS_S
        ]
        orig_rows = result["evaluations"][-1]["evaluation"]["rows"]
        for i, (current, original) in enumerate(zip(rows, orig_rows)):
            for metric in ("hits", "extra_keydowns", "physical_keydowns", "matched_keys"):
                if current[metric] != original[metric]:
                    raise ValueError(
                        f"original score mismatch {arm} seed={seed} row={i} {metric}: "
                        f"{current[metric]} != {original[metric]}"
                    )
            if (current["interval_ms"] != original["interval_ms"]
                    or current["start_s"] != original["start_s"]):
                raise ValueError("evaluation split mismatch")
        if len(rows) != len(orig_rows):
            raise ValueError("evaluation episode count mismatch")
        baseline = result["evaluations"][0]["evaluation"]
        half = result["evaluations"][1]["evaluation"]
        last = result["evaluations"][2]["evaluation"]
        results.append({
            "arm": arm,
            "seed": seed,
            "model_sha256_verified": EXPECTED_MODEL_SHA256[(arm, seed)],
            "training": [
                {
                    "episodes_completed": block["episodes_completed"],
                    "training_hit_rate": block["training_hit_rate"],
                    "training_extras": block["training_extras"],
                    "entropy_nats": block["entropy"],
                    "training_reward_mean": block["training_reward_mean"],
                }
                for block in result["training_blocks"]
            ],
            "evaluation_checkpoints": [
                {"episode": n, "hits": e["hits"], "targets": e["targets"],
                 "extra_keydowns": e["extra_keydowns"]}
                for n, e in ((0, baseline), (48, half), (96, last))
            ],
            "final_diagnostic": aggregate(rows),
            "episodes": rows,
        })
    return {
        "schema": "research-a002p-posthoc-no-retraining-v1",
        "status": "post-hoc on unchanged final models; formal 002 unapproved",
        "source_workflow_run_id": SOURCE_RUN,
        "source_head": SOURCE_HEAD,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "original_score_consistency_verified": True,
        "intermediate_weights_not_saved": True,
        "warning": (
            "Training entropy is from the original recorded PPO blocks. "
            "Policy probabilities and physical timing traces are from the "
            "saved final model only; no historical policy reconstruction."
        ),
        "all_arms_and_seeds": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite posthoc report")
    torch.set_num_threads(1)
    report = diagnose(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                           encoding="utf-8")
    print("A002P_POSTHOC_SUMMARY", json.dumps({
        "source_run": SOURCE_RUN,
        "consistency": report["original_score_consistency_verified"],
        "results": [
            {"arm": row["arm"], "seed": row["seed"],
             **row["final_diagnostic"],
             "entropy_first": row["training"][0]["entropy_nats"],
             "entropy_last": row["training"][-1]["entropy_nats"],
             "evaluation_progress": row["evaluation_checkpoints"]}
            for row in report["all_arms_and_seeds"]
        ],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
