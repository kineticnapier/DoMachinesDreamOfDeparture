#!/usr/bin/env python3
"""Research A002-P: two-arm teacher-free PPO pilot with physical Motor.

Protocol: docs/research-a-experiment-002p-protocol.md
The study is EXPLORATORY and does not establish emergent fingering, even if a
single seed improves. Full results include every arm, seed, and failure.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:
    raise SystemExit("This pilot requires the CPU PyTorch package") from exc

from dmdod.research.a002_timing_only_env import (
    OBS_DIM, TimingOnlyTwoFingerEnv,
)

TRAIN_INTERVALS_MS = (100, 110, 120, 130)
EVAL_INTERVALS_MS = (105, 115, 125)
EVAL_STARTS_S = (.83, .97, 1.21)
ARMS = ("a0", "a1")
SEEDS = (0, 1)
EPISODES = 96
BATCH_EPISODES = 8
PPO_EPOCHS = 2
LR = .0005
GAMMA = .99
CLIP = .2
ENTROPY_COEF = .01
VALUE_COEF = .5
START_MIN_S = .78
START_MAX_S = 1.12


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
        )
        self.actor = nn.Linear(32, 4)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features).squeeze(-1)


def discounted_returns(rewards: list[float]) -> list[float]:
    running = 0.0
    result = [0.0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + GAMMA * running
        result[i] = running
    return result


def collect_episode(model: ActorCritic, *, arm: str, rng: random.Random,
                    env_cls=TimingOnlyTwoFingerEnv) -> dict:
    interval = rng.choice(TRAIN_INTERVALS_MS)
    start = rng.uniform(START_MIN_S, START_MAX_S)
    env = env_cls(interval_ms=interval, start_s=start, arm=arm)
    observation = env.reset()
    xs, actions, old_logps, old_values, rewards = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        while True:
            obs = torch.tensor(observation, dtype=torch.float32)
            logits, value = model(obs)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            observation, reward, done = env.step(int(action.item()))
            xs.append(obs)
            actions.append(int(action.item()))
            old_logps.append(float(distribution.log_prob(action).item()))
            old_values.append(float(value.item()))
            rewards.append(float(reward))
            if done:
                break
    score = env.result()
    return {
        "observations": torch.stack(xs),
        "actions": torch.tensor(actions, dtype=torch.long),
        "old_log_probs": torch.tensor(old_logps, dtype=torch.float32),
        "old_values": torch.tensor(old_values, dtype=torch.float32),
        "returns": torch.tensor(discounted_returns(rewards), dtype=torch.float32),
        "score": asdict(score),
        "interval_ms": interval, "start_s": start,
    }


def update_ppo(model: ActorCritic, optimizer, batch: list[dict]) -> dict:
    observations = torch.cat([ep["observations"] for ep in batch])
    actions = torch.cat([ep["actions"] for ep in batch])
    old_log_probs = torch.cat([ep["old_log_probs"] for ep in batch])
    returns = torch.cat([ep["returns"] for ep in batch])
    old_values = torch.cat([ep["old_values"] for ep in batch])
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    history = []
    model.train()
    for _ in range(PPO_EPOCHS):
        logits, values = model(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        ratio = (log_probs - old_log_probs).exp()
        objective = torch.min(
            ratio * advantages, ratio.clamp(1-CLIP, 1+CLIP) * advantages
        )
        policy_loss = -objective.mean()
        value_loss = (returns - values).pow(2).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({
            "loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()), "entropy": float(entropy.item())
        })
    return history[-1]


def evaluate(model: ActorCritic, *, arm: str, ablate_right: bool = False,
             env_cls=TimingOnlyTwoFingerEnv) -> dict:
    rows = []
    model.eval()
    with torch.no_grad():
        for interval in EVAL_INTERVALS_MS:
            for start in EVAL_STARTS_S:
                # For A1 ablation, evaluate the unchanged A1 policy on A0 body.
                env_arm = "a0" if ablate_right else arm
                env = env_cls(
                    arm=env_arm, interval_ms=interval, start_s=start
                )
                observation = env.reset()
                while True:
                    logits, _ = model(torch.tensor(observation, dtype=torch.float32))
                    action = int(logits.argmax().item())
                    observation, _, done = env.step(action)
                    if done:
                        break
                result = env.result()
                rows.append({
                    "interval_ms": interval, "start_s": start,
                    "hits": result.hits, "targets": result.targets,
                    "extra_keydowns": result.extra_keydowns,
                    "physical_keydowns": result.physical_keydowns,
                    "switch_fraction": result.switch_fraction,
                    "matched_keys": "".join("L" if k == "left" else "R" for k in result.matched_keys),
                    "full_hit": result.full_hit,
                })
    hits = sum(r["hits"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    total_switched = 0
    total_transitions = 0
    for row in rows:
        sequence = row["matched_keys"]
        total_transitions += max(0, len(sequence)-1)
        total_switched += sum(x!=y for x,y in zip(sequence, sequence[1:]))
    return {
        "episodes": len(rows), "hits": hits, "targets": targets,
        "hit_rate": hits / targets,
        "extra_keydowns": sum(row["extra_keydowns"] for row in rows),
        "full_hit_episodes": sum(row["full_hit"] for row in rows),
        "switch_fraction": (
            total_switched / total_transitions if total_transitions else None
        ),
        "rows": rows,
    }


def run_seed(*, arm: str, seed: int, episodes: int, checkpoint_dir: Path,
             env_cls=TimingOnlyTwoFingerEnv) -> dict:
    # Same initialization and environment-sequence seed for A0 and A1;
    # no weight transfer or teacher/finger-based initialization.
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    progress = [{"episode": 0, "evaluation": evaluate(model, arm=arm, env_cls=env_cls)}]
    blocks = []

    for first in range(0, episodes, BATCH_EPISODES):
        batch = [collect_episode(model, arm=arm, rng=rng, env_cls=env_cls)
                 for _ in range(min(BATCH_EPISODES, episodes-first))]
        losses = update_ppo(model, optimizer, batch)
        end = min(first+BATCH_EPISODES, episodes)
        blocks.append({
            "episodes_completed": end,
            "training_hit_rate": sum(ep["score"]["hits"] for ep in batch)
                                 / sum(ep["score"]["targets"] for ep in batch),
            "training_extras": sum(ep["score"]["extra_keydowns"] for ep in batch),
            "training_reward_mean": sum(ep["score"]["reward_sum"] for ep in batch)/len(batch),
            "train_interval_ms": [ep["interval_ms"] for ep in batch],
            "train_starts_s": [round(ep["start_s"], 6) for ep in batch],
            **losses,
        })
        if end == episodes // 2 or end == episodes:
            progress.append({"episode": end, "evaluation": evaluate(model, arm=arm, env_cls=env_cls)})
        print(
            f"{arm} seed={seed} episodes={end}/{episodes} "
            f"trainHit={blocks[-1]['training_hit_rate']:.3f} "
            f"entropy={losses['entropy']:.3f}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"a002p-{arm}-seed{seed}-ep{episodes}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint}")
    torch.save({
        "format": "research-a002p-pilot-v1", "arm": arm, "seed": seed,
        "episodes": episodes, "policy": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_class": "ActorCritic",
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "arm": arm, "seed": seed, "episodes": episodes,
        "checkpoint_sha256": digest, "checkpoint_file": checkpoint.name,
        "evaluations": progress, "training_blocks": blocks,
    }
    if arm == "a1":
        result["right_disabled_intervention"] = evaluate(model, arm="a1", ablate_right=True, env_cls=env_cls)
    return result


def run_study(*, quick: bool, checkpoint_dir: Path) -> dict:
    torch.set_num_threads(1)
    episodes = 8 if quick else EPISODES
    seeds = (0,) if quick else SEEDS
    results = [run_seed(arm=arm, seed=seed, episodes=episodes,
                        checkpoint_dir=checkpoint_dir)
               for arm in ARMS for seed in seeds]
    return {
        "schema": "research-a002p-pilot-ppo-v1",
        "status": "exploratory only; full study 002 preregistration NOT approved",
        "quick": quick,
        "source_protocol": "docs/research-a-experiment-002p-protocol.md",
        "settings": {
            "train_interval_ms": TRAIN_INTERVALS_MS,
            "train_start_range_s": [START_MIN_S, START_MAX_S],
            "eval_interval_ms": EVAL_INTERVALS_MS,
            "eval_starts_s": EVAL_STARTS_S, "targets_per_episode": 16,
            "seeds": seeds, "arms": ARMS, "episodes_per_seed": episodes,
            "batch_episodes": BATCH_EPISODES, "ppo_epochs": PPO_EPOCHS,
            "lr": LR, "gamma": GAMMA, "clip": CLIP,
            "entropy_coef": ENTROPY_COEF, "value_coef": VALUE_COEF,
            "device": "cpu",
        },
        "no_fingering_teacher": True,
        "no_absolute_time_observation": True,
        "relative_target_time_cue_permitted": True,
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    result = run_study(quick=args.quick, checkpoint_dir=args.checkpoints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("FINAL_RESULT_SUMMARY", json.dumps({
        "quick": args.quick,
        "results": [
            {
                "arm": r["arm"], "seed": r["seed"],
                "initial_hit_rate": r["evaluations"][0]["evaluation"]["hit_rate"],
                "final_hit_rate": r["evaluations"][-1]["evaluation"]["hit_rate"],
                "final_switch_fraction": r["evaluations"][-1]["evaluation"]["switch_fraction"],
                "final_extras": r["evaluations"][-1]["evaluation"]["extra_keydowns"],
                "final_full_episodes": r["evaluations"][-1]["evaluation"]["full_hit_episodes"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "ablation_hit_rate": (
                    r["right_disabled_intervention"]["hit_rate"]
                    if "right_disabled_intervention" in r else None
                ),
            }
            for r in result["all_results"]
        ],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
