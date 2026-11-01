#!/usr/bin/env python3
"""Research A v2-L0: from-scratch, no-teacher low-speed A0/A1 PPO probe.

Precommitted: docs/research-a-v2-low-speed-l0-protocol.md.
This is exploratory; nothing here certifies formal study 002.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.distributions import Categorical

from a002_exploratory_ppo import (
    ActorCritic, discounted_returns, update_ppo,
)
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv

ARMS=("a0","a1")
SEEDS=(0,1)
TRAIN_INTERVALS_MS=(200,260)
EVAL_INTERVALS_MS=(210,240,250)
EVAL_STARTS_S=(.835,1.075,1.290)
TRAIN_START_MIN_S=.78
TRAIN_START_MAX_S=1.12
EPISODES=256
BATCH_EPISODES=8
CHECKPOINTS=(0,128,256)
REPETITIONS=4
PROTOCOL="docs/research-a-v2-low-speed-l0-protocol.md"
ACTION_NAMES=("none","right","left","both")


class LoggedLowSpeedEnv(NoPositiveNearMissEnv):
    """Evaluation trace only; identical physical state, reward and scoring."""

    def reset(self):
        self.down_trace=[]
        return super().reset()

    def _judge_keydown(self, event_time_s: float, key: str) -> float:
        previous_hits=self.hits
        previous_cursor=self.cursor
        reward=super()._judge_keydown(event_time_s,key)
        hit=self.hits>previous_hits
        index=previous_cursor if hit else min(
            range(len(self.targets)),key=lambda i:abs(self.targets[i]-event_time_s)
        )
        self.down_trace.append({
            "time_s":round(event_time_s,6),"key":key,
            "hit":hit,
            "closest_target_ordinal":index,
            "offset_ms":round(1000*(event_time_s-self.targets[index]),4),
            "immediate_reward":round(reward,8),
        })
        return reward


def eval_seed(case_index: int, repetition: int) -> int:
    if not 0<=case_index<9 or not 0<=repetition<REPETITIONS:
        raise ValueError("bad eval seed indices")
    return 320000+100*case_index+repetition


def evaluate_episode(*,arm:str,interval_ms:int,start_s:float,mode:str,
                     model:ActorCritic|None, seed:int|None=None) -> dict:
    if mode not in ("argmax","sampled","random"):
        raise ValueError("unknown policy evaluation mode")
    if mode!="random" and model is None:
        raise ValueError("trained model required")
    if mode in ("sampled","random") and seed is None:
        raise ValueError("probability evaluations require seed")
    env=LoggedLowSpeedEnv(arm=arm,interval_ms=interval_ms,start_s=start_s)
    observation=env.reset()
    rng=torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)
    controls=[0]*4
    entropy_total=0.0
    steps=0
    with torch.no_grad():
        while True:
            if mode=="random":
                action=int(torch.randint(4,(1,),generator=rng).item())
            else:
                logits,_=model(torch.tensor(observation,dtype=torch.float32))
                probs=torch.softmax(logits,dim=-1)
                entropy_total+=float(
                    -(probs*torch.log(probs.clamp_min(1e-12))).sum().item()
                )
                action=(int(probs.argmax().item()) if mode=="argmax" else
                        int(torch.multinomial(probs,1,generator=rng).item()))
            controls[action]+=1
            steps+=1
            observation,_reward,finished=env.step(action)
            if finished:
                break
    result=env.result()
    if len(env.down_trace)!=result.physical_keydowns:
        raise AssertionError("physical KeyDown trace count mismatch")
    if result.hits+result.extra_keydowns!=result.physical_keydowns:
        raise AssertionError("physical KeyDown accounting mismatch")
    return {
        "arm":arm,"interval_ms":interval_ms,"start_s":start_s,
        "mode":mode,"sampling_seed":seed,
        **asdict(result),
        "physical_left":sum(e["key"]=="left" for e in env.down_trace),
        "physical_right":sum(e["key"]=="right" for e in env.down_trace),
        "controls":controls,"steps":steps,
        "mean_action_entropy":(entropy_total/steps if mode!="random" else None),
        "physical_trace":env.down_trace,
        "matched_keys":"".join("L" if x=="left" else "R"
                               for x in result.matched_keys),
    }


def evaluate_set(*,arm:str,model:ActorCritic|None,mode:str)->dict:
    rows=[]
    case=0
    for interval in EVAL_INTERVALS_MS:
        for start in EVAL_STARTS_S:
            repetitions=(range(REPETITIONS) if mode in ("sampled","random") else (0,))
            for repetition in repetitions:
                rows.append({
                    "case_index":case,"repetition":repetition,
                    **evaluate_episode(
                        arm=arm,interval_ms=interval,start_s=start,
                        mode=mode,model=model,
                        seed=(eval_seed(case,repetition) if mode!="argmax" else None),
                    ),
                })
            case+=1
    total_targets=sum(x["targets"] for x in rows)
    total_hits=sum(x["hits"] for x in rows)
    total_extras=sum(x["extra_keydowns"] for x in rows)
    actions=[sum(row["controls"][i] for row in rows) for i in range(4)]
    steps=sum(row["steps"] for row in rows)
    return {
        "episodes":len(rows),"targets":total_targets,
        "hits":total_hits,"hit_rate":total_hits/total_targets,
        "misses":sum(x["misses"] for x in rows),
        "extra_keydowns":total_extras,
        "extra_per_target":total_extras/total_targets,
        "full_hit_episodes":sum(x["full_hit"] for x in rows),
        "physical_left":sum(x["physical_left"] for x in rows),
        "physical_right":sum(x["physical_right"] for x in rows),
        "episodes_with_both_keys":sum(
            x["physical_left"]>0 and x["physical_right"]>0 for x in rows
        ),
        "action_selection_fraction":dict(zip(ACTION_NAMES,[x/steps for x in actions])),
        "mean_entropy":(
            sum(x["mean_action_entropy"]*x["steps"] for x in rows)/steps
            if mode!="random" else None
        ),
        "rows":rows,
    }


def train_episode(model:ActorCritic,*,arm:str,rng:random.Random)->dict:
    period=rng.choice(TRAIN_INTERVALS_MS)
    start=rng.uniform(TRAIN_START_MIN_S,TRAIN_START_MAX_S)
    env=LoggedLowSpeedEnv(interval_ms=period,start_s=start,arm=arm)
    observation=env.reset()
    xs,actions,logprobs,values,rewards=[],[],[],[],[]
    model.eval()
    with torch.no_grad():
        while True:
            obs=torch.tensor(observation,dtype=torch.float32)
            logits,value=model(obs)
            dist=Categorical(logits=logits)
            action=dist.sample()
            observation,reward,finished=env.step(int(action.item()))
            xs.append(obs)
            actions.append(int(action.item()))
            logprobs.append(float(dist.log_prob(action).item()))
            values.append(float(value.item()))
            rewards.append(float(reward))
            if finished:
                break
    result=env.result()
    return {
        "observations":torch.stack(xs),
        "actions":torch.tensor(actions,dtype=torch.long),
        "old_log_probs":torch.tensor(logprobs,dtype=torch.float32),
        "old_values":torch.tensor(values,dtype=torch.float32),
        "returns":torch.tensor(discounted_returns(rewards),dtype=torch.float32),
        "score":asdict(result),
        "interval_ms":period,"start_s":start,
    }


def train_model(*,arm:str,seed:int,checkpoint_dir:Path)->dict:
    torch.manual_seed(seed)
    rng=random.Random(seed)
    model=ActorCritic()
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    initial_weights=hashlib.sha256(
        b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
    ).hexdigest()
    evaluations=[]
    learning=[]
    def snapshot(at:int)->None:
        evaluations.append({
            "episode":at,
            "argmax":evaluate_set(arm=arm,model=model,mode="argmax"),
            "sampled":evaluate_set(arm=arm,model=model,mode="sampled"),
        })
    snapshot(0)
    for first in range(0,EPISODES,BATCH_EPISODES):
        batch=[train_episode(model,arm=arm,rng=rng) for _ in range(BATCH_EPISODES)]
        losses=update_ppo(model,optimizer,batch)
        end=first+BATCH_EPISODES
        learning.append({
            "episode":end,
            "train_hit_rate":sum(x["score"]["hits"] for x in batch)/(len(batch)*16),
            "train_extra_keydowns":sum(x["score"]["extra_keydowns"] for x in batch),
            "mean_train_reward":sum(x["score"]["reward_sum"] for x in batch)/len(batch),
            "periods_ms":[x["interval_ms"] for x in batch],
            "starts_s":[round(x["start_s"],6) for x in batch],
            **losses,
        })
        if end in CHECKPOINTS:
            snapshot(end)
        if end % 32==0:
            print("LOW_SPEED_PROGRESS",json.dumps({
                "arm":arm,"seed":seed,"episode":end,
                "training_hit_rate":learning[-1]["train_hit_rate"],
                "entropy":losses["entropy"],
                "eval_hit_rate":(evaluations[-1]["sampled"]["hit_rate"]
                                  if end in CHECKPOINTS else None),
            }),flush=True)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path=checkpoint_dir/f"research-a-v2-l0-{arm}-seed{seed}-ep{EPISODES}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    torch.save({
        "format":"research-a-v2-l0-v1",
        "arm":arm,"seed":seed,"episodes":EPISODES,
        "policy":model.state_dict(),"optimizer":optimizer.state_dict(),
    },path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "arm":arm,"seed":seed,
        "initial_state_sha256":initial_weights,
        "checkpoint_file":path.name,"checkpoint_sha256":sha,
        "training":learning,"evaluations":evaluations,
    }


def run_study(checkpoint_dir:Path)->dict:
    torch.set_num_threads(1)
    models=[]
    for arm in ARMS:
        for seed in SEEDS:
            r=train_model(arm=arm,seed=seed,checkpoint_dir=checkpoint_dir)
            models.append(r)
            final=r["evaluations"][-1]
            print("LOW_SPEED_MODEL_COMPLETE",json.dumps({
                "arm":arm,"seed":seed,
                "argmax_hit":final["argmax"]["hit_rate"],
                "sampled_hit":final["sampled"]["hit_rate"],
                "sampled_extras":final["sampled"]["extra_keydowns"],
                "checkpoint_sha256":r["checkpoint_sha256"],
            }),flush=True)
    baselines=[{"arm":arm,"random_uniform":evaluate_set(
        arm=arm,model=None,mode="random"
    )} for arm in ARMS]
    return {
        "schema":"research-a-v2-l0-low-speed-teacher-free-v1",
        "status":"exploratory low-speed acquisition only; formal study 002 not approved",
        "protocol":PROTOCOL,
        "settings":{
            "arms":ARMS,"seeds":SEEDS,
            "train_intervals_ms":TRAIN_INTERVALS_MS,
            "train_start_range_s":[TRAIN_START_MIN_S,TRAIN_START_MAX_S],
            "eval_intervals_ms":EVAL_INTERVALS_MS,
            "eval_starts_s":EVAL_STARTS_S,
            "episodes":EPISODES,"batch_episodes":BATCH_EPISODES,
            "evaluation_checkpoints":CHECKPOINTS,
            "evaluation_repetitions":REPETITIONS,
            "sample_seed_formula":"320000 + 100*case_index + repetition",
            "body_modified":False,"teacher_used":False,
            "reward":"R1 no positive near-miss, no finger-specific reward",
            "ppo_backend":"original A002-P ActorCritic and update_ppo",
        },
        "models":models,"random_baselines":baselines,
    }


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report=run_study(args.checkpoints)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("LOW_SPEED_FINAL",json.dumps({
        "models":[{
            "arm":r["arm"],"seed":r["seed"],
            "progress":[{
                "episode":e["episode"],"greedy_hit":e["argmax"]["hit_rate"],
                "sample_hit":e["sampled"]["hit_rate"],
                "sample_extra":e["sampled"]["extra_keydowns"],
                "sample_physical_keys":{
                    "left":e["sampled"]["physical_left"],
                    "right":e["sampled"]["physical_right"],
                },
            } for e in r["evaluations"]],
            "checkpoint_sha256":r["checkpoint_sha256"],
        } for r in report["models"]],
        "A3":[{"arm":r["arm"],
                "hits":r["random_uniform"]["hits"],
                "targets":r["random_uniform"]["targets"],
                "extras":r["random_uniform"]["extra_keydowns"]}
              for r in report["random_baselines"]],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
