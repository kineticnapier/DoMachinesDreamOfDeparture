#!/usr/bin/env python3
"""V2 preflight: archived R0/R1 checkpoints vs random on mixed-tempo targets.

NO TRAINING and NO formal success claims. See frozen preflight protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from a002p_s1_sampled_evaluation import load_checkpoint
from dmdod.research.a002v2_mixed_tempo_env import MixedTempoMotorEnv

SOURCE_RUN=41300077879
SOURCE_ARTIFACT=13080738533
PROTOCOL="docs/research-a-v2-mixed-tempo-preflight-protocol.md"
SCHEDULES=(
    (220,)*8+(120,)*8+(220,)*8,
    (210,)*8+(115,)*8+(230,)*8,
    (250,)*8+(125,)*8+(200,)*8,
)
STARTS=(.837,1.071,1.293)
REPETITIONS=4
REWARDS=("r0_original","r1_no_positive_near_miss")
ARMS=("a0","a1")
SEEDS=(0,1)


def fixed_seed(case: int, repetition: int) -> int:
    if not (0 <= case < 9 and 0 <= repetition < REPETITIONS):
        raise ValueError("invalid case/repetition")
    return 250000+100*case+repetition


def rollout(*, schedule: tuple[int,...], start_s: float, arm: str,
            reward: str, mode: str, model=None, disabled_side=None,
            generator_seed=None) -> dict:
    if mode not in ("argmax","sampled","random"):
        raise ValueError("unknown action mode")
    if mode != "random" and model is None:
        raise ValueError("trained model required for policy evaluation")
    if mode in ("sampled","random") and generator_seed is None:
        raise ValueError("sampling requires seed")
    env=MixedTempoMotorEnv(
        intervals_ms=schedule,start_s=start_s,arm=arm,
        disabled_side=disabled_side,
        near_reward="r0" if reward=="r0_original" else "r1",
    )
    observation=env.reset()
    rng=torch.Generator(device="cpu")
    if generator_seed is not None:
        rng.manual_seed(generator_seed)
    counts=[0,0,0,0]
    mean_entropy=0.0
    steps=0
    with torch.no_grad():
        while True:
            if mode=="random":
                action=int(torch.randint(4,(1,),generator=rng).item())
            else:
                logits,_=model(torch.tensor(observation,dtype=torch.float32))
                probs=torch.softmax(logits,dim=-1)
                mean_entropy+=float(-(probs*probs.clamp_min(1e-12).log()).sum().item())
                action=(int(probs.argmax().item()) if mode=="argmax" else
                        int(torch.multinomial(probs,1,generator=rng).item()))
            counts[action]+=1
            steps+=1
            observation,_,done=env.step(action)
            if done:
                break
    score=env.segmented_result()
    score.update({
        "start_s": start_s,
        "schedule_ms": list(schedule),
        "mode":mode,
        "sampling_seed":generator_seed,
        "control_action_counts":counts,
        "mean_policy_entropy_nats":mean_entropy/steps if mode!="random" else None,
    })
    return score


def summary(rows: list[dict]) -> dict:
    total_hits=sum(r["hits"] for r in rows)
    total_targets=sum(r["targets"] for r in rows)
    total_extras=sum(r["extra_keydowns"] for r in rows)
    combined=[]
    for j in range(3):
        data=[r["segments"][j] for r in rows]
        combined.append({
            "region":j,
            "hits":sum(x["hits"] for x in data),
            "targets":sum(x["targets"] for x in data),
            "misses":sum(x["misses"] for x in data),
            "extras_nearest":sum(x["extra_keydowns_nearest"] for x in data),
        })
    both=sum(r["left_physical_keydowns"]>0 and r["right_physical_keydowns"]>0 for r in rows)
    return {
        "episodes":len(rows),
        "hits":total_hits,
        "targets":total_targets,
        "hit_rate":total_hits/total_targets,
        "extra_keydowns":total_extras,
        "extras_per_episode":total_extras/len(rows),
        "left_physical_keydowns":sum(r["left_physical_keydowns"] for r in rows),
        "right_physical_keydowns":sum(r["right_physical_keydowns"] for r in rows),
        "episodes_using_both_physical_keys":both,
        "full_hit_episodes":sum(r["full_hit"] for r in rows),
        "by_segment":combined,
    }


def all_cases(*, arm: str, reward: str, mode: str, model=None,
              disabled_side=None) -> list[dict]:
    rows=[]
    case=0
    for schedule in SCHEDULES:
        for start in STARTS:
            reps=range(REPETITIONS) if mode in ("sampled","random") else (0,)
            for rep in reps:
                row=rollout(
                    schedule=schedule,start_s=start,arm=arm,reward=reward,
                    mode=mode,model=model,disabled_side=disabled_side,
                    generator_seed=(fixed_seed(case,rep)
                                    if mode in ("sampled","random") else None),
                )
                row.update({"case_index":case,"repetition":rep})
                rows.append(row)
            case+=1
    return rows


def evaluate_archive(input_dir: Path) -> dict:
    original=json.loads((input_dir/"paired-results.json").read_text(encoding="utf-8"))
    if (original["schema"]!="research-a002p-r1-paired-reward-ablation-v1"
            or not original["control_r0_matches_original_all_evaluations"]):
        raise ValueError("original eight-policy R0/R1 artifact not compatible")
    old=original["all_results"]
    expected={(r,arm,seed) for r in REWARDS for arm in ARMS for seed in SEEDS}
    actual={(x["reward_condition"],x["arm"],x["seed"]) for x in old}
    if len(old)!=8 or actual!=expected:
        raise ValueError("eight frozen-policy results missing or repeated")
    all_results=[]
    for entry in old:
        reward,arm,seed=(entry["reward_condition"],entry["arm"],entry["seed"])
        checkpoint=entry["training_record"]
        file=input_dir/"checkpoints"/reward/checkpoint["checkpoint_file"]
        model=load_checkpoint(file,checkpoint["checkpoint_sha256"],arm,seed)
        deterministic=all_cases(
            arm=arm,reward=reward,mode="argmax",model=model)
        sampled=all_cases(
            arm=arm,reward=reward,mode="sampled",model=model)
        interventions={}
        if arm=="a1":
            for disabled in ("left","right"):
                interventions[disabled]={
                    "summary":summary(all_cases(
                        arm=arm,reward=reward,mode="argmax",
                        model=model,disabled_side=disabled
                    )),
                }
        all_results.append({
            "reward":reward,"arm":arm,"seed":seed,
            "checkpoint_sha256_verified":checkpoint["checkpoint_sha256"],
            "argmax":summary(deterministic),
            "sampled":summary(sampled),
            "interventions_argmax":interventions,
            "all_argmax_cases":deterministic,
            "all_sampled_cases":sampled,
        })
        print("V2_EVALUATED_MODEL",json.dumps({
            "reward":reward,"arm":arm,"seed":seed,
            "argmax_hit_rate":all_results[-1]["argmax"]["hit_rate"],
            "sampled_hit_rate":all_results[-1]["sampled"]["hit_rate"],
            "sampled_physically_both":all_results[-1]["sampled"]["episodes_using_both_physical_keys"],
            "argmax_interventions":interventions,
        },ensure_ascii=False),flush=True)
    random_results=[]
    for arm in ARMS:
        random_rows=all_cases(arm=arm,reward="r0_original",mode="random")
        random_results.append({
            "arm":arm,"summary":summary(random_rows),"cases":random_rows,
        })
    return {
        "schema":"research-a-v2-mixed-tempo-preflight-v0",
        "status":"exploratory evaluation only; old models frozen, formal study 002 not begun",
        "protocol":PROTOCOL,
        "source_run":SOURCE_RUN,"source_artifact":SOURCE_ARTIFACT,
        "schedules":SCHEDULES,"start_times":STARTS,"sample_repetitions":REPETITIONS,
        "models":all_results,
        "untrained_uniform_random_a3":random_results,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite evaluation report")
    torch.set_num_threads(1)
    report=evaluate_archive(args.input)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V2_PREFLIGHT_FINAL",json.dumps({
        "models":[{
            "reward":m["reward"],"arm":m["arm"],"seed":m["seed"],
            "argmax":m["argmax"],"sampled":m["sampled"],
            "interventions_argmax":m["interventions_argmax"],
        } for m in report["models"]],
        "random":[{"arm":r["arm"],"summary":r["summary"]}
                  for r in report["untrained_uniform_random_a3"]],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
