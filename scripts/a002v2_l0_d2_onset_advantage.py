#!/usr/bin/env python3
"""D2: no-training, frozen R1/R2 policy-onset PPO advantage proxy audit.

Not historical training gradients and not causal action attribution.
Replays all held-out sampled evaluations against their archived exact rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median

import torch

from a002_exploratory_ppo import ActorCritic, discounted_returns
from a002v2_l0_low_speed_ppo import (
    EVAL_INTERVALS_MS, EVAL_STARTS_S, REPETITIONS, ARMS, SEEDS,
    LoggedLowSpeedEnv, eval_seed,
)
from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS

PROTOCOL="docs/research-a-v2-l0-d2-onset-advantage-protocol.md"
SOURCE_R1_RUN=41309837474
SOURCE_R1_ARTIFACT=13084957586
SOURCE_R2_RUN=41312136897
SOURCE_R2_ARTIFACT=13086306292
MODES={
    "r1":(LoggedLowSpeedEnv,"research-a-v2-l0-low-speed-teacher-free-v1"),
    "r2":(PenalizedLoggedLowSpeedEnv,"research-a-v2-l0-r2-penalty-comparison-v1"),
}
STEP_FIELDS=("hits","misses","targets","extra_keydowns","physical_keydowns",
             "full_hit","matched_keys","physical_left","physical_right",
             "controls","steps","physical_trace","sampling_seed",
             "interval_ms","start_s")


def source_models(root:Path,mode:str)->tuple[list[dict],dict]:
    filename="full-results.json" if mode=="r1" else "full-comparison.json"
    source=json.loads((root/filename).read_text(encoding="utf-8"))
    if source["schema"]!=MODES[mode][1]:
        raise ValueError("unexpected "+mode+" archive schema")
    models=source["models"] if mode=="r1" else source["r2_models"]
    if len(models)!=4 or {(x["arm"],x["seed"]) for x in models}!={
        (arm,seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("archive must contain four unique arm/seed models")
    if mode=="r2" and not source["r1_original_all_0_and_256_scores_reproduced"]:
        raise ValueError("R2 archive did not validate original R1")
    return models,source


def load_final_model(root:Path,entry:dict)->ActorCritic:
    checkpoint=root/"checkpoints"/entry["checkpoint_file"]
    digest=sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=entry["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch: "+checkpoint.name)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (state["arm"]!=entry["arm"] or state["seed"]!=entry["seed"]
        or state["episodes"]!=256 or state["format"]!="research-a-v2-l0-v1"):
        raise ValueError("checkpoint metadata mismatch: "+checkpoint.name)
    net=ActorCritic()
    net.load_state_dict(state["policy"],strict=True)
    net.eval()
    return net


def verify_replay(row:dict,original:dict,label:str)->None:
    for field in STEP_FIELDS:
        if row[field]!=original[field]:
            raise AssertionError(f"{label}: mismatch {field}")
    for field in ("reward_sum","mean_action_entropy"):
        if not math.isclose(row[field],original[field],rel_tol=1e-6,abs_tol=1e-6):
            raise AssertionError(f"{label}: mismatch {field}")


def replay_one(model:ActorCritic,*,arm:str,mode:str,interval_ms:int,
               start_s:float,seed:int)->dict:
    if mode not in MODES:
        raise ValueError("unknown R1/R2 mode")
    env=MODES[mode][0](arm=arm,interval_ms=interval_ms,start_s=start_s)
    obs=env.reset()
    rng=torch.Generator(device="cpu")
    rng.manual_seed(seed)
    last_positive={"left":False,"right":False}
    latest_onset={"left":None,"right":None}
    onset_used=Counter()
    controls=[0,0,0,0]
    steps=[]
    events=[]
    entropy_sum=0.0
    with torch.no_grad():
        while True:
            now=env.motor.sim.time_s
            logits,value=model(torch.tensor(obs,dtype=torch.float32))
            probs=torch.softmax(logits,dim=-1)
            action=int(torch.multinomial(probs,1,generator=rng).item())
            entropy_sum+=float(-(probs*torch.log(probs.clamp_min(1e-12))).sum())
            step_idx=len(steps)
            for key,positive in (
                ("left",ACTION_COMMANDS[action].left>0),
                ("right",ACTION_COMMANDS[action].right>0),
            ):
                if positive and not last_positive[key]:
                    latest_onset[key]=(step_idx,now)
                last_positive[key]=positive
            controls[action]+=1
            prev=len(env.down_trace)
            obs,reward,done=env.step(action)
            steps.append({
                "step":step_idx,"time_s":round(now,6),
                "action":action,"value":float(value.item()),
                "reward":float(reward),
            })
            for event in env.down_trace[prev:]:
                key=event["key"]
                onset=latest_onset[key]
                used=False
                if onset is not None:
                    used=onset_used[(key,onset[0])]>0
                    onset_used[(key,onset[0])]+=1
                events.append({
                    **event,
                    "onset_step":None if onset is None else onset[0],
                    "onset_time_s":None if onset is None else round(onset[1],6),
                    "onset_to_keydown_ms":(
                        None if onset is None else
                        round((event["time_s"]-onset[1])*1000,4)
                    ),
                    "onset_reused":used,
                    "onset_to_nearest_target_ms":(
                        None if onset is None else
                        round((env.targets[event["closest_target_ordinal"]]-onset[1])*1000,4)
                    ),
                })
            if done:
                break
    score=env.result()
    if len(events)!=score.physical_keydowns:
        raise AssertionError("KeyDown trace count mismatch")
    if score.hits+score.extra_keydowns!=score.physical_keydowns:
        raise AssertionError("Hit/Extra KeyDown accounting mismatch")
    return {
        "interval_ms":interval_ms,"start_s":start_s,"sampling_seed":seed,
        "hits":score.hits,"misses":score.misses,"targets":score.targets,
        "extra_keydowns":score.extra_keydowns,
        "physical_keydowns":score.physical_keydowns,
        "full_hit":score.full_hit,
        "matched_keys":"".join("L" if x=="left" else "R" for x in score.matched_keys),
        "physical_left":sum(x["key"]=="left" for x in events),
        "physical_right":sum(x["key"]=="right" for x in events),
        "controls":controls,"steps":len(steps),
        "reward_sum":score.reward_sum,
        "mean_action_entropy":entropy_sum/len(steps),
        "physical_trace":env.down_trace,
        "step_data":steps,
        "event_data":events,
    }


def normalized_advantages(episodes:list[dict])->None:
    raw=[]
    for episode in episodes:
        returns=discounted_returns([x["reward"] for x in episode["step_data"]])
        for step,g in zip(episode["step_data"],returns):
            step["return_to_go"]=float(g)
            step["advantage_raw"]=float(g-step["value"])
            raw.append(step["advantage_raw"])
    if not raw:
        raise ValueError("no sampled actions")
    center=mean(raw)
    # Match PPO's unbiased=False population standard deviation and epsilon.
    std=(sum((x-center)**2 for x in raw)/len(raw))**.5
    for episode in episodes:
        for step in episode["step_data"]:
            step["advantage_normalized"]=(step["advantage_raw"]-center)/(std+1e-6)
        for event in episode["event_data"]:
            idx=event["onset_step"]
            event["onset_advantage_normalized"]=(
                None if idx is None else
                episode["step_data"][idx]["advantage_normalized"]
            )
    return None


def describe(values:list[float])->dict:
    return {
        "count":len(values),
        "mean":mean(values) if values else None,
        "median":median(values) if values else None,
        "positive_fraction":sum(x>0 for x in values)/len(values) if values else None,
    }


def aggregate(episodes:list[dict])->dict:
    events=[x for ep in episodes for x in ep["event_data"]]
    hits=[x for x in events if x["hit"]]
    extras=[x for x in events if not x["hit"]]
    hit_adv=[x["onset_advantage_normalized"] for x in hits
             if x["onset_advantage_normalized"] is not None]
    extra_adv=[x["onset_advantage_normalized"] for x in extras
               if x["onset_advantage_normalized"] is not None]
    def event_summary(xs:list[dict],advs:list[float])->dict:
        return {
            "events":len(xs),
            "matched_onsets":len(advs),
            "unmatched_onsets":len(xs)-len(advs),
            "reused_onsets":sum(x["onset_reused"] for x in xs),
            "onset_advantage":describe(advs),
            "onset_to_keydown_ms":describe([
                x["onset_to_keydown_ms"] for x in xs
                if x["onset_to_keydown_ms"] is not None
            ]),
            "onset_to_nearest_target_ms":describe([
                x["onset_to_nearest_target_ms"] for x in xs
                if x["onset_to_nearest_target_ms"] is not None
            ]),
        }
    h=event_summary(hits,hit_adv)
    e=event_summary(extras,extra_adv)
    return {
        "episodes":len(episodes),
        "targets":sum(x["targets"] for x in episodes),
        "hits":len(hits),"extras":len(extras),
        "full_hit_episodes":sum(x["full_hit"] for x in episodes),
        "physical_left":sum(x["physical_left"] for x in episodes),
        "physical_right":sum(x["physical_right"] for x in episodes),
        "hit":h,"extra":e,
        "mean_onset_advantage_hit_minus_extra":(
            mean(hit_adv)-mean(extra_adv) if hit_adv and extra_adv else None
        ),
        "positive_fraction_hit_minus_extra":(
            h["onset_advantage"]["positive_fraction"]-
            e["onset_advantage"]["positive_fraction"]
            if hit_adv and extra_adv else None
        ),
        "all_action_advantages":{
            str(action):describe([
                step["advantage_normalized"]
                for ep in episodes for step in ep["step_data"]
                if step["action"]==action
            ]) for action in range(4)
        },
    }


def run(r1_dir:Path,r2_dir:Path)->dict:
    torch.set_num_threads(1)
    roots={"r1":r1_dir,"r2":r2_dir}
    all_models=[]
    r1_models,r1_source=source_models(r1_dir,"r1")
    r2_models,r2_source=source_models(r2_dir,"r2")
    for mode,models in (("r1",r1_models),("r2",r2_models)):
        for entry in models:
            arm,seed=entry["arm"],entry["seed"]
            model=load_final_model(roots[mode],entry)
            source_stage=next(x for x in entry["evaluations"] if x["episode"]==256)
            original_rows=source_stage["sampled"]["rows"]
            if len(original_rows)!=36:
                raise ValueError("original sampled evaluation missing rows")
            rows=[]
            case=0
            for interval in EVAL_INTERVALS_MS:
                for start in EVAL_STARTS_S:
                    for rep in range(REPETITIONS):
                        original=original_rows[len(rows)]
                        row=replay_one(model,arm=arm,mode=mode,interval_ms=interval,
                                       start_s=start,seed=eval_seed(case,rep))
                        verify_replay(row,original,
                                      f"{mode}/{arm}/seed{seed}/case{case}/rep{rep}")
                        row["case_index"]=case
                        row["repetition"]=rep
                        rows.append(row)
                    case+=1
            normalized_advantages(rows)
            summary=aggregate(rows)
            if summary["hits"]!=source_stage["sampled"]["hits"]:
                raise AssertionError("aggregate Hit mismatch")
            if summary["extras"]!=source_stage["sampled"]["extra_keydowns"]:
                raise AssertionError("aggregate Extra mismatch")
            all_models.append({
                "mode":mode,"arm":arm,"seed":seed,
                "checkpoint_sha256":entry["checkpoint_sha256"],
                "all_original_36_rows_reproduced":True,
                "summary":summary,"episodes":rows,
            })
            print("D2_MODEL_VERIFIED",json.dumps({
                "mode":mode,"arm":arm,"seed":seed,
                "hits":summary["hits"],"extras":summary["extras"],
                "hit_onset_adv":summary["hit"]["onset_advantage"],
                "extra_onset_adv":summary["extra"]["onset_advantage"],
                "mean_hit_minus_extra":summary["mean_onset_advantage_hit_minus_extra"],
            }),flush=True)
    if len(all_models)!=8:
        raise AssertionError("not all eight models were replayed")
    a3=[{
        "arm":x["arm"],"hits":x["random_uniform"]["hits"],
        "targets":x["random_uniform"]["targets"],
        "extra_keydowns":x["random_uniform"]["extra_keydowns"],
    } for x in r1_source["random_baselines"]]
    return {
        "schema":"research-a-v2-l0-d2-frozen-onset-advantage-v1",
        "status":"no training; retrospective frozen-policy advantage proxy only",
        "protocol":PROTOCOL,
        "source_r1_run":SOURCE_R1_RUN,
        "source_r1_artifact":SOURCE_R1_ARTIFACT,
        "source_r2_run":SOURCE_R2_RUN,
        "source_r2_artifact":SOURCE_R2_ARTIFACT,
        "not_historical_ppo_gradient":True,
        "normalization":"all held-out sampled steps per model; not 8-episode train batch",
        "all_288_archival_sampled_episodes_reproduced":True,
        "models":all_models,"a3_random_uniform_archival":a3,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1",type=Path,required=True)
    parser.add_argument("--r2",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite existing result")
    report=run(args.r1,args.r2)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("D2_FINAL_SUMMARY",json.dumps({
        "all_288_archival_sampled_episodes_reproduced":True,
        "models":[{k:v for k,v in m.items() if k!="episodes"}
                  for m in report["models"]],
        "a3":report["a3_random_uniform_archival"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
