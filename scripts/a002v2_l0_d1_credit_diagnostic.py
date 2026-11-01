#!/usr/bin/env python3
"""Research A v2-L0-D1 — original-model no-training action/KeyDown diagnosis.

Verifies all 0/256 greedy + sampled original per-episode scores/physical
traces before reporting. Intermediate 128 is archival stats ONLY.
PPO gradients and credit assignment are NOT reconstructed.
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

from a002_exploratory_ppo import ActorCritic
from a002v2_l0_low_speed_ppo import (
    ACTION_NAMES, ARMS, EVAL_INTERVALS_MS, EVAL_STARTS_S,
    REPETITIONS, SEEDS, LoggedLowSpeedEnv, eval_seed,
)
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS

PROTOCOL="docs/research-a-v2-l0-d1-credit-diagnostic-protocol.md"
ORIGINAL_RUN=41309837474
ORIGINAL_ARTIFACT=13084957586
SOURCE_COMMIT="fa456d0e07e90fb740d969715f593f81c1ee48c3"
PHASE_LABELS=("far_early_gt100ms","early_20_to_100ms",
              "judgement_window_20ms","late_gt20ms")


def phase_name(remaining_ms:float)->str:
    if remaining_ms > 100.0:
        return PHASE_LABELS[0]
    if remaining_ms > 20.0:
        return PHASE_LABELS[1]
    if remaining_ms >= -20.0:
        return PHASE_LABELS[2]
    return PHASE_LABELS[3]


def model_from_archive(entry:dict, root:Path, episode:int)->ActorCritic:
    arm,seed=entry["arm"],entry["seed"]
    if episode == 0:
        torch.manual_seed(seed)
        net=ActorCritic()
    elif episode == 256:
        path=root/"checkpoints"/entry["checkpoint_file"]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["checkpoint_sha256"]:
            raise ValueError("original checkpoint SHA-256 mismatch "+path.name)
        state=torch.load(path,map_location="cpu",weights_only=True)
        if (state["format"]!="research-a-v2-l0-v1" or
            state["arm"]!=arm or state["seed"]!=seed or state["episodes"]!=256):
            raise ValueError("saved model metadata mismatch")
        net=ActorCritic()
        net.load_state_dict(state["policy"],strict=True)
    else:
        raise ValueError("intermediate weights unavailable; only 0 and 256")
    net.eval()
    return net


def replay_one(model:ActorCritic, *, arm:str, period:int, start:float,
               mode:str, random_seed:int|None)->dict:
    if mode not in ("argmax","sampled"):
        raise ValueError("invalid model evaluation mode")
    if mode == "sampled" and random_seed is None:
        raise ValueError("sampled requires original evaluation RNG seed")
    env=LoggedLowSpeedEnv(arm=arm,interval_ms=period,start_s=start)
    observation=env.reset()
    generator=torch.Generator(device="cpu")
    if random_seed is not None:
        generator.manual_seed(random_seed)

    pressed_command={"left":False,"right":False}
    latest_onset_s={"left":None,"right":None}
    phases={label:{
        "steps":0,"chosen":[0,0,0,0],
        "prob_sums":[0.0]*4,"entropy_sum":0.0,
        "top2_prob_gap_sum":0.0,
    } for label in PHASE_LABELS}
    hit_events=[]
    all_keydown_events=[]
    physical_lags_ms=[]

    with torch.no_grad():
        while True:
            now=env.motor.sim.time_s
            if env.cursor < len(env.targets):
                remaining_ms=(env.targets[env.cursor]-now)*1000.
            else:
                remaining_ms=-999.0
            phase=phase_name(remaining_ms)
            logits,_=model(torch.tensor(observation,dtype=torch.float32))
            probs=logits.softmax(dim=-1)
            if mode=="argmax":
                action=int(probs.argmax().item())
            else:
                action=int(torch.multinomial(probs,1,generator=generator).item())

            for key,value in (
                ("left",ACTION_COMMANDS[action].left),
                ("right",ACTION_COMMANDS[action].right),
            ):
                # A0's physical right finger is disabled by Motor, not by policy.
                positive=value>0
                if positive and not pressed_command[key]:
                    latest_onset_s[key]=now
                pressed_command[key]=positive
            row=phases[phase]
            row["steps"]+=1
            row["chosen"][action]+=1
            ps=[float(v) for v in probs.tolist()]
            for k in range(4):
                row["prob_sums"][k]+=ps[k]
            row["entropy_sum"]+=-sum(p*math.log(max(1e-12,p)) for p in ps)
            sorted_ps=sorted(ps,reverse=True)
            row["top2_prob_gap_sum"]+=sorted_ps[0]-sorted_ps[1]

            previous_trace_count=len(env.down_trace)
            observation,_reward,done=env.step(action)
            for event in env.down_trace[previous_trace_count:]:
                key=event["key"]
                lag=(None if latest_onset_s[key] is None else
                     round((event["time_s"]-latest_onset_s[key])*1000,4))
                record={**event,"command_onset_to_keydown_ms":lag}
                all_keydown_events.append(record)
                if lag is not None and lag>=-0.0001:
                    physical_lags_ms.append(lag)
                if event["hit"]:
                    hit_events.append(record)
            if done:
                break

    score=env.result()
    controls=[sum(b["chosen"][k] for b in phases.values()) for k in range(4)]
    steps=sum(b["steps"] for b in phases.values())
    keydown_by_key=dict(Counter(x["key"] for x in all_keydown_events))
    by_phase={}
    for label,b in phases.items():
        n=b["steps"]
        by_phase[label]={
            "steps":n,
            "chosen_fraction":dict(zip(ACTION_NAMES,[
                b["chosen"][k]/n if n else None for k in range(4)
            ])),
            "mean_softmax":dict(zip(ACTION_NAMES,[
                b["prob_sums"][k]/n if n else None for k in range(4)
            ])),
            "entropy_nats":b["entropy_sum"]/n if n else None,
            "top_two_prob_gap":b["top2_prob_gap_sum"]/n if n else None,
        }
    return {
        "interval_ms":period,"start_s":start,"mode":mode,
        "sampling_seed":random_seed,
        "hits":score.hits,"misses":score.misses,"targets":score.targets,
        "extra_keydowns":score.extra_keydowns,
        "physical_keydowns":score.physical_keydowns,
        "full_hit":score.full_hit,
        "reward_sum":score.reward_sum,
        "matched_keys":"".join("L" if k=="left" else "R" for k in score.matched_keys),
        "physical_left":keydown_by_key.get("left",0),
        "physical_right":keydown_by_key.get("right",0),
        "controls":controls,"steps":steps,
        "mean_action_entropy":sum(
            b["entropy_sum"] for b in phases.values()
        )/steps,
        "physical_trace":env.down_trace,
        "events_with_command_lag":all_keydown_events,
        "by_time_to_next_target":by_phase,
        "onset_to_physical_down_lag_ms":physical_lags_ms,
    }


def check_exact_original(candidate:dict, original:dict, *, label:str)->None:
    for key in ("interval_ms","start_s","hits","misses","targets",
                "extra_keydowns","physical_keydowns","full_hit",
                "matched_keys","physical_left","physical_right",
                "controls","steps","physical_trace"):
        if candidate[key]!=original[key]:
            raise AssertionError(
                f"original replay mismatch {label}, {key}: "
                f"{candidate[key]!r} != {original[key]!r}"
            )
    # Float-only values should be indistinguishable within numeric roundoff.
    for key in ("reward_sum","mean_action_entropy"):
        if not math.isclose(candidate[key],original[key],rel_tol=1e-6,abs_tol=1e-6):
            raise AssertionError(f"original replay float mismatch {label}, {key}")


def compress_stats(values:list[float])->dict:
    return {
        "n":len(values),
        "median":round(median(values),4) if values else None,
        "mean":round(mean(values),4) if values else None,
        "min":round(min(values),4) if values else None,
        "max":round(max(values),4) if values else None,
    }


def aggregate_replays(rows:list[dict])->dict:
    count=sum(x["steps"] for x in rows)
    hits=sum(x["hits"] for x in rows)
    targets=sum(x["targets"] for x in rows)
    extras=sum(x["extra_keydowns"] for x in rows)
    aggregate_phases={}
    for label in PHASE_LABELS:
        all_counts=[0.0]*4
        all_probs=[0.0]*4
        entropy=gap=0.0
        n=0
        for episode in rows:
            p=episode["by_time_to_next_target"][label]
            steps=p["steps"]
            n+=steps
            if steps:
                for i,name in enumerate(ACTION_NAMES):
                    all_counts[i]+=p["chosen_fraction"][name]*steps
                    all_probs[i]+=p["mean_softmax"][name]*steps
                entropy+=p["entropy_nats"]*steps
                gap+=p["top_two_prob_gap"]*steps
        aggregate_phases[label]={
            "steps":n,
            "chosen_fraction":dict(zip(ACTION_NAMES,[
                x/n if n else None for x in all_counts
            ])),
            "mean_softmax":dict(zip(ACTION_NAMES,[
                x/n if n else None for x in all_probs
            ])),
            "entropy_nats":entropy/n if n else None,
            "top_two_prob_gap":gap/n if n else None,
        }

    all_lags=[x for r in rows for x in r["onset_to_physical_down_lag_ms"]]
    hit_offsets=[x["offset_ms"] for r in rows for x in
                 r["events_with_command_lag"] if x["hit"]]
    extra_offsets=[x["offset_ms"] for r in rows for x in
                   r["events_with_command_lag"] if not x["hit"]]
    return {
        "episodes":len(rows),"hits":hits,"targets":targets,
        "hit_rate":hits/targets,
        "extra_keydowns":extras,"extra_per_target":extras/targets,
        "mean_reward_per_episode":mean(x["reward_sum"] for x in rows),
        "physical_left":sum(x["physical_left"] for x in rows),
        "physical_right":sum(x["physical_right"] for x in rows),
        "mean_action_entropy":sum(
            x["mean_action_entropy"]*x["steps"] for x in rows
        )/count,
        "by_time_to_next_target":aggregate_phases,
        "press_onset_to_keydown_lag_ms":compress_stats(all_lags),
        "hit_offset_ms":compress_stats(hit_offsets),
        "extra_nearest_offset_ms":compress_stats(extra_offsets),
        "hit_abs_offset_within_5ms":sum(abs(x)<=5 for x in hit_offsets),
        "hit_abs_offset_5_to_20ms":sum(5<abs(x)<=20 for x in hit_offsets),
        "extra_abs_offset_within_20ms":sum(abs(x)<=20 for x in extra_offsets),
        "extra_abs_offset_20_to_80ms":sum(20<abs(x)<=80 for x in extra_offsets),
    }


def archival_history(entry:dict)->dict:
    records=[]
    if len(entry["training"])!=32:
        raise ValueError("original PPO update history incomplete")
    for b in entry["training"]:
        records.append({
            "episode":b["episode"],
            "train_hit_rate":b["train_hit_rate"],
            "train_extra_keydowns":b["train_extra_keydowns"],
            "mean_train_reward":b["mean_train_reward"],
            "ppo_policy_loss":b["policy_loss"],
            "ppo_value_loss":b["value_loss"],
            "ppo_entropy":b["entropy"],
            "ppo_total_loss":b["loss"],
        })
    eval_history=[]
    for e in entry["evaluations"]:
        modes={}
        for mode in ("argmax","sampled"):
            orig=e[mode]
            modes[mode]={
                "hit_rate":orig["hit_rate"],
                "extra_per_target":orig["extra_per_target"],
                "action_selection_fraction":orig["action_selection_fraction"],
                "mean_entropy":orig["mean_entropy"],
                "mean_reward_per_episode":mean(x["reward_sum"] for x in orig["rows"]),
                "physical_left":orig["physical_left"],
                "physical_right":orig["physical_right"],
            }
        eval_history.append({"episode":e["episode"],**modes})
    return {"ppo_updates":records,"eval_0_128_256":eval_history}


def run_diagnosis(root:Path)->dict:
    source=json.loads((root/"full-results.json").read_text(encoding="utf-8"))
    if source["schema"]!="research-a-v2-l0-low-speed-teacher-free-v1":
        raise ValueError("original learning report schema mismatch")
    all_entries=source["models"]
    if len(all_entries)!=4 or {(x["arm"],x["seed"]) for x in all_entries}!={
        (arm,seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("not exactly four archived L0 models")
    output=[]
    for entry in all_entries:
        arm,seed=entry["arm"],entry["seed"]
        timepoints=[]
        for stage in (0,256):
            model=model_from_archive(entry,root,stage)
            source_eval=next(x for x in entry["evaluations"] if x["episode"]==stage)
            replayed_modes={}
            for mode in ("argmax","sampled"):
                original_rows=source_eval[mode]["rows"]
                cases=[]
                idx=0
                for period in EVAL_INTERVALS_MS:
                    for start in EVAL_STARTS_S:
                        reps=range(REPETITIONS) if mode=="sampled" else (0,)
                        for rep in reps:
                            episode=replay_one(model,arm=arm,period=period,start=start,
                                               mode=mode,random_seed=(
                                                   eval_seed(idx,rep) if mode=="sampled" else None
                                               ))
                            compare_entry=original_rows[len(cases)]
                            check_exact_original(episode,compare_entry,
                                                 label=f"{arm}/{seed}/ep{stage}/{mode}/{len(cases)}")
                            episode["case_index"]=idx
                            episode["repetition"]=rep
                            cases.append(episode)
                        idx+=1
                if len(cases)!=len(original_rows):
                    raise AssertionError("archived episode count mismatch")
                replayed_modes[mode]={
                    "original_score_reproduced":True,
                    "aggregate":aggregate_replays(cases),
                    "all_replayed_episodes":cases,
                }
            timepoints.append({"episode":stage,**replayed_modes})
        output.append({
            "arm":arm,"seed":seed,
            "original_checkpoint_sha256":entry["checkpoint_sha256"],
            "archival":archival_history(entry),
            "replays":timepoints,
        })
        final=timepoints[-1]
        print("D1_MODEL_VERIFIED",json.dumps({
            "arm":arm,"seed":seed,
            "final_sampled_hits":final["sampled"]["aggregate"]["hits"],
            "final_argmax_hits":final["argmax"]["aggregate"]["hits"],
            "original_scores_identical":True,
        }),flush=True)
    return {
        "schema":"research-a-v2-l0-d1-saved-policy-credit-proxy-v1",
        "status":"posthoc, 0/256 original replay; no training, no intermediate weights",
        "protocol":PROTOCOL,
        "source_run":ORIGINAL_RUN,"source_artifact":ORIGINAL_ARTIFACT,
        "source_commit":SOURCE_COMMIT,
        "what_is_not_measured":"actual per-action PPO gradient attribution",
        "all_models":output,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite previous output")
    torch.set_num_threads(1)
    result=run_diagnosis(args.input)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("D1_FINAL_SUMMARY",json.dumps({
        "models":[{
            "arm":x["arm"],"seed":x["seed"],
            "history":x["archival"]["eval_0_128_256"],
            "replay":{
                str(r["episode"]):{
                    mode:{
                        "hit_rate":r[mode]["aggregate"]["hit_rate"],
                        "extra_per_target":r[mode]["aggregate"]["extra_per_target"],
                        "entropy":r[mode]["aggregate"]["mean_action_entropy"],
                        "lag":r[mode]["aggregate"]["press_onset_to_keydown_lag_ms"],
                        "phase":r[mode]["aggregate"]["by_time_to_next_target"],
                    } for mode in ("argmax","sampled")
                } for r in x["replays"]
            },
        } for x in result["all_models"]],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
