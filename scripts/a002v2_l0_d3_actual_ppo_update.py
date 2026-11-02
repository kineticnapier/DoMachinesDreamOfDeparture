#!/usr/bin/env python3
"""D3: exact archived PPO training replay, instrumented without changing updates.

Only label as historical training signal if ALL original 32 update records,
0/128/256 evaluations, initial and final checkpoint SHA-256 are identical.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
from statistics import mean, median

import torch
from torch.distributions import Categorical

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv
from a002v2_l0_d2_onset_advantage import source_models
from dmdod.motor.env import MotorAction
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS

PROTOCOL="docs/research-a-v2-l0-d3-actual-ppo-update-protocol.md"
R1_RUN=41309837474
R2_RUN=41312136897
ORIGINAL_TRAIN=l0.train_episode
ORIGINAL_UPDATE=l0.update_ppo


class TraceMixin:
    """Side-channel recording only. All scoring and Motor steps are inherited."""
    capture_enabled=False
    completed=[]

    def reset(self):
        self._trace_step=0
        self._last_positive={"left":False,"right":False}
        self._latest_onset={"left":None,"right":None}
        self._onset_usage=Counter()
        self._training_events=[]
        return super().reset()

    def step(self, action_index):
        if not type(self).capture_enabled:
            return super().step(action_index)
        index=self._trace_step
        now=self.motor.sim.time_s
        desired=ACTION_COMMANDS[action_index]
        actual=(desired if self.arm=="a1" else MotorAction(desired.left,-1.0))
        for key,positive in (("left",actual.left>0),("right",actual.right>0)):
            if positive and not self._last_positive[key]:
                self._latest_onset[key]=(index,now)
            self._last_positive[key]=positive
        prev=len(self.down_trace)
        observation,reward,done=super().step(action_index)
        for e in self.down_trace[prev:]:
            onset=self._latest_onset[e["key"]]
            reused=False
            if onset is not None:
                tag=(e["key"],onset[0])
                reused=self._onset_usage[tag]>0
                self._onset_usage[tag]+=1
            self._training_events.append({
                "hit":e["hit"],"key":e["key"],
                "keydown_time_s":e["time_s"],
                "onset_step":None if onset is None else onset[0],
                "onset_time_s":None if onset is None else round(onset[1],6),
                "onset_to_keydown_ms":None if onset is None else
                    round(1000*(e["time_s"]-onset[1]),4),
                "onset_reused":reused,
                "target_offset_ms":e["offset_ms"],
            })
        self._trace_step+=1
        if done:
            type(self).completed.append({
                "steps":self._trace_step,
                "hits":self.hits,
                "extras":self.extra_keydowns,
                "events":self._training_events,
            })
        return observation,reward,done


class TraceR1(TraceMixin,l0.LoggedLowSpeedEnv):
    completed=[]


class TraceR2(TraceMixin,PenalizedLoggedLowSpeedEnv):
    completed=[]


def describe(events):
    if not events:
        return {"count":0,"positive_adv":0,"increased_logp":0,
                "both_positive":0,"positive_adv_fraction":None,
                "increased_logp_fraction":None,"both_positive_fraction":None,
                "mean_advantage":None,"mean_delta_logp":None,
                "median_onset_to_keydown_ms":None}
    n=len(events)
    adv=sum(x["advantage_normalized"]>0 for x in events)
    inc=sum(x["delta_logp"]>0 for x in events)
    both=sum(x["advantage_normalized"]>0 and x["delta_logp"]>0 for x in events)
    delays=[x["onset_to_keydown_ms"] for x in events
            if x["onset_to_keydown_ms"] is not None]
    return {"count":n,"positive_adv":adv,"increased_logp":inc,
            "both_positive":both,"positive_adv_fraction":adv/n,
            "increased_logp_fraction":inc/n,"both_positive_fraction":both/n,
            "mean_advantage":mean(x["advantage_normalized"] for x in events),
            "mean_delta_logp":mean(x["delta_logp"] for x in events),
            "median_onset_to_keydown_ms":median(delays) if delays else None}


def summarize(events):
    return {
        "hit":describe([e for e in events if e["hit"]]),
        "extra":describe([e for e in events if not e["hit"]]),
        "unmatched_onsets":sum(e["onset_step"] is None for e in events),
        "reused_onsets":sum(e["onset_reused"] for e in events),
        "all_physical_keydowns":len(events),
        "left_keydowns":sum(e["key"]=="left" for e in events),
        "right_keydowns":sum(e["key"]=="right" for e in events),
    }


class Recorder:
    def __init__(self,mode,arm,seed,env_cls):
        self.mode=mode
        self.arm=arm
        self.seed=seed
        self.env_cls=env_cls
        self.updates=[]
        self.events=[]

    def train_episode(self,*args,**kwargs):
        cls=self.env_cls
        before=len(cls.completed)
        cls.capture_enabled=True
        try:
            result=ORIGINAL_TRAIN(*args,**kwargs)
        finally:
            cls.capture_enabled=False
        if len(cls.completed)!=before+1:
            raise AssertionError("training episode side-channel missing")
        return result

    def update_ppo(self,model,optimizer,batch):
        cls=self.env_cls
        if len(batch)!=l0.BATCH_EPISODES or len(cls.completed)!=len(batch):
            raise AssertionError("training trace batch count mismatch")
        traces=cls.completed[:]
        cls.completed.clear()
        observations=torch.cat([ep["observations"] for ep in batch])
        actions=torch.cat([ep["actions"] for ep in batch])
        returns=torch.cat([ep["returns"] for ep in batch])
        old_values=torch.cat([ep["old_values"] for ep in batch])
        old_log_probs=torch.cat([ep["old_log_probs"] for ep in batch])
        advantages=returns-old_values
        advantages=(advantages-advantages.mean())/(
            advantages.std(unbiased=False)+1e-6
        )
        with torch.no_grad():
            logits,_=model(observations)
            before=Categorical(logits=logits).log_prob(actions)
        if not torch.allclose(before,old_log_probs,rtol=1e-5,atol=1e-5):
            raise AssertionError("pre-update action probabilities differ")
        gradient_epochs=[]
        original_step=optimizer.step

        def traced_optimizer_step(*args,**kwargs):
            # These are the actual clipped actor-layer gradients consumed by Adam.
            weight=model.actor.weight.grad
            bias=model.actor.bias.grad
            if weight is None or bias is None:
                raise AssertionError("actor gradient missing")
            with torch.no_grad():
                flat=torch.cat([weight.detach().flatten(),bias.detach().flatten()])
                gradient_epochs.append({
                    "actor_gradient_l2":float(torch.linalg.vector_norm(flat)),
                    "actor_weight_gradient":weight.detach().cpu().flatten().tolist(),
                    "actor_bias_gradient":bias.detach().cpu().flatten().tolist(),
                })
            return original_step(*args,**kwargs)

        optimizer.step=traced_optimizer_step
        try:
            # Call original PPO verbatim, including both epochs, clipping,
            # entropy, critic loss, Adam update, and gradient clipping.
            losses=ORIGINAL_UPDATE(model,optimizer,batch)
        finally:
            optimizer.step=original_step
        if len(gradient_epochs)!=2:
            raise AssertionError("original PPO did not execute two updates")
        with torch.no_grad():
            logits,_=model(observations)
            after=Categorical(logits=logits).log_prob(actions)
        offset=0
        update_index=len(self.updates)+1
        batch_events=[]
        for i,(episode,trace) in enumerate(zip(batch,traces)):
            n=len(episode["actions"])
            if n!=trace["steps"]:
                raise AssertionError("trace steps mismatch")
            score=episode["score"]
            if (trace["hits"]!=score["hits"] or
                trace["extras"]!=score["extra_keydowns"] or
                len(trace["events"])!=score["physical_keydowns"]):
                raise AssertionError("training physical event count mismatch")
            for e in trace["events"]:
                step=e["onset_step"]
                row={**e,"update":update_index,"episode_in_batch":i}
                if step is not None:
                    ix=offset+step
                    row.update({
                        "action":int(actions[ix]),
                        "advantage_normalized":float(advantages[ix]),
                        "return_to_go":float(returns[ix]),
                        "old_value":float(old_values[ix]),
                        "pre_logp":float(before[ix]),
                        "post_logp":float(after[ix]),
                        "delta_logp":float(after[ix]-before[ix]),
                    })
                else:
                    row.update({
                        "action":None,"advantage_normalized":None,
                        "return_to_go":None,"old_value":None,
                        "pre_logp":None,"post_logp":None,"delta_logp":None,
                    })
                batch_events.append(row)
            offset+=n
        if offset!=len(actions):
            raise AssertionError("PPO observation offset mismatch")
        self.events.extend(batch_events)
        matched=[e for e in batch_events if e["onset_step"] is not None]
        self.updates.append({
            "update":update_index,
            "episodes_completed":update_index*l0.BATCH_EPISODES,
            "observations":len(actions),
            "training_hits":sum(ep["score"]["hits"] for ep in batch),
            "training_extras":sum(ep["score"]["extra_keydowns"] for ep in batch),
            "event_summary":summarize(matched),
            "unmatched_onsets":len(batch_events)-len(matched),
            "reused_onsets":sum(e["onset_reused"] for e in batch_events),
            "actor_gradients_by_epoch":gradient_epochs,
            "ppo_losses":losses,
        })
        return losses


@contextmanager
def instrument(recorder):
    cls=recorder.env_cls
    cls.completed=[]
    cls.capture_enabled=False
    old_env=l0.LoggedLowSpeedEnv
    old_train=l0.train_episode
    old_update=l0.update_ppo
    l0.LoggedLowSpeedEnv=cls
    l0.train_episode=recorder.train_episode
    l0.update_ppo=recorder.update_ppo
    try:
        yield
    finally:
        l0.LoggedLowSpeedEnv=old_env
        l0.train_episode=old_train
        l0.update_ppo=old_update
        cls.capture_enabled=False
        cls.completed.clear()


def compare_archived(original,actual,label):
    if actual["initial_state_sha256"]!=original["initial_state_sha256"]:
        raise AssertionError(label+": original initial weights mismatch")
    if actual["checkpoint_sha256"]!=original["checkpoint_sha256"]:
        raise AssertionError(label+": final checkpoint SHA-256 mismatch")
    if actual["training"]!=original["training"]:
        raise AssertionError(label+": 32 original PPO update records mismatch")
    if actual["evaluations"]!=original["evaluations"]:
        raise AssertionError(label+": original 0/128/256 evaluation rows mismatch")
    if len(actual["training"])!=32 or len(actual["evaluations"])!=3:
        raise AssertionError(label+": expected 32 updates and 3 eval stages")


def run(r1_dir:Path,r2_dir:Path,output_dir:Path)->dict:
    torch.set_num_threads(1)
    r1,_=source_models(r1_dir,"r1")
    r2,_=source_models(r2_dir,"r2")
    report={
        "schema":"research-a-v2-l0-d3-original-ppo-update-replay-v1",
        "status":"in_progress",
        "protocol":PROTOCOL,
        "source_runs":{"r1":R1_RUN,"r2":R2_RUN},
        "original_model_replay_exact":False,
        "not_new_training_condition":True,
        "models":[],
        "a3_random_uniform_archival":{
            "a0":{"hits":46,"extras":250,"targets":576},
            "a1":{"hits":78,"extras":502,"targets":576},
        },
    }
    for mode,originals,env_cls in (
        ("r1",r1,TraceR1),("r2",r2,TraceR2)
    ):
        for original in originals:
            arm,seed=original["arm"],original["seed"]
            label=f"{mode}/{arm}/seed{seed}"
            recorder=Recorder(mode,arm,seed,env_cls)
            checkpoint_dir=output_dir/"replayed-checkpoints"/mode
            with instrument(recorder):
                trained=l0.train_model(
                    arm=arm,seed=seed,checkpoint_dir=checkpoint_dir
                )
            # Save the new run's outcome even if the exactness check fails.
            entry={
                "mode":mode,"arm":arm,"seed":seed,
                "source_checkpoint_sha256":original["checkpoint_sha256"],
                "replay_checkpoint_sha256":trained["checkpoint_sha256"],
                "original_reproduced":False,
                "update_count":len(recorder.updates),
                "event_count":len(recorder.events),
                "all_updates":recorder.updates,
                "all_training_keydown_events":recorder.events,
                "training_log":trained["training"],
                "evaluations":trained["evaluations"],
            }
            report["models"].append(entry)
            output_dir.mkdir(parents=True,exist_ok=True)
            (output_dir/"full-diagnostic.json").write_text(
                json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                encoding="utf-8"
            )
            compare_archived(original,trained,label)
            if len(recorder.updates)!=32:
                raise AssertionError(label+": expected exactly 32 updates")
            entry["original_reproduced"]=True
            summary=summarize([e for e in recorder.events
                               if e["onset_step"] is not None])
            entry["overall"]=summary
            entry["first_16_updates"]=summarize([
                e for e in recorder.events
                if e["update"]<=16 and e["onset_step"] is not None
            ])
            entry["last_16_updates"]=summarize([
                e for e in recorder.events
                if e["update"]>16 and e["onset_step"] is not None
            ])
            entry["unmatched_onsets"]=sum(
                e["onset_step"] is None for e in recorder.events
            )
            entry["reused_onsets"]=sum(
                e["onset_reused"] for e in recorder.events
            )
            entry["training_hits"]=sum(
                x["train_hit_rate"]*128 for x in trained["training"]
            )
            entry["training_extras"]=sum(
                x["train_extra_keydowns"] for x in trained["training"]
            )
            final=trained["evaluations"][-1]["sampled"]
            entry["final_sampled_evaluation"]={
                "hits":final["hits"],"targets":final["targets"],
                "extras":final["extra_keydowns"],
                "full_hit_episodes":final["full_hit_episodes"],
            }
            (output_dir/"full-diagnostic.json").write_text(
                json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                encoding="utf-8"
            )
            print("D3_ORIGINAL_TRAINING_REPRODUCED",json.dumps({
                "model":label,
                "checkpoint_sha256":trained["checkpoint_sha256"],
                "training_events":len(recorder.events),
                "hit":summary["hit"],"extra":summary["extra"],
                "unmatched":entry["unmatched_onsets"],
                "reused":entry["reused_onsets"],
            }),flush=True)
    if len(report["models"])!=8 or not all(
        x["original_reproduced"] for x in report["models"]
    ):
        raise AssertionError("not all eight original models reproduced")
    report["original_model_replay_exact"]=True
    report["status"]="complete; exact original PPO replay; diagnostic only"
    (output_dir/"full-diagnostic.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("D3_FINAL_SUMMARY",json.dumps({
        "original_model_replay_exact":True,
        "models":[{
            "mode":x["mode"],"arm":x["arm"],"seed":x["seed"],
            "updates":x["update_count"],"events":x["event_count"],
            "overall":x["overall"],
            "first_16_updates":x["first_16_updates"],
            "last_16_updates":x["last_16_updates"],
            "final_sampled_evaluation":x["final_sampled_evaluation"],
        } for x in report["models"]],
    },ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1",type=Path,required=True)
    parser.add_argument("--r2",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if (args.output_dir/"full-diagnostic.json").exists():
        raise FileExistsError("refuse to overwrite existing D3 output")
    run(args.r1,args.r2,args.output_dir)


if __name__=="__main__":
    main()
