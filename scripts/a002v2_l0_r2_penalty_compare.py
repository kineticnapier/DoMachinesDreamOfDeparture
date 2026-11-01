#!/usr/bin/env python3
"""R1 archival control vs new R2 (-0.05 near extra) PPO, 4 fixed models.

No retraining of the original R1 models. The original evaluation runs and
final weights are validated before paired comparisons. R2 calls the identical
existing L0 PPO, with only its environment constructor replaced temporarily.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv

ORIGINAL_RUN=41309837474
ORIGINAL_ARTIFACT=13084957586
ORIGINAL_HEAD="fa456d0e07e90fb740d969715f593f81c1ee48c3"
PROTOCOL="docs/research-a-v2-l0-r2-penalty-protocol.md"
R1_NAME="r1_original_saved"
R2_NAME="r2_near_extra_penalty_minus_0p05"


def checked_original(path:Path)->dict:
    report=json.loads((path/"full-results.json").read_text(encoding="utf-8"))
    if report["schema"]!="research-a-v2-l0-low-speed-teacher-free-v1":
        raise ValueError("wrong original L0 source report")
    settings=report["settings"]
    if (tuple(settings["arms"])!=l0.ARMS or
        tuple(settings["seeds"])!=l0.SEEDS or
        settings["episodes"]!=l0.EPISODES or
        tuple(settings["train_intervals_ms"])!=l0.TRAIN_INTERVALS_MS or
        tuple(settings["eval_intervals_ms"])!=l0.EVAL_INTERVALS_MS or
        tuple(settings["eval_starts_s"])!=l0.EVAL_STARTS_S or
        settings["evaluation_repetitions"]!=l0.REPETITIONS):
        raise ValueError("source protocol mismatch")
    if (len(report["models"])!=4 or
        {(m["arm"],m["seed"]) for m in report["models"]}!=
        {(arm,seed) for arm in l0.ARMS for seed in l0.SEEDS}):
        raise ValueError("missing or repeated archived original models")
    return report


def assert_original_evaluation(got:dict, expected:dict, label:str):
    """Verify *all* scoring rows, not just the aggregate or hit rate."""
    for key in ("hits","misses","targets","extra_keydowns","physical_left",
                "physical_right","full_hit_episodes","episodes"):
        if got[key]!=expected[key]:
            raise AssertionError(f"R1 original mismatch {label}/{key}")
    if len(got["rows"])!=len(expected["rows"]):
        raise AssertionError(f"R1 original episode count mismatch {label}")
    for i,(a,b) in enumerate(zip(got["rows"],expected["rows"])):
        for field in (
            "interval_ms","start_s","hits","misses","targets",
            "extra_keydowns","physical_keydowns","full_hit",
            "matched_keys","physical_left","physical_right",
            "controls","steps","physical_trace","sampling_seed"
        ):
            if a[field]!=b[field]:
                raise AssertionError(
                    f"R1 original mismatch {label}/row{i}/{field}: "
                    f"{a[field]!r} vs {b[field]!r}"
                )


def verify_original_models(input_dir:Path,original:dict)->list[dict]:
    result=[]
    for item in original["models"]:
        arm,seed=item["arm"],item["seed"]
        file=input_dir/"checkpoints"/item["checkpoint_file"]
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest!=item["checkpoint_sha256"]:
            raise ValueError(f"R1 original checkpoint digest mismatch {arm}/{seed}")
        ck=torch.load(file,map_location="cpu",weights_only=True)
        if (ck["arm"]!=arm or ck["seed"]!=seed or
            ck["format"]!="research-a-v2-l0-v1" or ck["episodes"]!=256):
            raise ValueError("R1 original checkpoint metadata mismatch")
        model=l0.ActorCritic()
        model.load_state_dict(ck["policy"],strict=True)
        model.eval()

        # Retain the exact initial seed and verify identical stage-0 evaluation.
        torch.manual_seed(seed)
        initial_model=l0.ActorCritic()
        buf=BytesIO()
        torch.save(initial_model.state_dict(),buf)
        if hashlib.sha256(buf.getvalue()).hexdigest()!=item["initial_state_sha256"]:
            raise ValueError(f"original initial policy bytes differ for {arm}/{seed}")
        # Evaluation always uses its own seeded CPU generator, so no global RNG.
        for stage,network in ((0,initial_model),(256,model)):
            from_original=next(e for e in item["evaluations"] if e["episode"]==stage)
            for mode in ("argmax","sampled"):
                replay=l0.evaluate_set(arm=arm,model=network,mode=mode)
                assert_original_evaluation(replay,from_original[mode],
                                           f"{arm}/{seed}/ep{stage}/{mode}")
        result.append({
            "arm":arm,"seed":seed,
            "checkpoint_sha256":digest,
            "initial_weights_sha256":item["initial_state_sha256"],
            "replayed_0_and_256_original_evaluations":True,
            "source_evaluations":item["evaluations"],
            "source_train_history":item["training"],
        })
        print("R1_ORIGINAL_VERIFIED",json.dumps({
            "arm":arm,"seed":seed,"checkpoint_sha256":digest,
            "evaluations_reproduced":True,
        }),flush=True)
    return result


def summary_row(arm:str,seed:int,stage:int,r1:dict,r2:dict)->dict:
    def detail(e:dict)->dict:
        out={}
        for mode in ("sampled","argmax"):
            z=e[mode]
            out[mode]={
                "hits":z["hits"],"targets":z["targets"],
                "hit_rate":z["hit_rate"],
                "extra_keydowns":z["extra_keydowns"],
                "extra_per_target":z["extra_per_target"],
                "physical_left":z["physical_left"],
                "physical_right":z["physical_right"],
                "mean_entropy":z["mean_entropy"],
                "full_hit_episodes":z["full_hit_episodes"],
            }
        return out
    return {
        "arm":arm,"seed":seed,"episode":stage,
        R1_NAME:detail(r1),R2_NAME:detail(r2),
        "sampled_hit_delta_pp":100*(r2["sampled"]["hit_rate"]-r1["sampled"]["hit_rate"]),
        "sampled_extra_per_target_delta":(
            r2["sampled"]["extra_per_target"]-r1["sampled"]["extra_per_target"]
        ),
    }


def run(input_dir:Path, checkpoint_dir:Path)->dict:
    torch.set_num_threads(1)
    old=checked_original(input_dir)
    r1=verify_original_models(input_dir,old)
    old_by_key={(x["arm"],x["seed"]):x for x in r1}
    new=[]
    comparison=[]
    for arm in l0.ARMS:
        for seed in l0.SEEDS:
            # The only changed function is which Env is instantiated.
            # Do not alter ActorCritic, update_ppo, reward of the original R1,
            # train/eval grids, seeds, stopping budget or original models.
            with patch.object(l0,"LoggedLowSpeedEnv",PenalizedLoggedLowSpeedEnv):
                trained=l0.train_model(
                    arm=arm,seed=seed,checkpoint_dir=checkpoint_dir
                )
            new.append(trained)
            old_item=old_by_key[(arm,seed)]
            if trained["initial_state_sha256"]!=old_item["initial_weights_sha256"]:
                raise AssertionError("R1/R2 did not use identical initial weights")
            for new_e in trained["evaluations"]:
                previous=next(x for x in old_item["source_evaluations"]
                              if x["episode"]==new_e["episode"])
                comparison.append(summary_row(
                    arm,seed,new_e["episode"],previous,new_e
                ))
            final=comparison[-1]
            print("R2_TRAIN_COMPLETE",json.dumps({
                "arm":arm,"seed":seed,
                "r1_sampled_hit":final[R1_NAME]["sampled"]["hit_rate"],
                "r2_sampled_hit":final[R2_NAME]["sampled"]["hit_rate"],
                "r1_sampled_extras":final[R1_NAME]["sampled"]["extra_keydowns"],
                "r2_sampled_extras":final[R2_NAME]["sampled"]["extra_keydowns"],
                "r2_checkpoint_sha256":trained["checkpoint_sha256"],
            }),flush=True)

    if len(comparison)!=12:
        raise AssertionError("not all four models and three stages evaluated")
    return {
        "schema":"research-a-v2-l0-r2-penalty-comparison-v1",
        "status":"exploratory; four new R2 models only; old R1 frozen",
        "protocol":PROTOCOL,
        "source_original_run":ORIGINAL_RUN,
        "source_original_artifact":ORIGINAL_ARTIFACT,
        "source_original_head":ORIGINAL_HEAD,
        "r1_original_all_0_and_256_scores_reproduced":True,
        "r1_models":r1,
        "r2_models":new,
        "all_stage_comparisons":comparison,
        "R2_only_change":"-0.05 reward for each formerly 0-reward near Extra KeyDown",
        "motor_changed":False,"fingering_teacher_used":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite results")
    report=run(args.original,args.checkpoints)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("R2_FINAL_SUMMARY",json.dumps({
        "original_replay_pass":report["r1_original_all_0_and_256_scores_reproduced"],
        "rows":[x for x in report["all_stage_comparisons"] if x["episode"]==256],
        "r2_checkpoint_sha256":{f"{x['arm']}/{x['seed']}":x["checkpoint_sha256"]
                                 for x in report["r2_models"]},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
