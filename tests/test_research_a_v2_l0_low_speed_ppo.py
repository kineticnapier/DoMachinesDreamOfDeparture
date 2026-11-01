"""Research A v2-L0: contracts for no-teacher low-speed learning (no study claim)."""
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from a002_exploratory_ppo import ActorCritic, update_ppo
from a002v2_l0_low_speed_ppo import (
    ARMS,SEEDS,EPISODES,BATCH_EPISODES,CHECKPOINTS,
    TRAIN_INTERVALS_MS,EVAL_INTERVALS_MS,EVAL_STARTS_S,
    REPETITIONS,LoggedLowSpeedEnv,eval_seed,
    train_episode,evaluate_episode,evaluate_set,
)
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv


def test_frozen_low_speed_protocol():
    assert ARMS==("a0","a1") and SEEDS==(0,1)
    assert TRAIN_INTERVALS_MS==(200,260)
    assert EVAL_INTERVALS_MS==(210,240,250)
    assert EVAL_STARTS_S==(.835,1.075,1.290)
    assert EPISODES==256 and BATCH_EPISODES==8
    assert CHECKPOINTS==(0,128,256)
    assert REPETITIONS==4
    seeds=[eval_seed(i,r) for i in range(9) for r in range(REPETITIONS)]
    assert len(set(seeds))==36
    assert seeds[0]==320000 and seeds[-1]==320803
    with pytest.raises(ValueError):
        eval_seed(9,0)


def test_evaluation_has_no_finger_reward():
    left=LoggedLowSpeedEnv(interval_ms=200,start_s=.8,arm="a1")
    right=LoggedLowSpeedEnv(interval_ms=200,start_s=.8,arm="a1")
    r0=left._judge_keydown(.779,"left")
    r1=right._judge_keydown(.779,"right")
    assert r0==r1==pytest.approx(0.0)
    assert left.hits==right.hits==0
    assert left.extra_keydowns==right.extra_keydowns==1
    assert left.down_trace[0]["key"]=="left"
    assert right.down_trace[0]["key"]=="right"


def test_same_action_sequence_has_same_motor_judgement_as_original():
    original=NoPositiveNearMissEnv(interval_ms=260,start_s=.92,arm="a1")
    logged=LoggedLowSpeedEnv(interval_ms=260,start_s=.92,arm="a1")
    for i in range(560):
        command=(i//6+i//11)%4
        o0,r0,d0=original.step(command)
        o1,r1,d1=logged.step(command)
        assert o0==o1
        assert r0==pytest.approx(r1)
        assert d0==d1
        if d0:
            break
    assert original.result()==logged.result()
    assert len(logged.down_trace)==logged.physical_keydowns


@pytest.mark.parametrize("arm",("a0","a1"))
def test_fixed_seed_sampling_exactly_reproducible(arm):
    torch.manual_seed(4)
    model=ActorCritic()
    a=evaluate_episode(arm=arm,interval_ms=240,start_s=1.075,
                       mode="sampled",model=model,seed=320100)
    b=evaluate_episode(arm=arm,interval_ms=240,start_s=1.075,
                       mode="sampled",model=model,seed=320100)
    assert a==b
    assert a["targets"]==16
    assert a["physical_left"]+a["physical_right"]==a["physical_keydowns"]
    assert a["hits"]+a["extra_keydowns"]==a["physical_keydowns"]
    if arm=="a0":
        assert a["physical_right"]==0


def test_random_uniform_baseline_same_physics_not_a_teacher():
    a=evaluate_episode(arm="a1",interval_ms=210,start_s=.835,
                       mode="random",model=None,seed=320000)
    b=evaluate_episode(arm="a1",interval_ms=210,start_s=.835,
                       mode="random",model=None,seed=320000)
    assert a==b
    assert a["mode"]=="random"
    assert a["mean_action_entropy"] is None


def test_sampled_evaluation_reports_all_nine_cases_and_both_fingers():
    torch.manual_seed(0)
    model=ActorCritic()
    x=evaluate_set(arm="a1",model=model,mode="sampled")
    y=evaluate_set(arm="a0",model=model,mode="argmax")
    assert x["episodes"]==36 and x["targets"]==576
    assert y["episodes"]==9 and y["targets"]==144
    assert x["hits"]+x["misses"]==x["targets"]
    assert x["physical_left"]+x["physical_right"]==x["hits"]+x["extra_keydowns"]
    assert y["physical_right"]==0


def test_ppo_smoke_trains_from_scratch_without_finger_labels():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model=ActorCritic()
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    rng=random.Random(0)
    batch=[train_episode(model,arm="a0",rng=rng) for _ in range(2)]
    assert all(row["interval_ms"] in TRAIN_INTERVALS_MS for row in batch)
    assert all(row["score"]["targets"]==16 for row in batch)
    before=[p.detach().clone() for p in model.parameters()]
    losses=update_ppo(model,optimizer,batch)
    assert "entropy" in losses
    assert any(not torch.equal(x,y) for x,y in zip(before,model.parameters()))
