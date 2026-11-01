"""Research A v2-L0-R2: reward-only isolation and archive fidelity tests."""
from __future__ import annotations

from pathlib import Path
import random
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv
from a002v2_l0_r2_penalty_compare import (
    ORIGINAL_RUN, ORIGINAL_ARTIFACT, checked_original,
    assert_original_evaluation,
)


def make_env(cls, arm="a1", start=.8, interval=200):
    return cls(arm=arm,start_s=start,interval_ms=interval)


def test_protocol_and_original_source_are_fixed():
    assert ORIGINAL_RUN==41309837474
    assert ORIGINAL_ARTIFACT==13084957586
    assert l0.ARMS==("a0","a1")
    assert l0.SEEDS==(0,1)
    assert l0.TRAIN_INTERVALS_MS==(200,260)
    assert l0.EVAL_INTERVALS_MS==(210,240,250)
    assert l0.EPISODES==256
    assert l0.CHECKPOINTS==(0,128,256)
    assert l0.REPETITIONS==4


@pytest.mark.parametrize("key",("left","right"))
def test_near_miss_reward_is_exactly_minus_point_zero_five(key):
    r1,r2=make_env(l0.LoggedLowSpeedEnv),make_env(PenalizedLoggedLowSpeedEnv)
    old=r1._judge_keydown(.779,key)  # 21ms early: outside the 20ms hit window
    new=r2._judge_keydown(.779,key)
    assert old==pytest.approx(0.)
    assert new==pytest.approx(-.05)
    assert r2.near_extra_penalties==1
    assert r1.extra_keydowns==r2.extra_keydowns==1
    assert r1.hits==r2.hits==0
    assert r2.down_trace[-1]["immediate_reward"]==pytest.approx(-.05)


def test_outside_near_window_and_real_hit_reward_are_unchanged():
    for when in (.600,.800,.801,1.010):
        r1,r2=make_env(l0.LoggedLowSpeedEnv),make_env(PenalizedLoggedLowSpeedEnv)
        a=r1._judge_keydown(when,"left")
        b=r2._judge_keydown(when,"left")
        assert a==pytest.approx(b)
        assert r1.hits==r2.hits
        assert r1.extra_keydowns==r2.extra_keydowns
        assert r2.near_extra_penalties==0


@pytest.mark.parametrize("arm",("a0","a1"))
def test_fixed_controls_produce_identical_physics_and_exact_reward_delta(arm):
    r1,r2=make_env(l0.LoggedLowSpeedEnv,arm=arm),make_env(PenalizedLoggedLowSpeedEnv,arm=arm)
    cumulative=0.
    for step in range(550):
        action=(step//3+step//11)%4
        o1,reward1,done1=r1.step(action)
        o2,reward2,done2=r2.step(action)
        assert o1==o2
        assert done1==done2
        assert r1.motor.observe()==r2.motor.observe()
        assert r1.hits==r2.hits
        assert r1.misses==r2.misses
        assert r1.extra_keydowns==r2.extra_keydowns
        assert r1.physical_keydowns==r2.physical_keydowns
        assert tuple(r1.matched_keys)==tuple(r2.matched_keys)
        cumulative+=reward2-reward1
        if done1:
            break
    assert r1.finished and r2.finished
    assert cumulative==pytest.approx(-.05*r2.near_extra_penalties,abs=1e-7)
    assert r2.reward_sum-r1.reward_sum==pytest.approx(-.05*r2.near_extra_penalties,abs=1e-7)
    assert [x["key"] for x in r1.down_trace]==[x["key"] for x in r2.down_trace]
    assert [x["time_s"] for x in r1.down_trace]==[x["time_s"] for x in r2.down_trace]


def test_ppo_module_can_train_r2_with_no_network_or_trainer_changes():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model=l0.ActorCritic()
    with patch.object(l0,"LoggedLowSpeedEnv",PenalizedLoggedLowSpeedEnv):
        episode=l0.train_episode(model,arm="a1",rng=random.Random(0))
        score=episode["score"]
        assert score["targets"]==16
        assert len(episode["actions"])>0
        assert score["hits"]+score["misses"]==16
        optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
        before=[p.detach().clone() for p in model.parameters()]
        stats=l0.update_ppo(model,optimizer,[episode])
        assert "entropy" in stats
        assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))


def test_original_evaluation_detects_per_episode_key_tampering():
    example={
        "hits":1,"misses":15,"targets":16,"extra_keydowns":1,
        "physical_left":2,"physical_right":0,
        "full_hit_episodes":0,"episodes":1,
        "rows":[{
            "interval_ms":210,"start_s":.835,"hits":1,"misses":15,
            "targets":16,"extra_keydowns":1,"physical_keydowns":2,
            "full_hit":False,"matched_keys":"L",
            "physical_left":2,"physical_right":0,
            "controls":[4,5,6,7],"steps":22,
            "physical_trace":[{"key":"left","time_s":.83}],
            "sampling_seed":320000,
        }]
    }
    assert_original_evaluation(example,example,"ok")
    bad={**example,"rows":[{**example["rows"][0],"matched_keys":"R"}]}
    with pytest.raises(AssertionError,match="matched_keys"):
        assert_original_evaluation(bad,example,"tampered")


def test_original_missing_report_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        checked_original(tmp_path)
