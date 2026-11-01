"""A002-P-R1 preregistered reward-only isolation tests."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from a002_exploratory_ppo import (
    ActorCritic, collect_episode, evaluate, run_seed,
    ARMS, SEEDS, EPISODES, TRAIN_INTERVALS_MS,
    EVAL_INTERVALS_MS, EVAL_STARTS_S,
)
from a002p_reward_ablation_train import (
    MODES, check_original_r0, brief_report,
)
from dmdod.research.a002_timing_only_env import (
    TimingOnlyTwoFingerEnv, WINDOW_S, NEAR_WINDOW_S,
)
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv


def test_protocol_remains_original_small_seed_budget_and_exact_action_model():
    assert ARMS == ("a0","a1")
    assert SEEDS == (0,1)
    assert EPISODES == 96
    assert TRAIN_INTERVALS_MS == (100,110,120,130)
    assert EVAL_INTERVALS_MS == (105,115,125)
    assert EVAL_STARTS_S == (.83,.97,1.21)
    assert len(MODES) == 2
    assert WINDOW_S == .020 and NEAR_WINDOW_S == .080


@pytest.mark.parametrize("when", (.721, .779, .801, .819, .821, .871, .882))
@pytest.mark.parametrize("key", ("left", "right"))
def test_reward_changes_only_for_near_miss_without_state_change(when,key):
    orig=TimingOnlyTwoFingerEnv(arm="a1",interval_ms=100,start_s=.8)
    new=NoPositiveNearMissEnv(arm="a1",interval_ms=100,start_s=.8)
    base=orig._judge_keydown(when,key)
    ablated=new._judge_keydown(when,key)
    assert orig.hits == new.hits
    assert orig.misses == new.misses
    assert orig.extra_keydowns == new.extra_keydowns
    assert orig.physical_keydowns == new.physical_keydowns
    assert orig.cursor == new.cursor
    assert orig.matched_keys == new.matched_keys
    # No added reward in the R1 implementation.
    assert ablated <= base+1e-10


def test_failed_keydown_cannot_be_locally_net_positive_under_r1():
    old=TimingOnlyTwoFingerEnv(arm="a1", interval_ms=100,start_s=.8)
    new=NoPositiveNearMissEnv(arm="a1", interval_ms=100,start_s=.8)
    old_step=old._judge_keydown(.779,"right")
    new_step=new._judge_keydown(.779,"right")
    assert old_step == pytest.approx(.110625)
    assert new_step == pytest.approx(0.0)
    assert old_step + old._expire_before(.821) > 0
    assert new_step + new._expire_before(.821) == pytest.approx(-.1)
    assert old.hits == new.hits == 0
    assert old.extra_keydowns == new.extra_keydowns == 1


@pytest.mark.parametrize("arm",ARMS)
def test_identical_action_sequence_has_identical_physics_judgements(arm):
    environments=[
        TimingOnlyTwoFingerEnv(arm=arm,interval_ms=110,start_s=.96),
        NoPositiveNearMissEnv(arm=arm,interval_ms=110,start_s=.96)
    ]
    timeline=[]
    for env in environments:
        obs=env.reset()
        data=[]
        for i in range(360):
            # Not a finger or timing teacher: identical arbitrary action sequence.
            command=[0,1,2,3,0,2,0,3,1][i%9]
            obs,reward,done=env.step(command)
            data.append((
                env.motor.sim.time_s,
                env.motor.observe(),
                env.hits,env.misses,env.extra_keydowns,
                tuple(env.matched_keys),
                env.physical_keydowns,
                done
            ))
            if done:
                break
        timeline.append(data)
    assert timeline[0] == timeline[1]


def test_scoring_does_not_favor_left_over_right():
    for env_type in (TimingOnlyTwoFingerEnv,NoPositiveNearMissEnv):
        left=env_type(arm="a1",interval_ms=100,start_s=.8)
        right=env_type(arm="a1",interval_ms=100,start_s=.8)
        assert left._judge_keydown(.779,"left") == right._judge_keydown(.779,"right")
        assert left.hits == right.hits
        assert left.extra_keydowns == right.extra_keydowns


def test_r0_environment_injection_does_not_change_baseline(tmp_path):
    # Both are freshly seeded; no teacher data. 8 episodes is an infrastructure
    # check and must not be presented as pilot results.
    torch.set_num_threads(1)
    default=run_seed(arm="a1",seed=0,episodes=8,checkpoint_dir=tmp_path/"default")
    injected=run_seed(
        arm="a1",seed=0,episodes=8,checkpoint_dir=tmp_path/"injected",
        env_cls=TimingOnlyTwoFingerEnv,
    )
    assert default["evaluations"] == injected["evaluations"]
    assert default["training_blocks"] == injected["training_blocks"]
    assert default["checkpoint_sha256"] == injected["checkpoint_sha256"]


def test_original_comparison_is_at_episode_and_per_episode_detail():
    example={
        "arm":"a1","seed":0,
        "evaluations":[
            {"episode":0,"evaluation":{
                "hits":1,"targets":16,"extra_keydowns":0,
                "full_hit_episodes":0,
                "rows":[{
                    "interval_ms":105,"start_s":.83,
                    "hits":1,"targets":16,
                    "extra_keydowns":0,
                    "physical_keydowns":1,
                    "matched_keys":"L","full_hit":False
                }]
            }}
        ]
    }
    assert check_original_r0(example,example) == []
    diff={
        **example,
        "evaluations":[{
            **example["evaluations"][0],
            "evaluation":{**example["evaluations"][0]["evaluation"],"rows":[{
                **example["evaluations"][0]["evaluation"]["rows"][0],
                "matched_keys":"R",
            }]}
        }]
    }
    issues=check_original_r0(diff,example)
    assert len(issues)==1 and issues[0]["metric"]=="matched_keys"


def test_intervention_and_objective_are_not_passing_teacher_labels():
    env=NoPositiveNearMissEnv(arm="a1",interval_ms=120,start_s=.9)
    observation=env.reset()
    assert len(observation)==8
    assert all(isinstance(x,float) for x in observation)
    assert isinstance(env.targets[0],float)
    assert not any("finger" in x for x in env.__dict__.keys())
