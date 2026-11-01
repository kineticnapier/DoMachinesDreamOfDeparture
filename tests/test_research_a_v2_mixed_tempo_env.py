"""Contract tests for v2 mixed-tempo evaluation; no RL training."""
from __future__ import annotations

from math import isclose
from pathlib import Path
import sys

import pytest
import torch

from dmdod.research.a002v2_mixed_tempo_env import MixedTempoMotorEnv
from a002_exploratory_ppo import ActorCritic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))


C0=(220,)*8+(120,)*8+(220,)*8


def test_exact_24_targets_and_two_boundaries():
    env=MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm="a1")
    assert len(env.targets)==24
    assert isclose(env.targets[1]-env.targets[0],.220,abs_tol=1e-9)
    assert isclose(env.targets[7]-env.targets[6],.220,abs_tol=1e-9)
    assert isclose(env.targets[8]-env.targets[7],.120,abs_tol=1e-9)
    assert isclose(env.targets[15]-env.targets[14],.120,abs_tol=1e-9)
    assert isclose(env.targets[16]-env.targets[15],.220,abs_tol=1e-9)
    assert len(env.observe())==8
    assert env.observe()[1] == pytest.approx(220/120-1.0)


def test_no_absolute_time_or_segment_id_in_observation():
    x=MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm="a1")
    y=MixedTempoMotorEnv(intervals_ms=C0,start_s=1.293,arm="a1")
    a=x.observe()
    b=y.observe()
    assert len(a)==len(b)==8
    assert all(isinstance(v,float) for v in a)
    assert all(isinstance(v,float) for v in b)
    # Both expose only the time-to-target (normalized), interval and physical body.
    assert a[0]==pytest.approx(4.)
    assert b[0]==pytest.approx(4.)
    assert a[1:]==b[1:]


@pytest.mark.parametrize("arm,disabled,expect",[
    ("a0",None,"left"),("a1","right","left"),("a1","left","right"),
])
def test_physical_finger_disable_never_produces_forbidden_key(arm,disabled,expect):
    env=MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm=arm,disabled_side=disabled)
    for i in range(700):
        _,_,done=env.step((i//7)%4)
        if done:
            break
    assert env.finished
    report=env.segmented_result()
    assert report["targets"]==24
    if expect=="left":
        assert report["right_physical_keydowns"]==0
    else:
        assert report["left_physical_keydowns"]==0


@pytest.mark.parametrize("arm",("a0","a1"))
def test_random_actions_preserve_exact_accounting(arm):
    env=MixedTempoMotorEnv(intervals_ms=C0,start_s=1.071,arm=arm,near_reward="r1")
    for i in range(600):
        _,_,done=env.step((i*11+i//13)%4)
        if done:
            break
    assert env.finished
    r=env.segmented_result()
    assert r["hits"]+r["misses"]==24
    assert r["hits"]+r["extra_keydowns"]==r["physical_keydowns"]
    assert sum(s["hits"] for s in r["segments"])==r["hits"]
    assert sum(s["misses"] for s in r["segments"])==r["misses"]
    assert sum(s["extra_keydowns_nearest"] for s in r["segments"])==r["extra_keydowns"]


def test_simultaneous_physical_keys_have_distinct_extra_or_hit_records():
    env=MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm="a1")
    while not env.finished:
        env.step(3)
    r=env.segmented_result()
    assert r["physical_keydowns"]==r["left_physical_keydowns"]+r["right_physical_keydowns"]
    assert r["physical_keydowns"]==len(r["hits_trace"])+len(r["extras_trace"])


def test_fixed_action_list_is_deterministic_and_reward_mode_is_not_physics():
    modes=[]
    for mode in ("r0","r1"):
        env=MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm="a1",near_reward=mode)
        trajectory=[]
        for i in range(700):
            action=(i//5+i//19)%4
            obs,reward,done=env.step(action)
            trajectory.append((obs,done))
            if done:
                break
        modes.append((trajectory,env.segmented_result()))
    a,b=modes
    assert a[0]==b[0]
    for name in ("hits","misses","extra_keydowns","physical_keydowns","hits_trace","extras_trace","segments"):
        assert a[1][name]==b[1][name]


def test_invalid_schedule_and_wrong_arm_rejected():
    with pytest.raises(ValueError):
        MixedTempoMotorEnv(intervals_ms=C0[:16],start_s=.837,arm="a1")
    with pytest.raises(ValueError):
        MixedTempoMotorEnv(intervals_ms=C0,start_s=.837,arm="a0",disabled_side="left")
