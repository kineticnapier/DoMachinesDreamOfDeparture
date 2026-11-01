"""Contract tests for teacher-free research A002-P.

These tests do NOT show that fingering emerged: only that training does not
receive target finger labels and that the A0 restriction is physical.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dmdod.research.a002_timing_only_env import (
    TimingOnlyTwoFingerEnv, ACTION_COMMANDS, OBS_DIM,
)
from a002_exploratory_ppo import (
    ARMS, SEEDS, EPISODES, TRAIN_INTERVALS_MS, EVAL_INTERVALS_MS,
    EVAL_STARTS_S, discounted_returns, evaluate, ActorCritic, run_seed,
)

import torch


def test_input_is_relative_timing_and_body_only_no_key_target_or_absolute_time():
    env = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=110, start_s=.93)
    x = env.reset()
    assert len(x) == OBS_DIM == 8
    assert isinstance(x, tuple)
    assert abs(x[0] - .93 / .11) > 0  # clipped, not absolute time
    assert not any(isinstance(v, str) for v in x)
    assert len(env.targets) == 16
    assert all(isinstance(t, float) for t in env.targets)


def test_timing_reward_exactly_invariant_under_key_renaming():
    left = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.84)
    right = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.84)
    reward_l = left._judge_keydown(.849, "left")
    reward_r = right._judge_keydown(.849, "right")
    assert reward_l == reward_r
    assert left.hits == right.hits == 1
    assert left.misses == right.misses == 0
    assert left.matched_keys == ["left"]
    assert right.matched_keys == ["right"]


def test_near_miss_reward_independent_of_finger_and_does_not_register_hit():
    left = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.84)
    right = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.84)
    assert left._judge_keydown(.79, "left") == right._judge_keydown(.79, "right")
    assert left.hits == right.hits == 0
    assert left.extra_keydowns == right.extra_keydowns == 1


def test_a0_second_finger_cannot_become_pressed():
    env = TimingOnlyTwoFingerEnv(arm="a0", interval_ms=100, start_s=.8)
    obs = env.reset()
    for _ in range(160):
        obs, _, done = env.step(3)  # both fingers requested downward
        assert not env.motor.sim.keyboard.right.pressed
        assert obs[7] == 0.0
        if done:
            break


def test_extras_and_expiry_cannot_be_awarded_as_hits():
    env = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=100, start_s=.8)
    env._judge_keydown(.60, "left")
    assert env.hits == 0 and env.extra_keydowns == 1
    score = env._expire_before(.95)
    assert score < 0
    assert env.misses >= 1


def test_reproducible_empty_policy_rollout():
    a = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=110, start_s=.82)
    b = TimingOnlyTwoFingerEnv(arm="a1", interval_ms=110, start_s=.82)
    path = (0,1,2,3)*70
    scores = []
    for env in (a,b):
        env.reset()
        for command in path:
            _,_,done=env.step(command)
            if done:
                break
        while not env.finished:
            env.step(0)
        scores.append(env.result())
    assert scores[0] == scores[1]


def test_train_and_eval_splits_not_overlapping():
    assert ARMS == ("a0", "a1")
    assert SEEDS == (0,1)
    assert EPISODES == 96
    assert TRAIN_INTERVALS_MS == (100,110,120,130)
    assert EVAL_INTERVALS_MS == (105,115,125)
    assert not set(TRAIN_INTERVALS_MS).intersection(EVAL_INTERVALS_MS)
    assert EVAL_STARTS_S == (.83,.97,1.21)


def test_ppo_discounted_returns_contract():
    value = discounted_returns([0., 0., 1.])
    assert len(value) == 3
    assert value[0] > 0
    assert value[-1] == 1


def test_actor_uses_identical_architecture_across_arms():
    torch.manual_seed(0)
    a = ActorCritic()
    torch.manual_seed(0)
    b = ActorCritic()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_zero_train_evaluation_and_no_finger_metadata_in_reward():
    torch.manual_seed(8)
    model = ActorCritic()
    baseline = evaluate(model,arm="a0")
    assert baseline["episodes"] == 9
    assert baseline["targets"] == 144
    assert baseline["hits"] <= baseline["targets"]
    assert isinstance(baseline["rows"],list)


def test_smoke_real_ppo_update_and_checksum(tmp_path):
    torch.set_num_threads(1)
    result = run_seed(arm="a0", seed=0, episodes=8, checkpoint_dir=tmp_path)
    assert result["episodes"] == 8
    assert len(result["evaluations"]) == 2
    assert len(result["training_blocks"]) == 1
    assert len(result["checkpoint_sha256"]) == 64
    assert (tmp_path/result["checkpoint_file"]).exists()
