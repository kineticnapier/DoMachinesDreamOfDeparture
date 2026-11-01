"""Audit the original A002-P reward function without changing training.

The near-miss reward is allowed by the exploratory protocol, but may offer
positive net return for a note which was not successfully hit.
"""
from __future__ import annotations

from dmdod.research.a002_timing_only_env import TimingOnlyTwoFingerEnv, WINDOW_S, NEAR_WINDOW_S


def test_near_miss_is_rewarded_before_expiration_despite_no_hit():
    env=TimingOnlyTwoFingerEnv(arm="a1", interval_ms=100, start_s=.8)
    # .779s is 21ms early. 20ms judging window is not met, but 80ms
    # near reward is positive. This is a synthetic scoring test only.
    event_reward=env._judge_keydown(.779, "left")
    assert env.hits == 0
    assert env.extra_keydowns == 1
    assert event_reward > .10
    assert abs(event_reward - .15*(1-.021/.080)) < 1e-10

    later_expiration=env._expire_before(.821)
    assert env.hits == 0 and env.misses == 1
    assert later_expiration == -.10
    assert event_reward+later_expiration > 0


def test_reward_does_not_depend_on_finger_even_for_early_near_miss():
    env_l=TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.8)
    env_r=TimingOnlyTwoFingerEnv(arm="a1", interval_ms=120, start_s=.8)
    assert env_l._judge_keydown(.779,"left") == env_r._judge_keydown(.779,"right")
    assert env_l.hits == env_r.hits == 0


def test_judgement_window_and_near_reward_window_still_orig_values():
    assert WINDOW_S == .020
    assert NEAR_WINDOW_S == .080
