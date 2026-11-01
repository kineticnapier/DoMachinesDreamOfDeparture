"""A002-P-R1: controlled removal of positive reward for an unhit KeyDown.

Does not change the original TimingOnlyTwoFingerEnv or Motor at all.
The inherited evaluator performs all state transitions and hit/extras counting.
Only the positive near-miss scalar is subtracted from its returned reward.
"""
from __future__ import annotations

from dmdod.research.a002_timing_only_env import (
    TimingOnlyTwoFingerEnv,
    NEAR_WINDOW_S,
    WINDOW_S,
)


class NoPositiveNearMissEnv(TimingOnlyTwoFingerEnv):
    """Exact A002-P environment with the single near-miss reward term removed."""

    def _judge_keydown(self, event_time_s: float, key: str) -> float:
        hits_before = self.hits
        original_reward = super()._judge_keydown(event_time_s, key)
        if self.hits != hits_before:
            return original_reward
        if self.cursor >= len(self.targets):
            return original_reward
        error = event_time_s - self.targets[self.cursor]
        if WINDOW_S < abs(error) <= NEAR_WINDOW_S:
            return original_reward - 0.15 * (1.0 - abs(error) / NEAR_WINDOW_S)
        return original_reward
