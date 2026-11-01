"""Research A v2-L0-R2: all physical extra KeyDowns receive -0.05.

R1's unused near-miss reward is zero for 20..80ms. This is the ONLY
change to scoring: near-miss extra KeyDowns now get -0.05, like far extras.
No physical Motor, judgement, observation or fingering teacher changes.
"""
from __future__ import annotations

from a002v2_l0_low_speed_ppo import LoggedLowSpeedEnv
from dmdod.research.a002_timing_only_env import WINDOW_S, NEAR_WINDOW_S

EXTRA_PENALTY = .05


class PenalizedLoggedLowSpeedEnv(LoggedLowSpeedEnv):
    def reset(self):
        self.near_extra_penalties = 0
        return super().reset()

    def _judge_keydown(self, event_time_s: float, key: str) -> float:
        old_extras = self.extra_keydowns
        reward = super()._judge_keydown(event_time_s, key)
        if self.extra_keydowns != old_extras:
            # The inherited R1 scorer has already performed all transitions.
            # Due may have expired during its call, so inspect the *current*
            # unresolved target, just as R1's reward logic does.
            if self.cursor < len(self.targets):
                error = event_time_s - self.targets[self.cursor]
                if WINDOW_S < abs(error) <= NEAR_WINDOW_S:
                    reward -= EXTRA_PENALTY
                    self.near_extra_penalties += 1
                    # Keep diagnostic trace consistent with the actual reward.
                    self.down_trace[-1]["immediate_reward"] = round(reward, 8)
        return reward
