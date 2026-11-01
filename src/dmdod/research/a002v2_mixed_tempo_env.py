"""Research A v2: mixed-tempo evaluator on the unchanged physical two-finger Motor.

This adapter exposes no key labels or tempo-segment index to the Student.
Relative time-to-next-target is an explicitly allowed timing-only cue.
The inherited KeyDown judge keeps the original A002-P matching semantics.
"""
from __future__ import annotations

from math import isfinite

from dmdod.motor.env import MotorAction, MotorEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.research.a002_timing_only_env import (
    TimingOnlyTwoFingerEnv, ACTION_COMMANDS, CONTROL_DT_S,
    WINDOW_S, NEAR_WINDOW_S,
)


class MixedTempoMotorEnv(TimingOnlyTwoFingerEnv):
    def __init__(self, *, intervals_ms: tuple[int, ...], start_s: float,
                 arm: str, disabled_side: str | None = None,
                 near_reward: str = "r0"):
        if len(intervals_ms) != 24:
            raise ValueError("three 8-target blocks are required")
        if any(not isfinite(x) or x < 60 or x > 400 for x in intervals_ms):
            raise ValueError("all intervals must be finite 60..400 ms")
        if not isfinite(start_s) or not 0.3 <= start_s <= 2.0:
            raise ValueError("invalid start time")
        if arm not in ("a0","a1"):
            raise ValueError("invalid arm")
        if disabled_side not in (None,"left","right"):
            raise ValueError("invalid disabled side")
        if arm == "a0" and disabled_side not in (None,"right"):
            raise ValueError("a0 is physically single-left, not single-right")
        if near_reward not in ("r0","r1"):
            raise ValueError("invalid near reward mode")
        self.arm = arm
        self.disabled_side = disabled_side
        self.near_reward = near_reward
        self.start_s = float(start_s)
        self.interval_s = 0.12  # legacy field; observe() uses per-target intervals
        self.target_intervals_s = tuple(float(x)/1000.0 for x in intervals_ms)
        targets = [self.start_s]
        for gap in self.target_intervals_s[1:]:
            targets.append(targets[-1] + gap)
        self.targets = tuple(targets)
        self.motor = MotorEnv(same_hand=True, control_dt_s=CONTROL_DT_S)
        self.reset()

    def reset(self):
        self.hit_records: list[dict] = []
        self.extra_records: list[dict] = []
        return super().reset()

    def observe(self) -> tuple[float,...]:
        state = self.motor.observe()
        index = min(self.cursor, len(self.targets)-1)
        gap = self.target_intervals_s[index]
        remaining = (self.targets[self.cursor] - self.motor.sim.time_s
                     if self.cursor < len(self.targets) else 0.0)
        return (
            max(-2.0, min(4.0, remaining / gap)),
            gap / .12 - 1.0,
            state.left_position_m*500.0,
            state.right_position_m*500.0,
            state.left_velocity_m_s*10.0,
            state.right_velocity_m_s*10.0,
            float(state.left_pressed),
            float(state.right_pressed),
        )

    def _judge_keydown(self, event_time_s: float, key: str) -> float:
        prior_hits = self.hits
        reward = super()._judge_keydown(event_time_s, key)
        if self.hits != prior_hits:
            ordinal = self.cursor-1
            self.hit_records.append({
                "ordinal": ordinal,
                "time_s": event_time_s,
                "key": key,
                "error_ms": (event_time_s-self.targets[ordinal])*1000.0,
            })
        else:
            nearest = min(range(len(self.targets)),
                          key=lambda i:abs(self.targets[i]-event_time_s))
            self.extra_records.append({
                "time_s": event_time_s, "key": key,
                "nearest_target": nearest,
                "nearest_error_ms": (event_time_s-self.targets[nearest])*1000.0,
            })
            if self.near_reward == "r1" and self.cursor < len(self.targets):
                error = event_time_s-self.targets[self.cursor]
                if WINDOW_S < abs(error) <= NEAR_WINDOW_S:
                    reward -= .15*(1.0-abs(error)/NEAR_WINDOW_S)
        return reward

    def step(self, action_index: int):
        if self.finished:
            raise RuntimeError("episode already finished")
        if not 0 <= action_index < len(ACTION_COMMANDS):
            raise ValueError("invalid action")
        desired=ACTION_COMMANDS[action_index]
        if self.arm == "a0" or self.disabled_side == "right":
            action=MotorAction(desired.left,-1.0)
        elif self.disabled_side == "left":
            action=MotorAction(-1.0,desired.right)
        else:
            action=desired
        physical=self.motor.step(action)
        reward=0.0
        for event in physical.evaluator_events:
            if event.event is KeyEvent.DOWN:
                reward+=self._judge_keydown(event.time_s,event.key)
        reward+=self._expire_before(self.motor.sim.time_s)
        if self.motor.sim.time_s >= self.targets[-1]+WINDOW_S+.100:
            reward+=self._expire_before(float("inf"))
            self.finished=True
        self.reward_sum+=reward
        return self.observe(),reward,self.finished

    def segmented_result(self) -> dict:
        score=self.result()
        if self.hits+ self.misses != len(self.targets):
            raise AssertionError("missing or double-counted targets")
        if self.physical_keydowns != self.hits+self.extra_keydowns:
            raise AssertionError("missing or double-counted physical KeyDowns")
        hits_by_ord={r["ordinal"] for r in self.hit_records}
        if len(hits_by_ord) != self.hits:
            raise AssertionError("same target matched more than once")
        regions=[]
        for segment in range(3):
            first=8*segment
            end=first+8
            rows=[r for r in self.hit_records if first<=r["ordinal"]<end]
            extras=[r for r in self.extra_records
                    if first<=r["nearest_target"]<end]
            regions.append({
                "segment": segment,
                "targets":8,
                "hits":len(rows),
                "misses":8-len(rows),
                # Attribution is to the closest target in time and not to a Hit.
                "extra_keydowns_nearest":len(extras),
                "left_hits":sum(r["key"]=="left" for r in rows),
                "right_hits":sum(r["key"]=="right" for r in rows),
            })
        if sum(s["hits"] for s in regions) != self.hits:
            raise AssertionError("hits missing from segment partition")
        if sum(s["extra_keydowns_nearest"] for s in regions)!=self.extra_keydowns:
            raise AssertionError("extra KeyDowns missing from segment partition")
        return {
            "arm": self.arm,
            "disabled_side": self.disabled_side,
            "near_reward": self.near_reward,
            "hits": score.hits,
            "misses": score.misses,
            "targets": score.targets,
            "extra_keydowns": score.extra_keydowns,
            "physical_keydowns": score.physical_keydowns,
            "full_hit": score.full_hit,
            "matched_keys": "".join("L" if k=="left" else "R"
                                    for k in score.matched_keys),
            "left_physical_keydowns":sum(r["key"]=="left" for r in self.hit_records+self.extra_records),
            "right_physical_keydowns":sum(r["key"]=="right" for r in self.hit_records+self.extra_records),
            "segments":regions,
            "hits_trace":self.hit_records,
            "extras_trace":self.extra_records,
        }
