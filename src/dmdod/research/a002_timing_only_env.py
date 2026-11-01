"""Research A002-P: time-only scoring over the existing physical two-finger Motor.

No target specifies a finger. Only the evaluator accesses due timestamps; the
policy receives relative time to the next unresolved target as an explicitly
permitted timing cue, plus physical state. Absolute time is never observed.

This is an exploratory RL setup; it is NOT the preregistered full study 002.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from dmdod.motor.env import MotorAction, MotorEnv, MotorObservation
from dmdod.motor.keyboard import KeyEvent

ACTION_COMMANDS = (
    MotorAction(-1.0, -1.0),
    MotorAction(-1.0, +1.0),
    MotorAction(+1.0, -1.0),
    MotorAction(+1.0, +1.0),
)
OBS_DIM = 8
WINDOW_S = 0.020
NEAR_WINDOW_S = 0.080
COUNT = 16
CONTROL_DT_S = 0.010


@dataclass(frozen=True)
class EpisodeResult:
    hits: int
    targets: int
    misses: int
    extra_keydowns: int
    physical_keydowns: int
    full_hit: bool
    switch_fraction: float | None
    matched_keys: tuple[str, ...]
    reward_sum: float


class TimingOnlyTwoFingerEnv:
    """Goal-time-only physical rhythm experiment.

    arm 'a0': second finger is physically disabled; otherwise same network,
    key-independent rewards, and observation shape as arm 'a1'.
    """

    def __init__(
        self, *, interval_ms: float, start_s: float, arm: str,
        targets_count: int = COUNT
    ) -> None:
        if arm not in ("a0", "a1"):
            raise ValueError("arm must be 'a0' or 'a1'")
        if not (isfinite(interval_ms) and 0 < interval_ms):
            raise ValueError("positive finite interval_ms required")
        if not (isfinite(start_s) and 0.3 <= start_s <= 2.0):
            raise ValueError("valid start_s required")
        if targets_count < 3:
            raise ValueError("at least three targets required")
        self.arm = arm
        self.interval_s = float(interval_ms) / 1000.0
        self.start_s = float(start_s)
        self.targets = tuple(self.start_s + i * self.interval_s for i in range(targets_count))
        self.motor = MotorEnv(same_hand=True, control_dt_s=CONTROL_DT_S)
        self.reset()

    def reset(self) -> tuple[float, ...]:
        self.motor.reset()
        self.cursor = 0
        self.hits = 0
        self.misses = 0
        self.extra_keydowns = 0
        self.physical_keydowns = 0
        self.reward_sum = 0.0
        self.matched_keys: list[str] = []
        self.finished = False
        return self.observe()

    def observe(self) -> tuple[float, ...]:
        state: MotorObservation = self.motor.observe()
        # A timing cue, not absolute time and not a fingering prescription.
        if self.cursor < len(self.targets):
            remaining = self.targets[self.cursor] - self.motor.sim.time_s
        else:
            remaining = 0.0
        normalized_remaining = max(-2.0, min(4.0, remaining / self.interval_s))
        return (
            normalized_remaining,
            self.interval_s / .12 - 1.0,
            state.left_position_m * 500.0,
            state.right_position_m * 500.0,
            state.left_velocity_m_s * 10.0,
            state.right_velocity_m_s * 10.0,
            float(state.left_pressed),
            float(state.right_pressed),
        )

    def _expire_before(self, time_s: float) -> float:
        reward = 0.0
        while (self.cursor < len(self.targets)
               and self.targets[self.cursor] + WINDOW_S < time_s - 1e-10):
            self.misses += 1
            self.cursor += 1
            reward -= .10
        return reward

    def _judge_keydown(self, event_time_s: float, key: str) -> float:
        """Completely key-invariant reward and hit assignment."""
        self.physical_keydowns += 1
        reward = self._expire_before(event_time_s)
        if self.cursor < len(self.targets):
            error = event_time_s - self.targets[self.cursor]
            if abs(error) <= WINDOW_S + 1e-10:
                self.hits += 1
                self.matched_keys.append(key)  # logging only; not a reward
                self.cursor += 1
                return reward + 1.0 - .15 * abs(error) / WINDOW_S
            if abs(error) <= NEAR_WINDOW_S:
                # Dense timing information only; no axis identity involved.
                self.extra_keydowns += 1
                return reward + .15 * (1.0 - abs(error) / NEAR_WINDOW_S)
        self.extra_keydowns += 1
        return reward - .05

    def step(self, action_index: int) -> tuple[tuple[float, ...], float, bool]:
        if self.finished:
            raise RuntimeError("episode already finished")
        if not 0 <= action_index < len(ACTION_COMMANDS):
            raise ValueError("action out of range")
        desired = ACTION_COMMANDS[action_index]
        action = (desired if self.arm == "a1"
                  else MotorAction(desired.left, -1.0))
        physical = self.motor.step(action)
        reward = 0.0
        for event in physical.evaluator_events:
            if event.event is KeyEvent.DOWN:
                reward += self._judge_keydown(event.time_s, event.key)
        reward += self._expire_before(self.motor.sim.time_s)
        if self.motor.sim.time_s >= self.targets[-1] + WINDOW_S + .100:
            reward += self._expire_before(float("inf"))
            self.finished = True
        self.reward_sum += reward
        return self.observe(), reward, self.finished

    def result(self) -> EpisodeResult:
        if not self.finished:
            raise RuntimeError("episode must finish before scoring")
        switches = sum(
            a != b for a, b in zip(self.matched_keys, self.matched_keys[1:])
        )
        return EpisodeResult(
            hits=self.hits, targets=len(self.targets), misses=self.misses,
            extra_keydowns=self.extra_keydowns,
            physical_keydowns=self.physical_keydowns,
            full_hit=self.hits == len(self.targets) and self.extra_keydowns == 0,
            switch_fraction=(
                switches / (len(self.matched_keys) - 1)
                if len(self.matched_keys) >= 2 else None
            ),
            matched_keys=tuple(self.matched_keys),
            reward_sum=self.reward_sum,
        )
