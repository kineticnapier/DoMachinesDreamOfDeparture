from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from dmdod.motor.body import FingerState, TwoFingerBody, _finite_body_value
from dmdod.motor.keyboard import KeyEvent, TwoKeyKeyboard


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.001

    def __post_init__(self) -> None:
        dt = _finite_body_value(self.dt_s, error="dt_s must be finite and positive")
        if dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")


@dataclass(frozen=True)
class StepResult:
    time_s: float
    left: FingerState
    right: FingerState
    events: tuple[tuple[str, KeyEvent], ...]


@dataclass
class Simulation:
    config: SimulationConfig = field(default_factory=SimulationConfig)
    body: TwoFingerBody = field(default_factory=TwoFingerBody)
    keyboard: TwoKeyKeyboard = field(default_factory=TwoKeyKeyboard)
    time_s: float = 0.0

    def reset(self) -> None:
        self.time_s = 0.0
        self.body.reset()
        self.keyboard.reset()

    def step(self, left_command: float, right_command: float) -> StepResult:
        left, right = self.body.step(left_command, right_command, self.config.dt_s)
        events = self.keyboard.step(left.position_m, right.position_m)
        self.time_s += self.config.dt_s
        return StepResult(self.time_s, left, right, tuple(events))

    def run_constant(self, left_command: float, right_command: float, duration_s: float) -> list[StepResult]:
        duration = _finite_body_value(duration_s, error="duration_s must be finite and non-negative")
        if duration < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        ratio = duration / self.config.dt_s
        if not isfinite(ratio):
            raise ValueError("duration_s / dt_s must be finite")
        steps = round(ratio)
        return [self.step(left_command, right_command) for _ in range(steps)]
