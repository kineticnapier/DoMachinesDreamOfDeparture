from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KeyEvent(str, Enum):
    DOWN = "down"
    UP = "up"


@dataclass(frozen=True)
class KeyConfig:
    actuation_m: float = 0.0020
    reset_m: float = 0.0018

    def __post_init__(self) -> None:
        if not 0.0 <= self.reset_m < self.actuation_m:
            raise ValueError("reset_m must be non-negative and below actuation_m")


@dataclass
class KeyState:
    pressed: bool = False


@dataclass
class TwoKeyKeyboard:
    config: KeyConfig = field(default_factory=KeyConfig)
    left: KeyState = field(default_factory=KeyState)
    right: KeyState = field(default_factory=KeyState)

    def reset(self) -> None:
        self.left = KeyState()
        self.right = KeyState()

    def step(self, left_position_m: float, right_position_m: float) -> list[tuple[str, KeyEvent]]:
        events: list[tuple[str, KeyEvent]] = []
        self._step_key("left", self.left, left_position_m, events)
        self._step_key("right", self.right, right_position_m, events)
        return events

    def _step_key(
        self,
        name: str,
        state: KeyState,
        position_m: float,
        events: list[tuple[str, KeyEvent]],
    ) -> None:
        if not state.pressed and position_m >= self.config.actuation_m:
            state.pressed = True
            events.append((name, KeyEvent.DOWN))
        elif state.pressed and position_m <= self.config.reset_m:
            state.pressed = False
            events.append((name, KeyEvent.UP))
