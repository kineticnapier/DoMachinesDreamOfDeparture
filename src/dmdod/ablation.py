from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import random

from .perception import VisualCueObservation


class CueAblationMode(str, Enum):
    """Policy-side cue perturbations used only for offline evaluation."""

    NORMAL = "normal"
    ZERO = "zero"
    RANDOM = "random"
    DELAY = "delay"


@dataclass
class CueAblator:
    """Transform agent-visible visual cues without changing hidden game truth.

    The evaluator/environment still scores against the original targets.  Only
    the cue presented to the policy is changed, so these modes can test whether
    a trained policy actually depends on the visual timing signal.
    """

    mode: CueAblationMode | str = CueAblationMode.NORMAL
    control_dt_s: float = 0.010
    delay_s: float = 0.100
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _delay_steps: int = field(init=False, repr=False)
    _queue: deque[VisualCueObservation] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = CueAblationMode(self.mode)
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if self.delay_s < 0.0:
            raise ValueError("delay_s must be non-negative")
        self._rng = random.Random(self.seed)
        self._delay_steps = int(round(self.delay_s / self.control_dt_s))
        self._queue = deque()

    def transform(self, cue: VisualCueObservation) -> VisualCueObservation:
        if self.mode is CueAblationMode.NORMAL:
            return cue
        if self.mode is CueAblationMode.ZERO:
            return VisualCueObservation(0.0, 0.0)
        if self.mode is CueAblationMode.RANDOM:
            return VisualCueObservation(self._rng.random(), self._rng.random())
        if self.mode is CueAblationMode.DELAY:
            if self._delay_steps == 0:
                return cue
            self._queue.append(cue)
            if len(self._queue) <= self._delay_steps:
                return VisualCueObservation(0.0, 0.0)
            return self._queue.popleft()
        raise AssertionError(f"unhandled cue ablation mode: {self.mode}")
