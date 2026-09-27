from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .evaluator import TargetHit


@dataclass(frozen=True)
class VisualCueObservation:
    """Agent-visible visual channels with no explicit timestamp information."""

    left: float
    right: float


@dataclass(frozen=True)
class VisualCueConfig:
    """Toy visual-perception model for the first RL experiments.

    Exact target timestamps remain private.  The policy sees only smooth cue
    intensities, analogous to seeing an object approach a hit position.  This is
    deliberately a toy perception layer, not yet an ADOFAI renderer model.
    """

    latency_s: float = 0.050
    width_s: float = 0.090
    horizon_s: float = 0.450

    def __post_init__(self) -> None:
        if self.latency_s < 0.0:
            raise ValueError("latency_s must be non-negative")
        if self.width_s <= 0.0:
            raise ValueError("width_s must be positive")
        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")


class VisualCueEncoder:
    """Convert privileged target timing into low-bandwidth visual cue channels."""

    def __init__(self, targets: tuple[TargetHit, ...] | list[TargetHit], *, config: VisualCueConfig | None = None) -> None:
        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.config = config or VisualCueConfig()

    def observe(self, simulation_time_s: float) -> VisualCueObservation:
        # The delayed perceptual clock is internal.  Only cue amplitudes leave
        # this object; neither simulation_time_s nor time-to-target does.
        perceived_time = simulation_time_s - self.config.latency_s
        left = 0.0
        right = 0.0

        for target in self._targets:
            dt = target.time_s - perceived_time
            if dt > self.config.horizon_s:
                break
            if dt < -self.config.horizon_s:
                continue
            cue = exp(-0.5 * (dt / self.config.width_s) ** 2)
            if target.key == "left":
                left = max(left, cue)
            elif target.key == "right":
                right = max(right, cue)
            else:
                raise ValueError(f"unsupported target key: {target.key!r}")

        return VisualCueObservation(min(1.0, left), min(1.0, right))
