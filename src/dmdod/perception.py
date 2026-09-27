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
    """Convert privileged target timing into low-bandwidth visual cue channels.

    ``active_target_indices`` lets the environment expose only the currently
    relevant target(s).  The indices are privileged environment state and never
    leave this encoder; the policy still receives only the two cue amplitudes.
    """

    def __init__(
        self,
        targets: tuple[TargetHit, ...] | list[TargetHit],
        *,
        config: VisualCueConfig | None = None,
    ) -> None:
        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.config = config or VisualCueConfig()

    def observe(
        self,
        simulation_time_s: float,
        *,
        active_target_indices: tuple[int, ...] | None = None,
    ) -> VisualCueObservation:
        # The delayed perceptual clock is internal.  Only cue amplitudes leave
        # this object; neither simulation_time_s, target indices, nor
        # time-to-target do.
        perceived_time = simulation_time_s - self.config.latency_s
        left = 0.0
        right = 0.0

        if active_target_indices is None:
            indexed_targets = enumerate(self._targets)
        else:
            indices = tuple(active_target_indices)
            for index in indices:
                if index < 0 or index >= len(self._targets):
                    raise IndexError(f"active target index out of range: {index}")
            indexed_targets = ((index, self._targets[index]) for index in indices)

        for _, target in indexed_targets:
            dt = target.time_s - perceived_time
            if dt > self.config.horizon_s:
                # With the full sorted target stream, everything after this is
                # farther away too.  For an explicit active subset we cannot
                # assume that ordering, so just skip the target.
                if active_target_indices is None:
                    break
                continue
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
