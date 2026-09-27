from __future__ import annotations

import random
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
    """Toy visual-perception model used by the RL experiments.

    Exact target timestamps remain private. The default configuration preserves
    the original smooth deterministic cue. Optional latency jitter, sampling,
    noise and dropout make the observation less oracle-like while keeping the
    same two-channel policy interface.

    ``latency_jitter_s`` is sampled once per episode. ``sample_period_s`` holds
    the last observed frame between visual samples. Gaussian amplitude noise and
    dropout are applied only when a new visual sample is produced.
    """

    latency_s: float = 0.050
    width_s: float = 0.090
    horizon_s: float = 0.450
    latency_jitter_s: float = 0.0
    sample_period_s: float = 0.0
    amplitude_noise_std: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.latency_s < 0.0:
            raise ValueError("latency_s must be non-negative")
        if self.width_s <= 0.0:
            raise ValueError("width_s must be positive")
        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if self.latency_jitter_s < 0.0:
            raise ValueError("latency_jitter_s must be non-negative")
        if self.sample_period_s < 0.0:
            raise ValueError("sample_period_s must be non-negative")
        if self.amplitude_noise_std < 0.0:
            raise ValueError("amplitude_noise_std must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1")


class VisualCueEncoder:
    """Convert privileged target timing into low-bandwidth visual cue channels.

    ``active_target_indices`` lets the environment expose only the currently
    relevant target(s). The indices are privileged environment state and never
    leave this encoder; the policy still receives only the two cue amplitudes.

    The encoder is stateful only when sensor imperfections are enabled. Calling
    ``reset`` starts a new perception episode and samples its latency offset.
    """

    def __init__(
        self,
        targets: tuple[TargetHit, ...] | list[TargetHit],
        *,
        config: VisualCueConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.config = config or VisualCueConfig()
        self._rng = random.Random(seed)
        self._episode_latency_s = self.config.latency_s
        self._held_observation: VisualCueObservation | None = None
        self._held_active_signature: tuple[int, ...] | None = None
        self._next_sample_time_s = float("-inf")
        self.reset(seed=seed)

    @property
    def episode_latency_s(self) -> float:
        """Privileged diagnostic value; never include this in policy input."""

        return self._episode_latency_s

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        jitter = self.config.latency_jitter_s
        if jitter > 0.0:
            sampled = self.config.latency_s + self._rng.uniform(-jitter, jitter)
            self._episode_latency_s = max(0.0, sampled)
        else:
            self._episode_latency_s = self.config.latency_s
        self._held_observation = None
        self._held_active_signature = None
        self._next_sample_time_s = float("-inf")

    def _indexed_targets(
        self,
        active_target_indices: tuple[int, ...] | None,
    ):
        if active_target_indices is None:
            return None, enumerate(self._targets)

        indices = tuple(active_target_indices)
        for index in indices:
            if index < 0 or index >= len(self._targets):
                raise IndexError(f"active target index out of range: {index}")
        return indices, ((index, self._targets[index]) for index in indices)

    def _sample_raw(
        self,
        simulation_time_s: float,
        *,
        active_target_indices: tuple[int, ...] | None,
    ) -> VisualCueObservation:
        perceived_time = simulation_time_s - self._episode_latency_s
        left = 0.0
        right = 0.0
        _, indexed_targets = self._indexed_targets(active_target_indices)

        for _, target in indexed_targets:
            dt = target.time_s - perceived_time
            if dt > self.config.horizon_s:
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

        if self.config.dropout_probability > 0.0 and self._rng.random() < self.config.dropout_probability:
            return VisualCueObservation(0.0, 0.0)

        noise = self.config.amplitude_noise_std
        if noise > 0.0:
            left += self._rng.gauss(0.0, noise)
            right += self._rng.gauss(0.0, noise)

        return VisualCueObservation(
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, right)),
        )

    def observe(
        self,
        simulation_time_s: float,
        *,
        active_target_indices: tuple[int, ...] | None = None,
    ) -> VisualCueObservation:
        # Only cue amplitudes leave this object; neither simulation time, target
        # indices, time-to-target nor sampled latency are exposed to the policy.
        signature = None if active_target_indices is None else tuple(active_target_indices)
        sample_period = self.config.sample_period_s

        # Force a fresh visual sample when the active target changes. This avoids
        # keeping a resolved tile visible merely because the display sample clock
        # has not advanced yet.
        can_hold = (
            sample_period > 0.0
            and self._held_observation is not None
            and signature == self._held_active_signature
            and simulation_time_s + 1e-12 < self._next_sample_time_s
        )
        if can_hold:
            return self._held_observation

        observation = self._sample_raw(
            simulation_time_s,
            active_target_indices=active_target_indices,
        )
        self._held_observation = observation
        self._held_active_signature = signature
        self._next_sample_time_s = simulation_time_s + sample_period
        return observation
