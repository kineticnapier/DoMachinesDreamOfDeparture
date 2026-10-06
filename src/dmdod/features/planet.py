from __future__ import annotations

import random
from dataclasses import dataclass
from math import cos, pi, sin

from dmdod.evaluation.evaluator import TargetHit


@dataclass(frozen=True)
class PlanetGeometryObservation:
    """Agent-visible geometry in the current-pivot coordinate frame.

    ``orbit_*`` is the orbiting planet position relative to the chosen/pivot
    planet. ``next_*`` is the visible direction from the pivot to the next tile.
    No timestamp, BPM, target angle, angle error, or rotation-direction flag is
    exposed. Rotation direction must be inferred from successive visual frames.
    """

    orbit_x: float
    orbit_y: float
    next_x: float
    next_y: float


@dataclass(frozen=True)
class PlanetVisionConfig:
    """Low-bandwidth visual sensor for planet geometry.

    The simulator keeps exact chart timing privately to advance the synthetic
    orbit, but the policy receives only sampled/noisy positions. Latency jitter
    is sampled once per episode; position noise/dropout are sampled per visual
    frame.
    """

    latency_s: float = 0.050
    latency_jitter_s: float = 0.015
    sample_period_s: float = 1.0 / 60.0
    position_noise_std: float = 0.015
    dropout_probability: float = 0.01

    def __post_init__(self) -> None:
        if self.latency_s < 0.0:
            raise ValueError("latency_s must be non-negative")
        if self.latency_jitter_s < 0.0:
            raise ValueError("latency_jitter_s must be non-negative")
        if self.sample_period_s < 0.0:
            raise ValueError("sample_period_s must be non-negative")
        if self.position_noise_std < 0.0:
            raise ValueError("position_noise_std must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1")


class StraightPlanetGeometryEncoder:
    """Render straight-tile ADOFAI-like geometry without a timing cue.

    This is deliberately a small bridge between the old Gaussian timing cue and
    a future full .adofai geometry simulator. For a straight 180-degree tile the
    orbiting planet travels half a revolution per beat and reaches the visible
    next-floor vector at the target instant.

    The design follows the game-side structure visible in the decompiled DLL:
    floor placement is derived from an exit-angle vector, while hit evaluation
    compares the chosen planet angle with its target exit angle and then switches
    the chosen planet. We model only the geometry visible to a player, not those
    privileged internal angle values.
    """

    def __init__(
        self,
        targets: tuple[TargetHit, ...] | list[TargetHit],
        *,
        bpm: float,
        config: PlanetVisionConfig | None = None,
        seed: int | None = None,
        clockwise: bool = False,
    ) -> None:
        if bpm <= 0.0:
            raise ValueError("bpm must be positive")
        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.bpm = float(bpm)
        self.config = config or PlanetVisionConfig()
        self.clockwise = bool(clockwise)
        self._rng = random.Random(seed)
        self._episode_latency_s = self.config.latency_s
        self._held_observation: PlanetGeometryObservation | None = None
        self._held_target_index: int | None = None
        self._next_sample_time_s = float("-inf")
        self.reset(seed=seed)

    @property
    def episode_latency_s(self) -> float:
        """Privileged diagnostic value; never include it in policy input."""

        return self._episode_latency_s

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        jitter = self.config.latency_jitter_s
        if jitter > 0.0:
            self._episode_latency_s = max(
                0.0,
                self.config.latency_s + self._rng.uniform(-jitter, jitter),
            )
        else:
            self._episode_latency_s = self.config.latency_s
        self._held_observation = None
        self._held_target_index = None
        self._next_sample_time_s = float("-inf")

    def _sample_raw(
        self,
        simulation_time_s: float,
        *,
        active_target_index: int | None,
    ) -> PlanetGeometryObservation:
        if active_target_index is None:
            return PlanetGeometryObservation(0.0, 0.0, 0.0, 0.0)
        if active_target_index < 0 or active_target_index >= len(self._targets):
            raise IndexError(f"active target index out of range: {active_target_index}")

        target = self._targets[active_target_index]
        perceived_time = simulation_time_s - self._episode_latency_s

        # Straight tile: the visible next-floor vector is +X in the local pivot
        # frame. The orbiting planet covers pi radians per beat. Exact target
        # timing remains private and is used only to render where the planet is.
        next_angle = 0.0
        angular_speed = pi * self.bpm / 60.0
        direction = -1.0 if self.clockwise else 1.0
        time_to_target = target.time_s - perceived_time
        orbit_angle = next_angle - direction * angular_speed * time_to_target

        orbit_x = cos(orbit_angle)
        orbit_y = sin(orbit_angle)
        next_x = 1.0
        next_y = 0.0

        if (
            self.config.dropout_probability > 0.0
            and self._rng.random() < self.config.dropout_probability
        ):
            return PlanetGeometryObservation(0.0, 0.0, 0.0, 0.0)

        noise = self.config.position_noise_std
        if noise > 0.0:
            orbit_x += self._rng.gauss(0.0, noise)
            orbit_y += self._rng.gauss(0.0, noise)
            next_x += self._rng.gauss(0.0, noise)
            next_y += self._rng.gauss(0.0, noise)

        return PlanetGeometryObservation(orbit_x, orbit_y, next_x, next_y)

    def observe(
        self,
        simulation_time_s: float,
        *,
        active_target_index: int | None,
    ) -> PlanetGeometryObservation:
        sample_period = self.config.sample_period_s
        can_hold = (
            sample_period > 0.0
            and self._held_observation is not None
            and active_target_index == self._held_target_index
            and simulation_time_s + 1e-12 < self._next_sample_time_s
        )
        if can_hold:
            return self._held_observation

        observation = self._sample_raw(
            simulation_time_s,
            active_target_index=active_target_index,
        )
        self._held_observation = observation
        self._held_target_index = active_target_index
        self._next_sample_time_s = simulation_time_s + sample_period
        return observation
