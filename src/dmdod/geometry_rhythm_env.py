from __future__ import annotations

from dataclasses import dataclass

from .evaluator import TargetHit
from .motor_env import MotorAction, MotorObservation
from .planet_perception import (
    PlanetGeometryObservation,
    PlanetVisionConfig,
    StraightPlanetGeometryEncoder,
)
from .rhythm_env import EpisodeStats, RhythmMotorEnv


@dataclass(frozen=True)
class GeometryRhythmObservation:
    """Policy input: body state plus visible planet/tile geometry."""

    motor: MotorObservation
    geometry: PlanetGeometryObservation


@dataclass(frozen=True)
class GeometryRhythmStep:
    observation: GeometryRhythmObservation
    reward: float
    done: bool


class GeometryRhythmEnv:
    """Geometry-observation wrapper around the established rhythm evaluator.

    A private ``RhythmMotorEnv`` still owns ADOFAI-like timing/OVERLOAD/reward
    mechanics. Its Gaussian cue is discarded. The policy instead sees a
    straight-tile planet renderer derived from the active target and current
    simulation time.

    This keeps scoring identical to the previous toy experiment while changing
    only what the agent can perceive.
    """

    def __init__(
        self,
        targets: list[TargetHit] | tuple[TargetHit, ...],
        *,
        bpm: float,
        same_hand: bool = True,
        control_dt_s: float = 0.010,
        vision_config: PlanetVisionConfig | None = None,
        perception_seed: int | None = None,
        clockwise: bool = False,
    ) -> None:
        if not targets:
            raise ValueError("at least one target is required")
        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self._perception_seed = perception_seed
        self._base = RhythmMotorEnv(
            self._targets,
            bpm=bpm,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        self._geometry_encoder = StraightPlanetGeometryEncoder(
            self._targets,
            bpm=bpm,
            config=vision_config,
            seed=perception_seed,
            clockwise=clockwise,
        )

    def _active_target_index(self) -> int | None:
        resolved = self._base.stats.hits + self._base.stats.misses
        return resolved if resolved < len(self._targets) else None

    def _observation(self) -> GeometryRhythmObservation:
        now = self._base.motor.diagnostics().time_s
        return GeometryRhythmObservation(
            self._base.motor.observe(),
            self._geometry_encoder.observe(
                now,
                active_target_index=self._active_target_index(),
            ),
        )

    def reset(self) -> GeometryRhythmObservation:
        self._base.reset()
        self._geometry_encoder.reset(seed=self._perception_seed)
        return self._observation()

    def step(self, action: MotorAction) -> GeometryRhythmStep:
        transition = self._base.step(action)
        return GeometryRhythmStep(
            observation=self._observation(),
            reward=transition.reward,
            done=transition.done,
        )

    @property
    def stats(self) -> EpisodeStats:
        return self._base.stats

    @property
    def timing_errors_ms(self) -> tuple[float, ...]:
        return self._base.timing_errors_ms

    @property
    def timing_windows(self):
        return self._base.timing_windows

    @property
    def motor(self):
        return self._base.motor

    @property
    def bpm(self) -> float:
        return self._base.bpm
