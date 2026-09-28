from __future__ import annotations

from dataclasses import dataclass

from .adofai_rules import TimingDifficulty
from .evaluator import TargetHit
from .geometry_rhythm_env import GeometryRhythmEnv, GeometryRhythmObservation
from .motor_env import MotorAction, MotorObservation
from .planet_perception import PlanetGeometryObservation, PlanetVisionConfig
from .rhythm_env import EpisodeStats, RewardConfig


@dataclass(frozen=True)
class VisualMotionFeatures:
    """Two-frame visual motion available without privileged timing state."""

    delta_orbit_x: float
    delta_orbit_y: float
    delta_next_x: float
    delta_next_y: float


@dataclass(frozen=True)
class MotionGeometryObservation:
    """Body + current geometry + explicit difference from the previous frame."""

    motor: MotorObservation
    geometry: PlanetGeometryObservation
    motion: VisualMotionFeatures


@dataclass(frozen=True)
class MotionGeometryStep:
    observation: MotionGeometryObservation
    reward: float
    done: bool


class MotionGeometryEnv:
    """Geometry environment exposing only a short visual motion cue.

    The simulator/evaluator still owns exact chart time and BPM privately.  The
    policy receives the same current geometry as before plus the difference from
    the immediately preceding *visible* frame.  This prevents the early
    curriculum from requiring the GRU to discover a finite difference before it
    can even represent apparent angular speed, while still withholding BPM,
    target time, target angle, timing error, and rotation direction.
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
        reward_config: RewardConfig | None = None,
        difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
        timing_scale: float = 1.0,
        controller_speed: float = 1.0,
        pitch: float = 1.0,
        speed_trial: float = 1.0,
        mobile: bool = False,
    ) -> None:
        self._base = GeometryRhythmEnv(
            targets,
            bpm=bpm,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            vision_config=vision_config,
            perception_seed=perception_seed,
            clockwise=clockwise,
            reward_config=reward_config,
            difficulty=difficulty,
            timing_scale=timing_scale,
            controller_speed=controller_speed,
            pitch=pitch,
            speed_trial=speed_trial,
            mobile=mobile,
        )
        self._previous_geometry: PlanetGeometryObservation | None = None

    def _augment(self, observation: GeometryRhythmObservation) -> MotionGeometryObservation:
        previous = self._previous_geometry
        geometry = observation.geometry
        if previous is None:
            motion = VisualMotionFeatures(0.0, 0.0, 0.0, 0.0)
        else:
            motion = VisualMotionFeatures(
                geometry.orbit_x - previous.orbit_x,
                geometry.orbit_y - previous.orbit_y,
                geometry.next_x - previous.next_x,
                geometry.next_y - previous.next_y,
            )
        self._previous_geometry = geometry
        return MotionGeometryObservation(observation.motor, geometry, motion)

    def reset(self) -> MotionGeometryObservation:
        self._previous_geometry = None
        return self._augment(self._base.reset())

    def step(self, action: MotorAction) -> MotionGeometryStep:
        transition = self._base.step(action)
        return MotionGeometryStep(
            observation=self._augment(transition.observation),
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
