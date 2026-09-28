from __future__ import annotations

from dataclasses import dataclass

from .adofai_rules import TimingDifficulty
from .evaluator import TargetHit
from .geometry_rhythm_env import GeometryRhythmEnv, GeometryRhythmObservation
from .motor_env import MotorAction, MotorObservation
from .pattern_memory import PatternLookup, PatternMemory, PatternMemoryFeatures
from .planet_perception import PlanetGeometryObservation, PlanetVisionConfig
from .rhythm_env import EpisodeStats, RewardConfig


@dataclass(frozen=True)
class PatternGeometryObservation:
    """Body + visible geometry + learned pattern/practice context."""

    motor: MotorObservation
    geometry: PlanetGeometryObservation
    pattern: PatternMemoryFeatures


@dataclass(frozen=True)
class PatternGeometryStep:
    observation: PatternGeometryObservation
    reward: float
    done: bool


class PatternMemoryGeometryEnv:
    """Geometry environment with human-like reusable/practice memory.

    The wrapped timing evaluator stays private. Pattern lookup is keyed only by
    recent visible geometry. After a successful hit, the already-observed visual
    context and the resulting signed timing error are used as feedback for future
    attempts. No exact target time, BPM, target angle or chart progress index is
    exposed to the policy.
    """

    def __init__(
        self,
        targets: list[TargetHit] | tuple[TargetHit, ...],
        *,
        bpm: float,
        pattern_memory: PatternMemory,
        chart_id: str | None,
        practice_memory: bool = True,
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
        self.pattern_memory = pattern_memory
        self.chart_id = chart_id
        self.practice_memory = bool(practice_memory)
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
        self._last_lookup: PatternLookup | None = None

    def _augment(self, observation: GeometryRhythmObservation) -> PatternGeometryObservation:
        lookup = self.pattern_memory.observe(
            observation.geometry,
            use_memory=self.practice_memory,
        )
        self._last_lookup = lookup
        return PatternGeometryObservation(
            motor=observation.motor,
            geometry=observation.geometry,
            pattern=lookup.features,
        )

    def reset(self) -> PatternGeometryObservation:
        self.pattern_memory.begin_episode(self.chart_id)
        self._last_lookup = None
        return self._augment(self._base.reset())

    def step(self, action: MotorAction) -> PatternGeometryStep:
        lookup = self._last_lookup
        previous_error_count = len(self._base.timing_errors_ms)
        transition = self._base.step(action)

        # A timing error is appended only for a valid hit. Associate that
        # post-action feedback with the context the policy actually saw before
        # choosing this action. This is practice, not future information.
        if self.practice_memory and lookup is not None:
            new_errors = self._base.timing_errors_ms[previous_error_count:]
            for error_ms in new_errors:
                self.pattern_memory.learn(
                    lookup.keys,
                    error_ms=error_ms,
                    action=action,
                )

        return PatternGeometryStep(
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
