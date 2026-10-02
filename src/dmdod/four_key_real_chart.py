from __future__ import annotations

"""Four-key Level A real-chart environment and structured HUD encoder.

This module is the bridge between the existing four-finger motor body and the
real-chart evaluator used by Level A.  It intentionally does not modify the
legacy two-key path.  The visible chart geometry/HUD contract is unchanged;
only the motor slice grows from 6D to 12D, so the structured HUD input grows
from 245D to 251D.
"""

from dataclasses import dataclass

from .adofai_rules import TimingJudgement
from .four_key_motor import (
    FOUR_KEY_OBSERVATION_DIM,
    FourKeyAction,
    FourKeyMotorEnv,
    FourKeyObservation,
)
from .keyboard import KeyEvent
from .real_chart_env import (
    RealChartMotorEnv,
    RelativeVisibleFloor,
)
from .real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_FLOOR_FEATURE_DIM,
    RealChartFeatureConfig,
    _floor_features,
)
from .real_chart_hud import (
    DEFAULT_FEEDBACK_HOLD_S,
    HudRealChartMotorEnv,
)
from .real_chart_hud_features import (
    HUD_BPM_LOG_CLIP,
    HUD_ERROR_CLIP,
    HUD_ERROR_SCALE_MS,
    HUD_FEATURE_DIM,
    HUD_JUDGEMENTS,
    encode_bpm,
)


FOUR_KEY_REAL_CHART_FEATURE_VERSION = "four-key-visible-geometry-v1"
FOUR_KEY_HUD_FEATURE_VERSION = "four-key-tbpm-rbpm-feedback-v1"
FOUR_KEY_REAL_CHART_INPUT_DIM = (
    FOUR_KEY_OBSERVATION_DIM
    + 2
    + DEFAULT_REAL_CHART_FEATURE_CONFIG.floor_slots * REAL_CHART_FLOOR_FEATURE_DIM
)
FOUR_KEY_HUD_REAL_CHART_INPUT_DIM = FOUR_KEY_REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class FourKeyRealChartObservation:
    motor: FourKeyObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]


@dataclass(frozen=True, slots=True)
class FourKeyHudRealChartObservation:
    motor: FourKeyObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]
    tile_bpm: float
    real_bpm: float
    feedback_visible: bool
    last_judgement: TimingJudgement | None
    last_timing_error_ms: float


@dataclass(frozen=True, slots=True)
class FourKeyRealChartStep:
    observation: FourKeyRealChartObservation
    reward: float
    done: bool


class FourKeyRealChartMotorEnv(RealChartMotorEnv):
    """Real-chart evaluator driven by the four-key physical body.

    Scoring remains finger-agnostic: every physical KeyDown can claim the next
    playable target exactly as in the legacy two-key environment.  ``same_hand``
    is accepted only for compatibility with the legacy trainer constructor; the
    four-key body has its own explicit two-hand layout and therefore ignores it.
    """

    def __init__(
        self,
        segment,
        *,
        same_hand: bool | None = None,
        control_dt_s: float = 0.010,
        physics_dt_s: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(
            segment,
            same_hand=True,
            control_dt_s=control_dt_s,
            **kwargs,
        )
        self.motor = FourKeyMotorEnv(
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
        )

    def step(self, action: FourKeyAction) -> FourKeyRealChartStep:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")

        transition = self.motor.step(action)
        reward = 0.0
        for event in transition.evaluator_events:
            self._advance_overload(event.time_s)
            reward += self._expire_misses(event.time_s)
            if self._failed_on_miss:
                break
            reward += self._score_event(event)
            if self._overload.overloaded:
                break

        now = transition.diagnostics.time_s
        if not self._failed_on_miss and not self._overload.overloaded:
            self._advance_overload(now)
            reward += self._expire_misses(now)
        reward -= self.reward_config.effort_penalty * sum(
            abs(value) for value in action.as_tuple()
        )

        all_resolved = all(used or missed for used, missed in zip(self._used, self._missed))
        self._done = (
            self._failed_on_miss
            or self._overload.overloaded
            or now >= self._episode_end_s
            or all_resolved
        )
        if self._done and now >= self._episode_end_s and not all_resolved:
            reward += self._expire_misses(float("inf"))

        self._total_reward += reward
        return FourKeyRealChartStep(
            self._observation(transition.observation, now),
            reward,
            self._done,
        )

    def _observation(
        self,
        motor: FourKeyObservation,
        episode_time_s: float,
    ) -> FourKeyRealChartObservation:
        base = super()._observation(motor, episode_time_s)
        return FourKeyRealChartObservation(
            motor=motor,
            orbiting_x=base.orbiting_x,
            orbiting_y=base.orbiting_y,
            floors=base.floors,
        )


class HudFourKeyRealChartMotorEnv(HudRealChartMotorEnv, FourKeyRealChartMotorEnv):
    """Four-key real-chart environment with the same player-visible HUD slice."""

    def __init__(
        self,
        *args,
        feedback_hold_s: float = DEFAULT_FEEDBACK_HOLD_S,
        **kwargs,
    ) -> None:
        super().__init__(*args, feedback_hold_s=feedback_hold_s, **kwargs)

    def _observation(
        self,
        motor: FourKeyObservation,
        episode_time_s: float,
    ) -> FourKeyHudRealChartObservation:
        base = super()._observation(motor, episode_time_s)
        return FourKeyHudRealChartObservation(
            motor=motor,
            orbiting_x=base.orbiting_x,
            orbiting_y=base.orbiting_y,
            floors=base.floors,
            tile_bpm=base.tile_bpm,
            real_bpm=base.real_bpm,
            feedback_visible=base.feedback_visible,
            last_judgement=base.last_judgement,
            last_timing_error_ms=base.last_timing_error_ms,
        )


class DiagnosticHudFourKeyRealChartMotorEnv(HudFourKeyRealChartMotorEnv):
    """Four-key HUD environment with evaluator-only physical KeyDown counting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.physical_keydowns = 0

    def reset(self) -> FourKeyHudRealChartObservation:
        self.physical_keydowns = 0
        return super().reset()

    def _score_event(self, event) -> float:
        if event.event is KeyEvent.DOWN:
            self.physical_keydowns += 1
        return super()._score_event(event)


def encode_four_key_real_chart_observation(
    observation: FourKeyRealChartObservation,
    *,
    config: RealChartFeatureConfig = DEFAULT_REAL_CHART_FEATURE_CONFIG,
) -> tuple[float, ...]:
    """Encode the 12D four-key motor slice plus the existing visible geometry."""

    motor = observation.motor
    values: list[float] = [
        motor.left_outer_position_m / 0.006,
        motor.left_inner_position_m / 0.006,
        motor.right_inner_position_m / 0.006,
        motor.right_outer_position_m / 0.006,
        motor.left_outer_velocity_m_s / 1.0,
        motor.left_inner_velocity_m_s / 1.0,
        motor.right_inner_velocity_m_s / 1.0,
        motor.right_outer_velocity_m_s / 1.0,
        1.0 if motor.left_outer_pressed else 0.0,
        1.0 if motor.left_inner_pressed else 0.0,
        1.0 if motor.right_inner_pressed else 0.0,
        1.0 if motor.right_outer_pressed else 0.0,
        observation.orbiting_x / config.tile_size,
        observation.orbiting_y / config.tile_size,
    ]

    by_relative_index = {floor.relative_index: floor for floor in observation.floors}
    for relative_index in range(-config.behind_floors, config.ahead_floors + 1):
        floor = by_relative_index.get(relative_index)
        if floor is None:
            values.extend((0.0,) * REAL_CHART_FLOOR_FEATURE_DIM)
        else:
            values.extend(_floor_features(floor, config))

    expected = (
        FOUR_KEY_OBSERVATION_DIM
        + 2
        + config.floor_slots * REAL_CHART_FLOOR_FEATURE_DIM
    )
    if len(values) != expected:
        raise RuntimeError(
            f"four-key real-chart feature size mismatch: expected {expected}, got {len(values)}"
        )
    return tuple(values)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def encode_four_key_hud_real_chart_observation(
    observation: FourKeyHudRealChartObservation,
) -> tuple[float, ...]:
    """Encode four-key geometry/motor state plus the unchanged 12D HUD slice."""

    base = FourKeyRealChartObservation(
        motor=observation.motor,
        orbiting_x=observation.orbiting_x,
        orbiting_y=observation.orbiting_y,
        floors=observation.floors,
    )
    values = list(encode_four_key_real_chart_observation(base))
    visible = bool(observation.feedback_visible and observation.last_judgement is not None)
    values.extend(
        (
            encode_bpm(observation.tile_bpm),
            encode_bpm(observation.real_bpm),
            1.0 if visible else 0.0,
            _clip(
                observation.last_timing_error_ms / HUD_ERROR_SCALE_MS,
                -HUD_ERROR_CLIP,
                HUD_ERROR_CLIP,
            )
            if visible
            else 0.0,
        )
    )
    values.extend(
        1.0 if visible and observation.last_judgement is judgement else 0.0
        for judgement in HUD_JUDGEMENTS
    )
    if len(values) != FOUR_KEY_HUD_REAL_CHART_INPUT_DIM:
        raise RuntimeError(
            "four-key HUD feature size mismatch: "
            f"expected {FOUR_KEY_HUD_REAL_CHART_INPUT_DIM}, got {len(values)}"
        )
    return tuple(values)
