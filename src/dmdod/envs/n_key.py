from __future__ import annotations

"""Configurable N-key Level A real-chart environment and HUD encoder.

The chart geometry and HUD contract are identical to the existing 2K/4K Level A
paths.  Only the motor slice scales with key count: position, velocity and press
state contribute 3 values per physical key.  This makes the HUD input dimension
``239 + 3 * key_count`` (251D for 4K, 257D for 6K, 263D for 8K).
"""

from dataclasses import dataclass

from dmdod.adofai_rules import TimingJudgement
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.capacity import NKeyRealChartMotorEnv
from dmdod.motor.n_key import NKeyObservation, n_key_names
from dmdod.envs.real_chart import RelativeVisibleFloor
from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_FLOOR_FEATURE_DIM,
    RealChartFeatureConfig,
    _floor_features,
)
from dmdod.features.hud import DEFAULT_FEEDBACK_HOLD_S, hud_bpms_for_floor
from dmdod.features.hud_features import (
    HUD_ERROR_CLIP,
    HUD_ERROR_SCALE_MS,
    HUD_FEATURE_DIM,
    HUD_JUDGEMENTS,
    encode_bpm,
)


N_KEY_REAL_CHART_FEATURE_VERSION = "n-key-visible-geometry-v1"
N_KEY_HUD_FEATURE_VERSION = "n-key-tbpm-rbpm-feedback-v1"


def n_key_motor_observation_dim(key_count: int) -> int:
    n_key_names(key_count)  # validate the configurable body contract
    return 3 * int(key_count)


def n_key_real_chart_input_dim(
    key_count: int,
    *,
    config: RealChartFeatureConfig = DEFAULT_REAL_CHART_FEATURE_CONFIG,
) -> int:
    return (
        n_key_motor_observation_dim(key_count)
        + 2
        + config.floor_slots * REAL_CHART_FLOOR_FEATURE_DIM
    )


def n_key_hud_real_chart_input_dim(key_count: int) -> int:
    return n_key_real_chart_input_dim(key_count) + HUD_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class NKeyRealChartObservation:
    motor: NKeyObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]


@dataclass(frozen=True, slots=True)
class NKeyHudRealChartObservation:
    motor: NKeyObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]
    tile_bpm: float
    real_bpm: float
    feedback_visible: bool
    last_judgement: TimingJudgement | None
    last_timing_error_ms: float


@dataclass(frozen=True, slots=True)
class TooEarlyKeyDownTrace:
    time_s: float
    key: str
    target_index: int
    error_ms: float
    overload_before: float
    overload_after: float
    fail_overload: bool


@dataclass(frozen=True, slots=True)
class EpisodeTerminationTrace:
    time_s: float
    reason: str
    hits: int
    too_early: int
    keydowns: int
    next_target_index: int | None
    overload_value: float


@dataclass(frozen=True, slots=True)
class NKeyOverloadTrace:
    termination: EpisodeTerminationTrace
    keydown_times_s: tuple[float, ...]
    too_early_events: tuple[TooEarlyKeyDownTrace, ...]


class DiagnosticHudNKeyRealChartMotorEnv(NKeyRealChartMotorEnv):
    """N-key real-chart evaluator with the same human-visible HUD slice.

    Exact target times remain evaluator-only.  ``physical_keydowns`` is inherited
    from ``NKeyRealChartMotorEnv`` and is likewise diagnostic-only.
    """

    def __init__(
        self,
        *args,
        feedback_hold_s: float = DEFAULT_FEEDBACK_HOLD_S,
        capture_overload_trace: bool = False,
        **kwargs,
    ) -> None:
        if feedback_hold_s < 0.0:
            raise ValueError("feedback_hold_s must be non-negative")
        self.feedback_hold_s = float(feedback_hold_s)
        self.capture_overload_trace = bool(capture_overload_trace)
        self._trace_keydowns: list[float] = []
        self._trace_too_early: list[TooEarlyKeyDownTrace] = []
        self._trace_termination: EpisodeTerminationTrace | None = None
        self._trace_overload_event_time_s: float | None = None
        self._hud_judgement: TimingJudgement | None = None
        self._hud_error_ms = 0.0
        self._hud_feedback_until_s = -1.0
        super().__init__(*args, **kwargs)

    def reset(self) -> NKeyHudRealChartObservation:
        self._trace_keydowns = []
        self._trace_too_early = []
        self._trace_termination = None
        self._trace_overload_event_time_s = None
        self._hud_judgement = None
        self._hud_error_ms = 0.0
        self._hud_feedback_until_s = -1.0
        return super().reset()

    def _score_event(self, event) -> float:
        if event.event is not KeyEvent.DOWN:
            return super()._score_event(event)

        if self.capture_overload_trace:
            self._trace_keydowns.append(float(event.time_s))
        target_index = self._next_target_index()
        if target_index is None:
            return super()._score_event(event)

        target = self.segment.targets[target_index]
        error_s = float(event.time_s) - float(target.episode_time_s)
        margin_count_before = len(self._hit_margins)
        if self.capture_overload_trace:
            early_before = self._too_early
            gauge_before = float(self._overload.value)
            overloaded_before = bool(self._overload.overloaded)
        reward = super()._score_event(event)
        if self.capture_overload_trace and self._too_early > early_before:
            failed = bool(self._overload.overloaded) and not overloaded_before
            self._trace_too_early.append(
                TooEarlyKeyDownTrace(
                    time_s=float(event.time_s),
                    key=str(event.key),
                    target_index=target_index,
                    error_ms=error_s * 1000.0,
                    overload_before=gauge_before,
                    overload_after=float(self._overload.value),
                    fail_overload=failed,
                )
            )
            if failed:
                self._trace_overload_event_time_s = float(event.time_s)
        if len(self._hit_margins) > margin_count_before:
            self._hud_judgement = self._hit_margins[-1]
            self._hud_error_ms = error_s * 1000.0
            self._hud_feedback_until_s = float(event.time_s) + self.feedback_hold_s
        return reward

    def step(self, action):
        result = super().step(action)
        if self.capture_overload_trace and result.done:
            all_resolved = all(
                used or missed for used, missed in zip(self._used, self._missed)
            )
            if self._overload.overloaded:
                reason = "Overload"
            elif self._failed_on_miss:
                reason = "Miss failure"
            elif all_resolved:
                reason = "All targets resolved"
            else:
                reason = "Time limit"
            episode_time_s = (
                self._trace_overload_event_time_s
                if reason == "Overload" and self._trace_overload_event_time_s is not None
                else self.privileged_episode_time_s()
            )
            self._trace_termination = EpisodeTerminationTrace(
                time_s=float(episode_time_s),
                reason=reason,
                hits=int(self._hits),
                too_early=int(self._too_early),
                keydowns=int(self.physical_keydowns),
                next_target_index=self._next_target_index(),
                overload_value=float(self._overload.value),
            )
        return result

    @property
    def overload_trace(self) -> NKeyOverloadTrace | None:
        if not self.capture_overload_trace:
            return None
        if self._trace_termination is None:
            raise RuntimeError("overload trace requested before episode termination")
        return NKeyOverloadTrace(
            termination=self._trace_termination,
            keydown_times_s=tuple(self._trace_keydowns),
            too_early_events=tuple(self._trace_too_early),
        )

    def _observation(
        self,
        motor: NKeyObservation,
        episode_time_s: float,
    ) -> NKeyHudRealChartObservation:
        base = super()._observation(motor, episode_time_s)
        chart_time = self.segment.chart_time_from_episode(episode_time_s)
        floor_index = self._floor_index_at_chart_time(chart_time)
        floor = self.segment.chart.floors[floor_index]
        tile_bpm, real_bpm = hud_bpms_for_floor(self.segment, floor)
        feedback_visible = (
            self._hud_judgement is not None
            and episode_time_s <= self._hud_feedback_until_s + 1e-12
        )
        return NKeyHudRealChartObservation(
            motor=motor,
            orbiting_x=base.orbiting_x,
            orbiting_y=base.orbiting_y,
            floors=base.floors,
            tile_bpm=tile_bpm,
            real_bpm=real_bpm,
            feedback_visible=feedback_visible,
            last_judgement=self._hud_judgement if feedback_visible else None,
            last_timing_error_ms=self._hud_error_ms if feedback_visible else 0.0,
        )


def encode_n_key_real_chart_observation(
    observation: NKeyRealChartObservation,
    *,
    config: RealChartFeatureConfig = DEFAULT_REAL_CHART_FEATURE_CONFIG,
) -> tuple[float, ...]:
    motor = observation.motor
    key_count = len(motor.key_names)
    n_key_names(key_count)

    values: list[float] = [position / 0.006 for position in motor.positions_m]
    values.extend(velocity / 1.0 for velocity in motor.velocities_m_s)
    values.extend(1.0 if pressed else 0.0 for pressed in motor.pressed_flags)
    values.extend(
        (
            observation.orbiting_x / config.tile_size,
            observation.orbiting_y / config.tile_size,
        )
    )

    by_relative_index = {floor.relative_index: floor for floor in observation.floors}
    for relative_index in range(-config.behind_floors, config.ahead_floors + 1):
        floor = by_relative_index.get(relative_index)
        if floor is None:
            values.extend((0.0,) * REAL_CHART_FLOOR_FEATURE_DIM)
        else:
            values.extend(_floor_features(floor, config))

    expected = n_key_real_chart_input_dim(key_count, config=config)
    if len(values) != expected:
        raise RuntimeError(
            f"N-key real-chart feature size mismatch: expected {expected}, got {len(values)}"
        )
    return tuple(values)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def encode_n_key_hud_real_chart_observation(
    observation: NKeyHudRealChartObservation,
) -> tuple[float, ...]:
    key_count = len(observation.motor.key_names)
    base = NKeyRealChartObservation(
        motor=observation.motor,
        orbiting_x=observation.orbiting_x,
        orbiting_y=observation.orbiting_y,
        floors=observation.floors,
    )
    values = list(encode_n_key_real_chart_observation(base))
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

    expected = n_key_hud_real_chart_input_dim(key_count)
    if len(values) != expected:
        raise RuntimeError(
            f"N-key HUD feature size mismatch: expected {expected}, got {len(values)}"
        )
    return tuple(values)
