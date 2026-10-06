from __future__ import annotations

from dataclasses import dataclass

from dmdod.adofai_rules import TimingJudgement
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.env import MotorObservation, TimedKeyEvent
from dmdod.envs.real_chart import RealChartMotorEnv, RelativeVisibleFloor


DEFAULT_FEEDBACK_HOLD_S = 0.75


@dataclass(frozen=True, slots=True)
class HudRealChartObservation:
    """Player-visible real-chart state with a small HUD slice.

    This deliberately models information a strong player may actually see:
    current Tile BPM / effective Real BPM plus the most recent timing feedback
    for a short display interval. It still excludes exact chart time, target
    timestamps, absolute floor indices, internal overload, fatigue, and other
    evaluator-only state.
    """

    motor: MotorObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]
    tile_bpm: float
    real_bpm: float
    feedback_visible: bool
    last_judgement: TimingJudgement | None
    last_timing_error_ms: float


def hud_bpms_for_floor(segment, floor) -> tuple[float, float]:
    """Return wall-clock Tile BPM and angle-normalized Real BPM.

    ``floor.bpm`` is the chart's current configured tile speed after SetSpeed.
    Pitch changes wall-clock speed, so both HUD values include pitch. Real BPM
    is the 180-degree-equivalent cadence implied by the tile's actual rotation
    duration, excluding Pause time. Floor 0 is countdown/start state rather
    than a playable tile, so it reports the tile speed for both values.
    """

    pitch = max(1e-6, float(segment.pitch_ratio))
    tile_bpm = max(1e-6, float(floor.bpm)) * pitch
    if int(floor.index) == 0:
        return tile_bpm, tile_bpm

    rotation_chart_s = float(floor.exit_time_s - floor.target_time_s - floor.pause_s)
    if rotation_chart_s <= 1e-9:
        return tile_bpm, tile_bpm
    real_bpm = 60.0 * pitch / rotation_chart_s
    return tile_bpm, real_bpm


class HudRealChartMotorEnv(RealChartMotorEnv):
    """RealChartMotorEnv with human-visible BPM and transient judgement HUD."""

    def __init__(self, *args, feedback_hold_s: float = DEFAULT_FEEDBACK_HOLD_S, **kwargs) -> None:
        if feedback_hold_s < 0.0:
            raise ValueError("feedback_hold_s must be non-negative")
        self.feedback_hold_s = float(feedback_hold_s)
        self._hud_judgement: TimingJudgement | None = None
        self._hud_error_ms = 0.0
        self._hud_feedback_until_s = -1.0
        super().__init__(*args, **kwargs)

    def reset(self) -> HudRealChartObservation:
        self._hud_judgement = None
        self._hud_error_ms = 0.0
        self._hud_feedback_until_s = -1.0
        return super().reset()

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is not KeyEvent.DOWN:
            return super()._score_event(event)

        target_index = self._next_target_index()
        if target_index is None:
            return super()._score_event(event)

        target = self.segment.targets[target_index]
        error_s = event.time_s - target.episode_time_s
        margin_count_before = len(self._hit_margins)
        reward = super()._score_event(event)

        if len(self._hit_margins) > margin_count_before:
            # Use the recorded margin rather than the pre-call classification so
            # a TooEarly that crosses the fail bar is exposed as FailOverload.
            self._hud_judgement = self._hit_margins[-1]
            self._hud_error_ms = error_s * 1000.0
            self._hud_feedback_until_s = event.time_s + self.feedback_hold_s
        return reward

    def _observation(self, motor: MotorObservation, episode_time_s: float) -> HudRealChartObservation:
        base = super()._observation(motor, episode_time_s)
        chart_time = self.segment.chart_time_from_episode(episode_time_s)
        floor_index = self._floor_index_at_chart_time(chart_time)
        floor = self.segment.chart.floors[floor_index]
        tile_bpm, real_bpm = hud_bpms_for_floor(self.segment, floor)
        feedback_visible = (
            self._hud_judgement is not None
            and episode_time_s <= self._hud_feedback_until_s + 1e-12
        )
        return HudRealChartObservation(
            motor=base.motor,
            orbiting_x=base.orbiting_x,
            orbiting_y=base.orbiting_y,
            floors=base.floors,
            tile_bpm=tile_bpm,
            real_bpm=real_bpm,
            feedback_visible=feedback_visible,
            last_judgement=self._hud_judgement if feedback_visible else None,
            last_timing_error_ms=self._hud_error_ms if feedback_visible else 0.0,
        )


class DiagnosticHudRealChartMotorEnv(HudRealChartMotorEnv):
    """HUD environment with evaluator-only physical key-down counting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.physical_keydowns = 0

    def reset(self) -> HudRealChartObservation:
        self.physical_keydowns = 0
        return super().reset()

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is KeyEvent.DOWN:
            self.physical_keydowns += 1
        return super()._score_event(event)
