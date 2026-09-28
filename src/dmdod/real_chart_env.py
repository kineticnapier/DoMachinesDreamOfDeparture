from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import cos, exp, pi, sin

from .adofai_playable import PlayableChartSegment, PlayableChartTarget
from .adofai_rules import (
    OverloadCounter,
    TimingDifficulty,
    TimingJudgement,
    classify_timing,
    normal_accuracy_percent,
    timing_windows,
    x_accuracy_components,
    x_accuracy_percent,
    x_accuracy_weight,
)
from .adofai_timing import PI_STOCK
from .keyboard import KeyEvent
from .motor_env import MotorAction, MotorEnv, MotorObservation, TimedKeyEvent
from .rhythm_env import EpisodeStats, RewardConfig


@dataclass(frozen=True, slots=True)
class RelativeVisibleFloor:
    """One policy-visible floor relative to the current stationary floor."""

    relative_index: int
    x: float
    y: float
    entry_angle_rad: float
    exit_angle_rad: float
    midspin: bool
    is_ccw: bool
    num_planets: int
    event_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealChartObservation:
    """Policy input for a real ADOFAI chart.

    Exact chart time, target time, BPM, absolute floor index, and evaluator
    state are intentionally absent.  The orbiting planet and future floors are
    rendered from privileged chart truth, just as a game renderer uses its
    internal clock without exposing that clock to the player.
    """

    motor: MotorObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]


@dataclass(frozen=True, slots=True)
class RealChartStep:
    observation: RealChartObservation
    reward: float
    done: bool


class RealChartMotorEnv:
    """Motor environment for an actual compiled ADOFAI chart segment.

    Any physical key-down can claim the next playable floor; target choice is
    not hard-wired to a finger.  Midspin floors remain visible geometry but are
    excluded from playable targets by ``build_playable_segment``.

    Multipress/OverPress queue semantics are still future work.  Ordinary
    timing windows, TooEarly overload, misses, X-Accuracy reward, variable BPM,
    pitch scaling, body physics, and multi-floor lookahead are active here.
    """

    def __init__(
        self,
        segment: PlayableChartSegment,
        *,
        same_hand: bool = True,
        control_dt_s: float = 0.010,
        reward_config: RewardConfig | None = None,
        tail_s: float = 0.400,
        difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
        timing_scale: float = 1.0,
        behind_floors: int = 2,
        ahead_floors: int = 12,
    ) -> None:
        if not segment.targets:
            raise ValueError("real chart segment must contain at least one playable target")
        if tail_s < 0.0:
            raise ValueError("tail_s must be non-negative")
        if behind_floors < 0 or ahead_floors < 0:
            raise ValueError("visible floor counts must be non-negative")

        self.segment = segment
        self.motor = MotorEnv(same_hand=same_hand, control_dt_s=control_dt_s)
        self.reward_config = reward_config or RewardConfig()
        self.tail_s = tail_s
        self.difficulty = (
            TimingDifficulty(str(difficulty).lower())
            if not isinstance(difficulty, TimingDifficulty)
            else difficulty
        )
        self.timing_scale = timing_scale
        self.behind_floors = behind_floors
        self.ahead_floors = ahead_floors
        self._floor_entry_times = tuple(floor.target_time_s for floor in segment.chart.floors)
        self._episode_end_s = max(
            segment.duration_s,
            segment.targets[-1].episode_time_s,
        ) + tail_s

        self._used: list[bool] = []
        self._missed: list[bool] = []
        self._errors_s: list[float] = []
        self._hits = 0
        self._misses = 0
        self._too_early = 0
        self._overload = OverloadCounter()
        self._overload_clock_s = 0.0
        self._judgement_counts: dict[TimingJudgement, int] = {}
        self._hit_margins: list[TimingJudgement] = []
        self._total_reward = 0.0
        self._done = False
        self._failed_on_miss = False

    def reset(self) -> RealChartObservation:
        motor = self.motor.reset()
        self._used = [False] * len(self.segment.targets)
        self._missed = [False] * len(self.segment.targets)
        self._errors_s = []
        self._hits = 0
        self._misses = 0
        self._too_early = 0
        self._overload = OverloadCounter()
        self._overload_clock_s = 0.0
        self._judgement_counts = {judgement: 0 for judgement in TimingJudgement}
        self._hit_margins = []
        self._total_reward = 0.0
        self._done = False
        self._failed_on_miss = False
        return self._observation(motor, 0.0)

    def observe(self) -> RealChartObservation:
        now = self.motor.diagnostics().time_s
        return self._observation(self.motor.observe(), now)

    def step(self, action: MotorAction) -> RealChartStep:
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
        reward -= self.reward_config.effort_penalty * (abs(action.left) + abs(action.right))

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
        return RealChartStep(
            self._observation(transition.observation, now),
            reward,
            self._done,
        )

    def _next_target_index(self) -> int | None:
        for index, (used, missed) in enumerate(zip(self._used, self._missed)):
            if not used and not missed:
                return index
        return None

    def _record_margin(self, judgement: TimingJudgement) -> None:
        self._hit_margins.append(judgement)
        self._judgement_counts[judgement] += 1

    def _windows(self, target: PlayableChartTarget):
        return timing_windows(
            target.bpm,
            difficulty=self.difficulty,
            timing_scale=self.timing_scale,
            pitch=self.segment.pitch_ratio,
        )

    def _classify(self, error_s: float, target: PlayableChartTarget) -> TimingJudgement:
        return classify_timing(
            error_s,
            target.bpm,
            difficulty=self.difficulty,
            timing_scale=self.timing_scale,
            pitch=self.segment.pitch_ratio,
        )

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is not KeyEvent.DOWN:
            return 0.0
        target_index = self._next_target_index()
        if target_index is None:
            return 0.0

        target = self.segment.targets[target_index]
        error_s = event.time_s - target.episode_time_s
        judgement = self._classify(error_s, target)

        if judgement is TimingJudgement.TOO_EARLY:
            self._too_early += 1
            overloaded_now = self._overload.record_too_early()
            self._record_margin(
                TimingJudgement.FAIL_OVERLOAD
                if overloaded_now
                else TimingJudgement.TOO_EARLY
            )
            reward = -self.reward_config.too_early_penalty
            if overloaded_now:
                reward -= self.reward_config.overload_penalty
            return reward

        if judgement is TimingJudgement.TOO_LATE:
            self._record_margin(TimingJudgement.TOO_LATE)
            self._missed[target_index] = True
            self._misses += 1
            if self.reward_config.fail_on_miss:
                self._failed_on_miss = True
            return -self.reward_config.miss_penalty

        self._record_margin(judgement)
        self._used[target_index] = True
        self._hits += 1
        self._errors_s.append(error_s)
        self._overload.record_valid_hit()

        windows = self._windows(target)
        sigma = self.reward_config.timing_sigma_s
        if sigma is None:
            sigma = max(
                1e-6,
                windows.perfect_s * self.reward_config.perfect_sigma_fraction,
            )
        center_quality = exp(-0.5 * (error_s / sigma) ** 2)
        return (
            self.reward_config.hit_reward
            + self.reward_config.xacc_reward * x_accuracy_weight(judgement)
            + self.reward_config.timing_bonus * center_quality
        )

    def _expire_misses(self, now_s: float) -> float:
        reward = 0.0
        for index, target in enumerate(self.segment.targets):
            if self._used[index] or self._missed[index]:
                continue
            if target.episode_time_s + self._windows(target).pass_s < now_s:
                self._missed[index] = True
                self._misses += 1
                self._record_margin(TimingJudgement.FAIL_MISS)
                reward -= self.reward_config.miss_penalty
                if self.reward_config.fail_on_miss:
                    self._failed_on_miss = True
        return reward

    def _advance_overload(self, to_time_s: float) -> None:
        if to_time_s <= self._overload_clock_s:
            return
        delta_s = to_time_s - self._overload_clock_s
        midpoint_s = self._overload_clock_s + delta_s * 0.5
        chart_time = self.segment.chart_time_from_episode(midpoint_s)
        floor = self.segment.chart.floors[self._floor_index_at_chart_time(chart_time)]
        beat_delta = delta_s * max(1e-6, floor.bpm) * self.segment.pitch_ratio / 60.0
        self._overload.advance_beats(beat_delta)
        self._overload_clock_s = to_time_s

    def _floor_index_at_chart_time(self, chart_time_s: float) -> int:
        index = bisect_right(self._floor_entry_times, chart_time_s) - 1
        return max(0, min(index, len(self._floor_entry_times) - 1))

    def _observation(self, motor: MotorObservation, episode_time_s: float) -> RealChartObservation:
        chart_time = self.segment.chart_time_from_episode(episode_time_s)
        floor_index = self._floor_index_at_chart_time(chart_time)
        current = self.segment.chart.floors[floor_index]

        start = max(0, floor_index - self.behind_floors)
        end = min(len(self.segment.chart.floors), floor_index + self.ahead_floors + 1)
        visible = tuple(
            RelativeVisibleFloor(
                relative_index=floor.index - current.index,
                x=floor.x - current.x,
                y=floor.y - current.y,
                entry_angle_rad=floor.entry_angle_rad,
                exit_angle_rad=floor.exit_angle_rad,
                midspin=floor.midspin,
                is_ccw=floor.is_ccw,
                num_planets=floor.num_planets,
                event_markers=floor.event_markers,
            )
            for floor in self.segment.chart.floors[start:end]
        )
        orbit_x, orbit_y = self._orbiting_relative_position(current, chart_time)
        return RealChartObservation(
            motor=motor,
            orbiting_x=orbit_x,
            orbiting_y=orbit_y,
            floors=visible,
        )

    def _orbiting_relative_position(self, floor, chart_time_s: float) -> tuple[float, float]:
        rotation_s = max(0.0, floor.exit_time_s - floor.target_time_s - floor.pause_s)
        elapsed = max(0.0, chart_time_s - floor.target_time_s - floor.pause_s)
        progress = 1.0 if rotation_s <= 1e-12 else min(1.0, elapsed / rotation_s)

        # duration = angle/pi * 60/bpm.  The compiled floor BPM is the stock
        # effective speed for mid-floor SetSpeed, so this also reconstructs its
        # visible total turn.  Floor 0 includes countdown rotations in duration.
        moved = rotation_s * PI_STOCK * max(1e-6, floor.bpm) / 60.0
        entry = floor.entry_angle_rad
        if floor.index == 0 and self.segment.chart.countdown_ticks > 0:
            entry = PI_STOCK * (0.5 - self.segment.chart.countdown_ticks)
        direction = -1.0 if floor.is_ccw else 1.0
        angle = entry + direction * moved * progress
        return 1.5 * sin(angle), 1.5 * cos(angle)

    @property
    def stats(self) -> EpisodeStats:
        mean_abs_error_ms = None
        if self._errors_s:
            mean_abs_error_ms = (
                sum(abs(error) for error in self._errors_s) / len(self._errors_s) * 1000.0
            )
        perfects = self._judgement_counts.get(TimingJudgement.PERFECT, 0)
        xacc_points, xacc_denominator = x_accuracy_components(self._hit_margins)
        margin_count = len(self._hit_margins)
        return EpisodeStats(
            targets=len(self.segment.targets),
            hits=self._hits,
            misses=self._misses,
            too_early_presses=self._too_early,
            overload_counter=self._overload.value,
            overloaded=self._overload.overloaded,
            perfects=perfects,
            early_late_perfects=(
                self._judgement_counts.get(TimingJudgement.EARLY_PERFECT, 0)
                + self._judgement_counts.get(TimingJudgement.LATE_PERFECT, 0)
            ),
            early_late_hits=(
                self._judgement_counts.get(TimingJudgement.VERY_EARLY, 0)
                + self._judgement_counts.get(TimingJudgement.VERY_LATE, 0)
            ),
            mean_abs_error_ms=mean_abs_error_ms,
            x_accuracy_percent=x_accuracy_percent(self._hit_margins),
            perfect_rate=perfects / max(1, margin_count),
            total_reward=self._total_reward,
            accuracy_percent=normal_accuracy_percent(self._hit_margins),
            fail_misses=self._judgement_counts.get(TimingJudgement.FAIL_MISS, 0),
            fail_overloads=self._judgement_counts.get(TimingJudgement.FAIL_OVERLOAD, 0),
            hit_margin_count=margin_count,
            x_accuracy_points=xacc_points,
            x_accuracy_denominator=xacc_denominator,
        )

    @property
    def hit_margins(self) -> tuple[TimingJudgement, ...]:
        return tuple(self._hit_margins)

    @property
    def timing_errors_ms(self) -> tuple[float, ...]:
        return tuple(error * 1000.0 for error in self._errors_s)

    def privileged_next_target(self) -> PlayableChartTarget | None:
        """Evaluator/debug access only; never include this in policy input."""

        index = self._next_target_index()
        return None if index is None else self.segment.targets[index]

    def privileged_episode_time_s(self) -> float:
        """Evaluator/debug access only."""

        return self.motor.diagnostics().time_s
