from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .adofai_rules import (
    OverloadCounter,
    TimingDifficulty,
    TimingJudgement,
    TimingWindows,
    classify_timing,
    normal_accuracy_percent,
    timing_windows,
    x_accuracy_components,
    x_accuracy_percent,
    x_accuracy_weight,
)
from .evaluator import TargetHit
from .keyboard import KeyEvent
from .motor_env import MotorAction, MotorEnv, MotorObservation, TimedKeyEvent
from .perception import VisualCueConfig, VisualCueEncoder, VisualCueObservation


@dataclass(frozen=True)
class RhythmObservation:
    """Policy input: body state plus toy visual cues, with no exact timing."""

    motor: MotorObservation
    cue: VisualCueObservation


@dataclass(frozen=True)
class RewardConfig:
    """Accuracy-first learning reward on top of ADOFAI-like mechanics.

    The main quality term uses ADOFAI's X-Accuracy judgement weights. A smaller
    continuous center bonus breaks the otherwise-flat Perfect plateau so the
    policy still benefits from moving toward the middle of Perfect.

    These are RL reward magnitudes, not ADOFAI score values.
    """

    hit_reward: float = 0.20
    xacc_reward: float = 1.80
    timing_bonus: float = 0.50
    perfect_sigma_fraction: float = 0.35
    # Optional compatibility override. When None, sigma follows the current
    # DLL-derived Perfect window instead of using a fixed number of milliseconds.
    timing_sigma_s: float | None = None
    miss_penalty: float = 1.0
    too_early_penalty: float = 0.20
    overload_penalty: float = 4.0
    effort_penalty: float = 0.001
    fail_on_miss: bool = False

    def __post_init__(self) -> None:
        if self.timing_sigma_s is not None and self.timing_sigma_s <= 0.0:
            raise ValueError("timing_sigma_s must be positive when supplied")
        if self.perfect_sigma_fraction <= 0.0:
            raise ValueError("perfect_sigma_fraction must be positive")
        if self.xacc_reward < 0.0 or self.timing_bonus < 0.0:
            raise ValueError("accuracy rewards must be non-negative")
        if self.overload_penalty < 0.0:
            raise ValueError("overload_penalty must be non-negative")


@dataclass(frozen=True)
class EpisodeStats:
    targets: int
    hits: int
    misses: int
    too_early_presses: int
    overload_counter: float
    overloaded: bool
    perfects: int
    early_late_perfects: int
    early_late_hits: int
    mean_abs_error_ms: float | None
    x_accuracy_percent: float
    perfect_rate: float
    total_reward: float
    accuracy_percent: float
    fail_misses: int
    fail_overloads: int
    hit_margin_count: int
    x_accuracy_points: float
    x_accuracy_denominator: int

    @property
    def stray_presses(self) -> int:
        """Compatibility alias for the old toy metric."""

        return self.too_early_presses


@dataclass(frozen=True)
class RhythmStep:
    observation: RhythmObservation
    reward: float
    done: bool


class RhythmMotorEnv:
    """Rhythm task around MotorEnv with DLL-derived ADOFAI timing/fail rules.

    Exact target timestamps remain private. The policy receives only body state
    and a cue for the next unresolved target. Timing classification supports
    Lenient/Normal/Strict, ScaleMargin (``timing_scale``), speed, pitch,
    speed-trial adjustment, and the DLL's mobile timing minima.

    The ordinary TooEarly fail-bar path follows the inspected DLL: TooEarly adds
    0.5, valid hits do not heal it, song progress decays it by 0.4 per beat, and
    FailOverload occurs only when the counter is strictly greater than 1.0.
    Multipress remains a separate future implementation.
    """

    def __init__(
        self,
        targets: list[TargetHit] | tuple[TargetHit, ...],
        *,
        bpm: float | None = None,
        same_hand: bool = True,
        control_dt_s: float = 0.010,
        reward_config: RewardConfig | None = None,
        cue_config: VisualCueConfig | None = None,
        perception_seed: int | None = None,
        tail_s: float = 0.400,
        difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
        timing_scale: float = 1.0,
        controller_speed: float = 1.0,
        pitch: float = 1.0,
        speed_trial: float = 1.0,
        mobile: bool = False,
    ) -> None:
        if not targets:
            raise ValueError("at least one target is required")
        if tail_s < 0.0:
            raise ValueError("tail_s must be non-negative")

        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.bpm = self._resolve_bpm(bpm)
        self.difficulty = (
            TimingDifficulty(str(difficulty).lower())
            if not isinstance(difficulty, TimingDifficulty)
            else difficulty
        )
        self.timing_scale = timing_scale
        self.controller_speed = controller_speed
        self.pitch = pitch
        self.speed_trial = speed_trial
        self.mobile = mobile
        self.timing_windows = timing_windows(
            self.bpm,
            difficulty=self.difficulty,
            timing_scale=self.timing_scale,
            controller_speed=self.controller_speed,
            pitch=self.pitch,
            speed_trial=self.speed_trial,
            mobile=self.mobile,
        )
        self.motor = MotorEnv(same_hand=same_hand, control_dt_s=control_dt_s)
        self.reward_config = reward_config or RewardConfig()
        self._perception_seed = perception_seed
        self._cue_encoder = VisualCueEncoder(
            self._targets,
            config=cue_config,
            seed=perception_seed,
        )
        self._episode_end_s = self._targets[-1].time_s + tail_s

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

    def _resolve_bpm(self, bpm: float | None) -> float:
        if bpm is not None:
            if bpm <= 0.0:
                raise ValueError("bpm must be positive")
            return bpm
        if len(self._targets) < 2:
            raise ValueError("bpm is required when fewer than two targets are supplied")
        interval = self._targets[1].time_s - self._targets[0].time_s
        if interval <= 0.0:
            raise ValueError("cannot infer bpm from non-increasing targets")
        return 60.0 / interval

    def reset(self) -> RhythmObservation:
        motor_observation = self.motor.reset()
        self._used = [False] * len(self._targets)
        self._missed = [False] * len(self._targets)
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
        self._cue_encoder.reset(seed=self._perception_seed)
        return RhythmObservation(motor_observation, self._cue_observation(0.0))

    def _record_margin(self, judgement: TimingJudgement) -> None:
        self._hit_margins.append(judgement)
        self._judgement_counts[judgement] += 1

    def _advance_overload(self, to_time_s: float) -> None:
        """Apply DLL overload recovery for elapsed synthetic song time."""

        if to_time_s <= self._overload_clock_s:
            return
        delta_s = to_time_s - self._overload_clock_s
        beat_delta = delta_s / (60.0 / self.bpm)
        self._overload.advance_beats(beat_delta)
        self._overload_clock_s = to_time_s

    def _next_unresolved_target_index(self) -> int | None:
        for i, (used, missed) in enumerate(zip(self._used, self._missed)):
            if not used and not missed:
                return i
        return None

    def _cue_observation(self, now_s: float) -> VisualCueObservation:
        target_index = self._next_unresolved_target_index()
        active = () if target_index is None else (target_index,)
        return self._cue_encoder.observe(now_s, active_target_indices=active)

    def _next_target_index(self, key: str) -> int | None:
        for i, target in enumerate(self._targets):
            if self._used[i] or self._missed[i] or target.key != key:
                continue
            return i
        return None

    def _classify(self, signed_error: float) -> TimingJudgement:
        return classify_timing(
            signed_error,
            self.bpm,
            difficulty=self.difficulty,
            timing_scale=self.timing_scale,
            controller_speed=self.controller_speed,
            pitch=self.pitch,
            speed_trial=self.speed_trial,
            mobile=self.mobile,
        )

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is not KeyEvent.DOWN:
            return 0.0

        target_index = self._next_target_index(event.key)
        if target_index is None:
            return 0.0

        target = self._targets[target_index]
        signed_error = event.time_s - target.time_s
        judgement = self._classify(signed_error)

        if judgement is TimingJudgement.TOO_EARLY:
            self._too_early += 1
            overloaded_now = self._overload.record_too_early()
            # scrPlanet.SwitchChosen replaces the triggering TooEarly margin with
            # FailOverload when OnDamage reports that the fail bar crossed > 1.
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
            if not self._missed[target_index]:
                self._missed[target_index] = True
                self._misses += 1
                if self.reward_config.fail_on_miss:
                    self._failed_on_miss = True
                return -self.reward_config.miss_penalty
            return 0.0

        self._record_margin(judgement)
        self._used[target_index] = True
        self._hits += 1
        self._errors_s.append(signed_error)
        # Kept as a compatibility call; the DLL valid-hit path does not heal
        # the overload counter directly.
        self._overload.record_valid_hit()

        xacc_quality = x_accuracy_weight(judgement)
        sigma = self.reward_config.timing_sigma_s
        if sigma is None:
            sigma = max(
                1e-6,
                self.timing_windows.perfect_s * self.reward_config.perfect_sigma_fraction,
            )
        center_quality = exp(-0.5 * (signed_error / sigma) ** 2)
        return (
            self.reward_config.hit_reward
            + self.reward_config.xacc_reward * xacc_quality
            + self.reward_config.timing_bonus * center_quality
        )

    def _expire_misses(self, now_s: float) -> float:
        reward = 0.0
        pass_window = self.timing_windows.pass_s
        for i, target in enumerate(self._targets):
            if self._used[i] or self._missed[i]:
                continue
            if target.time_s + pass_window < now_s:
                self._missed[i] = True
                self._misses += 1
                self._record_margin(TimingJudgement.FAIL_MISS)
                reward -= self.reward_config.miss_penalty
                if self.reward_config.fail_on_miss:
                    self._failed_on_miss = True
        return reward

    def observe(self) -> RhythmObservation:
        now = self.motor.diagnostics().time_s
        return RhythmObservation(self.motor.observe(), self._cue_observation(now))

    def step(self, action: MotorAction) -> RhythmStep:
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
        observation = RhythmObservation(transition.observation, self._cue_observation(now))
        return RhythmStep(observation, reward, self._done)

    @property
    def stats(self) -> EpisodeStats:
        mean_abs_error_ms = None
        if self._errors_s:
            mean_abs_error_ms = (
                sum(abs(error) for error in self._errors_s)
                / len(self._errors_s)
                * 1000.0
            )
        perfects = self._judgement_counts.get(TimingJudgement.PERFECT, 0)
        xacc_points, xacc_denominator = x_accuracy_components(self._hit_margins)
        margin_count = len(self._hit_margins)
        return EpisodeStats(
            targets=len(self._targets),
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
            # A stray/non-Perfect HitMargin must prevent a nominal 100% PP rate,
            # even if the eventual target hit itself was Perfect.
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
    def timing_errors_ms(self) -> tuple[float, ...]:
        """Privileged signed timing errors for offline evaluation only."""

        return tuple(error * 1000.0 for error in self._errors_s)

    @property
    def hit_margins(self) -> tuple[TimingJudgement, ...]:
        """Privileged DLL-style score history for offline evaluation/tests."""

        return tuple(self._hit_margins)


def make_regular_targets(
    *,
    bpm: float,
    count: int,
    start_s: float = 0.750,
    pattern: str = "left",
) -> list[TargetHit]:
    """Build a synthetic straight-tile training chart.

    Straight 180 degree tiles are one beat apart, so the interval is 60/BPM.
    Exact returned timestamps stay environment-private once the task is built.
    """

    if bpm <= 0.0 or count <= 0:
        raise ValueError("bpm and count must be positive")
    if start_s < 0.0:
        raise ValueError("start_s must be non-negative")
    if pattern not in {"left", "alternate"}:
        raise ValueError("pattern must be 'left' or 'alternate'")

    interval = 60.0 / bpm
    targets: list[TargetHit] = []
    for i in range(count):
        key = "left" if pattern == "left" or i % 2 == 0 else "right"
        targets.append(TargetHit(start_s + i * interval, key))
    return targets
