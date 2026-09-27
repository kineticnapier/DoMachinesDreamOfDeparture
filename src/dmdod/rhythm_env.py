from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .adofai_rules import (
    OverloadCounter,
    TimingJudgement,
    TimingWindows,
    classify_normal_timing,
    normal_timing_windows,
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
    """RL reward shaping layered on top of ADOFAI-like game mechanics.

    Timing categories and overload are structural game rules. These numeric
    rewards are only learning signals and are not ADOFAI score/accuracy values.
    """

    timing_sigma_s: float = 0.045
    hit_reward: float = 1.0
    timing_bonus: float = 1.0
    miss_penalty: float = 1.0
    too_early_penalty: float = 0.20
    overload_penalty: float = 4.0
    effort_penalty: float = 0.001
    fail_on_miss: bool = False

    def __post_init__(self) -> None:
        if self.timing_sigma_s <= 0.0:
            raise ValueError("timing_sigma_s must be positive")
        if self.overload_penalty < 0.0:
            raise ValueError("overload_penalty must be non-negative")


@dataclass(frozen=True)
class EpisodeStats:
    targets: int
    hits: int
    misses: int
    too_early_presses: int
    overload_counter: int
    overloaded: bool
    perfects: int
    early_late_perfects: int
    early_late_hits: int
    mean_abs_error_ms: float | None
    total_reward: float

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
    """Toy rhythm task around MotorEnv using ADOFAI Normal timing mechanics.

    Exact target timestamps remain private. The policy receives only body state
    and a cue for the next unresolved target. Once a tile is hit or missed, its
    cue disappears and perception advances to the following target. This keeps
    the toy observation closer to ADOFAI's progressing current/next-tile state
    instead of blending already-resolved notes with future notes.

    Timing judgements use the Normal timing option: 30 degree Perfect, 45 degree
    E/L Perfect, 60 degree Pass, with the timing window tightening with BPM until
    310 BPM and staying fixed above it.

    Too Early inputs update the overload counter (+2); valid tile hits reduce it
    by 1 without going below zero; reaching 6 ends the episode with OVERLOAD.
    ``fail_on_miss`` stays configurable because early toy-RL curriculum benefits
    from continuing after misses, while a later gameplay environment can enable
    real fail-on-miss semantics.
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
        tail_s: float = 0.400,
    ) -> None:
        if not targets:
            raise ValueError("at least one target is required")
        if tail_s < 0.0:
            raise ValueError("tail_s must be non-negative")

        self._targets = tuple(sorted(targets, key=lambda target: target.time_s))
        self.bpm = self._resolve_bpm(bpm)
        self.timing_windows = normal_timing_windows(self.bpm)
        self.motor = MotorEnv(same_hand=same_hand, control_dt_s=control_dt_s)
        self.reward_config = reward_config or RewardConfig()
        self._cue_encoder = VisualCueEncoder(self._targets, config=cue_config)
        self._episode_end_s = self._targets[-1].time_s + tail_s

        self._used: list[bool] = []
        self._missed: list[bool] = []
        self._errors_s: list[float] = []
        self._hits = 0
        self._misses = 0
        self._too_early = 0
        self._overload = OverloadCounter()
        self._judgement_counts: dict[TimingJudgement, int] = {}
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
        self._judgement_counts = {judgement: 0 for judgement in TimingJudgement}
        self._total_reward = 0.0
        self._done = False
        self._failed_on_miss = False
        return RhythmObservation(motor_observation, self._cue_observation(0.0))

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

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is not KeyEvent.DOWN:
            return 0.0

        target_index = self._next_target_index(event.key)
        if target_index is None:
            return 0.0

        target = self._targets[target_index]
        signed_error = event.time_s - target.time_s
        judgement = classify_normal_timing(signed_error, self.bpm)
        self._judgement_counts[judgement] += 1

        if judgement is TimingJudgement.TOO_EARLY:
            self._too_early += 1
            overloaded_now = self._overload.record_too_early()
            reward = -self.reward_config.too_early_penalty
            if overloaded_now:
                reward -= self.reward_config.overload_penalty
            return reward

        if judgement is TimingJudgement.TOO_LATE:
            if not self._missed[target_index]:
                self._missed[target_index] = True
                self._misses += 1
                if self.reward_config.fail_on_miss:
                    self._failed_on_miss = True
                return -self.reward_config.miss_penalty
            return 0.0

        self._used[target_index] = True
        self._hits += 1
        self._errors_s.append(signed_error)
        self._overload.record_valid_hit()
        timing_quality = exp(-0.5 * (signed_error / self.reward_config.timing_sigma_s) ** 2)
        return self.reward_config.hit_reward + self.reward_config.timing_bonus * timing_quality

    def _expire_misses(self, now_s: float) -> float:
        reward = 0.0
        pass_window = self.timing_windows.pass_s
        for i, target in enumerate(self._targets):
            if self._used[i] or self._missed[i]:
                continue
            if target.time_s + pass_window < now_s:
                self._missed[i] = True
                self._misses += 1
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
            reward += self._expire_misses(event.time_s)
            if self._failed_on_miss:
                break
            reward += self._score_event(event)
            if self._overload.overloaded:
                break

        now = transition.diagnostics.time_s
        if not self._failed_on_miss and not self._overload.overloaded:
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
            mean_abs_error_ms = sum(abs(error) for error in self._errors_s) / len(self._errors_s) * 1000.0
        return EpisodeStats(
            targets=len(self._targets),
            hits=self._hits,
            misses=self._misses,
            too_early_presses=self._too_early,
            overload_counter=self._overload.value,
            overloaded=self._overload.overloaded,
            perfects=self._judgement_counts.get(TimingJudgement.PERFECT, 0),
            early_late_perfects=(
                self._judgement_counts.get(TimingJudgement.EARLY_PERFECT, 0)
                + self._judgement_counts.get(TimingJudgement.LATE_PERFECT, 0)
            ),
            early_late_hits=(
                self._judgement_counts.get(TimingJudgement.EARLY, 0)
                + self._judgement_counts.get(TimingJudgement.LATE, 0)
            ),
            mean_abs_error_ms=mean_abs_error_ms,
            total_reward=self._total_reward,
        )

    @property
    def timing_errors_ms(self) -> tuple[float, ...]:
        """Privileged signed timing errors for offline evaluation only."""

        return tuple(error * 1000.0 for error in self._errors_s)


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
