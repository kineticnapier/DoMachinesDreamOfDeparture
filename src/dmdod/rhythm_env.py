from __future__ import annotations

from dataclasses import dataclass
from math import exp

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
    hit_window_s: float = 0.120
    timing_sigma_s: float = 0.045
    hit_reward: float = 1.0
    timing_bonus: float = 1.0
    miss_penalty: float = 1.0
    stray_penalty: float = 0.20
    effort_penalty: float = 0.001

    def __post_init__(self) -> None:
        if self.hit_window_s <= 0.0 or self.timing_sigma_s <= 0.0:
            raise ValueError("hit_window_s and timing_sigma_s must be positive")


@dataclass(frozen=True)
class EpisodeStats:
    targets: int
    hits: int
    misses: int
    stray_presses: int
    mean_abs_error_ms: float | None
    total_reward: float


@dataclass(frozen=True)
class RhythmStep:
    observation: RhythmObservation
    reward: float
    done: bool


class RhythmMotorEnv:
    """First RL task wrapper around MotorEnv.

    Target timestamps are private to this environment.  The policy receives only
    MotorObservation and smooth visual cue amplitudes.  Reward may depend on the
    privileged timing truth, as is normal in reinforcement learning, but the
    timing error itself is never included in the observation.
    """

    def __init__(
        self,
        targets: list[TargetHit] | tuple[TargetHit, ...],
        *,
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
        self.motor = MotorEnv(same_hand=same_hand, control_dt_s=control_dt_s)
        self.reward_config = reward_config or RewardConfig()
        self._cue_encoder = VisualCueEncoder(self._targets, config=cue_config)
        self._episode_end_s = self._targets[-1].time_s + tail_s

        self._used: list[bool] = []
        self._missed: list[bool] = []
        self._errors_s: list[float] = []
        self._hits = 0
        self._misses = 0
        self._strays = 0
        self._total_reward = 0.0
        self._done = False

    def reset(self) -> RhythmObservation:
        motor_observation = self.motor.reset()
        self._used = [False] * len(self._targets)
        self._missed = [False] * len(self._targets)
        self._errors_s = []
        self._hits = 0
        self._misses = 0
        self._strays = 0
        self._total_reward = 0.0
        self._done = False
        return RhythmObservation(motor_observation, self._cue_encoder.observe(0.0))

    def _score_event(self, event: TimedKeyEvent) -> float:
        if event.event is not KeyEvent.DOWN:
            return 0.0

        cfg = self.reward_config
        best_index: int | None = None
        best_abs_error = cfg.hit_window_s + 1.0
        signed_error = 0.0

        for i, target in enumerate(self._targets):
            if self._used[i] or self._missed[i] or target.key != event.key:
                continue
            error = event.time_s - target.time_s
            abs_error = abs(error)
            if abs_error <= cfg.hit_window_s and abs_error < best_abs_error:
                best_index = i
                best_abs_error = abs_error
                signed_error = error

        if best_index is None:
            self._strays += 1
            return -cfg.stray_penalty

        self._used[best_index] = True
        self._hits += 1
        self._errors_s.append(signed_error)
        timing_quality = exp(-0.5 * (signed_error / cfg.timing_sigma_s) ** 2)
        return cfg.hit_reward + cfg.timing_bonus * timing_quality

    def _expire_misses(self, now_s: float) -> float:
        reward = 0.0
        cfg = self.reward_config
        for i, target in enumerate(self._targets):
            if self._used[i] or self._missed[i]:
                continue
            if target.time_s + cfg.hit_window_s < now_s:
                self._missed[i] = True
                self._misses += 1
                reward -= cfg.miss_penalty
        return reward

    def observe(self) -> RhythmObservation:
        # simulation time is deliberately consumed only inside the perception
        # module; it is not stored in RhythmObservation.
        now = self.motor.diagnostics().time_s
        return RhythmObservation(self.motor.observe(), self._cue_encoder.observe(now))

    def step(self, action: MotorAction) -> RhythmStep:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")

        transition = self.motor.step(action)
        reward = 0.0
        for event in transition.evaluator_events:
            reward += self._score_event(event)

        now = transition.diagnostics.time_s
        reward += self._expire_misses(now)
        reward -= self.reward_config.effort_penalty * (abs(action.left) + abs(action.right))

        self._done = now >= self._episode_end_s
        if self._done:
            reward += self._expire_misses(float("inf"))

        self._total_reward += reward
        observation = RhythmObservation(transition.observation, self._cue_encoder.observe(now))
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
            stray_presses=self._strays,
            mean_abs_error_ms=mean_abs_error_ms,
            total_reward=self._total_reward,
        )

    @property
    def timing_errors_ms(self) -> tuple[float, ...]:
        """Privileged signed timing errors for offline evaluation only.

        This property must not be copied into policy observations.  It exists so
        deterministic evaluation can report distribution statistics such as P95
        without weakening the agent/ground-truth separation.
        """

        return tuple(error * 1000.0 for error in self._errors_s)


def make_regular_targets(
    *,
    bpm: float,
    count: int,
    start_s: float = 0.750,
    pattern: str = "left",
) -> list[TargetHit]:
    """Build a synthetic training chart.  The returned times stay env-private."""
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
