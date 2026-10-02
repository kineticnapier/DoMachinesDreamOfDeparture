from __future__ import annotations

"""Teacher-only N-key real-chart capacity diagnostics.

This module is intentionally evaluator-side.  It uses privileged future target
times to answer a physical question: with the frozen motor profile and a given
number of keys, how much of a dense chart can a center-first scheduler actually
produce?  No policy is trained here.
"""

from dataclasses import dataclass

from .keyboard import KeyEvent
from .n_key_motor import (
    NKeyAction,
    NKeyMotorEnv,
    NKeyObservation,
    n_key_names,
    n_key_tiers,
)
from .real_chart_env import RealChartMotorEnv, RealChartStep


@dataclass(frozen=True, slots=True)
class NKeyLeadCalibration:
    key_count: int
    key_latencies_s: tuple[tuple[str, float], ...]
    lead_s: float
    control_dt_s: float
    physics_dt_s: float

    def latency_for(self, key: str) -> float:
        for candidate, latency in self.key_latencies_s:
            if candidate == key:
                return float(latency)
        raise KeyError(key)


def _single_key_action(key_count: int, index: int) -> NKeyAction:
    values = [0.0] * key_count
    values[index] = 1.0
    return NKeyAction(tuple(values))


def calibrate_n_key_press_lead(
    key_count: int,
    *,
    control_dt_s: float = 0.010,
    physics_dt_s: float = 0.001,
    max_wait_s: float = 1.0,
) -> NKeyLeadCalibration:
    names = n_key_names(key_count)
    if max_wait_s <= 0.0:
        raise ValueError("max_wait_s must be positive")

    latencies: list[tuple[str, float]] = []
    for index, key in enumerate(names):
        env = NKeyMotorEnv(
            key_count,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
        )
        env.reset()
        action = _single_key_action(key_count, index)
        max_steps = max(1, int(max_wait_s / control_dt_s) + 1)
        found = None
        for _ in range(max_steps):
            transition = env.step(action)
            for event in transition.evaluator_events:
                if event.event is KeyEvent.DOWN and event.key == key:
                    found = float(event.time_s)
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"N-key calibration produced no {key} key-down event")
        latencies.append((key, found))

    inner = n_key_tiers(key_count)[0]
    lead = max(dict(latencies)[inner[0]], dict(latencies)[inner[1]]) + 0.5 * control_dt_s
    return NKeyLeadCalibration(
        key_count=int(key_count),
        key_latencies_s=tuple(latencies),
        lead_s=float(lead),
        control_dt_s=float(control_dt_s),
        physics_dt_s=float(physics_dt_s),
    )


class NKeyRealChartMotorEnv(RealChartMotorEnv):
    """Real-chart evaluator driven by the configurable N-key body."""

    def __init__(
        self,
        segment,
        *,
        key_count: int,
        control_dt_s: float = 0.010,
        physics_dt_s: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(segment, same_hand=True, control_dt_s=control_dt_s, **kwargs)
        self.key_count = int(key_count)
        self.key_names = n_key_names(self.key_count)
        self.motor = NKeyMotorEnv(
            self.key_count,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
        )
        self.physical_keydowns = 0

    def reset(self):
        self.physical_keydowns = 0
        return super().reset()

    def _score_event(self, event) -> float:
        if event.event is KeyEvent.DOWN:
            self.physical_keydowns += 1
        return super()._score_event(event)

    def step(self, action: NKeyAction) -> RealChartStep:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        if len(action.values) != self.key_count:
            raise ValueError(f"expected {self.key_count} action values")

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
        reward -= self.reward_config.effort_penalty * sum(abs(value) for value in action.values)

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


class CenterFirstNKeyTeacher:
    """Reserve unresolved future targets from the center pair outward."""

    def __init__(self, key_count: int) -> None:
        self.key_count = int(key_count)
        self.key_names = n_key_names(self.key_count)
        self.tiers = n_key_tiers(self.key_count)
        self.reset()

    def reset(self) -> None:
        self._last_by_tier = {
            tier_index: pair[1]
            for tier_index, pair in enumerate(self.tiers)
        }
        self._reservations: dict[str, tuple[object, float]] = {}

    @staticmethod
    def _contains_token(targets: tuple[tuple[object, float], ...], token: object) -> bool:
        return any(candidate == token for candidate, _ in targets)

    def _choose_key(self, observation: NKeyObservation, occupied: set[str]) -> str | None:
        for tier_index, pair in enumerate(self.tiers):
            last = self._last_by_tier[tier_index]
            order = (pair[1], pair[0]) if last == pair[0] else pair
            for key in order:
                if key not in occupied and not observation.pressed(key):
                    self._last_by_tier[tier_index] = key
                    return key
        return None

    def pipeline_action(
        self,
        observation: NKeyObservation,
        *,
        now_s: float,
        targets: tuple[tuple[object, float], ...],
        lead_s: float,
    ) -> NKeyAction:
        now_s = float(now_s)
        lead_s = float(lead_s)

        for key, (token, _) in tuple(self._reservations.items()):
            if not self._contains_token(targets, token):
                del self._reservations[key]

        commands = {
            key: (-1.0 if observation.pressed(key) else 0.0)
            for key in self.key_names
        }
        for key, (_, target_time_s) in self._reservations.items():
            if not observation.pressed(key) and now_s + lead_s >= float(target_time_s):
                commands[key] = 1.0

        occupied = {
            key for key in self.key_names if observation.pressed(key)
        } | set(self._reservations)
        reserved_tokens = [token for token, _ in self._reservations.values()]

        for token, target_time_s in targets:
            target_time_s = float(target_time_s)
            if now_s + lead_s < target_time_s:
                break
            if any(token == reserved for reserved in reserved_tokens):
                continue
            key = self._choose_key(observation, occupied)
            if key is None:
                break
            self._reservations[key] = (token, target_time_s)
            reserved_tokens.append(token)
            occupied.add(key)
            commands[key] = 1.0

        return NKeyAction(tuple(commands[key] for key in self.key_names))


@dataclass(frozen=True, slots=True)
class NKeyTeacherCapacityResult:
    key_count: int
    stats: object
    physical_keydowns: int
    max_due_targets: int
    max_reservations: int
    max_pressed_keys: int
    max_blocked_due_targets: int
    blocked_unique_targets: int
    reservation_capacity_frames: int
    release_wait_frames: int
    other_block_frames: int
    missed_targets: int
    missed_after_capacity_block: int
    missed_without_capacity_block: int
    peak_targets_per_lead_window: int
    min_target_gap_ms: float | None


def _target_token(target) -> tuple[int, int, float]:
    return (int(target.ordinal), int(target.floor_index), float(target.episode_time_s))


def _unresolved_tail(env: NKeyRealChartMotorEnv) -> tuple[tuple[object, float], ...]:
    target = env.privileged_next_target()
    if target is None:
        return ()
    return tuple(
        (_target_token(future), float(future.episode_time_s))
        for future in env.segment.targets[int(target.ordinal) :]
    )


def _peak_targets_in_window(segment, window_s: float) -> int:
    times = [float(target.episode_time_s) for target in segment.targets]
    if not times:
        return 0
    left = 0
    best = 0
    for right, time_s in enumerate(times):
        while left <= right and time_s - times[left] > window_s + 1e-12:
            left += 1
        best = max(best, right - left + 1)
    return best


def _min_target_gap_ms(segment) -> float | None:
    times = [float(target.episode_time_s) for target in segment.targets]
    if len(times) < 2:
        return None
    return min((b - a) * 1000.0 for a, b in zip(times, times[1:]))


def diagnose_n_key_teacher_capacity(
    segment,
    *,
    key_count: int,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float = 0.001,
) -> NKeyTeacherCapacityResult:
    env = NKeyRealChartMotorEnv(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
    )
    teacher = CenterFirstNKeyTeacher(key_count)
    observation = env.reset()

    max_due = 0
    max_reservations = 0
    max_pressed = 0
    max_blocked = 0
    reservation_capacity_frames = 0
    release_wait_frames = 0
    other_block_frames = 0
    blocked_tokens: set[object] = set()

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        now_s = float(env.privileged_episode_time_s())
        targets = _unresolved_tail(env)
        due_tokens = [
            token
            for token, target_time_s in targets
            if now_s + float(lead_s) >= float(target_time_s)
        ]
        max_due = max(max_due, len(due_tokens))

        action = teacher.pipeline_action(
            observation.motor,
            now_s=now_s,
            targets=targets,
            lead_s=lead_s,
        )
        reservations = dict(teacher._reservations)  # debug-only privileged state
        reserved_tokens = {token for token, _ in reservations.values()}
        pressed = {
            key for key in env.key_names if observation.motor.pressed(key)
        }
        occupied = pressed | set(reservations)
        blocked = [token for token in due_tokens if token not in reserved_tokens]

        max_reservations = max(max_reservations, len(reservations))
        max_pressed = max(max_pressed, len(pressed))
        max_blocked = max(max_blocked, len(blocked))
        blocked_tokens.update(blocked)

        if blocked:
            if len(reservations) >= key_count:
                reservation_capacity_frames += 1
            elif len(occupied) >= key_count:
                release_wait_frames += 1
            else:
                other_block_frames += 1

        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("N-key teacher capacity diagnostic exceeded step budget")

    missed_tokens = {
        _target_token(target)
        for target, missed in zip(segment.targets, env._missed)  # debug-only evaluator state
        if missed
    }
    capacity_misses = missed_tokens & blocked_tokens

    return NKeyTeacherCapacityResult(
        key_count=int(key_count),
        stats=env.stats,
        physical_keydowns=int(env.physical_keydowns),
        max_due_targets=max_due,
        max_reservations=max_reservations,
        max_pressed_keys=max_pressed,
        max_blocked_due_targets=max_blocked,
        blocked_unique_targets=len(blocked_tokens),
        reservation_capacity_frames=reservation_capacity_frames,
        release_wait_frames=release_wait_frames,
        other_block_frames=other_block_frames,
        missed_targets=len(missed_tokens),
        missed_after_capacity_block=len(capacity_misses),
        missed_without_capacity_block=len(missed_tokens - blocked_tokens),
        peak_targets_per_lead_window=_peak_targets_in_window(segment, float(lead_s)),
        min_target_gap_ms=_min_target_gap_ms(segment),
    )
