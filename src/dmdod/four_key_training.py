from __future__ import annotations

"""Training-side primitives for the four-key Level A path.

This module deliberately stops short of installing a full trainer.  It provides
exactly the pieces the existing BC/DAgger pipeline needs next: a stateful
center-first privileged teacher, 251D/4D trajectory containers, the four-output
actuation loss, and expert/mixture rollout collection on the real-chart HUD
environment.
"""

from dataclasses import dataclass
import random

import torch

from .four_key_motor import (
    CENTER_KEY_NAMES,
    FOUR_KEY_NAMES,
    OUTER_KEY_NAMES,
    FourKeyAction,
    FourKeyObservation,
)
from .four_key_policy import FourKeyRecurrentActorCritic
from .four_key_real_chart import (
    FOUR_KEY_HUD_REAL_CHART_INPUT_DIM,
    DiagnosticHudFourKeyRealChartMotorEnv,
    FourKeyHudRealChartObservation,
    encode_four_key_hud_real_chart_observation,
)
from .real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


TEACHER_ACTIVE_THRESHOLD = 0.25
PRESS_MARGIN = 0.70
RELEASE_MARGIN = -0.30
NEUTRAL_PUSH_LIMIT = 0.05
MSE_COEF = 0.20
PRESS_MARGIN_COEF = 6.0
RELEASE_MARGIN_COEF = 2.0
NEUTRAL_PUSH_COEF = 12.0


@dataclass(frozen=True, slots=True)
class FourKeyDAggerSequence:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        if (
            self.observations.ndim != 2
            or self.observations.shape[1] != FOUR_KEY_HUD_REAL_CHART_INPUT_DIM
        ):
            raise ValueError(
                "four-key DAgger observations must have shape "
                f"[T, {FOUR_KEY_HUD_REAL_CHART_INPUT_DIM}]"
            )
        if self.teacher_actions.shape != (self.observations.shape[0], 4):
            raise ValueError("four-key DAgger teacher actions must have shape [T, 4]")

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class FourKeyRollout:
    sequence: FourKeyDAggerSequence
    stats: object
    physical_keydowns: int
    teacher_fraction: float


class CenterFirstFourKeyTeacher:
    """Privileged timing teacher with a latched center-first finger choice.

    The inner pair is always the first tier.  Outer keys are considered only
    while both inner keys are physically held.  Once a key is chosen for a
    target, the choice is latched until that key actually presses or the target
    changes; otherwise a 100 Hz teacher would alternate free inner keys on every
    control frame while waiting for the physical switch to actuate.

    During DAgger, a student's positive command may choose among currently
    eligible keys, but it cannot skip the center-first tier.
    """

    def __init__(self, *, preference_threshold: float = 0.15) -> None:
        self.preference_threshold = float(preference_threshold)
        self.reset()

    def reset(self) -> None:
        self._last_center = "right_inner"
        self._last_outer = "right_outer"
        self._pending_key: str | None = None
        self._target_token: object | None = None

    @staticmethod
    def _ordered_pair(pair: tuple[str, str], last: str) -> tuple[str, str]:
        return (pair[1], pair[0]) if last == pair[0] else pair

    @staticmethod
    def _preferred_value(action: FourKeyAction | None, key: str) -> float:
        if action is None:
            return float("-inf")
        return float(getattr(action, key))

    def _choose_from_tier(
        self,
        observation: FourKeyObservation,
        pair: tuple[str, str],
        *,
        last: str,
        preferred_action: FourKeyAction | None,
    ) -> str | None:
        available = [key for key in pair if not observation.pressed(key)]
        if not available:
            return None

        if preferred_action is not None:
            best = max(
                available,
                key=lambda key: self._preferred_value(preferred_action, key),
            )
            if self._preferred_value(preferred_action, best) >= self.preference_threshold:
                return best

        for key in self._ordered_pair(pair, last):
            if key in available:
                return key
        return available[0]

    def _choose_key(
        self,
        observation: FourKeyObservation,
        preferred_action: FourKeyAction | None,
    ) -> str | None:
        center = self._choose_from_tier(
            observation,
            CENTER_KEY_NAMES,
            last=self._last_center,
            preferred_action=preferred_action,
        )
        if center is not None:
            self._last_center = center
            return center

        outer = self._choose_from_tier(
            observation,
            OUTER_KEY_NAMES,
            last=self._last_outer,
            preferred_action=preferred_action,
        )
        if outer is not None:
            self._last_outer = outer
        return outer

    def action(
        self,
        observation: FourKeyObservation,
        *,
        now_s: float,
        target_time_s: float,
        lead_s: float,
        target_token: object | None,
        preferred_action: FourKeyAction | None = None,
    ) -> FourKeyAction:
        if target_token != self._target_token:
            self._target_token = target_token
            self._pending_key = None

        commands = {
            key: (-1.0 if observation.pressed(key) else 0.0)
            for key in FOUR_KEY_NAMES
        }

        if target_token is None or float(now_s) + float(lead_s) < float(target_time_s):
            self._pending_key = None
            return FourKeyAction(*(commands[key] for key in FOUR_KEY_NAMES))

        if self._pending_key is not None and observation.pressed(self._pending_key):
            self._pending_key = None

        if self._pending_key is None:
            self._pending_key = self._choose_key(observation, preferred_action)

        if self._pending_key is not None:
            commands[self._pending_key] = 1.0
        return FourKeyAction(*(commands[key] for key in FOUR_KEY_NAMES))


def four_key_actuation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Four-output equivalent of the mature two-key actuation-aware BC loss."""

    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("predicted and target must both have shape [T, 4]")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)

    mse = (predicted - target).square().mean()

    press_gap = torch.relu(PRESS_MARGIN - predicted).square()
    press_loss = (press_gap * press).sum() / press.sum().clamp_min(1)

    release_gap = torch.relu(predicted - RELEASE_MARGIN).square()
    release_loss = (release_gap * release).sum() / release.sum().clamp_min(1)

    unsafe_neutral_push = torch.relu(predicted - NEUTRAL_PUSH_LIMIT).square()
    neutral_loss = (unsafe_neutral_push * neutral).sum() / neutral.sum().clamp_min(1)

    return (
        MSE_COEF * mse
        + PRESS_MARGIN_COEF * press_loss
        + RELEASE_MARGIN_COEF * release_loss
        + NEUTRAL_PUSH_COEF * neutral_loss
    )


def _teacher_action(
    env: DiagnosticHudFourKeyRealChartMotorEnv,
    observation: FourKeyHudRealChartObservation,
    teacher: CenterFirstFourKeyTeacher,
    *,
    lead_s: float,
    preferred_action: FourKeyAction | None = None,
) -> FourKeyAction:
    target = env.privileged_next_target()
    if target is None:
        return teacher.action(
            observation.motor,
            now_s=env.privileged_episode_time_s(),
            target_time_s=float("inf"),
            lead_s=lead_s,
            target_token=None,
            preferred_action=preferred_action,
        )
    return teacher.action(
        observation.motor,
        now_s=env.privileged_episode_time_s(),
        target_time_s=target.episode_time_s,
        lead_s=lead_s,
        target_token=(target.floor_index, target.episode_time_s),
        preferred_action=preferred_action,
    )


def collect_four_key_expert_sequence(
    segment,
    *,
    lead_s: float,
    control_dt_s: float,
    device: torch.device,
    source: str = "four-key-center-first-expert",
) -> FourKeyRollout:
    env = DiagnosticHudFourKeyRealChartMotorEnv(
        segment,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    teacher = CenterFirstFourKeyTeacher()
    observation = env.reset()
    observations: list[tuple[float, ...]] = []
    actions: list[tuple[float, float, float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        action = _teacher_action(env, observation, teacher, lead_s=lead_s)
        observations.append(encode_four_key_hud_real_chart_observation(observation))
        actions.append(action.as_tuple())
        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("four-key expert episode exceeded step budget")

    sequence = FourKeyDAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(actions, dtype=torch.float32, device=device),
        source=source,
    )
    return FourKeyRollout(sequence, env.stats, env.physical_keydowns, 1.0)


def collect_four_key_mixture_rollout(
    model: FourKeyRecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    teacher_fraction: float,
    control_dt_s: float,
    device: torch.device,
    source: str,
    seed: int,
) -> FourKeyRollout:
    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must be in [0, 1]")

    env = DiagnosticHudFourKeyRealChartMotorEnv(
        segment,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    teacher = CenterFirstFourKeyTeacher()
    observation = env.reset()
    state = model.initial_state(device)
    rng = random.Random(seed)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, float, float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            encoded = encode_four_key_hud_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            student, state = model.deterministic_action(x, state)
            teacher_action = _teacher_action(
                env,
                observation,
                teacher,
                lead_s=lead_s,
                preferred_action=student,
            )

            observations.append(encoded)
            labels.append(teacher_action.as_tuple())
            applied = teacher_action if rng.random() < teacher_fraction else student
            step = env.step(applied)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("four-key DAgger rollout exceeded step budget")

    sequence = FourKeyDAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        source=source,
    )
    return FourKeyRollout(
        sequence,
        env.stats,
        env.physical_keydowns,
        float(teacher_fraction),
    )
