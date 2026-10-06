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

from dmdod.legacy.four_key.motor import (
    CENTER_KEY_NAMES,
    FOUR_KEY_NAMES,
    OUTER_KEY_NAMES,
    FourKeyAction,
    FourKeyObservation,
)
from dmdod.legacy.four_key.policy import FourKeyRecurrentActorCritic
from dmdod.legacy.four_key.real_chart import (
    FOUR_KEY_HUD_REAL_CHART_INPUT_DIM,
    DiagnosticHudFourKeyRealChartMotorEnv,
    FourKeyHudRealChartObservation,
    encode_four_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG


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
    """Privileged center-first teacher with an in-flight target pipeline.

    A dense chart can expose the next launch point before the key for the
    previous target has physically actuated.  The teacher therefore reserves
    different fingers for multiple future targets at once instead of waiting
    for ``privileged_next_target()`` to advance after every KeyDown.

    Reservations are center-first: the two inner keys are occupied first and
    the outer tier is unlocked only while both inner keys are physically held
    or already reserved for earlier targets.  A reservation stays latched until
    its target resolves, preventing 100 Hz routing from swapping fingers while
    the switch is still travelling.  During DAgger, student preference may pick
    a key only inside the currently eligible tier.
    """

    def __init__(self, *, preference_threshold: float = 0.15) -> None:
        self.preference_threshold = float(preference_threshold)
        self.reset()

    def reset(self) -> None:
        self._last_center = "right_inner"
        self._last_outer = "right_outer"
        self._reservations: dict[str, tuple[object, float]] = {}

    @staticmethod
    def _ordered_pair(pair: tuple[str, str], last: str) -> tuple[str, str]:
        return (pair[1], pair[0]) if last == pair[0] else pair

    @staticmethod
    def _preferred_value(action: FourKeyAction | None, key: str) -> float:
        if action is None:
            return float("-inf")
        return float(getattr(action, key))

    @staticmethod
    def _contains_token(
        targets: tuple[tuple[object, float], ...],
        token: object,
    ) -> bool:
        return any(candidate == token for candidate, _ in targets)

    def _choose_from_tier(
        self,
        observation: FourKeyObservation,
        pair: tuple[str, str],
        *,
        last: str,
        occupied: set[str],
        preferred_action: FourKeyAction | None,
    ) -> str | None:
        available = [
            key
            for key in pair
            if key not in occupied and not observation.pressed(key)
        ]
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
        *,
        occupied: set[str],
        preferred_action: FourKeyAction | None,
    ) -> str | None:
        center = self._choose_from_tier(
            observation,
            CENTER_KEY_NAMES,
            last=self._last_center,
            occupied=occupied,
            preferred_action=preferred_action,
        )
        if center is not None:
            self._last_center = center
            return center

        outer = self._choose_from_tier(
            observation,
            OUTER_KEY_NAMES,
            last=self._last_outer,
            occupied=occupied,
            preferred_action=preferred_action,
        )
        if outer is not None:
            self._last_outer = outer
        return outer

    def pipeline_action(
        self,
        observation: FourKeyObservation,
        *,
        now_s: float,
        targets: tuple[tuple[object, float], ...],
        lead_s: float,
        preferred_action: FourKeyAction | None = None,
    ) -> FourKeyAction:
        """Launch every due target that can be assigned to a free finger.

        ``targets`` must be the still-unresolved target tail in chronological
        order, represented as ``(stable_token, episode_time_s)`` pairs.  Exact
        target times are privileged teacher-only information and are never
        emitted in the 251D policy observation.
        """

        now_s = float(now_s)
        lead_s = float(lead_s)

        # Once a target disappears from the unresolved tail, its reservation is
        # complete (hit or miss) and the finger may be allocated again after it
        # physically resets.
        for key, (token, _) in tuple(self._reservations.items()):
            if not self._contains_token(targets, token):
                del self._reservations[key]

        commands = {
            key: (-1.0 if observation.pressed(key) else 0.0)
            for key in FOUR_KEY_NAMES
        }

        # Keep every in-flight reservation latched.  If an early press failed to
        # consume its target, the same reservation naturally retries once that
        # key has physically reset instead of immediately spraying another key.
        for key, (_, target_time_s) in self._reservations.items():
            if (
                not observation.pressed(key)
                and now_s + lead_s >= float(target_time_s)
            ):
                commands[key] = 1.0

        occupied = {
            key for key in FOUR_KEY_NAMES if observation.pressed(key)
        } | set(self._reservations)
        reserved_tokens = [token for token, _ in self._reservations.values()]

        for token, target_time_s in targets:
            target_time_s = float(target_time_s)
            if now_s + lead_s < target_time_s:
                break
            if any(token == reserved for reserved in reserved_tokens):
                continue

            key = self._choose_key(
                observation,
                occupied=occupied,
                preferred_action=preferred_action,
            )
            if key is None:
                break

            self._reservations[key] = (token, target_time_s)
            reserved_tokens.append(token)
            occupied.add(key)
            commands[key] = 1.0

        return FourKeyAction(*(commands[key] for key in FOUR_KEY_NAMES))

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
        """Single-target compatibility wrapper around the pipelined scheduler."""

        targets = (
            ()
            if target_token is None
            else ((target_token, float(target_time_s)),)
        )
        return self.pipeline_action(
            observation,
            now_s=now_s,
            targets=targets,
            lead_s=lead_s,
            preferred_action=preferred_action,
        )


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
        targets: tuple[tuple[object, float], ...] = ()
    else:
        # ``ordinal`` is the target's index in segment.targets.  Feed the whole
        # unresolved tail to the privileged teacher so targets closer than one
        # physical press latency can already occupy different fingers.
        targets = tuple(
            (
                (future.ordinal, future.floor_index, future.episode_time_s),
                float(future.episode_time_s),
            )
            for future in env.segment.targets[int(target.ordinal) :]
        )

    return teacher.pipeline_action(
        observation.motor,
        now_s=env.privileged_episode_time_s(),
        targets=targets,
        lead_s=lead_s,
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
