from __future__ import annotations

"""Bootstrap-side training primitives for configurable N-key Level A policies."""

from dataclasses import dataclass

import torch

from .n_key_capacity import CenterFirstNKeyTeacher
from .n_key_motor import NKeyAction, n_key_names
from .n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
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
class NKeyBCSequence:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    key_count: int
    source: str

    def __post_init__(self) -> None:
        key_count = int(self.key_count)
        n_key_names(key_count)
        expected_input = n_key_hud_real_chart_input_dim(key_count)
        if self.observations.ndim != 2 or self.observations.shape[1] != expected_input:
            raise ValueError(
                f"N-key observations must have shape [T, {expected_input}]"
            )
        if self.teacher_actions.shape != (self.observations.shape[0], key_count):
            raise ValueError(
                f"N-key teacher actions must have shape [T, {key_count}]"
            )

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class NKeyRollout:
    sequence: NKeyBCSequence
    stats: object
    physical_keydowns: int


def _macro_key_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a masked loss per key first, then equally across active keys.

    N-key routing is intentionally center-first, so inner keys appear in many
    more press/release labels while outer keys spend most frames neutral.  A
    single global denominator lets those frequency differences hide severe
    per-key errors.  Macro averaging makes every key that has at least one
    sample of the requested class in this chunk contribute equally.
    """

    counts = mask.sum(dim=0)
    active = counts > 0
    per_key = (values * mask).sum(dim=0) / counts.clamp_min(1)
    return (per_key * active).sum() / active.sum().clamp_min(1)


def n_key_actuation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Actuation-aware BC loss with per-key class balancing.

    Press, release and unsafe-neutral penalties are macro-averaged across keys
    before the existing coefficients are applied.  This preserves the mature
    2K/4K margins while preventing an 8K center-first dataset from letting
    high-frequency inner-key labels or high-volume outer-key neutral labels
    dominate the objective.
    """

    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have matching shape [T, K]")
    if predicted.shape[1] < 2 or predicted.shape[1] % 2 != 0:
        raise ValueError("N-key action width must be an even integer >= 2")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)

    mse = (predicted - target).square().mean()

    press_gap = torch.relu(PRESS_MARGIN - predicted).square()
    press_loss = _macro_key_masked_mean(press_gap, press)

    release_gap = torch.relu(predicted - RELEASE_MARGIN).square()
    release_loss = _macro_key_masked_mean(release_gap, release)

    unsafe_neutral_push = torch.relu(predicted - NEUTRAL_PUSH_LIMIT).square()
    neutral_loss = _macro_key_masked_mean(unsafe_neutral_push, neutral)

    return (
        MSE_COEF * mse
        + PRESS_MARGIN_COEF * press_loss
        + RELEASE_MARGIN_COEF * release_loss
        + NEUTRAL_PUSH_COEF * neutral_loss
    )


def _target_token(target) -> tuple[int, int, float]:
    return (int(target.ordinal), int(target.floor_index), float(target.episode_time_s))


def _unresolved_tail(env: DiagnosticHudNKeyRealChartMotorEnv) -> tuple[tuple[object, float], ...]:
    target = env.privileged_next_target()
    if target is None:
        return ()
    return tuple(
        (_target_token(future), float(future.episode_time_s))
        for future in env.segment.targets[int(target.ordinal) :]
    )


def _teacher_action(
    env: DiagnosticHudNKeyRealChartMotorEnv,
    teacher: CenterFirstNKeyTeacher,
    observation,
    *,
    lead_s: float,
) -> NKeyAction:
    return teacher.pipeline_action(
        observation.motor,
        now_s=env.privileged_episode_time_s(),
        targets=_unresolved_tail(env),
        lead_s=lead_s,
    )


def collect_n_key_expert_sequence(
    segment,
    *,
    key_count: int,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    source: str = "n-key-center-first-expert",
) -> NKeyRollout:
    key_count = int(key_count)
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    teacher = CenterFirstNKeyTeacher(key_count)
    observation = env.reset()
    observations: list[tuple[float, ...]] = []
    actions: list[tuple[float, ...]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        action = _teacher_action(
            env,
            teacher,
            observation,
            lead_s=lead_s,
        )
        observations.append(encode_n_key_hud_real_chart_observation(observation))
        actions.append(action.as_tuple())
        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("N-key expert episode exceeded step budget")

    sequence = NKeyBCSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(actions, dtype=torch.float32, device=device),
        key_count=key_count,
        source=source,
    )
    return NKeyRollout(
        sequence=sequence,
        stats=env.stats,
        physical_keydowns=int(env.physical_keydowns),
    )
