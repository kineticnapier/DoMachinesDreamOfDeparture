from __future__ import annotations

"""Bootstrap/DAgger training primitives for configurable N-key Level A policies."""

from dataclasses import dataclass

import torch

from dmdod.motor.capacity import CenterFirstNKeyTeacher
from dmdod.motor.n_key import NKeyAction, n_key_names
from dmdod.policies.n_key import NKeyRecurrentActorCritic
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    return (values * mask).sum() / count.clamp_min(1)


def n_key_actuation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant actuation loss for interchangeable free fingers.

    ADOFAI scoring does not care which physical key claims a target.  The
    center-first privileged teacher still needs a concrete routing choice to
    drive the body, but that key identity is not a unique BC label.  Treating
    it as exact creates contradictory supervision in larger bodies: a valid
    press on another free finger is simultaneously scored as a missed teacher
    press and an unsafe neutral push.

    For each frame this objective therefore keeps only the action semantics:

    * keys the teacher is releasing stay identity-specific, because they are
      the keys physically held in the teacher/student state;
    * the *number* of requested presses is preserved, but those presses may be
      assigned to any currently non-held output;
    * every remaining non-held output is neutral and is penalized for an extra
      positive push.

    Sorting the non-held outputs implements the minimum-cost assignment without
    enumerating key permutations: the strongest ``press_count`` outputs are the
    candidate presses, and the rest are candidate neutrals.
    """

    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have matching shape [T, K]")
    key_count = int(predicted.shape[1])
    if key_count < 2 or key_count % 2 != 0:
        raise ValueError("N-key action width must be an even integer >= 2")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    available = ~release

    press_count = press.sum(dim=1, keepdim=True)
    available_count = available.sum(dim=1, keepdim=True)
    if bool((press_count > available_count).any()):
        raise ValueError("teacher requests more presses than non-held keys")

    # tanh policy outputs are in [-1, 1], so -2 safely moves held/release keys
    # behind every available key before sorting.  Release outputs are trained
    # separately against their physical identity below.
    ranked, _ = torch.sort(
        predicted.masked_fill(release, -2.0),
        dim=1,
        descending=True,
    )
    ranks = torch.arange(key_count, device=predicted.device).reshape(1, key_count)
    ranked_press = ranks < press_count
    ranked_available = ranks < available_count
    ranked_neutral = ranked_available & ~ranked_press

    desired_ranked = ranked_press.to(dtype=predicted.dtype)
    available_mse = ((ranked - desired_ranked).square() * ranked_available).sum()
    release_mse = ((predicted + 1.0).square() * release).sum()
    mse = (available_mse + release_mse) / max(1, predicted.numel())

    press_gap = torch.relu(PRESS_MARGIN - ranked).square()
    press_loss = _masked_mean(press_gap, ranked_press)

    release_gap = torch.relu(predicted - RELEASE_MARGIN).square()
    release_loss = _masked_mean(release_gap, release)

    unsafe_neutral_push = torch.relu(ranked - NEUTRAL_PUSH_LIMIT).square()
    neutral_loss = _masked_mean(unsafe_neutral_push, ranked_neutral)

    return (
        MSE_COEF * mse
        + PRESS_MARGIN_COEF * press_loss
        + RELEASE_MARGIN_COEF * release_loss
        + NEUTRAL_PUSH_COEF * neutral_loss
    )


def discretize_n_key_action(
    values: tuple[float, ...],
    *,
    press_threshold: float,
    release_threshold: float,
) -> NKeyAction:
    """Convert soft policy outputs to {-1, 0, +1} for physical execution."""

    press_threshold = float(press_threshold)
    release_threshold = float(release_threshold)
    if not (0.0 < press_threshold <= 1.0):
        raise ValueError("press_threshold must be in (0, 1]")
    if not (-1.0 <= release_threshold < 0.0):
        raise ValueError("release_threshold must be in [-1, 0)")

    return NKeyAction(
        tuple(
            1.0
            if value >= press_threshold
            else -1.0
            if value <= release_threshold
            else 0.0
            for value in values
        )
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


def _make_env(
    segment,
    *,
    key_count: int,
    control_dt_s: float,
    physics_dt_s: float,
) -> DiagnosticHudNKeyRealChartMotorEnv:
    return DiagnosticHudNKeyRealChartMotorEnv(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
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
    env = _make_env(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
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


def collect_n_key_dagger_sequence(
    model: NKeyRecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    press_threshold: float,
    release_threshold: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    source: str = "n-key-student-state-dagger",
    action_mode: str = "hard",
) -> NKeyRollout:
    """Collect privileged set-valued labels on a student-state trajectory.

    The student alone drives the physical body.  At every visited observation,
    the privileged center-first teacher is queried only for a supervision label.
    ``action_mode='hard'`` discretizes the policy with the supplied thresholds;
    ``action_mode='continuous'`` applies the tanh policy output directly.  The
    latter is useful once DAgger has taught the policy amplitudes well enough for
    the physical body to respond reliably without thresholding.

    Because ``n_key_actuation_loss`` treats free-finger press identity as a set,
    the teacher's concrete routing choice is not imposed as the unique answer.
    Release identity remains tied to the actually held keys visible in the
    student-state observation.
    """

    if action_mode not in {"hard", "continuous"}:
        raise ValueError("action_mode must be 'hard' or 'continuous'")

    key_count = int(model.key_count)
    env = _make_env(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
    )
    teacher = CenterFirstNKeyTeacher(key_count)
    observation = env.reset()
    state = model.initial_state(device)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, ...]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            encoded = encode_n_key_hud_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            # Use the policy's deterministic inference path instead of the
            # training forward. Connectome policies can omit critic/std work
            # here and transfer all action values to CPU in one synchronization.
            deterministic, state = model.deterministic_action(x, state)
            soft_values = deterministic.as_tuple()
            if action_mode == "continuous":
                student_action = deterministic
            else:
                student_action = discretize_n_key_action(
                    soft_values,
                    press_threshold=press_threshold,
                    release_threshold=release_threshold,
                )
            teacher_action = _teacher_action(
                env,
                teacher,
                observation,
                lead_s=lead_s,
            )

            observations.append(encoded)
            labels.append(teacher_action.as_tuple())
            step = env.step(student_action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key DAgger rollout exceeded step budget")

    sequence = NKeyBCSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        key_count=key_count,
        source=source,
    )
    return NKeyRollout(
        sequence=sequence,
        stats=env.stats,
        physical_keydowns=int(env.physical_keydowns),
    )
