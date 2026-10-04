from __future__ import annotations

"""Failure-continuing DAgger collection for configurable N-key policies.

Evaluation semantics remain unchanged: overload or fail-on-miss still ends an
ordinary episode.  This module is training-only.  When a student reaches a
terminal failure, it snapshots the terminal evaluation result, then keeps the
physical body, policy state, chart clock, target bookkeeping, and privileged
teacher labels running until the normal episode horizon.  That gives DAgger
supervision on failure/recovery states without making failed evaluation scores
look successful.
"""

import torch

from .n_key_motor import NKeyAction
from .n_key_policy import NKeyPolicyBase
from .n_key_training import (
    NKeyBCSequence,
    NKeyRollout,
    _make_env,
    _teacher_action,
    discretize_n_key_action,
)
from .n_key_real_chart import encode_n_key_hud_real_chart_observation


def _diagnostic_step_after_failure(env, action: NKeyAction):
    """Advance one control tick after terminal failure without re-ending it.

    The normal evaluator refuses to step once ``done`` is true.  For DAgger
    collection only, advance the same motor and scoring bookkeeping directly so
    future targets can still become hits/misses and the teacher can move on.
    The caller keeps the first terminal stats separately; these post-failure
    diagnostics never replace the evaluation result.
    """

    transition = env.motor.step(action)
    for event in transition.evaluator_events:
        env._advance_overload(event.time_s)
        env._expire_misses(event.time_s)
        env._score_event(event)

    now = float(transition.diagnostics.time_s)
    env._advance_overload(now)
    env._expire_misses(now)
    return env._observation(transition.observation, now)


def collect_n_key_dagger_sequence_with_continuation(
    model: NKeyPolicyBase,
    segment,
    *,
    lead_s: float,
    press_threshold: float,
    release_threshold: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    source: str = "n-key-student-state-dagger-continuation",
    action_mode: str = "hard",
    continue_after_failure: bool = True,
) -> NKeyRollout:
    """Collect DAgger labels and optionally continue failed trajectories.

    Before failure this is intentionally equivalent to the normal N-key DAgger
    collector.  If overload or fail-on-miss terminates the evaluator, the first
    terminal stats/key-down count are frozen for reporting while the student,
    body, chart clock, target bookkeeping, HUD, and teacher continue to the
    ordinary episode end.  Evaluation itself is not modified.
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
    teacher = __import__(
        "dmdod.n_key_capacity", fromlist=["CenterFirstNKeyTeacher"]
    ).CenterFirstNKeyTeacher(key_count)
    observation = env.reset()
    state = model.initial_state(device)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, ...]] = []

    terminal_stats = None
    terminal_keydowns: int | None = None
    continuing_failure = False
    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            encoded = encode_n_key_hud_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            mean, _, _, state = model.forward_step(x, state)
            soft_values = tuple(float(value.item()) for value in torch.tanh(mean))
            if action_mode == "continuous":
                student_action = NKeyAction(soft_values)
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

            if continuing_failure:
                observation = _diagnostic_step_after_failure(env, student_action)
                if env.privileged_episode_time_s() >= env._episode_end_s - 1e-12:
                    break
                continue

            step = env.step(student_action)
            observation = step.observation
            if not step.done:
                continue

            terminal_stats = env.stats
            terminal_keydowns = int(env.physical_keydowns)
            failed = bool(terminal_stats.overloaded or env._failed_on_miss)
            before_horizon = env.privileged_episode_time_s() < env._episode_end_s - 1e-12
            if continue_after_failure and failed and before_horizon:
                continuing_failure = True
                continue
            break
        else:
            raise RuntimeError("N-key failure-continuation DAgger rollout exceeded step budget")

    if terminal_stats is None:
        terminal_stats = env.stats
        terminal_keydowns = int(env.physical_keydowns)

    sequence = NKeyBCSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        key_count=key_count,
        source=source,
    )
    return NKeyRollout(
        sequence=sequence,
        stats=terminal_stats,
        physical_keydowns=int(terminal_keydowns),
    )
