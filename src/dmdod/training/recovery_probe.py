from __future__ import annotations

"""Counterfactual teacher takeovers from actual student motor states.

Research-only diagnostics: student observations never receive privileged target
times; no optimizer, model weights, or checkpoint selection is modified.
The teacher starts with *fresh* reservations at takeover, so this tests that
particular teacher strategy, not an optimal recovery oracle.
"""

import copy
from dataclasses import dataclass
import math

import torch

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    NKeyOverloadTrace,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.capacity import CenterFirstNKeyTeacher
from dmdod.motor.n_key import NKeyAction
from dmdod.training.n_key import _teacher_action


DEFAULT_TAKEOVER_TIMES_S = (1.0, 2.0, 3.0)


@dataclass(frozen=True, slots=True)
class TakeoverResult:
    alpha: float
    requested_time_s: float
    actual_time_s: float
    starting_hits: int
    starting_too_early: int
    starting_gauge: float
    teacher_end_s: float
    teacher_hits: int
    teacher_too_early: int
    teacher_keydowns: int
    teacher_overloaded: bool
    teacher_reason: str


@dataclass(frozen=True, slots=True)
class RecoveryProbeResult:
    alpha: float
    student_end_s: float
    student_hits: int
    student_too_early: int
    student_keydowns: int
    student_overloaded: bool
    student_reason: str
    takeovers: tuple[TakeoverResult, ...]
    skipped_times_s: tuple[float, ...]


def _make_diagnostic_env(named, *, key_count, control_dt_s, physics_dt_s):
    return DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,
        key_count=key_count,
        capture_overload_trace=True,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )


def _takeover(env, *, lead_s: float, max_steps: int):
    """Run a fresh center-first teacher from a deep-copied student state."""
    teacher = CenterFirstNKeyTeacher(env.key_count)
    observation = env.observe()
    for _ in range(max_steps):
        action = _teacher_action(env, teacher, observation, lead_s=lead_s)
        step = env.step(action)
        observation = step.observation
        if step.done:
            return env.stats, env.overload_trace.termination
    raise RuntimeError("teacher takeover exceeded the normal episode step budget")


def run_teacher_takeover_probe(
    model,
    named,
    *,
    alpha: float,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    expected_trace: NKeyOverloadTrace,
    takeover_times_s: tuple[float, ...] = DEFAULT_TAKEOVER_TIMES_S,
) -> RecoveryProbeResult:
    """Replay #09 student exactly; clone live states for privileged takeover.

    The baseline student rollout MUST reproduce the existing Validation
    termination before any takeover result is accepted.  No post-failure
    continuation or change to evaluator scoring is made.
    """
    if not takeover_times_s or any(
        not math.isfinite(t) or t <= 0.0 for t in takeover_times_s
    ) or any(b <= a for a, b in zip(takeover_times_s, takeover_times_s[1:])):
        raise ValueError("takeover times must be positive, finite and increasing")
    if control_dt_s <= 0 or physics_dt_s <= 0 or not math.isfinite(lead_s):
        raise ValueError("invalid recovery probe timing")
    for t in takeover_times_s:
        if abs(round(t / control_dt_s) * control_dt_s - t) > 1e-7:
            raise ValueError("takeover times must align to control ticks")
    if expected_trace is None:
        raise ValueError("expected Validation trace is required")

    env = _make_diagnostic_env(
        named, key_count=model.key_count,
        control_dt_s=control_dt_s, physics_dt_s=physics_dt_s,
    )
    observation = env.reset()
    state = model.initial_state(device)
    max_steps = int((named.segment.duration_s + 2.0) / control_dt_s) + 200
    checkpoints: list[tuple[float, object]] = []
    next_index = 0

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            # Snapshot *before* applying the next control command.
            now = float(env.privileged_episode_time_s())
            while (
                next_index < len(takeover_times_s)
                and now + 1e-7 >= takeover_times_s[next_index]
            ):
                target = takeover_times_s[next_index]
                if abs(now - target) > 1e-6:
                    raise RuntimeError("recovery probe missed control-tick snapshot")
                checkpoints.append((target, copy.deepcopy(env)))
                next_index += 1
            encoded = encode_n_key_hud_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            mean, _, _, state = model.forward_step(x, state)
            action = NKeyAction(tuple(float(v.item()) for v in torch.tanh(mean)))
            result = env.step(action)
            observation = result.observation
            if result.done:
                break
        else:
            raise RuntimeError("recovery probe student exceeded step budget")

    student_stats = env.stats
    trace = env.overload_trace
    if trace is None:
        raise RuntimeError("recovery probe student trace missing")
    terminal = trace.termination
    reference = expected_trace.termination
    fields = ("hits", "too_early", "keydowns", "reason", "next_target_index")
    if (
        any(getattr(terminal, name) != getattr(reference, name) for name in fields)
        or abs(terminal.time_s - reference.time_s) > 1e-6
        or abs(terminal.overload_value - reference.overload_value) > 1e-5
        or bool(student_stats.overloaded) != (reference.reason == "Overload")
    ):
        raise RuntimeError(
            f"alpha={alpha:.6f} recovery probe #09 student reproduction FAIL: "
            f"actual={terminal} expected={reference}"
        )

    takeovers: list[TakeoverResult] = []
    for requested_time_s, snapshot in checkpoints:
        starting = snapshot.stats
        starting_time = float(snapshot.privileged_episode_time_s())
        teacher_stats, teacher_terminal = _takeover(
            snapshot, lead_s=lead_s, max_steps=max_steps,
        )
        takeovers.append(TakeoverResult(
            alpha=float(alpha),
            requested_time_s=requested_time_s,
            actual_time_s=starting_time,
            starting_hits=int(starting.hits),
            starting_too_early=int(starting.too_early_presses),
            starting_gauge=float(starting.overload_counter),
            teacher_end_s=float(teacher_terminal.time_s),
            teacher_hits=int(teacher_stats.hits),
            teacher_too_early=int(teacher_stats.too_early_presses),
            teacher_keydowns=int(snapshot.physical_keydowns),
            teacher_overloaded=bool(teacher_stats.overloaded),
            teacher_reason=teacher_terminal.reason,
        ))

    return RecoveryProbeResult(
        alpha=float(alpha),
        student_end_s=float(terminal.time_s),
        student_hits=int(student_stats.hits),
        student_too_early=int(student_stats.too_early_presses),
        student_keydowns=int(env.physical_keydowns),
        student_overloaded=bool(student_stats.overloaded),
        student_reason=terminal.reason,
        takeovers=tuple(takeovers),
        skipped_times_s=takeover_times_s[next_index:],
    )


def format_recovery_probe(result: RecoveryProbeResult) -> tuple[str, ...]:
    lines = [
        f"=== #09 teacher takeover alpha={result.alpha:.6f} (diagnostic only) ===",
        f"student baseline: end={result.student_end_s:.3f}s "
        f"H={result.student_hits} Early={result.student_too_early} "
        f"Keydowns={result.student_keydowns} Overload={int(result.student_overloaded)} "
        f"reason={result.student_reason} [matches fixed Validation]",
    ]
    for t in result.takeovers:
        lines.append(
            f"  teacher from t={t.actual_time_s:.3f}s: "
            f"initial H={t.starting_hits} Early={t.starting_too_early} "
            f"gauge={t.starting_gauge:.3f} -> "
            f"end={t.teacher_end_s:.3f}s H={t.teacher_hits} "
            f"Early={t.teacher_too_early} Keydowns={t.teacher_keydowns} "
            f"Overload={int(t.teacher_overloaded)} reason={t.teacher_reason}"
        )
    for t in result.skipped_times_s:
        lines.append(f"  t={t:.3f}s: SKIP (student already terminated)")
    lines.extend((
        "NOTE: this teacher has privileged future targets and fresh reservations at takeover.",
        "A failed takeover does not prove physical recovery is impossible.",
        "No training, checkpoint selection, optimizer or Validation ranking uses this result.",
    ))
    return tuple(lines)
