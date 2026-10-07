from __future__ import annotations

"""Closed-loop trajectory divergence diagnostics for action-trust updates."""

from dataclasses import asdict, dataclass
import json
from math import sqrt
from pathlib import Path

import torch

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.action_trust import (
    build_action_trust_sequences,
    freeze_actor_only,
    train_actor_action_trust,
)
from dmdod.training.n_key import (
    NKeyBCSequence,
    collect_n_key_expert_sequence,
)
from dmdod.training.real_chart import (
    build_connectome_policy_from_checkpoint,
    collect_student_state_sequences,
)


@dataclass(frozen=True, slots=True)
class DivergencePoint:
    step: int
    time_s: float
    target_ordinal: int | None
    baseline: object
    candidate: object


@dataclass(frozen=True, slots=True)
class EventGuardViolation:
    kind: str
    time_s: float
    target_ordinal: int | None
    duration_s: float


@dataclass(frozen=True, slots=True)
class BoundaryDivergence:
    step: int
    time_s: float
    target_ordinal: int | None
    key: str
    event: str
    threshold_m: float
    baseline_event: bool
    candidate_event: bool
    baseline_position_m: float
    candidate_position_m: float
    position_delta_m: float
    baseline_margin_m: float
    candidate_margin_m: float
    baseline_action: float
    candidate_action: float


@dataclass(frozen=True, slots=True)
class TrajectoryProbeResult:
    anchor_index: int
    chart_name: str
    steps: int
    action_rms: float
    action_max: float
    position_rms_m: float
    position_max_m: float
    velocity_rms_m_s: float
    activation_rms: float
    fatigue_rms: float
    hand_fatigue_rms: float
    coordination_rms: float
    pressed_mismatch_frames: int
    first_pressed_divergence: DivergencePoint | None
    first_keydown_divergence: DivergencePoint | None
    first_score_divergence: DivergencePoint | None
    first_overload_divergence: DivergencePoint | None
    baseline_hits: int
    candidate_hits: int
    baseline_misses: int
    candidate_misses: int
    baseline_too_early: int
    candidate_too_early: int
    baseline_overloaded: bool
    candidate_overloaded: bool
    baseline_keydowns: int
    candidate_keydowns: int
    stopped_side: str
    first_boundary_divergence: BoundaryDivergence | None = None
    event_guard_violation: EventGuardViolation | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class _SquaredAccumulator:
    __slots__ = ("squared", "count", "maximum")

    def __init__(self) -> None:
        self.squared = 0.0
        self.count = 0
        self.maximum = 0.0

    def add(self, left, right) -> None:
        for a, b in zip(left, right):
            delta = float(a) - float(b)
            self.squared += delta * delta
            self.count += 1
            self.maximum = max(self.maximum, abs(delta))

    @property
    def rms(self) -> float:
        return sqrt(self.squared / max(1, self.count))


def _next_target_ordinal(env) -> int | None:
    target = env.privileged_next_target()
    return None if target is None else int(target.ordinal)


def _point(
    *,
    step: int,
    env_a,
    env_b,
    baseline,
    candidate,
) -> DivergencePoint:
    return DivergencePoint(
        step=int(step),
        time_s=max(
            float(env_a.privileged_episode_time_s()),
            float(env_b.privileged_episode_time_s()),
        ),
        target_ordinal=_next_target_ordinal(env_a),
        baseline=baseline,
        candidate=candidate,
    )


def _coordination_tuple(diagnostics) -> tuple[float, float, float]:
    return (
        float(diagnostics.left_hand_coordination),
        float(diagnostics.right_hand_coordination),
        float(diagnostics.bilateral_coordination),
    )


def _hand_fatigue_tuple(diagnostics) -> tuple[float, float]:
    return (
        float(diagnostics.left_hand_fatigue),
        float(diagnostics.right_hand_fatigue),
    )


def _score_tuple(stats) -> tuple[int, int, int]:
    return (
        int(stats.hits),
        int(stats.misses),
        int(stats.too_early_presses),
    )


def _make_env(segment, *, key_count: int, control_dt_s: float, physics_dt_s: float):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    env.motor.capture_physics_trace = True
    return env


def _first_boundary_divergence(
    *,
    step: int,
    env_a,
    env_b,
    action_a: tuple[float, ...],
    action_b: tuple[float, ...],
) -> BoundaryDivergence | None:
    transition_a = env_a.motor.last_transition
    transition_b = env_b.motor.last_transition
    if transition_a is None or transition_b is None:
        return None

    samples_a = transition_a.physics_samples
    samples_b = transition_b.physics_samples
    if len(samples_a) != len(samples_b):
        raise RuntimeError("paired motor traces have different physics sample counts")

    key_names = env_a.motor.key_names
    config = env_a.motor.keyboard.config
    key_order = {name: index for index, name in enumerate(key_names)}

    for sample_a, sample_b in zip(samples_a, samples_b):
        events_a = set(sample_a.events)
        events_b = set(sample_b.events)
        if events_a == events_b:
            continue

        differing = events_a.symmetric_difference(events_b)
        key, event = min(
            differing,
            key=lambda item: (key_order[item[0]], item[1].value),
        )
        index = key_order[key]
        threshold = (
            float(config.actuation_m)
            if event is KeyEvent.DOWN
            else float(config.reset_m)
        )
        position_a = float(sample_a.positions_m[index])
        position_b = float(sample_b.positions_m[index])
        return BoundaryDivergence(
            step=int(step),
            time_s=max(float(sample_a.time_s), float(sample_b.time_s)),
            target_ordinal=_next_target_ordinal(env_a),
            key=str(key),
            event=str(event.value),
            threshold_m=threshold,
            baseline_event=(key, event) in events_a,
            candidate_event=(key, event) in events_b,
            baseline_position_m=position_a,
            candidate_position_m=position_b,
            position_delta_m=position_b - position_a,
            baseline_margin_m=position_a - threshold,
            candidate_margin_m=position_b - threshold,
            baseline_action=float(action_a[index]),
            candidate_action=float(action_b[index]),
        )
    return None


@torch.no_grad()
def probe_anchor_pair(
    baseline_model,
    candidate_model,
    named,
    *,
    anchor_index: int,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    stop_on_event_divergence: bool = False,
    mismatch_grace_s: float = 0.0,
) -> TrajectoryProbeResult:
    """Run two policies synchronously on independent copies of one anchor."""

    env_a = _make_env(
        named.segment,
        key_count=int(baseline_model.key_count),
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
    )
    env_b = _make_env(
        named.segment,
        key_count=int(candidate_model.key_count),
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
    )
    observation_a = env_a.reset()
    observation_b = env_b.reset()
    state_a = baseline_model.initial_state(device)
    state_b = candidate_model.initial_state(device)
    baseline_model.eval()
    candidate_model.eval()

    action_acc = _SquaredAccumulator()
    position_acc = _SquaredAccumulator()
    velocity_acc = _SquaredAccumulator()
    activation_acc = _SquaredAccumulator()
    fatigue_acc = _SquaredAccumulator()
    hand_fatigue_acc = _SquaredAccumulator()
    coordination_acc = _SquaredAccumulator()

    pressed_mismatch_frames = 0
    first_boundary = None
    first_pressed = None
    first_keydown = None
    first_score = None
    first_overload = None
    event_guard_violation = None
    pressed_mismatch_run = 0
    keydown_mismatch_run = 0
    steps = 0
    stopped_side = "budget"

    max_steps = int((named.segment.duration_s + 2.0) / control_dt_s) + 200
    for step_index in range(1, max_steps + 1):
        x_a = torch.tensor(
            encode_n_key_hud_real_chart_observation(observation_a),
            dtype=torch.float32,
            device=device,
        )
        x_b = torch.tensor(
            encode_n_key_hud_real_chart_observation(observation_b),
            dtype=torch.float32,
            device=device,
        )
        action_a, state_a = baseline_model.deterministic_action(x_a, state_a)
        action_b, state_b = candidate_model.deterministic_action(x_b, state_b)
        values_a = action_a.as_tuple()
        values_b = action_b.as_tuple()
        action_acc.add(values_a, values_b)

        transition_a = env_a.step(action_a)
        transition_b = env_b.step(action_b)
        observation_a = transition_a.observation
        observation_b = transition_b.observation
        steps = step_index

        if first_boundary is None:
            first_boundary = _first_boundary_divergence(
                step=step_index,
                env_a=env_a,
                env_b=env_b,
                action_a=values_a,
                action_b=values_b,
            )

        motor_a = observation_a.motor
        motor_b = observation_b.motor
        position_acc.add(motor_a.positions_m, motor_b.positions_m)
        velocity_acc.add(motor_a.velocities_m_s, motor_b.velocities_m_s)

        diagnostics_a = env_a.motor.diagnostics()
        diagnostics_b = env_b.motor.diagnostics()
        activation_acc.add(diagnostics_a.activations, diagnostics_b.activations)
        fatigue_acc.add(diagnostics_a.finger_fatigue, diagnostics_b.finger_fatigue)
        hand_fatigue_acc.add(
            _hand_fatigue_tuple(diagnostics_a),
            _hand_fatigue_tuple(diagnostics_b),
        )
        coordination_acc.add(
            _coordination_tuple(diagnostics_a),
            _coordination_tuple(diagnostics_b),
        )

        pressed_mismatch = motor_a.pressed_flags != motor_b.pressed_flags
        if pressed_mismatch:
            pressed_mismatch_frames += 1
            pressed_mismatch_run += 1
            if first_pressed is None:
                first_pressed = _point(
                    step=step_index,
                    env_a=env_a,
                    env_b=env_b,
                    baseline=tuple(bool(v) for v in motor_a.pressed_flags),
                    candidate=tuple(bool(v) for v in motor_b.pressed_flags),
                )
        else:
            pressed_mismatch_run = 0

        keydown_mismatch = (
            int(env_a.physical_keydowns) != int(env_b.physical_keydowns)
        )
        if keydown_mismatch:
            keydown_mismatch_run += 1
            if first_keydown is None:
                first_keydown = _point(
                    step=step_index,
                    env_a=env_a,
                    env_b=env_b,
                    baseline=int(env_a.physical_keydowns),
                    candidate=int(env_b.physical_keydowns),
                )
        else:
            keydown_mismatch_run = 0

        stats_a = env_a.stats
        stats_b = env_b.stats
        if first_score is None and _score_tuple(stats_a) != _score_tuple(stats_b):
            first_score = _point(
                step=step_index,
                env_a=env_a,
                env_b=env_b,
                baseline=_score_tuple(stats_a),
                candidate=_score_tuple(stats_b),
            )

        if first_overload is None and (
            bool(stats_a.overloaded) != bool(stats_b.overloaded)
        ):
            first_overload = _point(
                step=step_index,
                env_a=env_a,
                env_b=env_b,
                baseline=(
                    float(stats_a.overload_counter),
                    bool(stats_a.overloaded),
                ),
                candidate=(
                    float(stats_b.overload_counter),
                    bool(stats_b.overloaded),
                ),
            )

        if stop_on_event_divergence:
            if first_score is not None:
                event_guard_violation = EventGuardViolation(
                    kind="score-topology",
                    time_s=float(env_a.privileged_episode_time_s()),
                    target_ordinal=_next_target_ordinal(env_a),
                    duration_s=0.0,
                )
            else:
                pressed_duration = pressed_mismatch_run * control_dt_s
                keydown_duration = keydown_mismatch_run * control_dt_s
                if pressed_mismatch and pressed_duration + 1e-12 >= mismatch_grace_s:
                    event_guard_violation = EventGuardViolation(
                        kind="pressed-state",
                        time_s=float(env_a.privileged_episode_time_s()),
                        target_ordinal=_next_target_ordinal(env_a),
                        duration_s=float(pressed_duration),
                    )
                elif keydown_mismatch and keydown_duration + 1e-12 >= mismatch_grace_s:
                    event_guard_violation = EventGuardViolation(
                        kind="keydown-count",
                        time_s=float(env_a.privileged_episode_time_s()),
                        target_ordinal=_next_target_ordinal(env_a),
                        duration_s=float(keydown_duration),
                    )
            if event_guard_violation is not None:
                stopped_side = "event-divergence"
                break

        if transition_a.done or transition_b.done:
            if transition_a.done and transition_b.done:
                stopped_side = "both"
            elif transition_a.done:
                stopped_side = "baseline"
            else:
                stopped_side = "candidate"
            break
    else:
        raise RuntimeError("trajectory probe exceeded step budget")

    stats_a = env_a.stats
    stats_b = env_b.stats
    return TrajectoryProbeResult(
        anchor_index=int(anchor_index),
        chart_name=str(named.chart_name),
        steps=int(steps),
        action_rms=action_acc.rms,
        action_max=action_acc.maximum,
        position_rms_m=position_acc.rms,
        position_max_m=position_acc.maximum,
        velocity_rms_m_s=velocity_acc.rms,
        activation_rms=activation_acc.rms,
        fatigue_rms=fatigue_acc.rms,
        hand_fatigue_rms=hand_fatigue_acc.rms,
        coordination_rms=coordination_acc.rms,
        pressed_mismatch_frames=int(pressed_mismatch_frames),
        first_pressed_divergence=first_pressed,
        first_keydown_divergence=first_keydown,
        first_score_divergence=first_score,
        first_overload_divergence=first_overload,
        baseline_hits=int(stats_a.hits),
        candidate_hits=int(stats_b.hits),
        baseline_misses=int(stats_a.misses),
        candidate_misses=int(stats_b.misses),
        baseline_too_early=int(stats_a.too_early_presses),
        candidate_too_early=int(stats_b.too_early_presses),
        baseline_overloaded=bool(stats_a.overloaded),
        candidate_overloaded=bool(stats_b.overloaded),
        baseline_keydowns=int(env_a.physical_keydowns),
        candidate_keydowns=int(env_b.physical_keydowns),
        stopped_side=stopped_side,
        first_boundary_divergence=first_boundary,
        event_guard_violation=event_guard_violation,
    )


def boundary_event_guard_reason(
    *,
    anchor_index: int,
    reference_stats,
    result: TrajectoryProbeResult,
    preserve_safe_only: bool = True,
) -> str | None:
    """Reject meaningful event-topology changes on currently safe anchors."""

    if preserve_safe_only and bool(reference_stats.overloaded):
        return None

    violation = result.event_guard_violation
    if violation is None:
        return None

    detail = ""
    boundary = result.first_boundary_divergence
    if boundary is not None:
        detail = (
            f" first-boundary={boundary.key}:{boundary.event}@{boundary.time_s:.3f}s "
            f"dpos={boundary.position_delta_m * 1e6:+.3f}um"
        )
    duration = (
        ""
        if violation.duration_s <= 0.0
        else f" duration={violation.duration_s * 1000.0:.1f}ms"
    )
    return (
        f"anchor {anchor_index} {violation.kind} divergence at "
        f"{violation.time_s:.3f}s target={violation.target_ordinal}"
        f"{duration}{detail}"
    )


@torch.no_grad()
def evaluate_boundary_event_guard(
    baseline_model,
    candidate_model,
    anchors,
    reference_results,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    preserve_safe_only: bool = True,
    mismatch_grace_s: float = 0.030,
) -> tuple[bool, tuple[str, ...], list[TrajectoryProbeResult]]:
    if len(anchors) != len(reference_results):
        raise ValueError("boundary trust reference/anchor count mismatch")

    reasons: list[str] = []
    probes: list[TrajectoryProbeResult] = []
    for index, (named, reference) in enumerate(
        zip(anchors, reference_results),
        1,
    ):
        reference_stats, _ = reference
        if preserve_safe_only and bool(reference_stats.overloaded):
            continue

        result = probe_anchor_pair(
            baseline_model,
            candidate_model,
            named,
            anchor_index=index,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            stop_on_event_divergence=True,
            mismatch_grace_s=mismatch_grace_s,
        )
        probes.append(result)
        reason = boundary_event_guard_reason(
            anchor_index=index,
            reference_stats=reference_stats,
            result=result,
            preserve_safe_only=preserve_safe_only,
        )
        if reason is not None:
            reasons.append(reason)
            break

    return not reasons, tuple(reasons), probes


def collect_probe_training_sequences(
    model,
    anchors,
    *,
    round_index: int,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
) -> tuple[list[NKeyBCSequence], int, int]:
    expert_sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(anchors, 1):
        expert = collect_n_key_expert_sequence(
            named.segment,
            key_count=int(model.key_count),
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"trajectory-probe-expert-{index}-{named.chart_name}",
        )
        expert_sequences.append(expert.sequence)

    student_sequences, student_frames = collect_student_state_sequences(
        model,
        anchors,
        round_index=round_index,
        collection_index=0,
        lead_s=lead_s,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    expert_frames = sum(sequence.frames for sequence in expert_sequences)
    return [*expert_sequences, *student_sequences], expert_frames, student_frames


def build_probe_candidate(
    parent: dict,
    baseline_model,
    training_sequences: list[NKeyBCSequence],
    *,
    action_rms: float,
    actor_steps: int,
    lr: float,
    stay_coef: float,
    lr_backoffs: int,
    min_lr: float,
    chunk_steps: int,
    device: torch.device,
):
    cached = build_action_trust_sequences(baseline_model, training_sequences)
    candidate = build_connectome_policy_from_checkpoint(parent, device=device)
    candidate.load_state_dict(baseline_model.state_dict())
    candidate.prepare_recurrent_runtime()
    actor_parameters = freeze_actor_only(candidate)
    optimizer = torch.optim.SGD(actor_parameters, lr=lr)
    metrics = train_actor_action_trust(
        candidate,
        cached,
        optimizer=optimizer,
        actor_steps=actor_steps,
        chunk_steps=chunk_steps,
        base_lr=lr,
        stay_coef=stay_coef,
        max_action_rms=action_rms,
        lr_backoffs=lr_backoffs,
        min_lr=min_lr,
    )
    del cached
    return candidate, metrics


def save_probe_report(
    path: str | Path,
    *,
    candidate_metrics,
    results: list[TrajectoryProbeResult],
    source_checkpoint: str,
    configured_action_rms: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "trajectory-divergence-probe",
        "source_checkpoint": str(source_checkpoint),
        "configured_action_rms": float(configured_action_rms),
        "candidate_training": candidate_metrics.as_dict(),
        "anchors": [result.as_dict() for result in results],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def format_probe_result(result: TrajectoryProbeResult) -> str:
    def point(value: DivergencePoint | None) -> str:
        if value is None:
            return "-"
        target = "?" if value.target_ordinal is None else str(value.target_ordinal)
        return f"{value.time_s:.3f}s/t{target}"

    boundary = result.first_boundary_divergence
    boundary_text = "-"
    if boundary is not None:
        boundary_text = (
            f"{boundary.time_s:.3f}s {boundary.key} {boundary.event.upper()} "
            f"thr={boundary.threshold_m * 1000.0:.6f}mm "
            f"pos={boundary.baseline_position_m * 1000.0:.6f}/"
            f"{boundary.candidate_position_m * 1000.0:.6f}mm "
            f"d={boundary.position_delta_m * 1e6:+.3f}um "
            f"margin={boundary.baseline_margin_m * 1e6:+.3f}/"
            f"{boundary.candidate_margin_m * 1e6:+.3f}um "
            f"event={int(boundary.baseline_event)}/{int(boundary.candidate_event)} "
            f"action={boundary.baseline_action:.6f}/"
            f"{boundary.candidate_action:.6f}"
        )

    return (
        f"anchor {result.anchor_index:02d}: "
        f"actionRMS={result.action_rms:.3g} max={result.action_max:.3g} "
        f"posRMS={result.position_rms_m:.3g}m "
        f"actRMS={result.activation_rms:.3g} fatigueRMS={result.fatigue_rms:.3g} "
        f"boundary=[{boundary_text}] "
        f"press-div={point(result.first_pressed_divergence)} "
        f"key-div={point(result.first_keydown_divergence)} "
        f"score-div={point(result.first_score_divergence)} "
        f"over-div={point(result.first_overload_divergence)} "
        f"H={result.baseline_hits}->{result.candidate_hits} "
        f"early={result.baseline_too_early}->{result.candidate_too_early} "
        f"over={result.baseline_overloaded}->{result.candidate_overloaded}"
    )
