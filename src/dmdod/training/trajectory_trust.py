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
    return DiagnosticHudNKeyRealChartMotorEnv(
        segment,
        key_count=key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )


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
    first_pressed = None
    first_keydown = None
    first_score = None
    first_overload = None
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

        if motor_a.pressed_flags != motor_b.pressed_flags:
            pressed_mismatch_frames += 1
            if first_pressed is None:
                first_pressed = _point(
                    step=step_index,
                    env_a=env_a,
                    env_b=env_b,
                    baseline=tuple(bool(v) for v in motor_a.pressed_flags),
                    candidate=tuple(bool(v) for v in motor_b.pressed_flags),
                )

        if (
            first_keydown is None
            and int(env_a.physical_keydowns) != int(env_b.physical_keydowns)
        ):
            first_keydown = _point(
                step=step_index,
                env_a=env_a,
                env_b=env_b,
                baseline=int(env_a.physical_keydowns),
                candidate=int(env_b.physical_keydowns),
            )

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
    )


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

    return (
        f"anchor {result.anchor_index:02d}: "
        f"actionRMS={result.action_rms:.3g} max={result.action_max:.3g} "
        f"posRMS={result.position_rms_m:.3g}m "
        f"actRMS={result.activation_rms:.3g} fatigueRMS={result.fatigue_rms:.3g} "
        f"press-div={point(result.first_pressed_divergence)} "
        f"key-div={point(result.first_keydown_divergence)} "
        f"score-div={point(result.first_score_divergence)} "
        f"over-div={point(result.first_overload_divergence)} "
        f"H={result.baseline_hits}->{result.candidate_hits} "
        f"early={result.baseline_too_early}->{result.candidate_too_early} "
        f"over={result.baseline_overloaded}->{result.candidate_overloaded}"
    )
