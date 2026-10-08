from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.connectome.fly_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
    NKeyFlyConnectomeActorCritic,
)
from dmdod.n_key_dagger_continuation import (
    collect_n_key_dagger_sequence_with_continuation,
)
from dmdod.n_key_motor import NKeyAction
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.n_key_training import NKeyBCSequence
from dmdod.connectome.random_policy import (
    N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    NKeyRandomConnectomeActorCritic,
)
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


DEFAULT_HIT_DROP_FRACTION = 0.03
DEFAULT_HIT_DROP_ABSOLUTE = 3


@dataclass(frozen=True, slots=True)
class ChartRuntime:
    spec: object
    compiled: object

    @property
    def duration_s(self) -> float:
        return float(self.compiled.duration_s)


@dataclass(frozen=True, slots=True)
class NamedSegment:
    role: str
    chart_name: str
    chart_sha256: str
    start_s: float
    end_s: float
    segment: object


@dataclass(frozen=True, slots=True)
class RoleSummary:
    hits: int
    targets: int
    x_accuracy_percent: float
    early: int
    overloaded: bool
    keydowns: int


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def compile_role(items) -> list[ChartRuntime]:
    return [
        ChartRuntime(item, load_compiled_adofai(item.resolved_path))
        for item in items
    ]


def _window(duration_s: float, start_s: float, length_s: float) -> tuple[float, float]:
    if duration_s <= 0.0:
        raise ValueError("chart duration must be positive")
    length = min(float(length_s), duration_s)
    start = min(max(0.0, float(start_s)), max(0.0, duration_s - length))
    return start, start + length


def _anchor_windows(
    duration_s: float,
    window_s: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    if count <= 0:
        raise ValueError("anchor count must be positive")
    length = min(window_s, duration_s)
    max_start = max(0.0, duration_s - length)
    if count == 1 or max_start <= 1e-12:
        return ((0.0, length),)

    result: list[tuple[float, float]] = []
    for index in range(count):
        pair = _window(duration_s, max_start * index / (count - 1), length)
        if not result or abs(pair[0] - result[-1][0]) > 1e-9:
            result.append(pair)
    return tuple(result)


def _named_segment(
    runtime: ChartRuntime,
    role: str,
    start_s: float,
    end_s: float,
) -> NamedSegment:
    segment = build_playable_segment(
        runtime.compiled,
        start_s=start_s,
        end_s=end_s,
    )
    if not segment.targets:
        raise ValueError(
            f"segment contains no playable targets: "
            f"{runtime.spec.name} {start_s:.3f}..{end_s:.3f}s"
        )
    return NamedSegment(
        role=role,
        chart_name=runtime.spec.name,
        chart_sha256=runtime.spec.content_sha256,
        start_s=start_s,
        end_s=end_s,
        segment=segment,
    )


def build_anchor_segments(
    train_charts: list[ChartRuntime],
    *,
    window_s: float,
    anchors_per_chart: int,
) -> list[NamedSegment]:
    result: list[NamedSegment] = []
    for chart in train_charts:
        for index, (start, end) in enumerate(
            _anchor_windows(chart.duration_s, window_s, anchors_per_chart),
            1,
        ):
            result.append(_named_segment(chart, f"anchor-{index}", start, end))
    return result


def build_validation_segments(
    validation_charts: list[ChartRuntime],
    *,
    window_s: float,
) -> list[NamedSegment]:
    result: list[NamedSegment] = []
    for chart in validation_charts:
        length = min(float(window_s), chart.duration_s)
        start, end = _window(
            chart.duration_s,
            (chart.duration_s - length) * 0.5,
            length,
        )
        result.append(_named_segment(chart, "validation", start, end))
    return result


def summarize(results: list[tuple[object, int]]) -> RoleSummary:
    if not results:
        return RoleSummary(0, 0, 0.0, 0, False, 0)
    targets = sum(int(stats.targets) for stats, _ in results)
    hits = sum(int(stats.hits) for stats, _ in results)
    early = sum(int(stats.too_early_presses) for stats, _ in results)
    keydowns = sum(int(count) for _, count in results)
    x_points = sum(float(stats.x_accuracy_points) for stats, _ in results)
    x_den = sum(float(stats.x_accuracy_denominator) for stats, _ in results)
    return RoleSummary(
        hits=hits,
        targets=targets,
        x_accuracy_percent=100.0 * x_points / x_den if x_den > 0.0 else 0.0,
        early=early,
        overloaded=any(bool(stats.overloaded) for stats, _ in results),
        keydowns=keydowns,
    )


def format_summary(summary: RoleSummary) -> str:
    return (
        f"H={summary.hits}/{summary.targets} "
        f"X={summary.x_accuracy_percent:.2f}% "
        f"early={summary.early} over={summary.overloaded} "
        f"keydowns={summary.keydowns}"
    )


def aggregate(results: list[tuple[object, int]]) -> str:
    return "none" if not results else format_summary(summarize(results))


def train_safety_guard(
    references: list[tuple[object, int]],
    candidates: list[tuple[object, int]],
) -> tuple[bool, tuple[str, ...]]:
    if len(references) != len(candidates):
        raise ValueError("Train guard reference/candidate count mismatch")

    reasons: list[str] = []
    for index, ((reference, _), (candidate, _)) in enumerate(
        zip(references, candidates),
        1,
    ):
        if not bool(reference.overloaded) and bool(candidate.overloaded):
            reasons.append(f"anchor {index} safe->overload")
            continue

        tolerance = max(
            DEFAULT_HIT_DROP_ABSOLUTE,
            round(int(reference.targets) * DEFAULT_HIT_DROP_FRACTION),
        )
        floor = int(reference.hits) - tolerance
        if int(candidate.hits) < floor:
            reasons.append(
                f"anchor {index} hits {int(candidate.hits)}<{floor} "
                f"(reference={int(reference.hits)} tolerance={tolerance})"
            )
    return not reasons, tuple(reasons)


def train_survival_guard(
    references: list[tuple[object, int]],
    candidates: list[tuple[object, int]],
) -> tuple[bool, tuple[str, ...]]:
    """Reject only when a previously SAFE Train anchor becomes overloaded."""

    if len(references) != len(candidates):
        raise ValueError("Train guard reference/candidate count mismatch")

    reasons: list[str] = []
    for index, ((reference, _), (candidate, _)) in enumerate(
        zip(references, candidates),
        1,
    ):
        if not bool(reference.overloaded) and bool(candidate.overloaded):
            reasons.append(f"anchor {index} safe->overload")
    return not reasons, tuple(reasons)


def selection_key(results: list[tuple[object, int]]) -> tuple[float, ...]:
    summary = summarize(results)
    return (
        float(summary.hits),
        float(summary.x_accuracy_percent),
        -float(summary.early),
        -float(summary.keydowns),
    )


def safe_anchor_count(results: list[tuple[object, int]]) -> int:
    return sum(
        1
        for stats, _ in results
        if not bool(stats.overloaded)
    )


def survival_selection_key(
    results: list[tuple[object, int]],
) -> tuple[float, ...]:
    return (
        float(safe_anchor_count(results)),
        *selection_key(results),
    )


def clone_model_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _required(checkpoint: dict, name: str):
    if name not in checkpoint:
        raise SystemExit(f"connectome checkpoint is missing {name}")
    return checkpoint[name]


def build_connectome_policy_from_checkpoint(
    checkpoint: dict,
    *,
    device: torch.device,
):
    backend = str(checkpoint.get("n_key_policy_backend", "gru"))
    if backend not in {
        N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
        N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    }:
        raise SystemExit(
            "current training pipeline requires fly_connectome or random_connectome"
        )

    common = dict(
        input_dim=int(checkpoint["input_dim"]),
        key_count=int(checkpoint["key_count"]),
        core_path=str(_required(checkpoint, "fly_connectome_core_path")),
        sensory_dim=int(_required(checkpoint, "fly_connectome_sensory_dim")),
        recurrent_gain=float(_required(checkpoint, "fly_connectome_recurrent_gain")),
        projection_seed=int(_required(checkpoint, "fly_connectome_projection_seed")),
    )
    if backend == N_KEY_POLICY_BACKEND_FLY_CONNECTOME:
        model = NKeyFlyConnectomeActorCritic(**common)
    else:
        model = NKeyRandomConnectomeActorCritic(
            **common,
            topology_seed=int(_required(checkpoint, "random_connectome_topology_seed")),
        )

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.prepare_recurrent_runtime()
    return model


def _evaluate_continuous(
    model,
    named: NamedSegment,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,
        key_count=model.key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    max_steps = int((named.segment.duration_s + 2.0) / control_dt_s) + 200

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(
                encode_n_key_hud_real_chart_observation(observation),
                dtype=torch.float32,
                device=device,
            )
            mean, _, _, state = model.forward_step(x, state)
            action = NKeyAction(
                tuple(float(value.item()) for value in torch.tanh(mean))
            )
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key continuous evaluation exceeded step budget")
    return env.stats, int(env.physical_keydowns)


def evaluate_role_continuous(
    model,
    segments: list[NamedSegment],
    *,
    label: str,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
) -> list[tuple[object, int]]:
    results: list[tuple[object, int]] = []
    total = len(segments)
    for index, named in enumerate(segments, 1):
        stats, keydowns = _evaluate_continuous(
            model,
            named,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
        )
        results.append((stats, keydowns))
        print(
            f"{label} {index:02d}/{total} {named.chart_name}: "
            f"H={stats.hits}/{stats.targets} "
            f"X={stats.x_accuracy_percent:.2f}% "
            f"PP={stats.perfect_rate * 100.0:.1f}% "
            f"MAE={stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float('nan'):.2f}ms "
            f"early={stats.too_early_presses} "
            f"over={stats.overloaded} keydowns={keydowns}"
        )
    print(f"{label} aggregate: {aggregate(results)}")
    return results


def collect_student_state_sequences(
    model,
    anchors: list[NamedSegment],
    *,
    round_index: int,
    collection_index: int,
    lead_s: float,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
) -> tuple[list[NKeyBCSequence], int]:
    sequences: list[NKeyBCSequence] = []
    total = len(anchors)
    label = f"collect-m{collection_index}"
    for index, named in enumerate(anchors, 1):
        rollout = collect_n_key_dagger_sequence_with_continuation(
            model,
            named.segment,
            lead_s=lead_s,
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=(
                f"dagger{round_index}-student-m{collection_index}-"
                f"{index}-{named.chart_name}"
            ),
            action_mode="continuous",
            continue_after_failure=True,
        )
        sequences.append(rollout.sequence)
        print(
            f"{label} {index:02d}/{total} {named.chart_name}: "
            f"frames={rollout.sequence.frames} "
            f"H={rollout.stats.hits}/{rollout.stats.targets} "
            f"X={rollout.stats.x_accuracy_percent:.2f}% "
            f"early={rollout.stats.too_early_presses} "
            f"over={rollout.stats.overloaded} "
            f"keydowns={rollout.physical_keydowns}"
        )
    frames = sum(sequence.frames for sequence in sequences)
    print(f"{label} aggregate: student-state={frames} frames")
    return sequences, frames
