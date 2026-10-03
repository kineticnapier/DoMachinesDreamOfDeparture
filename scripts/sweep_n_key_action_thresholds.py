from __future__ import annotations

"""Sweep hard action thresholds for a trained N-key policy in closed loop.

This is a diagnostic for separating weak continuous action amplitude from true
closed-loop distribution shift.  The checkpoint is not retrained.  For each
press/release threshold pair, the policy's tanh output is converted to {-1,0,+1}
before it reaches the physical N-key body.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v080 as v080
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import NKeyAction, n_key_names
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
)
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


DEFAULT_PRESS_THRESHOLDS = (0.25, 0.40, 0.50, 0.60, 0.70)
DEFAULT_RELEASE_THRESHOLDS = (-0.15, -0.30, -0.45)


@dataclass(frozen=True, slots=True)
class SweepAggregate:
    hits: int
    targets: int
    x_accuracy_percent: float
    early: int
    overloaded: bool
    keydowns: int


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("threshold list must not be empty")
    return values


def discretize_action(
    values: tuple[float, ...],
    *,
    press_threshold: float,
    release_threshold: float,
) -> NKeyAction:
    press_threshold = float(press_threshold)
    release_threshold = float(release_threshold)
    if not (0.0 < press_threshold <= 1.0):
        raise ValueError("press_threshold must be in (0, 1]")
    if not (-1.0 <= release_threshold < 0.0):
        raise ValueError("release_threshold must be in [-1, 0)")

    hard = tuple(
        1.0
        if value >= press_threshold
        else -1.0
        if value <= release_threshold
        else 0.0
        for value in values
    )
    return NKeyAction(hard)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def _evaluate_segment(
    model: NKeyRecurrentActorCritic,
    named,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    press_threshold: float | None,
    release_threshold: float | None,
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
            soft_values = tuple(float(value.item()) for value in torch.tanh(mean))
            if press_threshold is None:
                action = NKeyAction(soft_values)
            else:
                if release_threshold is None:
                    raise RuntimeError("release threshold is required for hard actions")
                action = discretize_action(
                    soft_values,
                    press_threshold=press_threshold,
                    release_threshold=release_threshold,
                )
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key threshold sweep episode exceeded step budget")

    return env.stats, int(env.physical_keydowns)


def _aggregate(results: list[tuple[object, int]]) -> SweepAggregate:
    targets = sum(int(stats.targets) for stats, _ in results)
    hits = sum(int(stats.hits) for stats, _ in results)
    early = sum(int(stats.too_early_presses) for stats, _ in results)
    keydowns = sum(int(keydowns) for _, keydowns in results)
    x_points = sum(float(stats.x_accuracy_points) for stats, _ in results)
    x_den = sum(float(stats.x_accuracy_denominator) for stats, _ in results)
    xacc = 100.0 * x_points / x_den if x_den > 0.0 else 0.0
    overloaded = any(bool(stats.overloaded) for stats, _ in results)
    return SweepAggregate(
        hits=hits,
        targets=targets,
        x_accuracy_percent=xacc,
        early=early,
        overloaded=overloaded,
        keydowns=keydowns,
    )


def _run_setting(
    model: NKeyRecurrentActorCritic,
    anchors,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
    press_threshold: float | None,
    release_threshold: float | None,
) -> SweepAggregate:
    results = [
        _evaluate_segment(
            model,
            named,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            press_threshold=press_threshold,
            release_threshold=release_threshold,
        )
        for named in anchors
    ]
    return _aggregate(results)


def _format_result(label: str, result: SweepAggregate) -> str:
    return (
        f"{label}: H={result.hits}/{result.targets} "
        f"X={result.x_accuracy_percent:.2f}% early={result.early} "
        f"over={result.overloaded} keydowns={result.keydowns}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one N-key checkpoint closed-loop with continuous actions and "
            "a grid of hard press/release thresholds on Train anchors."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--press-thresholds",
        default=",".join(str(value) for value in DEFAULT_PRESS_THRESHOLDS),
    )
    parser.add_argument(
        "--release-thresholds",
        default=",".join(str(value) for value in DEFAULT_RELEASE_THRESHOLDS),
    )
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        press_thresholds = _parse_float_list(args.press_thresholds)
        release_thresholds = _parse_float_list(args.release_thresholds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(not (0.0 < value <= 1.0) for value in press_thresholds):
        raise SystemExit("all press thresholds must be in (0, 1]")
    if any(not (-1.0 <= value < 0.0) for value in release_thresholds):
        raise SystemExit("all release thresholds must be in [-1, 0)")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")

    device = _device_from_arg(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    key_count = int(payload["key_count"])
    key_names = n_key_names(key_count)
    input_dim = int(payload["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected {expected_input}"
        )

    model = NKeyRecurrentActorCritic(
        input_dim=input_dim,
        key_count=key_count,
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.gru.flatten_parameters()

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=float(payload["train_window"]),
        anchors_per_chart=int(payload["anchors_per_chart"]),
    )
    checkpoint_limit = payload.get("anchor_limit")
    limit = args.anchor_limit if args.anchor_limit is not None else checkpoint_limit
    if limit is not None:
        anchors = anchors[: int(limit)]
    if not anchors:
        raise SystemExit("no Train anchors selected")

    control_dt_s = float(payload["control_dt"])
    physics_dt_s = float(payload.get("physics_dt", 0.001))

    print("=== DMDOD N-Key Closed-Loop Action Threshold Sweep ===")
    print(
        f"checkpoint={checkpoint_path} keys={key_count} input={input_dim}D "
        f"anchors={len(anchors)} device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        "press-thresholds=" + ",".join(f"{value:.2f}" for value in press_thresholds)
    )
    print(
        "release-thresholds=" + ",".join(f"{value:.2f}" for value in release_thresholds)
    )

    continuous = _run_setting(
        model,
        anchors,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
        press_threshold=None,
        release_threshold=None,
    )
    print(_format_result("continuous", continuous))

    hard_results: list[tuple[float, float, SweepAggregate]] = []
    for press_threshold in press_thresholds:
        for release_threshold in release_thresholds:
            result = _run_setting(
                model,
                anchors,
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
                press_threshold=press_threshold,
                release_threshold=release_threshold,
            )
            hard_results.append((press_threshold, release_threshold, result))
            print(
                _format_result(
                    f"hard p>={press_threshold:.2f} r<={release_threshold:.2f}",
                    result,
                )
            )

    best_press, best_release, best = max(
        hard_results,
        key=lambda item: (
            item[2].hits,
            item[2].x_accuracy_percent,
            -item[2].early,
            item[2].keydowns,
        ),
    )
    print("=== best hard setting by hits, then XAcc ===")
    print(
        _format_result(
            f"p>={best_press:.2f} r<={best_release:.2f}",
            best,
        )
    )


if __name__ == "__main__":
    main()
