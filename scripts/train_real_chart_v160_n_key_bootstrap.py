from __future__ import annotations

"""v1.6 bootstrap trainer for configurable even-N Level A bodies.

This generalizes the proven v1.5 four-key bootstrap path to 4K/6K/8K while
keeping the same chart geometry and HUD contract.  Final is never evaluated or
used for selection here.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

import train_real_chart_v080 as v080
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_capacity import calibrate_n_key_press_lead
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
)
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_expert_sequence,
    n_key_actuation_loss,
)
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


TRAINER_VERSION = "1.6.0-n-key-bootstrap"
CHECKPOINT_FORMAT_VERSION = 17


def _train_bc_epoch(
    model: NKeyRecurrentActorCritic,
    sequences: list[NKeyBCSequence],
    *,
    optimizer: torch.optim.Optimizer,
    chunk_steps: int,
) -> float:
    if not sequences:
        raise ValueError("at least one N-key training sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    model.train()
    loss_sum = 0.0
    frame_count = 0
    for sequence in sequences:
        if sequence.key_count != model.key_count:
            raise ValueError("sequence key_count does not match model key_count")
        state = model.initial_state(sequence.observations.device)
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            state = state.detach()
            means, _, state = model.forward_sequence(
                sequence.observations[start:end],
                state,
            )
            predicted = torch.tanh(means)
            target = sequence.teacher_actions[start:end]
            loss = n_key_actuation_loss(predicted, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            frames = end - start
            loss_sum += float(loss.detach().item()) * frames
            frame_count += frames

    return loss_sum / max(1, frame_count)


def _evaluate(
    model: NKeyRecurrentActorCritic,
    named,
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
            action, state = model.deterministic_action(x, state)
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key validation episode exceeded step budget")
    return env.stats, int(env.physical_keydowns)


def _aggregate(results: list[tuple[object, int]]) -> str:
    if not results:
        return "none"
    targets = sum(int(stats.targets) for stats, _ in results)
    hits = sum(int(stats.hits) for stats, _ in results)
    early = sum(int(stats.too_early_presses) for stats, _ in results)
    keydowns = sum(int(keydowns) for _, keydowns in results)
    x_points = sum(float(stats.x_accuracy_points) for stats, _ in results)
    x_den = sum(float(stats.x_accuracy_denominator) for stats, _ in results)
    xacc = 100.0 * x_points / x_den if x_den > 0.0 else 0.0
    overloaded = any(bool(stats.overloaded) for stats, _ in results)
    return (
        f"H={hits}/{targets} X={xacc:.2f}% early={early} "
        f"over={overloaded} keydowns={keydowns}"
    )


def _checkpoint_payload(
    *,
    model: NKeyRecurrentActorCritic,
    args,
    dataset,
    calibration,
    epoch: int,
    losses: list[float],
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_version": TRAINER_VERSION,
        "input_dim": int(model.input_dim),
        "action_dim": int(model.action_dim),
        "key_count": int(model.key_count),
        "key_names": tuple(model.key_names),
        "hidden_dim": int(model.hidden_dim),
        "model_state": model.state_dict(),
        "dataset_root": dataset.root,
        "dataset_signature": dataset.signature(),
        "completed_bootstrap_epoch": int(epoch),
        "requested_bootstrap_epochs": int(args.bootstrap_epochs),
        "train_window": float(args.train_window),
        "validation_window": float(args.validation_window),
        "anchors_per_chart": int(args.anchors_per_chart),
        "anchor_limit": args.anchor_limit,
        "validation_limit": args.validation_limit,
        "control_dt": float(args.control_dt),
        "physics_dt": float(args.physics_dt),
        "chunk_steps": int(args.chunk_steps),
        "seed": int(args.seed),
        "calibration": asdict(calibration),
        "loss_history": list(losses),
        "final_used_for_selection": False,
        "finalized": False,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap a configurable N-key Level A policy on Train anchors and "
            "evaluate only Validation. Final is never touched."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--train-window", type=float, default=v080.DEFAULT_TRAIN_WINDOW_S)
    parser.add_argument("--validation-window", type=float, default=v080.DEFAULT_VALIDATION_WINDOW_S)
    parser.add_argument("--anchors-per-chart", type=int, default=v080.DEFAULT_ANCHORS_PER_CHART)
    parser.add_argument("--bootstrap-epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    try:
        key_names = n_key_names(args.keys)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.train_window <= 0.0 or args.validation_window <= 0.0:
        raise SystemExit("train/validation windows must be positive")
    if args.anchors_per_chart <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("anchors-per-chart/bootstrap-epochs must be positive")
    if args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("hidden/chunk-steps must be positive")
    if args.control_dt <= 0.0 or args.physics_dt <= 0.0:
        raise SystemExit("control-dt/physics-dt must be positive")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = _device_from_arg(args.device)

    try:
        dataset = discover_multichart_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    train_charts = v080._compile_role(dataset.train)
    validation_charts = v080._compile_role(dataset.validation)
    anchor_segments = v080._build_anchor_segments(
        train_charts,
        window_s=args.train_window,
        anchors_per_chart=args.anchors_per_chart,
    )
    validation_segments = v080._build_validation_segments(
        validation_charts,
        window_s=args.validation_window,
    )
    if args.anchor_limit is not None:
        anchor_segments = anchor_segments[: args.anchor_limit]
    if args.validation_limit is not None:
        validation_segments = validation_segments[: args.validation_limit]

    calibration = calibrate_n_key_press_lead(
        args.keys,
        control_dt_s=args.control_dt,
        physics_dt_s=args.physics_dt,
    )
    input_dim = n_key_hud_real_chart_input_dim(args.keys)
    checkpoint = Path(
        args.checkpoint
        or f"checkpoints/real_chart_v160_{args.keys}k_bootstrap.pt"
    )

    print("=== DMDOD v1.6.0 N-Key Bootstrap ===")
    print(
        f"keys={args.keys} input={input_dim}D action={args.keys} hidden={args.hidden} "
        f"device={device} lead={calibration.lead_s * 1000.0:.1f}ms "
        f"control={args.control_dt * 1000.0:.1f}ms physics={args.physics_dt * 1000.0:.1f}ms"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchor_segments)} validation={len(validation_segments)} "
        f"epochs={args.bootstrap_epochs} | FINAL untouched"
    )
    print(
        "press-latency-ms: "
        + " ".join(
            f"{key}={latency * 1000.0:.1f}"
            for key, latency in calibration.key_latencies_s
        )
    )

    sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(anchor_segments, 1):
        rollout = collect_n_key_expert_sequence(
            named.segment,
            key_count=args.keys,
            lead_s=calibration.lead_s,
            control_dt_s=args.control_dt,
            physics_dt_s=args.physics_dt,
            device=device,
            source=f"{args.keys}k-anchor-{index}-{named.chart_name}",
        )
        sequences.append(rollout.sequence)
        print(
            f"teacher {index:03d}/{len(anchor_segments)} {named.chart_name} "
            f"H={rollout.stats.hits}/{rollout.stats.targets} "
            f"X={rollout.stats.x_accuracy_percent:.1f}% "
            f"early={rollout.stats.too_early_presses} "
            f"over={rollout.stats.overloaded} keydowns={rollout.physical_keydowns}"
        )

    model = NKeyRecurrentActorCritic(
        input_dim=input_dim,
        key_count=args.keys,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    # Keep cuDNN's recurrent weights packed after moving the model to CUDA.
    model.gru.flatten_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: list[float] = []

    for epoch in range(1, args.bootstrap_epochs + 1):
        loss = _train_bc_epoch(
            model,
            sequences,
            optimizer=optimizer,
            chunk_steps=args.chunk_steps,
        )
        losses.append(loss)
        print(f"bootstrap {epoch:03d}/{args.bootstrap_epochs} loss={loss:.6f}")
        _save_checkpoint(
            checkpoint,
            _checkpoint_payload(
                model=model,
                args=args,
                dataset=dataset,
                calibration=calibration,
                epoch=epoch,
                losses=losses,
            ),
        )

    validation_results: list[tuple[object, int]] = []
    for index, named in enumerate(validation_segments, 1):
        stats, keydowns = _evaluate(
            model,
            named,
            control_dt_s=args.control_dt,
            physics_dt_s=args.physics_dt,
            device=device,
        )
        validation_results.append((stats, keydowns))
        print(
            f"validation {index:02d}/{len(validation_segments)} {named.chart_name}: "
            f"H={stats.hits}/{stats.targets} X={stats.x_accuracy_percent:.2f}% "
            f"PP={stats.perfect_rate * 100.0:.1f}% "
            f"MAE={stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float('nan'):.2f}ms "
            f"early={stats.too_early_presses} over={stats.overloaded} keydowns={keydowns}"
        )

    print("validation aggregate: " + _aggregate(validation_results))
    print(f"checkpoint final: {checkpoint}")


if __name__ == "__main__":
    main()
