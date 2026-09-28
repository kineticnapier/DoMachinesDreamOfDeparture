from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_rules import timing_windows
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.motor_env import MotorAction
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_env import RealChartMotorEnv
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.3-neutral-safe-bc"
CHECKPOINT_FORMAT_VERSION = 2
PRESS_THRESHOLD = 0.25
PRESS_WEIGHT = 6.0
RELEASE_WEIGHT = 2.0
NEUTRAL_WEIGHT = 1.0
NEUTRAL_PUSH_LIMIT = 0.05
UNSAFE_NEUTRAL_PUSH_COEF = 12.0


@dataclass(frozen=True, slots=True)
class StartupProbeResult:
    safe: bool
    early_presses: int
    overloaded: bool
    max_positive_action: float
    until_s: float


def _teacher_action(env: RealChartMotorEnv, observation, lead_s: float) -> MotorAction:
    """Privileged alternating-finger action used only to generate BC labels."""

    target = env.privileged_next_target()
    now = env.privileged_episode_time_s()
    motor = observation.motor

    left = -1.0 if motor.left_pressed else 0.0
    right = -1.0 if motor.right_pressed else 0.0
    if target is not None and now + lead_s >= target.episode_time_s:
        if target.ordinal & 1:
            if not motor.right_pressed:
                right = 1.0
        else:
            if not motor.left_pressed:
                left = 1.0
    return MotorAction(left, right)


def _collect_teacher_sequence(
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    env = RealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    xs: list[tuple[float, ...]] = []
    ys: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        action = _teacher_action(env, observation, lead_s)
        xs.append(encode_real_chart_observation(observation))
        ys.append((action.left, action.right))
        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("teacher real-chart episode exceeded step budget")

    x = torch.tensor(xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    return x, y, env.stats


def _action_frame_counts(actions: torch.Tensor) -> tuple[int, int, int]:
    """Return press/release/neutral frame counts for compact diagnostics."""

    press = (actions > PRESS_THRESHOLD).any(dim=1)
    release = (~press) & (actions < -PRESS_THRESHOLD).any(dim=1)
    neutral = ~(press | release)
    return (
        int(press.sum().item()),
        int(release.sum().item()),
        int(neutral.sum().item()),
    )


def _bc_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Neutral-safe imitation objective for continuous motor commands.

    Presses still receive extra weight, but releases no longer dominate every
    active frame.  More importantly, a positive command while the teacher asks
    for neutral is explicitly penalized once it exceeds a small dead-safe
    margin.  Negative neutral error is not given the same extra penalty because
    lifting a finger cannot create a premature KeyDown.
    """

    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and target must both have shape [T, 2]")

    weights = torch.full_like(target, NEUTRAL_WEIGHT)
    weights = torch.where(target > PRESS_THRESHOLD, torch.full_like(weights, PRESS_WEIGHT), weights)
    weights = torch.where(target < -PRESS_THRESHOLD, torch.full_like(weights, RELEASE_WEIGHT), weights)
    squared = (predicted - target).square()
    imitation = (squared * weights).sum() / weights.sum().clamp_min(1.0)

    neutral = target.abs() <= PRESS_THRESHOLD
    unsafe_push = torch.relu(predicted - NEUTRAL_PUSH_LIMIT).square()
    neutral_count = neutral.sum().clamp_min(1)
    safety = (unsafe_push * neutral).sum() / neutral_count
    return imitation + UNSAFE_NEUTRAL_PUSH_COEF * safety


def _train_bc(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
) -> None:
    parameters = [
        *model.input_layer.parameters(),
        *model.gru.parameters(),
        *model.post.parameters(),
        *model.actor_mean.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    for epoch in range(1, epochs + 1):
        state = model.initial_state(observations.device)
        loss_sum = 0.0
        frame_count = 0

        for start in range(0, observations.shape[0], chunk_steps):
            end = min(observations.shape[0], start + chunk_steps)
            state = state.detach()
            predictions: list[torch.Tensor] = []
            for x in observations[start:end]:
                mean, _, _, state = model.forward_step(x, state)
                predictions.append(torch.tanh(mean))

            predicted = torch.stack(predictions)
            loss = _bc_loss(predicted, actions[start:end])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            chunk_frames = end - start
            loss_sum += float(loss.detach().item()) * chunk_frames
            frame_count += chunk_frames

        if epoch == 1 or epoch == epochs or epoch % 2 == 0:
            print(f"bc epoch {epoch:02d}/{epochs} loss={loss_sum / max(frame_count, 1):.6f}")


def _startup_probe(
    model: RecurrentActorCritic,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> StartupProbeResult:
    """Check that the student can stay physically idle before the first hit window.

    The probe uses privileged timing only to decide when the diagnostic ends.
    The model itself still receives the exact same visible-only observation as
    normal evaluation.
    """

    env = RealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    first = segment.targets[0]
    pass_s = timing_windows(
        first.bpm,
        pitch=segment.pitch_ratio,
    ).pass_s
    until_s = max(0.0, first.episode_time_s - pass_s)
    max_positive = 0.0

    with torch.no_grad():
        while env.privileged_episode_time_s() + control_dt_s <= until_s + 1e-12:
            x = torch.tensor(
                encode_real_chart_observation(observation),
                dtype=torch.float32,
                device=device,
            )
            action, state = model.deterministic_action(x, state)
            max_positive = max(max_positive, action.left, action.right, 0.0)
            step = env.step(action)
            observation = step.observation
            if step.done or env.stats.too_early_presses > 0:
                break

    stats = env.stats
    safe = stats.too_early_presses == 0 and not stats.overloaded
    return StartupProbeResult(
        safe=safe,
        early_presses=stats.too_early_presses,
        overloaded=stats.overloaded,
        max_positive_action=max_positive,
        until_s=until_s,
    )


def _evaluate_student(
    model: RecurrentActorCritic,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    env = RealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(
                encode_real_chart_observation(observation),
                dtype=torch.float32,
                device=device,
            )
            action, state = model.deterministic_action(x, state)
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("student real-chart episode exceeded step budget")
    return env.stats


def _format_stats(label: str, stats) -> str:
    mae = stats.mean_abs_error_ms
    mae_text = "nan" if mae is None else f"{mae:.2f}"
    return (
        f"{label}: H={stats.hits}/{stats.targets} miss={stats.misses} "
        f"X={stats.x_accuracy_percent:.2f}% PP={stats.perfect_rate * 100.0:.1f}% "
        f"MAE={mae_text}ms early={stats.too_early_presses} overload={stats.overloaded}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real-chart student with neutral-safe behavior cloning, "
            "then compare practiced and sight-read segments."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-start", type=float, default=0.0)
    parser.add_argument("--train-end", type=float, default=30.0)
    parser.add_argument("--sight-start", type=float, default=None)
    parser.add_argument("--sight-end", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v053_neutral_safe.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cross-hand",
        action="store_true",
        help="use the bilateral two-finger body instead of same-hand fingers",
    )
    args = parser.parse_args()

    if args.train_end <= args.train_start:
        raise SystemExit("--train-end must be greater than --train-start")
    if args.epochs <= 0 or args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("epochs/hidden/chunk-steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand
    compiled = load_compiled_adofai(args.chart)
    train_segment = build_playable_segment(
        compiled,
        start_s=args.train_start,
        end_s=min(args.train_end, compiled.duration_s),
    )
    if not train_segment.targets:
        raise SystemExit("training segment contains no playable targets")

    sight_start = args.train_end if args.sight_start is None else args.sight_start
    sight_end = (
        sight_start + (args.train_end - args.train_start)
        if args.sight_end is None
        else args.sight_end
    )
    sight_end = min(sight_end, compiled.duration_s)
    sight_segment = build_playable_segment(compiled, start_s=sight_start, end_s=sight_end)
    if not sight_segment.targets:
        raise SystemExit("sight-read segment contains no playable targets")

    calibration = calibrate_single_press_lead(
        control_dt_s=args.control_dt,
        same_hand=same_hand,
    )
    print("=== DMDOD / Real Chart Student v0.5.3 ===")
    print(
        f"chart={args.chart}\n"
        f"train={args.train_start:g}..{min(args.train_end, compiled.duration_s):g}s "
        f"targets={len(train_segment.targets)} | "
        f"sight={sight_start:g}..{sight_end:g}s targets={len(sight_segment.targets)}"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D visible-only hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms"
    )
    print(
        f"neutral-safe BC press/release/neutral weights="
        f"{PRESS_WEIGHT:g}/{RELEASE_WEIGHT:g}/{NEUTRAL_WEIGHT:g} "
        f"push-limit=+{NEUTRAL_PUSH_LIMIT:.2f} coef={UNSAFE_NEUTRAL_PUSH_COEF:g}"
    )

    observations, teacher_actions, teacher_stats = _collect_teacher_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    press_frames, release_frames, neutral_frames = _action_frame_counts(teacher_actions)
    print(_format_stats("teacher train", teacher_stats))
    print(
        f"teacher frames={observations.shape[0]} press={press_frames} "
        f"release={release_frames} neutral={neutral_frames}"
    )

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
            raise SystemExit("unsupported real-chart v0.5.3 checkpoint format")
        if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
            raise SystemExit("checkpoint real-chart input dimension does not match current encoder")
        if int(payload.get("hidden_dim", -1)) != args.hidden:
            raise SystemExit("checkpoint hidden size does not match --hidden")
        model.load_state_dict(payload["model_state"])
        print(f"resume={checkpoint_path}")

    _train_bc(
        model,
        observations,
        teacher_actions,
        epochs=args.epochs,
        learning_rate=args.lr,
        chunk_steps=args.chunk_steps,
    )

    startup = _startup_probe(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    print(
        f"startup probe: {'PASS' if startup.safe else 'FAIL'} "
        f"early={startup.early_presses} overload={startup.overloaded} "
        f"max_push={startup.max_positive_action:+.3f} until={startup.until_s:.3f}s"
    )

    practiced = _evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    sight = _evaluate_student(
        model,
        sight_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    print(_format_stats("student practiced", practiced))
    print(_format_stats("student sight-read", sight))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "input_dim": REAL_CHART_INPUT_DIM,
            "hidden_dim": args.hidden,
            "model_state": model.state_dict(),
            "chart": str(args.chart),
            "train_start": args.train_start,
            "train_end": min(args.train_end, compiled.duration_s),
            "feature_config": {
                "behind_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
                "ahead_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
            },
            "neutral_safe_bc": {
                "press_weight": PRESS_WEIGHT,
                "release_weight": RELEASE_WEIGHT,
                "neutral_weight": NEUTRAL_WEIGHT,
                "push_limit": NEUTRAL_PUSH_LIMIT,
                "unsafe_push_coef": UNSAFE_NEUTRAL_PUSH_COEF,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
