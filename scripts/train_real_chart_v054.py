from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v053 as base
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.keyboard import KeyEvent
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_env import RealChartMotorEnv
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.4-actuation-aware-bc"
CHECKPOINT_FORMAT_VERSION = 3
TEACHER_ACTIVE_THRESHOLD = 0.25
PRESS_MARGIN = 0.70
RELEASE_MARGIN = -0.30
NEUTRAL_PUSH_LIMIT = 0.05
MSE_COEF = 0.20
PRESS_MARGIN_COEF = 6.0
RELEASE_MARGIN_COEF = 2.0
NEUTRAL_PUSH_COEF = 12.0


@dataclass(frozen=True, slots=True)
class BCDiagnostics:
    teacher_press_frames: int
    student_strong_press_frames: int
    press_commands: int
    press_recalled: int
    release_commands: int
    release_recalled: int
    neutral_commands: int
    neutral_false_positive: int
    max_neutral_push: float

    @property
    def press_recall(self) -> float:
        return self.press_recalled / max(1, self.press_commands)

    @property
    def release_recall(self) -> float:
        return self.release_recalled / max(1, self.release_commands)

    @property
    def neutral_false_positive_rate(self) -> float:
        return self.neutral_false_positive / max(1, self.neutral_commands)


@dataclass(frozen=True, slots=True)
class StudentEvalResult:
    stats: object
    physical_keydowns: int


class DiagnosticRealChartMotorEnv(RealChartMotorEnv):
    """Real-chart env with evaluator-only physical KeyDown counting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.physical_keydowns = 0

    def reset(self):
        self.physical_keydowns = 0
        return super().reset()

    def _score_event(self, event):
        if event.event is KeyEvent.DOWN:
            self.physical_keydowns += 1
        return super()._score_event(event)


def _actuation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Imitation loss aligned with whether continuous commands actuate the key.

    A teacher press is not considered good merely because the continuous output
    moved in the correct direction: the student is pushed past a strong positive
    command margin.  Release frames get their own negative margin, while neutral
    frames retain v0.5.3's asymmetric protection against accidental positive
    force.  A small MSE term keeps the commands shaped like the teacher without
    allowing MSE to dominate the discrete physical consequences.
    """

    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and target must both have shape [T, 2]")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)

    mse = (predicted - target).square().mean()

    press_gap = torch.relu(PRESS_MARGIN - predicted).square()
    press_count = press.sum().clamp_min(1)
    press_loss = (press_gap * press).sum() / press_count

    release_gap = torch.relu(predicted - RELEASE_MARGIN).square()
    release_count = release.sum().clamp_min(1)
    release_loss = (release_gap * release).sum() / release_count

    unsafe_neutral_push = torch.relu(predicted - NEUTRAL_PUSH_LIMIT).square()
    neutral_count = neutral.sum().clamp_min(1)
    neutral_loss = (unsafe_neutral_push * neutral).sum() / neutral_count

    return (
        MSE_COEF * mse
        + PRESS_MARGIN_COEF * press_loss
        + RELEASE_MARGIN_COEF * release_loss
        + NEUTRAL_PUSH_COEF * neutral_loss
    )


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
            loss = _actuation_loss(predicted, actions[start:end])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            chunk_frames = end - start
            loss_sum += float(loss.detach().item()) * chunk_frames
            frame_count += chunk_frames

        if epoch == 1 or epoch == epochs or epoch % 2 == 0:
            print(f"bc epoch {epoch:02d}/{epochs} loss={loss_sum / max(frame_count, 1):.6f}")


def _predict_teacher_forced(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
    state = model.initial_state(observations.device)
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for x in observations:
            mean, _, _, state = model.forward_step(x, state)
            predictions.append(torch.tanh(mean))
    return torch.stack(predictions)


def _bc_diagnostics(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
    target: torch.Tensor,
) -> BCDiagnostics:
    predicted = _predict_teacher_forced(model, observations)
    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)

    teacher_press_frames = int(press.any(dim=1).sum().item())
    student_strong_press_frames = int((predicted >= PRESS_MARGIN).any(dim=1).sum().item())
    press_recalled = int((press & (predicted >= PRESS_MARGIN)).sum().item())
    release_recalled = int((release & (predicted <= RELEASE_MARGIN)).sum().item())
    neutral_fp = neutral & (predicted > NEUTRAL_PUSH_LIMIT)
    neutral_values = torch.where(neutral, predicted, torch.full_like(predicted, float("-inf")))
    max_neutral_push = float(neutral_values.max().item()) if bool(neutral.any()) else float("nan")

    return BCDiagnostics(
        teacher_press_frames=teacher_press_frames,
        student_strong_press_frames=student_strong_press_frames,
        press_commands=int(press.sum().item()),
        press_recalled=press_recalled,
        release_commands=int(release.sum().item()),
        release_recalled=release_recalled,
        neutral_commands=int(neutral.sum().item()),
        neutral_false_positive=int(neutral_fp.sum().item()),
        max_neutral_push=max_neutral_push,
    )


def _evaluate_student(
    model: RecurrentActorCritic,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> StudentEvalResult:
    env = DiagnosticRealChartMotorEnv(
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

    return StudentEvalResult(env.stats, env.physical_keydowns)


def _format_eval(label: str, result: StudentEvalResult) -> str:
    return f"{base._format_stats(label, result.stats)} keydowns={result.physical_keydowns}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real-chart student with actuation-aware behavior cloning, "
            "then compare practiced and sight-read segments."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-start", type=float, default=0.0)
    parser.add_argument("--train-end", type=float, default=30.0)
    parser.add_argument("--sight-start", type=float, default=None)
    parser.add_argument("--sight-end", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v054_actuation.pt")
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
    print("=== DMDOD / Real Chart Student v0.5.4 ===")
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
        "actuation-aware BC "
        f"press>={PRESS_MARGIN:+.2f} release<={RELEASE_MARGIN:+.2f} "
        f"neutral<={NEUTRAL_PUSH_LIMIT:+.2f} "
        f"coef={PRESS_MARGIN_COEF:g}/{RELEASE_MARGIN_COEF:g}/{NEUTRAL_PUSH_COEF:g} "
        f"mse={MSE_COEF:g}"
    )

    observations, teacher_actions, teacher_stats = base._collect_teacher_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    press_frames, release_frames, neutral_frames = base._action_frame_counts(teacher_actions)
    print(base._format_stats("teacher train", teacher_stats))
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
            raise SystemExit("unsupported real-chart v0.5.4 checkpoint format")
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

    bc_diag = _bc_diagnostics(model, observations, teacher_actions)
    print(
        f"bc probe: strong-press frames={bc_diag.student_strong_press_frames}/"
        f"{bc_diag.teacher_press_frames} "
        f"press-recall={bc_diag.press_recall * 100.0:.1f}% "
        f"release-recall={bc_diag.release_recall * 100.0:.1f}% "
        f"neutral-fp={bc_diag.neutral_false_positive_rate * 100.0:.2f}% "
        f"max-neutral-push={bc_diag.max_neutral_push:+.3f}"
    )

    startup = base._startup_probe(
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
    print(_format_eval("student practiced", practiced))
    print(_format_eval("student sight-read", sight))

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
            "actuation_aware_bc": {
                "press_margin": PRESS_MARGIN,
                "release_margin": RELEASE_MARGIN,
                "neutral_push_limit": NEUTRAL_PUSH_LIMIT,
                "mse_coef": MSE_COEF,
                "press_margin_coef": PRESS_MARGIN_COEF,
                "release_margin_coef": RELEASE_MARGIN_COEF,
                "neutral_push_coef": NEUTRAL_PUSH_COEF,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
