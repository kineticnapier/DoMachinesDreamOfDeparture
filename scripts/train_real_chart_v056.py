from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v055 as v055
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.6-stable-dagger"
CHECKPOINT_FORMAT_VERSION = 5
DEFAULT_BOOTSTRAP = "checkpoints/real_chart_v054_actuation.pt"
DEFAULT_MIXTURE_BETAS = (0.50, 0.25, 0.10, 0.00)
DEFAULT_PRESS_RECOVERY_CAP = 8
DEFAULT_HIT_REGRESSION_FRACTION = 0.05
DEFAULT_HIT_REGRESSION_ABSOLUTE = 3


@dataclass(frozen=True, slots=True)
class StableSequence:
    sequence: v055.DAggerSequence
    loss_weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.loss_weights.shape != self.sequence.teacher_actions.shape:
            raise ValueError("stable DAgger loss weights must match teacher action shape")
        if bool((self.loss_weights < 0.0).any()):
            raise ValueError("stable DAgger loss weights must be non-negative")

    @property
    def frames(self) -> int:
        return self.sequence.frames


@dataclass(frozen=True, slots=True)
class StableRollout:
    sequence: StableSequence
    evaluation: v054.StudentEvalResult
    teacher_fraction: float


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str


def _mixture_beta(round_index: int, schedule: tuple[float, ...] = DEFAULT_MIXTURE_BETAS) -> float:
    if round_index <= 0:
        raise ValueError("round_index must be positive")
    if not schedule:
        return 0.0
    return float(schedule[min(round_index - 1, len(schedule) - 1)])


def _press_recovery_weights(actions: torch.Tensor, cap: int) -> torch.Tensor:
    """Mask excess repeated +1 recovery labels while preserving sequence context.

    DAgger can visit a state where a target remains unresolved for many frames;
    the privileged teacher then emits the same positive command every frame.
    Those frames still pass through the GRU, but only the first ``cap`` positive
    labels in each per-finger run contribute press loss.  Release/neutral labels
    always remain trainable.
    """

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [T, 2]")
    if cap <= 0:
        raise ValueError("press recovery cap must be positive")

    weights = torch.ones_like(actions)
    threshold = v054.TEACHER_ACTIVE_THRESHOLD
    for channel in range(actions.shape[1]):
        run = 0
        for index in range(actions.shape[0]):
            if float(actions[index, channel].item()) > threshold:
                run += 1
                if run > cap:
                    weights[index, channel] = 0.0
            else:
                run = 0
    return weights


def _make_stable_sequence(
    sequence: v055.DAggerSequence,
    *,
    press_recovery_cap: int,
    expert: bool = False,
) -> StableSequence:
    weights = (
        torch.ones_like(sequence.teacher_actions)
        if expert
        else _press_recovery_weights(sequence.teacher_actions, press_recovery_cap)
    )
    return StableSequence(sequence, weights)


def _weighted_actuation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or loss_weights.shape != target.shape:
        raise ValueError("predicted, target, and loss_weights must have the same shape")
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("actuation tensors must have shape [T, 2]")

    press = target > v054.TEACHER_ACTIVE_THRESHOLD
    release = target < -v054.TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)
    weights = loss_weights.to(dtype=predicted.dtype)

    mse_weight = weights.sum().clamp_min(1.0)
    mse = ((predicted - target).square() * weights).sum() / mse_weight

    press_weight = weights * press
    press_gap = torch.relu(v054.PRESS_MARGIN - predicted).square()
    press_loss = (press_gap * press_weight).sum() / press_weight.sum().clamp_min(1.0)

    release_weight = weights * release
    release_gap = torch.relu(predicted - v054.RELEASE_MARGIN).square()
    release_loss = (release_gap * release_weight).sum() / release_weight.sum().clamp_min(1.0)

    neutral_weight = weights * neutral
    unsafe_neutral = torch.relu(predicted - v054.NEUTRAL_PUSH_LIMIT).square()
    neutral_loss = (unsafe_neutral * neutral_weight).sum() / neutral_weight.sum().clamp_min(1.0)

    return (
        v054.MSE_COEF * mse
        + v054.PRESS_MARGIN_COEF * press_loss
        + v054.RELEASE_MARGIN_COEF * release_loss
        + v054.NEUTRAL_PUSH_COEF * neutral_loss
    )


def _train_stable_bc(
    model: RecurrentActorCritic,
    sequences: list[StableSequence],
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
) -> None:
    if not sequences:
        raise ValueError("at least one stable DAgger sequence is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    parameters = [
        *model.input_layer.parameters(),
        *model.gru.parameters(),
        *model.post.parameters(),
        *model.actor_mean.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        weighted_elements = 0.0
        order = sequences if epoch & 1 else list(reversed(sequences))
        for stable in order:
            sequence = stable.sequence
            state = model.initial_state(sequence.observations.device)
            for start in range(0, sequence.frames, chunk_steps):
                end = min(sequence.frames, start + chunk_steps)
                state = state.detach()
                predictions: list[torch.Tensor] = []
                for x in sequence.observations[start:end]:
                    mean, _, _, state = model.forward_step(x, state)
                    predictions.append(torch.tanh(mean))

                predicted = torch.stack(predictions)
                target = sequence.teacher_actions[start:end]
                weights = stable.loss_weights[start:end]
                loss = _weighted_actuation_loss(predicted, target, weights)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                weight = float(weights.sum().item())
                loss_sum += float(loss.detach().item()) * max(weight, 1.0)
                weighted_elements += max(weight, 1.0)

        if epoch == 1 or epoch == epochs or epoch % 2 == 0:
            print(
                f"stable bc epoch {epoch:02d}/{epochs} "
                f"loss={loss_sum / max(1.0, weighted_elements):.6f} "
                f"trajectories={len(sequences)}"
            )


def _trim_student_replay(
    replay: list[StableSequence],
    *,
    max_frames: int,
) -> list[StableSequence]:
    """Keep newest whole student trajectories inside one expert-length budget."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    kept: list[StableSequence] = []
    total = 0
    for sequence in reversed(replay):
        if sequence.frames > max_frames:
            # A normal full-segment rollout should fit.  If it does not, skip it
            # rather than slicing recurrent context at an arbitrary boundary.
            continue
        if total + sequence.frames > max_frames:
            continue
        kept.append(sequence)
        total += sequence.frames
    kept.reverse()
    return kept


def _student_replay_frames(replay: list[StableSequence]) -> int:
    return sum(sequence.frames for sequence in replay)


def _collect_mixture_rollout(
    model: RecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    teacher_fraction: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    source: str,
    seed: int,
    press_recovery_cap: int,
) -> StableRollout:
    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must be in [0, 1]")

    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    rng = random.Random(seed)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            encoded = encode_real_chart_observation(observation)
            teacher = v054.base._teacher_action(env, observation, lead_s)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            student, state = model.deterministic_action(x, state)

            observations.append(encoded)
            labels.append((teacher.left, teacher.right))
            applied = teacher if rng.random() < teacher_fraction else student

            step = env.step(applied)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("stable DAgger rollout exceeded real-chart step budget")

    sequence = v055.DAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        source=source,
    )
    stable = _make_stable_sequence(
        sequence,
        press_recovery_cap=press_recovery_cap,
        expert=False,
    )
    return StableRollout(
        sequence=stable,
        evaluation=v054.StudentEvalResult(env.stats, env.physical_keydowns),
        teacher_fraction=teacher_fraction,
    )


def _score_tuple(result: v054.StudentEvalResult) -> tuple:
    stats = result.stats
    return (
        0 if stats.overloaded else 1,
        stats.hits,
        -stats.too_early_presses,
        stats.x_accuracy_percent,
        stats.perfect_rate,
        -stats.misses,
    )


def _guard_decision(
    best: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
    *,
    hit_regression_fraction: float = DEFAULT_HIT_REGRESSION_FRACTION,
    hit_regression_absolute: int = DEFAULT_HIT_REGRESSION_ABSOLUTE,
) -> GuardDecision:
    if not best.stats.overloaded and candidate.stats.overloaded:
        return GuardDecision(False, "safe->overload")

    allowed_drop = max(
        hit_regression_absolute,
        int(round(best.stats.targets * hit_regression_fraction)),
    )
    if candidate.stats.hits < best.stats.hits - allowed_drop:
        return GuardDecision(False, f"hit regression>{allowed_drop}")

    if _score_tuple(candidate) > _score_tuple(best):
        return GuardDecision(True, "better")
    return GuardDecision(False, "not better")


def _print_rollout(label: str, rollout: StableRollout) -> None:
    actions = rollout.sequence.sequence.teacher_actions
    press, release, neutral = v054.base._action_frame_counts(actions)
    print(
        f"{v054._format_eval(label, rollout.evaluation)} "
        f"frames={rollout.sequence.frames} teacher={rollout.teacher_fraction * 100.0:.0f}% "
        f"labels(P/R/N)={press}/{release}/{neutral}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stable real-chart DAgger with teacher/student mixture rollouts, bounded "
            "student replay, recovery-label caps, and best-model rollback."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-start", type=float, default=0.0)
    parser.add_argument("--train-end", type=float, default=30.0)
    parser.add_argument("--sight-start", type=float, default=None)
    parser.add_argument("--sight-end", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--round-epochs", type=int, default=4)
    parser.add_argument("--bootstrap-epochs", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bootstrap-lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--press-recovery-cap", type=int, default=DEFAULT_PRESS_RECOVERY_CAP)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v056_stable_dagger.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cross-hand", action="store_true")
    args = parser.parse_args()

    if args.train_end <= args.train_start:
        raise SystemExit("--train-end must be greater than --train-start")
    if args.rounds <= 0 or args.round_epochs <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("round counts and epoch counts must be positive")
    if args.hidden <= 0 or args.chunk_steps <= 0 or args.press_recovery_cap <= 0:
        raise SystemExit("hidden/chunk-steps/press-recovery-cap must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand
    compiled = load_compiled_adofai(args.chart)
    train_end = min(args.train_end, compiled.duration_s)
    train_segment = build_playable_segment(compiled, start_s=args.train_start, end_s=train_end)
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
    print("=== DMDOD / Real Chart Student v0.5.6 Stable DAgger ===")
    print(
        f"chart={args.chart}\n"
        f"train={args.train_start:g}..{train_end:g}s targets={len(train_segment.targets)} | "
        f"sight={sight_start:g}..{sight_end:g}s targets={len(sight_segment.targets)}"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D visible-only hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs} lr={args.lr:g}"
    )
    print(
        "stable DAgger mixture=50/25/10/0% teacher | "
        f"student-replay<=expert frames | press-recovery-cap={args.press_recovery_cap}"
    )

    expert_x, expert_y, teacher_stats = v054.base._collect_teacher_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    expert_sequence = _make_stable_sequence(
        v055.DAggerSequence(expert_x, expert_y, "expert"),
        press_recovery_cap=args.press_recovery_cap,
        expert=True,
    )
    print(v054.base._format_stats("teacher train", teacher_stats))
    print(f"expert dataset frames={expert_sequence.frames}")

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)

    bootstrapped = False
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        v055._load_checkpoint_model(
            model,
            checkpoint_path,
            hidden_dim=args.hidden,
            expected_format=CHECKPOINT_FORMAT_VERSION,
            device=device,
        )
        print(f"resume={checkpoint_path}")
        bootstrapped = True
    elif not args.no_bootstrap:
        bootstrap_path = Path(args.bootstrap)
        if bootstrap_path.exists():
            v055._load_checkpoint_model(
                model,
                bootstrap_path,
                hidden_dim=args.hidden,
                expected_format=v054.CHECKPOINT_FORMAT_VERSION,
                device=device,
            )
            print(f"bootstrap={bootstrap_path} (v0.5.4 safe baseline)")
            bootstrapped = True

    if not bootstrapped:
        print(f"bootstrap=fresh expert BC epochs={args.bootstrap_epochs}")
        v054._train_bc(
            model,
            expert_x,
            expert_y,
            epochs=args.bootstrap_epochs,
            learning_rate=args.bootstrap_lr,
            chunk_steps=args.chunk_steps,
        )

    best_eval = v054._evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    best_state = copy.deepcopy(model.state_dict())
    print(v054._format_eval("baseline practiced", best_eval))

    replay: list[StableSequence] = []
    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        beta = _mixture_beta(round_index)
        rollout = _collect_mixture_rollout(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            teacher_fraction=beta,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"mixture-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        _print_rollout(f"stable {round_index:02d} rollout", rollout)

        replay.append(rollout.sequence)
        replay = _trim_student_replay(replay, max_frames=expert_sequence.frames)
        replay_frames = _student_replay_frames(replay)
        expert_share = expert_sequence.frames / max(1, expert_sequence.frames + replay_frames)
        print(
            f"replay expert={expert_sequence.frames} student={replay_frames} "
            f"expert-share={expert_share * 100.0:.1f}% trajectories={1 + len(replay)}"
        )

        pre_round_state = copy.deepcopy(model.state_dict())
        _train_stable_bc(
            model,
            [expert_sequence, *replay],
            epochs=args.round_epochs,
            learning_rate=args.lr,
            chunk_steps=args.chunk_steps,
        )
        candidate = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        decision = _guard_decision(best_eval, candidate)
        print(
            f"{v054._format_eval(f'stable {round_index:02d} candidate', candidate)} "
            f"guard={'ACCEPT' if decision.accepted else 'ROLLBACK'}({decision.reason})"
        )

        if decision.accepted:
            best_eval = candidate
            best_state = copy.deepcopy(model.state_dict())
        else:
            # Roll back to the global best, not merely the immediately previous
            # round.  This prevents DAgger ping-pong from ratcheting downward.
            model.load_state_dict(best_state)

        round_history.append(
            {
                "round": round_index,
                "teacher_fraction": beta,
                "rollout_frames": rollout.sequence.frames,
                "replay_frames": replay_frames,
                "expert_share": expert_share,
                "candidate_hits": candidate.stats.hits,
                "candidate_misses": candidate.stats.misses,
                "candidate_early": candidate.stats.too_early_presses,
                "candidate_overloaded": candidate.stats.overloaded,
                "accepted": decision.accepted,
                "guard_reason": decision.reason,
            }
        )
        del pre_round_state

    model.load_state_dict(best_state)
    startup = v054.base._startup_probe(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    practiced = v054._evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    sight = v054._evaluate_student(
        model,
        sight_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    print(
        f"startup probe: {'PASS' if startup.safe else 'FAIL'} "
        f"early={startup.early_presses} overload={startup.overloaded} "
        f"max_push={startup.max_positive_action:+.3f} until={startup.until_s:.3f}s"
    )
    print(v054._format_eval("best student practiced", practiced))
    print(v054._format_eval("best student sight-read", sight))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "input_dim": REAL_CHART_INPUT_DIM,
            "hidden_dim": args.hidden,
            "model_state": best_state,
            "chart": str(args.chart),
            "train_start": args.train_start,
            "train_end": train_end,
            "feature_config": {
                "behind_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
                "ahead_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
            },
            "stable_dagger": {
                "mixture_betas": list(DEFAULT_MIXTURE_BETAS),
                "student_replay_budget_frames": expert_sequence.frames,
                "press_recovery_cap": args.press_recovery_cap,
                "round_history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
