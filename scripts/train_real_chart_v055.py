from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.5-dagger-closed-loop-imitation"
CHECKPOINT_FORMAT_VERSION = 4
DEFAULT_BOOTSTRAP = "checkpoints/real_chart_v054_actuation.pt"


@dataclass(frozen=True, slots=True)
class DAggerSequence:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        if self.observations.ndim != 2 or self.observations.shape[1] != REAL_CHART_INPUT_DIM:
            raise ValueError("DAgger observations have the wrong shape")
        if self.teacher_actions.shape != (self.observations.shape[0], 2):
            raise ValueError("DAgger teacher actions must have shape [T, 2]")

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class DAggerRollout:
    sequence: DAggerSequence
    evaluation: v054.StudentEvalResult


def _collect_dagger_sequence(
    model: RecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    source: str,
) -> DAggerRollout:
    """Roll out the student, but label every visited state with the teacher.

    The policy action moves the physical body and therefore determines the next
    observation. The privileged teacher only supplies the training label for the
    already-visited state; its target timing is never added to student features.
    """

    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
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

            step = env.step(student)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("DAgger rollout exceeded real-chart step budget")

    sequence = DAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        source=source,
    )
    return DAggerRollout(
        sequence=sequence,
        evaluation=v054.StudentEvalResult(env.stats, env.physical_keydowns),
    )


def _train_aggregate_bc(
    model: RecurrentActorCritic,
    sequences: list[DAggerSequence],
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
) -> None:
    """Train on all expert-labelled trajectories with GRU resets at boundaries."""

    if not sequences:
        raise ValueError("at least one DAgger sequence is required")
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
        frame_count = 0

        # Sequence order stays deterministic for reproducible experiments, but
        # hidden state is reset for every trajectory. Concatenating trajectories
        # into one recurrent stream would leak one episode into the next.
        for sequence in sequences:
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
                loss = v054._actuation_loss(predicted, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                frames = end - start
                loss_sum += float(loss.detach().item()) * frames
                frame_count += frames

        if epoch == 1 or epoch == epochs or epoch % 2 == 0:
            print(
                f"dagger bc epoch {epoch:02d}/{epochs} "
                f"loss={loss_sum / max(1, frame_count):.6f} frames={frame_count}"
            )


def _dataset_frames(sequences: list[DAggerSequence]) -> int:
    return sum(sequence.frames for sequence in sequences)


def _load_checkpoint_model(
    model: RecurrentActorCritic,
    path: Path,
    *,
    hidden_dim: int,
    expected_format: int | None,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location=device)
    if expected_format is not None and int(payload.get("format_version", -1)) != expected_format:
        raise SystemExit(f"unsupported checkpoint format in {path}")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit(f"checkpoint input dimension does not match current encoder: {path}")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit(f"checkpoint hidden size does not match --hidden: {path}")
    model.load_state_dict(payload["model_state"])
    return payload


def _print_rollout(label: str, rollout: DAggerRollout) -> None:
    press, release, neutral = v054.base._action_frame_counts(rollout.sequence.teacher_actions)
    print(
        f"{v054._format_eval(label, rollout.evaluation)} "
        f"frames={rollout.sequence.frames} labels(P/R/N)={press}/{release}/{neutral}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real-chart student with closed-loop DAgger: the student "
            "drives the body, while the privileged teacher labels student-visited states."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-start", type=float, default=0.0)
    parser.add_argument("--train-end", type=float, default=30.0)
    parser.add_argument("--sight-start", type=float, default=None)
    parser.add_argument("--sight-end", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--round-epochs", type=int, default=5)
    parser.add_argument("--bootstrap-epochs", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bootstrap-lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v055_dagger.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cross-hand", action="store_true")
    args = parser.parse_args()

    if args.train_end <= args.train_start:
        raise SystemExit("--train-end must be greater than --train-start")
    if args.rounds <= 0 or args.round_epochs <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("round counts and epoch counts must be positive")
    if args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("hidden/chunk-steps must be positive")

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
    print("=== DMDOD / Real Chart Student v0.5.5 DAgger ===")
    print(
        f"chart={args.chart}\n"
        f"train={args.train_start:g}..{train_end:g}s targets={len(train_segment.targets)} | "
        f"sight={sight_start:g}..{sight_end:g}s targets={len(sight_segment.targets)}"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D visible-only hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs}"
    )

    expert_x, expert_y, teacher_stats = v054.base._collect_teacher_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    sequences = [DAggerSequence(expert_x, expert_y, "expert")]
    print(v054.base._format_stats("teacher train", teacher_stats))
    print(f"expert dataset frames={expert_x.shape[0]}")

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
        _load_checkpoint_model(
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
            _load_checkpoint_model(
                model,
                bootstrap_path,
                hidden_dim=args.hidden,
                expected_format=v054.CHECKPOINT_FORMAT_VERSION,
                device=device,
            )
            print(f"bootstrap={bootstrap_path} (v0.5.4 model)")
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

    initial = v054._evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    print(v054._format_eval("before DAgger practiced", initial))

    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        rollout = _collect_dagger_sequence(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"student-round-{round_index}",
        )
        sequences.append(rollout.sequence)
        _print_rollout(f"dagger {round_index:02d} rollout", rollout)
        print(
            f"dataset trajectories={len(sequences)} frames={_dataset_frames(sequences)}"
        )

        _train_aggregate_bc(
            model,
            sequences,
            epochs=args.round_epochs,
            learning_rate=args.lr,
            chunk_steps=args.chunk_steps,
        )

        practiced = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        print(v054._format_eval(f"dagger {round_index:02d} practiced", practiced))
        round_history.append(
            {
                "round": round_index,
                "rollout_frames": rollout.sequence.frames,
                "dataset_frames": _dataset_frames(sequences),
                "rollout_hits": rollout.evaluation.stats.hits,
                "rollout_misses": rollout.evaluation.stats.misses,
                "rollout_early": rollout.evaluation.stats.too_early_presses,
                "rollout_keydowns": rollout.evaluation.physical_keydowns,
                "post_hits": practiced.stats.hits,
                "post_misses": practiced.stats.misses,
                "post_early": practiced.stats.too_early_presses,
                "post_keydowns": practiced.physical_keydowns,
            }
        )

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
    print(v054._format_eval("student practiced", practiced))
    print(v054._format_eval("student sight-read", sight))

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
            "train_end": train_end,
            "dagger": {
                "rounds": args.rounds,
                "round_epochs": args.round_epochs,
                "learning_rate": args.lr,
                "aggregate_trajectories": len(sequences),
                "aggregate_frames": _dataset_frames(sequences),
                "student_drives_collection": True,
                "teacher_labels_only": True,
                "history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
