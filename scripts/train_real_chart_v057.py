from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v055 as v055
import train_real_chart_v056 as v056
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.7-conservative-dagger"
CHECKPOINT_FORMAT_VERSION = 6
DEFAULT_BOOTSTRAP = "checkpoints/real_chart_v054_actuation.pt"
DEFAULT_HIT_REGRESSION_FRACTION = 0.03
DEFAULT_HIT_REGRESSION_ABSOLUTE = 3
DEFAULT_XACC_REGRESSION_POINTS = 2.0
DEFAULT_EARLY_REGRESSION = 4


@dataclass(frozen=True, slots=True)
class ConservativeDecision:
    accepted: bool
    reason: str


def _accuracy_key(result: v054.StudentEvalResult) -> tuple[float, float, int, int, int]:
    """Accuracy-first ordering after hard completion/safety guards.

    X-Accuracy and PP lead the ordering; hit count then breaks ties, followed by
    TooEarly and miss counts.  Hard guards separately prevent this ordering from
    trading away too much completion or safety for a prettier accuracy number.
    """

    stats = result.stats
    return (
        float(stats.x_accuracy_percent),
        float(stats.perfect_rate),
        int(stats.hits),
        -int(stats.too_early_presses),
        -int(stats.misses),
    )


def _conservative_guard(
    best: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
    *,
    hit_regression_fraction: float = DEFAULT_HIT_REGRESSION_FRACTION,
    hit_regression_absolute: int = DEFAULT_HIT_REGRESSION_ABSOLUTE,
    xacc_regression_points: float = DEFAULT_XACC_REGRESSION_POINTS,
    early_regression: int = DEFAULT_EARLY_REGRESSION,
) -> ConservativeDecision:
    """Reject destructive BC steps before considering accuracy-first ranking."""

    if not best.stats.overloaded and candidate.stats.overloaded:
        return ConservativeDecision(False, "safe->overload")

    allowed_hit_drop = max(
        int(hit_regression_absolute),
        int(round(best.stats.targets * hit_regression_fraction)),
    )
    if candidate.stats.hits < best.stats.hits - allowed_hit_drop:
        return ConservativeDecision(False, f"hit regression>{allowed_hit_drop}")

    if candidate.stats.x_accuracy_percent < best.stats.x_accuracy_percent - xacc_regression_points:
        return ConservativeDecision(False, f"XAcc regression>{xacc_regression_points:g}pt")

    if candidate.stats.too_early_presses > best.stats.too_early_presses + early_regression:
        return ConservativeDecision(False, f"early regression>{early_regression}")

    if _accuracy_key(candidate) > _accuracy_key(best):
        return ConservativeDecision(True, "accuracy-first better")
    return ConservativeDecision(False, "not better")


def _policy_parameters(model: RecurrentActorCritic) -> list[torch.nn.Parameter]:
    return [
        *model.input_layer.parameters(),
        *model.gru.parameters(),
        *model.post.parameters(),
        *model.actor_mean.parameters(),
    ]


def _train_one_epoch(
    model: RecurrentActorCritic,
    sequences: list[v056.StableSequence],
    *,
    optimizer: torch.optim.Optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Run exactly one recurrent BC epoch and return its weighted loss.

    Every trajectory starts from a fresh GRU state.  The caller owns the
    optimizer so accepted consecutive epochs can retain Adam moments; after a
    rejected epoch the caller discards the optimizer together with that model
    step to avoid stale moments from a rolled-back policy.
    """

    if not sequences:
        raise ValueError("at least one stable DAgger sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    parameters = _policy_parameters(model)
    ordered = list(reversed(sequences)) if reverse_order else sequences
    loss_sum = 0.0
    weighted_elements = 0.0

    for stable in ordered:
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
            loss = v056._weighted_actuation_loss(predicted, target, weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            weight = max(float(weights.sum().item()), 1.0)
            loss_sum += float(loss.detach().item()) * weight
            weighted_elements += weight

    return loss_sum / max(1.0, weighted_elements)


def _new_optimizer(model: RecurrentActorCritic, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(_policy_parameters(model), lr=learning_rate)


def _load_bootstrap(
    model: RecurrentActorCritic,
    path: Path,
    *,
    hidden_dim: int,
    device: torch.device,
) -> str:
    payload = torch.load(path, map_location=device)
    format_version = int(payload.get("format_version", -1))
    supported = {
        v054.CHECKPOINT_FORMAT_VERSION: "v0.5.4",
        v056.CHECKPOINT_FORMAT_VERSION: "v0.5.6",
        CHECKPOINT_FORMAT_VERSION: "v0.5.7",
    }
    if format_version not in supported:
        raise SystemExit(f"unsupported bootstrap checkpoint format in {path}: {format_version}")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit(f"checkpoint input dimension does not match current encoder: {path}")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit(f"checkpoint hidden size does not match --hidden: {path}")
    model.load_state_dict(payload["model_state"])
    return supported[format_version]


def _format_candidate(
    label: str,
    result: v054.StudentEvalResult,
    decision: ConservativeDecision,
    *,
    loss: float,
) -> str:
    return (
        f"{v054._format_eval(label, result)} loss={loss:.6f} "
        f"guard={'ACCEPT' if decision.accepted else 'ROLLBACK'}({decision.reason})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative real-chart DAgger: evaluate after every BC epoch, "
            "roll back destructive updates immediately, and rank survivors by accuracy."
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
    parser.add_argument("--press-recovery-cap", type=int, default=v056.DEFAULT_PRESS_RECOVERY_CAP)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v057_conservative_dagger.pt")
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
    print("=== DMDOD / Real Chart Student v0.5.7 Conservative DAgger ===")
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
        "conservative guard: eval every epoch | "
        f"hit-drop<={max(DEFAULT_HIT_REGRESSION_ABSOLUTE, int(round(len(train_segment.targets) * DEFAULT_HIT_REGRESSION_FRACTION)))} "
        f"XAcc-drop<={DEFAULT_XACC_REGRESSION_POINTS:g}pt "
        f"early-rise<={DEFAULT_EARLY_REGRESSION} | accuracy-first ranking"
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
    expert_sequence = v056._make_stable_sequence(
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
        source_version = _load_bootstrap(
            model,
            checkpoint_path,
            hidden_dim=args.hidden,
            device=device,
        )
        print(f"resume={checkpoint_path} ({source_version})")
        bootstrapped = True
    elif not args.no_bootstrap:
        bootstrap_path = Path(args.bootstrap)
        if bootstrap_path.exists():
            source_version = _load_bootstrap(
                model,
                bootstrap_path,
                hidden_dim=args.hidden,
                device=device,
            )
            print(f"bootstrap={bootstrap_path} ({source_version})")
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

    replay: list[v056.StableSequence] = []
    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        model.load_state_dict(best_state)
        beta = v056._mixture_beta(round_index)
        rollout = v056._collect_mixture_rollout(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            teacher_fraction=beta,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"conservative-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        v056._print_rollout(f"conservative {round_index:02d} rollout", rollout)

        replay.append(rollout.sequence)
        replay = v056._trim_student_replay(replay, max_frames=expert_sequence.frames)
        replay_frames = v056._student_replay_frames(replay)
        expert_share = expert_sequence.frames / max(1, expert_sequence.frames + replay_frames)
        sequences = [expert_sequence, *replay]
        print(
            f"replay expert={expert_sequence.frames} student={replay_frames} "
            f"expert-share={expert_share * 100.0:.1f}% trajectories={len(sequences)}"
        )

        optimizer = _new_optimizer(model, args.lr)
        epoch_history: list[dict] = []
        accepted_epochs = 0
        for epoch_index in range(1, args.round_epochs + 1):
            loss = _train_one_epoch(
                model,
                sequences,
                optimizer=optimizer,
                chunk_steps=args.chunk_steps,
                reverse_order=not bool(epoch_index & 1),
            )
            candidate = v054._evaluate_student(
                model,
                train_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
            decision = _conservative_guard(best_eval, candidate)
            print(
                _format_candidate(
                    f"conservative {round_index:02d} epoch {epoch_index:02d}",
                    candidate,
                    decision,
                    loss=loss,
                )
            )

            epoch_history.append(
                {
                    "epoch": epoch_index,
                    "loss": loss,
                    "hits": candidate.stats.hits,
                    "misses": candidate.stats.misses,
                    "xacc": candidate.stats.x_accuracy_percent,
                    "pp": candidate.stats.perfect_rate,
                    "early": candidate.stats.too_early_presses,
                    "overloaded": candidate.stats.overloaded,
                    "accepted": decision.accepted,
                    "guard_reason": decision.reason,
                }
            )

            if decision.accepted:
                accepted_epochs += 1
                best_eval = candidate
                best_state = copy.deepcopy(model.state_dict())
            else:
                # Immediately return to the best known policy.  Adam moments
                # from a rejected policy are discarded as well; carrying them
                # across the rollback would reintroduce the rejected direction.
                model.load_state_dict(best_state)
                optimizer = _new_optimizer(model, args.lr)

        print(
            f"conservative {round_index:02d} summary: accepted-epochs="
            f"{accepted_epochs}/{args.round_epochs} best="
            f"H{best_eval.stats.hits}/{best_eval.stats.targets} "
            f"X{best_eval.stats.x_accuracy_percent:.2f}% "
            f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
            f"early={best_eval.stats.too_early_presses}"
        )
        round_history.append(
            {
                "round": round_index,
                "teacher_fraction": beta,
                "rollout_frames": rollout.sequence.frames,
                "replay_frames": replay_frames,
                "expert_share": expert_share,
                "accepted_epochs": accepted_epochs,
                "epochs": epoch_history,
            }
        )

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
            "conservative_dagger": {
                "mixture_betas": list(v056.DEFAULT_MIXTURE_BETAS),
                "student_replay_budget_frames": expert_sequence.frames,
                "press_recovery_cap": args.press_recovery_cap,
                "hit_regression_fraction": DEFAULT_HIT_REGRESSION_FRACTION,
                "hit_regression_absolute": DEFAULT_HIT_REGRESSION_ABSOLUTE,
                "xacc_regression_points": DEFAULT_XACC_REGRESSION_POINTS,
                "early_regression": DEFAULT_EARLY_REGRESSION,
                "round_history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
