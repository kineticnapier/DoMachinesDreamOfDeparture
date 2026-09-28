from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v055 as v055
import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.5.8-trust-region-dagger"
CHECKPOINT_FORMAT_VERSION = 7
DEFAULT_BOOTSTRAP = "checkpoints/real_chart_v054_actuation.pt"
DEFAULT_TRUST_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


@dataclass(frozen=True, slots=True)
class TrustCandidate:
    alpha: float
    evaluation: v054.StudentEvalResult
    decision: v057.ConservativeDecision


@dataclass(frozen=True, slots=True)
class TrustChoice:
    accepted: bool
    alpha: float | None
    evaluation: v054.StudentEvalResult | None
    reason: str


def _interpolate_state(
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate a proposal back toward the current trusted policy.

    ``alpha=1`` is the full BC proposal and ``alpha=0`` is exactly the trusted
    base.  Floating tensors are linearly interpolated.  Non-floating state is
    kept from the trusted base so counters/buffers can never be blended.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if base_state.keys() != proposal_state.keys():
        raise ValueError("base and proposal state dictionaries must have matching keys")

    result: dict[str, torch.Tensor] = {}
    for key, base in base_state.items():
        proposal = proposal_state[key]
        if base.shape != proposal.shape or base.dtype != proposal.dtype:
            raise ValueError(f"state mismatch for {key}")
        if torch.is_floating_point(base) or torch.is_complex(base):
            result[key] = torch.lerp(base, proposal, alpha)
        else:
            result[key] = base.clone()
    return result


def _choose_trust_candidate(
    best: v054.StudentEvalResult,
    candidates: list[TrustCandidate],
) -> TrustChoice:
    """Choose the best guard-passing interpolation candidate.

    Guard rejection remains authoritative.  Among survivors, use v0.5.7's
    accuracy-first ordering rather than preferring the largest alpha.
    """

    accepted = [candidate for candidate in candidates if candidate.decision.accepted]
    if not accepted:
        reasons = ",".join(candidate.decision.reason for candidate in candidates)
        return TrustChoice(False, None, None, reasons or "no candidates")

    chosen = max(accepted, key=lambda item: v057._accuracy_key(item.evaluation))
    # The conservative guard only marks a candidate accepted when it outranks
    # best, but retain the explicit check so this helper is safe in isolation.
    if v057._accuracy_key(chosen.evaluation) <= v057._accuracy_key(best):
        return TrustChoice(False, None, None, "no accuracy-first improvement")
    return TrustChoice(True, chosen.alpha, chosen.evaluation, chosen.decision.reason)


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
        v057.CHECKPOINT_FORMAT_VERSION: "v0.5.7",
        CHECKPOINT_FORMAT_VERSION: "v0.5.8",
    }
    if format_version not in supported:
        raise SystemExit(f"unsupported bootstrap checkpoint format in {path}: {format_version}")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit(f"checkpoint input dimension does not match current encoder: {path}")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit(f"checkpoint hidden size does not match --hidden: {path}")
    model.load_state_dict(payload["model_state"])
    return supported[format_version]


def _evaluate_line_search(
    model: RecurrentActorCritic,
    *,
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    best_eval: v054.StudentEvalResult,
    alphas: tuple[float, ...],
    train_segment,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    label_prefix: str,
) -> tuple[TrustChoice, list[TrustCandidate], dict[str, torch.Tensor] | None]:
    candidates: list[TrustCandidate] = []
    candidate_states: dict[float, dict[str, torch.Tensor]] = {}

    for alpha in alphas:
        state = _interpolate_state(base_state, proposal_state, alpha)
        model.load_state_dict(state)
        evaluation = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        decision = v057._conservative_guard(best_eval, evaluation)
        candidates.append(TrustCandidate(alpha, evaluation, decision))
        candidate_states[alpha] = state
        print(
            f"{v054._format_eval(f'{label_prefix} a={alpha:g}', evaluation)} "
            f"guard={'PASS' if decision.accepted else 'reject'}({decision.reason})"
        )

    choice = _choose_trust_candidate(best_eval, candidates)
    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None

    chosen_state = candidate_states[choice.alpha]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trust-region real-chart DAgger: make one BC proposal, then line-search "
            "between the trusted policy and that proposal using real gameplay evaluation."
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
    parser.add_argument("--checkpoint", default="checkpoints/real_chart_v058_trust_region.pt")
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
    print("=== DMDOD / Real Chart Student v0.5.8 Trust-Region DAgger ===")
    print(
        f"chart={args.chart}\n"
        f"train={args.train_start:g}..{train_end:g}s targets={len(train_segment.targets)} | "
        f"sight={sight_start:g}..{sight_end:g}s targets={len(sight_segment.targets)}"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D visible-only hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs} proposal-lr={args.lr:g}"
    )
    print(
        "trust alphas=" + "/".join(f"{alpha:g}" for alpha in DEFAULT_TRUST_ALPHAS)
        + " | gameplay guard at every alpha | accuracy-first selection"
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
            source=f"trust-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        v056._print_rollout(f"trust {round_index:02d} rollout", rollout)

        replay.append(rollout.sequence)
        replay = v056._trim_student_replay(replay, max_frames=expert_sequence.frames)
        replay_frames = v056._student_replay_frames(replay)
        expert_share = expert_sequence.frames / max(1, expert_sequence.frames + replay_frames)
        sequences = [expert_sequence, *replay]
        print(
            f"replay expert={expert_sequence.frames} student={replay_frames} "
            f"expert-share={expert_share * 100.0:.1f}% trajectories={len(sequences)}"
        )

        epoch_history: list[dict] = []
        accepted_epochs = 0
        for epoch_index in range(1, args.round_epochs + 1):
            # Every proposal starts from the currently trusted policy.  Adam is
            # deliberately fresh: optimizer moments from an untrusted proposal
            # must not influence the next line search.
            model.load_state_dict(best_state)
            base_state = copy.deepcopy(best_state)
            optimizer = v057._new_optimizer(model, args.lr)
            loss = v057._train_one_epoch(
                model,
                sequences,
                optimizer=optimizer,
                chunk_steps=args.chunk_steps,
                reverse_order=not bool(epoch_index & 1),
            )
            proposal_state = copy.deepcopy(model.state_dict())

            choice, candidates, chosen_state = _evaluate_line_search(
                model,
                base_state=base_state,
                proposal_state=proposal_state,
                best_eval=best_eval,
                alphas=DEFAULT_TRUST_ALPHAS,
                train_segment=train_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                label_prefix=f"trust {round_index:02d} epoch {epoch_index:02d}",
            )

            if choice.accepted and choice.evaluation is not None and chosen_state is not None:
                accepted_epochs += 1
                best_eval = choice.evaluation
                best_state = copy.deepcopy(chosen_state)
                print(
                    f"trust {round_index:02d} epoch {epoch_index:02d}: "
                    f"ACCEPT alpha={choice.alpha:g} loss={loss:.6f} "
                    f"best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
                    f"X{best_eval.stats.x_accuracy_percent:.2f}% "
                    f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
                    f"early={best_eval.stats.too_early_presses}"
                )
            else:
                model.load_state_dict(best_state)
                print(
                    f"trust {round_index:02d} epoch {epoch_index:02d}: "
                    f"ROLLBACK all alphas loss={loss:.6f}"
                )

            epoch_history.append(
                {
                    "epoch": epoch_index,
                    "proposal_loss": loss,
                    "accepted": choice.accepted,
                    "alpha": choice.alpha,
                    "candidates": [
                        {
                            "alpha": candidate.alpha,
                            "hits": candidate.evaluation.stats.hits,
                            "misses": candidate.evaluation.stats.misses,
                            "xacc": candidate.evaluation.stats.x_accuracy_percent,
                            "pp": candidate.evaluation.stats.perfect_rate,
                            "early": candidate.evaluation.stats.too_early_presses,
                            "overloaded": candidate.evaluation.stats.overloaded,
                            "guard": candidate.decision.reason,
                            "guard_pass": candidate.decision.accepted,
                        }
                        for candidate in candidates
                    ],
                }
            )

        print(
            f"trust {round_index:02d} summary: accepted-epochs={accepted_epochs}/"
            f"{args.round_epochs} best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
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
            "trust_region_dagger": {
                "trust_alphas": list(DEFAULT_TRUST_ALPHAS),
                "mixture_betas": list(v056.DEFAULT_MIXTURE_BETAS),
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
