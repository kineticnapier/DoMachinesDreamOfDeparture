from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
import train_real_chart_v058 as v058
import train_real_chart_v060 as v060
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.6.1-safety-selected-bootstrap"
CHECKPOINT_FORMAT_VERSION = 9
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v061_safety_selected.pt"


def _safe_escape_key(result: v054.StudentEvalResult) -> tuple[int, float, float, int, int]:
    """Rank safe recovery candidates without rewarding tiny high-XAcc runs.

    Escaping overload is a viability problem first: among safe candidates, keep
    completion primary, then accuracy. Once the trusted policy is safe, normal
    v0.5.7 accuracy-first ranking takes over.
    """

    stats = result.stats
    return (
        int(stats.hits),
        float(stats.x_accuracy_percent),
        float(stats.perfect_rate),
        -int(stats.too_early_presses),
        -int(stats.misses),
    )


def _safety_guard(
    best: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> v057.ConservativeDecision:
    """Safety-state-aware guard used by v0.6.1.

    - overloaded -> safe is always eligible as a safety escape;
    - safe -> overloaded is never eligible;
    - overloaded -> overloaded is never compared by XAcc;
    - safe -> safe uses the existing conservative accuracy-first guard.
    """

    best_failed = bool(best.stats.overloaded)
    candidate_failed = bool(candidate.stats.overloaded)
    if best_failed and not candidate_failed:
        return v057.ConservativeDecision(True, "escaped overload")
    if not best_failed and candidate_failed:
        return v057.ConservativeDecision(False, "safe->overload")
    if best_failed and candidate_failed:
        return v057.ConservativeDecision(False, "both overloaded")
    return v057._conservative_guard(best, candidate)


def _candidate_key(
    best: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> tuple:
    if best.stats.overloaded:
        return _safe_escape_key(candidate)
    return v057._accuracy_key(candidate)


def _choose_line_search_candidate(
    best: v054.StudentEvalResult,
    candidates: list[v058.TrustCandidate],
) -> v058.TrustChoice:
    passing = [candidate for candidate in candidates if candidate.decision.accepted]
    if not passing:
        reasons = ",".join(candidate.decision.reason for candidate in candidates)
        return v058.TrustChoice(False, None, None, reasons or "no candidates")

    chosen = max(passing, key=lambda item: _candidate_key(best, item.evaluation))
    return v058.TrustChoice(
        True,
        chosen.alpha,
        chosen.evaluation,
        chosen.decision.reason,
    )


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
) -> tuple[v058.TrustChoice, list[v058.TrustCandidate], dict[str, torch.Tensor] | None]:
    candidates: list[v058.TrustCandidate] = []
    candidate_states: dict[float, dict[str, torch.Tensor]] = {}

    for alpha in alphas:
        state = v058._interpolate_state(base_state, proposal_state, alpha)
        model.load_state_dict(state)
        evaluation = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        decision = _safety_guard(best_eval, evaluation)
        candidates.append(v058.TrustCandidate(alpha, evaluation, decision))
        candidate_states[alpha] = state
        print(
            f"{v054._format_eval(f'{label_prefix} a={alpha:g}', evaluation)} "
            f"guard={'PASS' if decision.accepted else 'reject'}({decision.reason})"
        )

    choice = _choose_line_search_candidate(best_eval, candidates)
    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None

    chosen_state = candidate_states[choice.alpha]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def _bootstrap_epoch(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    chunk_steps: int,
) -> float:
    """Run one v0.5.4 actuation-aware BC epoch while retaining Adam state."""

    state = model.initial_state(observations.device)
    parameters = v057._policy_parameters(model)
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
        loss = v054._actuation_loss(predicted, actions[start:end])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        frames = end - start
        loss_sum += float(loss.detach().item()) * frames
        frame_count += frames

    return loss_sum / max(1, frame_count)


def _bootstrap_fallback_key(result: v054.StudentEvalResult) -> tuple[int, float, int, int]:
    stats = result.stats
    return (
        int(stats.hits),
        float(stats.x_accuracy_percent),
        -int(stats.too_early_presses),
        -int(stats.misses),
    )


def _train_safety_selected_bootstrap(
    model: RecurrentActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    segment,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> tuple[v054.StudentEvalResult, dict[str, torch.Tensor], list[dict]]:
    """Train continuously, but retain only the best non-overloaded epoch.

    Epoch 0 (the random initial policy) is evaluated too. This guarantees that
    a later overloaded BC epoch cannot silently become the trusted baseline if
    an earlier safe policy existed.
    """

    initial = v054._evaluate_student(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
    )
    initial_state = copy.deepcopy(model.state_dict())

    best_safe_eval: v054.StudentEvalResult | None = None
    best_safe_state: dict[str, torch.Tensor] | None = None
    best_safe_epoch: int | None = None
    if not initial.stats.overloaded:
        best_safe_eval = initial
        best_safe_state = copy.deepcopy(initial_state)
        best_safe_epoch = 0

    fallback_eval = initial
    fallback_state = copy.deepcopy(initial_state)
    fallback_epoch = 0
    print(
        f"bootstrap 00: {v054._format_eval('eval', initial)} "
        f"{'SAFE' if not initial.stats.overloaded else 'OVERLOAD'}"
    )

    optimizer = torch.optim.Adam(v057._policy_parameters(model), lr=learning_rate)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        loss = _bootstrap_epoch(
            model,
            observations,
            actions,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
        )
        evaluation = v054._evaluate_student(
            model,
            segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        state = copy.deepcopy(model.state_dict())

        kept = False
        if not evaluation.stats.overloaded and (
            best_safe_eval is None
            or _safe_escape_key(evaluation) > _safe_escape_key(best_safe_eval)
        ):
            best_safe_eval = evaluation
            best_safe_state = copy.deepcopy(state)
            best_safe_epoch = epoch
            kept = True

        if _bootstrap_fallback_key(evaluation) > _bootstrap_fallback_key(fallback_eval):
            fallback_eval = evaluation
            fallback_state = copy.deepcopy(state)
            fallback_epoch = epoch

        print(
            f"bootstrap {epoch:02d}: loss={loss:.6f} "
            f"H={evaluation.stats.hits}/{evaluation.stats.targets} "
            f"X={evaluation.stats.x_accuracy_percent:.2f}% "
            f"early={evaluation.stats.too_early_presses} "
            f"overload={evaluation.stats.overloaded}"
            + (" KEEP" if kept else "")
        )
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "hits": evaluation.stats.hits,
                "misses": evaluation.stats.misses,
                "xacc": evaluation.stats.x_accuracy_percent,
                "pp": evaluation.stats.perfect_rate,
                "early": evaluation.stats.too_early_presses,
                "overloaded": evaluation.stats.overloaded,
                "kept_safe": kept,
            }
        )

    if best_safe_eval is not None and best_safe_state is not None and best_safe_epoch is not None:
        model.load_state_dict(best_safe_state)
        print(
            f"bootstrap selected: epoch={best_safe_epoch} SAFE "
            f"H={best_safe_eval.stats.hits}/{best_safe_eval.stats.targets} "
            f"X={best_safe_eval.stats.x_accuracy_percent:.2f}% "
            f"early={best_safe_eval.stats.too_early_presses}"
        )
        return best_safe_eval, best_safe_state, history

    model.load_state_dict(fallback_state)
    print(
        f"bootstrap selected: epoch={fallback_epoch} UNSAFE-FALLBACK "
        f"H={fallback_eval.stats.hits}/{fallback_eval.stats.targets} "
        f"X={fallback_eval.stats.x_accuracy_percent:.2f}% "
        f"early={fallback_eval.stats.too_early_presses}"
    )
    return fallback_eval, fallback_state, history


def _load_resume(
    model: RecurrentActorCritic,
    path: Path,
    *,
    hidden_dim: int,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location=device)
    format_version = int(payload.get("format_version", -1))
    if format_version not in (v060.CHECKPOINT_FORMAT_VERSION, CHECKPOINT_FORMAT_VERSION):
        raise SystemExit("v0.6.1 resumes only from v0.6.0/v0.6.1 finger-agnostic checkpoints")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit("checkpoint input dimension does not match current encoder")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit("checkpoint hidden size does not match --hidden")
    model.load_state_dict(payload["model_state"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finger-agnostic real-chart DAgger with gameplay-selected bootstrap "
            "and safety-state-aware trust-region guards."
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
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
    sight_end = sight_start + (args.train_end - args.train_start) if args.sight_end is None else args.sight_end
    sight_end = min(sight_end, compiled.duration_s)
    sight_segment = build_playable_segment(compiled, start_s=sight_start, end_s=sight_end)
    if not sight_segment.targets:
        raise SystemExit("sight-read segment contains no playable targets")

    calibration = calibrate_single_press_lead(control_dt_s=args.control_dt, same_hand=same_hand)
    print("=== DMDOD / Real Chart Student v0.6.1 Safety-Selected ===")
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
    print("teacher=finger-agnostic | bootstrap=gameplay-eval every epoch, best SAFE retained")
    print("guard: overload->safe priority | safe->overload reject | overload->overload reject")

    expert_x, expert_y, teacher_eval = v060._collect_expert_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    expert_sequence = v056._make_stable_sequence(
        v060.v055.DAggerSequence(expert_x, expert_y, "finger-agnostic-expert"),
        press_recovery_cap=args.press_recovery_cap,
        expert=True,
    )
    print(v054._format_eval("teacher train", teacher_eval))
    press, release, neutral = v054.base._action_frame_counts(expert_y)
    print(f"expert dataset frames={expert_sequence.frames} labels(P/R/N)={press}/{release}/{neutral}")

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    bootstrap_history: list[dict] = []

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        payload = _load_resume(model, checkpoint_path, hidden_dim=args.hidden, device=device)
        print(f"resume={checkpoint_path} format={payload.get('format_version')}")
        best_eval = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        best_state = copy.deepcopy(model.state_dict())
    else:
        print(f"bootstrap=fresh finger-agnostic BC epochs={args.bootstrap_epochs}")
        best_eval, best_state, bootstrap_history = _train_safety_selected_bootstrap(
            model,
            expert_x,
            expert_y,
            segment=train_segment,
            epochs=args.bootstrap_epochs,
            learning_rate=args.bootstrap_lr,
            chunk_steps=args.chunk_steps,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )

    model.load_state_dict(best_state)
    print(v054._format_eval("baseline practiced", best_eval))

    replay: list[v056.StableSequence] = []
    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        model.load_state_dict(best_state)
        beta = v056._mixture_beta(round_index)
        rollout = v060._collect_mixture_rollout(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            teacher_fraction=beta,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"safety-selected-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        v056._print_rollout(f"safe {round_index:02d} rollout", rollout)

        replay.append(rollout.sequence)
        replay = v056._trim_student_replay(replay, max_frames=expert_sequence.frames)
        replay_frames = v056._student_replay_frames(replay)
        expert_share = expert_sequence.frames / max(1, expert_sequence.frames + replay_frames)
        sequences = [expert_sequence, *replay]
        print(
            f"replay expert={expert_sequence.frames} student={replay_frames} "
            f"expert-share={expert_share * 100.0:.1f}% trajectories={len(sequences)}"
        )

        accepted_epochs = 0
        epoch_history: list[dict] = []
        for epoch_index in range(1, args.round_epochs + 1):
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
                alphas=v058.DEFAULT_TRUST_ALPHAS,
                train_segment=train_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                label_prefix=f"safe {round_index:02d} epoch {epoch_index:02d}",
            )

            if choice.accepted and choice.evaluation is not None and chosen_state is not None:
                accepted_epochs += 1
                best_eval = choice.evaluation
                best_state = copy.deepcopy(chosen_state)
                print(
                    f"safe {round_index:02d} epoch {epoch_index:02d}: "
                    f"ACCEPT alpha={choice.alpha:g} loss={loss:.6f} "
                    f"best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
                    f"X{best_eval.stats.x_accuracy_percent:.2f}% "
                    f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
                    f"early={best_eval.stats.too_early_presses} "
                    f"overload={best_eval.stats.overloaded}"
                )
            else:
                model.load_state_dict(best_state)
                print(f"safe {round_index:02d} epoch {epoch_index:02d}: ROLLBACK all alphas loss={loss:.6f}")

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
                            "xacc": candidate.evaluation.stats.x_accuracy_percent,
                            "pp": candidate.evaluation.stats.perfect_rate,
                            "early": candidate.evaluation.stats.too_early_presses,
                            "overloaded": candidate.evaluation.stats.overloaded,
                            "guard": candidate.decision.reason,
                        }
                        for candidate in candidates
                    ],
                }
            )

        print(
            f"safe {round_index:02d} summary: accepted-epochs={accepted_epochs}/{args.round_epochs} "
            f"best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
            f"X{best_eval.stats.x_accuracy_percent:.2f}% "
            f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
            f"early={best_eval.stats.too_early_presses} overload={best_eval.stats.overloaded}"
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
            "safety_selected": {
                "finger_agnostic": True,
                "bootstrap_eval_every_epoch": True,
                "bootstrap_history": bootstrap_history,
                "overload_to_safe_priority": True,
                "both_overloaded_comparison": False,
                "trust_alphas": list(v058.DEFAULT_TRUST_ALPHAS),
                "mixture_betas": list(v056.DEFAULT_MIXTURE_BETAS),
                "press_recovery_cap": args.press_recovery_cap,
                "round_history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
