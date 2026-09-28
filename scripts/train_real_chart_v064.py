from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
import train_real_chart_v058 as v058
import train_real_chart_v060 as v060
import train_real_chart_v062 as v062
import train_real_chart_v063 as v063
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.6.4-anchor-guard"
CHECKPOINT_FORMAT_VERSION = 12
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v064_anchor_guard.pt"
ANCHOR_GUARD_VERSION = "per-anchor-v1"


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    alpha: float
    train_eval: v054.StudentEvalResult
    validation_eval: v054.StudentEvalResult
    anchor_evals: tuple[v054.StudentEvalResult, ...]
    train_decision: v057.ConservativeDecision
    validation_decision: v057.ConservativeDecision
    anchor_decisions: tuple[v057.ConservativeDecision, ...]

    @property
    def accepted(self) -> bool:
        return (
            self.train_decision.accepted
            and self.validation_decision.accepted
            and all(decision.accepted for decision in self.anchor_decisions)
        )


@dataclass(frozen=True, slots=True)
class AnchorChoice:
    accepted: bool
    alpha: float | None
    train_eval: v054.StudentEvalResult | None
    validation_eval: v054.StudentEvalResult | None
    anchor_evals: tuple[v054.StudentEvalResult, ...] | None
    reason: str


def _anchor_guard(
    reference: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> v057.ConservativeDecision:
    """Apply the fixed validation-floor semantics to one training anchor.

    Anchor references only move upward, so the tolerated hit/XAcc/early drift is
    always measured from a retained floor rather than accumulating every round.
    """

    decision = v062._validation_guard(reference, candidate)
    reason = decision.reason.replace("validation", "anchor", 1)
    return v057.ConservativeDecision(decision.accepted, reason)


def _reference_key(result: v054.StudentEvalResult) -> tuple:
    return v062._validation_reference_key(result)


def _update_reference(
    reference: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> v054.StudentEvalResult:
    return candidate if _reference_key(candidate) > _reference_key(reference) else reference


def _choose_anchor_candidate(
    base_train: v054.StudentEvalResult,
    candidates: list[AnchorCandidate],
) -> AnchorChoice:
    passing = [candidate for candidate in candidates if candidate.accepted]
    if not passing:
        reasons = ";".join(
            f"a={candidate.alpha:g}:"
            f"{candidate.train_decision.reason}/"
            f"{candidate.validation_decision.reason}/"
            + ",".join(decision.reason for decision in candidate.anchor_decisions)
            for candidate in candidates
        )
        return AnchorChoice(False, None, None, None, None, reasons or "no candidates")

    chosen = max(
        passing,
        key=lambda item: v061_candidate_key(base_train, item.train_eval),
    )
    return AnchorChoice(
        True,
        chosen.alpha,
        chosen.train_eval,
        chosen.validation_eval,
        chosen.anchor_evals,
        "train+validation+anchors preserved",
    )


def v061_candidate_key(
    base_train: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> tuple:
    # Keep the v0.6.1 safety-aware ranking without importing another large
    # trainer namespace through callers/tests.
    import train_real_chart_v061 as v061

    return v061._candidate_key(base_train, candidate)


def _candidate_brief(candidate: AnchorCandidate) -> str:
    t = candidate.train_eval.stats
    v = candidate.validation_eval.stats
    anchor_pass = sum(decision.accepted for decision in candidate.anchor_decisions)
    anchor_total = len(candidate.anchor_decisions)
    anchor_over = any(result.stats.overloaded for result in candidate.anchor_evals)
    mark = "+" if candidate.accepted else "-"
    return (
        f"{candidate.alpha:g}{mark} "
        f"T{t.hits}/{t.targets} X{t.x_accuracy_percent:.1f} "
        f"V{v.hits}/{v.targets} X{v.x_accuracy_percent:.1f}"
        f"{'!' if v.overloaded else ''} "
        f"A{anchor_pass}/{anchor_total}{'!' if anchor_over else ''}"
    )


def _evaluate_anchor_line_search(
    model: RecurrentActorCritic,
    *,
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    base_train_eval: v054.StudentEvalResult,
    validation_reference: v054.StudentEvalResult,
    anchor_references: tuple[v054.StudentEvalResult, ...],
    alphas: tuple[float, ...],
    train_segment,
    validation_segment,
    anchor_segments: list,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    label_prefix: str,
    verbose: bool,
) -> tuple[AnchorChoice, list[AnchorCandidate], dict[str, torch.Tensor] | None]:
    if len(anchor_references) != len(anchor_segments):
        raise ValueError("anchor reference count must match anchor segment count")

    candidates: list[AnchorCandidate] = []
    states: dict[float, dict[str, torch.Tensor]] = {}

    for alpha in alphas:
        state = v058._interpolate_state(base_state, proposal_state, alpha)
        model.load_state_dict(state)
        train_eval = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        validation_eval = v054._evaluate_student(
            model,
            validation_segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        anchor_evals = tuple(
            v062._evaluate_many(
                model,
                anchor_segments,
                same_hand=same_hand,
                control_dt_s=control_dt_s,
                device=device,
            )
        )

        train_decision = v063.v062.v061._safety_guard(base_train_eval, train_eval)
        validation_decision = v062._validation_guard(validation_reference, validation_eval)
        anchor_decisions = tuple(
            _anchor_guard(reference, evaluation)
            for reference, evaluation in zip(anchor_references, anchor_evals)
        )
        candidate = AnchorCandidate(
            alpha,
            train_eval,
            validation_eval,
            anchor_evals,
            train_decision,
            validation_decision,
            anchor_decisions,
        )
        candidates.append(candidate)
        states[alpha] = state

        if verbose:
            anchor_text = " ".join(
                f"A{i + 1}=H{result.stats.hits}/{result.stats.targets} "
                f"X{result.stats.x_accuracy_percent:.1f}% "
                f"over={result.stats.overloaded} {decision.reason}"
                for i, (result, decision) in enumerate(zip(anchor_evals, anchor_decisions))
            )
            print(
                f"{label_prefix} a={alpha:g}: "
                f"T H={train_eval.stats.hits}/{train_eval.stats.targets} "
                f"X={train_eval.stats.x_accuracy_percent:.2f}% {train_decision.reason} | "
                f"V H={validation_eval.stats.hits}/{validation_eval.stats.targets} "
                f"X={validation_eval.stats.x_accuracy_percent:.2f}% "
                f"over={validation_eval.stats.overloaded} {validation_decision.reason} | "
                f"{anchor_text}"
            )

    choice = _choose_anchor_candidate(base_train_eval, candidates)
    if not verbose:
        print(f"{label_prefix}: " + " | ".join(_candidate_brief(candidate) for candidate in candidates))

    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None

    chosen_state = states[choice.alpha]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def _run_signature(args, *, chart_path: str, train_pool, validation_window, sight_window) -> dict:
    signature = v063._run_signature(
        args,
        chart_path=chart_path,
        train_pool=train_pool,
        validation_window=validation_window,
        sight_window=sight_window,
    )
    signature["anchor_guard"] = ANCHOR_GUARD_VERSION
    return signature


def _checkpoint_payload(
    *,
    model_state: dict[str, torch.Tensor],
    args,
    chart_path: str,
    signature: dict,
    completed_round: int,
    rng_state,
    validation_reference: v054.StudentEvalResult,
    anchor_references: tuple[v054.StudentEvalResult, ...],
    bootstrap_history: list[dict],
    round_history: list[dict],
    finalized: bool,
    final_metrics: dict | None = None,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_version": TRAINER_VERSION,
        "input_dim": REAL_CHART_INPUT_DIM,
        "hidden_dim": int(args.hidden),
        "model_state": model_state,
        "chart": chart_path,
        "signature": signature,
        "completed_round": int(completed_round),
        "rng_state": rng_state,
        "validation_reference": v063._eval_to_payload(validation_reference),
        "anchor_references": [v063._eval_to_payload(result) for result in anchor_references],
        "bootstrap_history": bootstrap_history,
        "round_history": round_history,
        "requested_rounds": int(args.rounds),
        "finalized": bool(finalized),
        "final_metrics": final_metrics,
        "sight_used_for_selection": False,
        "anchor_guard": ANCHOR_GUARD_VERSION,
    }


def _save_progress(
    path: Path,
    *,
    model_state: dict[str, torch.Tensor],
    args,
    chart_path: str,
    signature: dict,
    completed_round: int,
    rng_state,
    validation_reference: v054.StudentEvalResult,
    anchor_references: tuple[v054.StudentEvalResult, ...],
    bootstrap_history: list[dict],
    round_history: list[dict],
    finalized: bool = False,
    final_metrics: dict | None = None,
) -> None:
    payload = _checkpoint_payload(
        model_state=model_state,
        args=args,
        chart_path=chart_path,
        signature=signature,
        completed_round=completed_round,
        rng_state=rng_state,
        validation_reference=validation_reference,
        anchor_references=anchor_references,
        bootstrap_history=bootstrap_history,
        round_history=round_history,
        finalized=finalized,
        final_metrics=final_metrics,
    )
    v063._atomic_torch_save(payload, path)
    if args.keep_round_checkpoints and completed_round > 0 and not finalized:
        v063._atomic_torch_save(payload, v063._round_snapshot_path(path, completed_round))


def _load_progress(
    model: RecurrentActorCritic,
    path: Path,
    *,
    current_signature: dict,
    anchor_count: int,
    device: torch.device,
) -> tuple[dict, v054.StudentEvalResult, tuple[v054.StudentEvalResult, ...]]:
    payload = torch.load(path, map_location=device)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise SystemExit("v0.6.4 --resume requires a v0.6.4 anchor-guard progress checkpoint")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit("checkpoint input dimension does not match current encoder")
    if int(payload.get("hidden_dim", -1)) != model.hidden_dim:
        raise SystemExit("checkpoint hidden size does not match --hidden")

    saved_signature = payload.get("signature")
    if not isinstance(saved_signature, dict):
        raise SystemExit("checkpoint has no resumable run signature")
    mismatches = v063._signature_mismatches(saved_signature, current_signature)
    if mismatches:
        raise SystemExit("resume configuration differs from checkpoint: " + ", ".join(mismatches))
    anchors_payload = payload.get("anchor_references")
    if not isinstance(anchors_payload, list) or len(anchors_payload) != anchor_count:
        raise SystemExit("checkpoint anchor reference count does not match current bootstrap anchors")
    if "rng_state" not in payload or "validation_reference" not in payload:
        raise SystemExit("checkpoint is missing exact resume state")

    model.load_state_dict(payload["model_state"])
    validation_reference = v063._eval_from_payload(payload["validation_reference"])
    anchor_references = tuple(v063._eval_from_payload(item) for item in anchors_payload)
    return payload, validation_reference, anchor_references


def _anchor_summary(results: tuple[v054.StudentEvalResult, ...] | list[v054.StudentEvalResult]) -> str:
    hits, targets, mean_xacc, overloaded = v063._summary_metrics(list(results))
    return f"H{hits}/{targets} meanX{mean_xacc:.1f}% over={overloaded}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable multi-segment finger-agnostic DAgger with validation and "
            "per-anchor anti-forgetting guards on every trust-region candidate."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-pool-start", type=float, default=v062.DEFAULT_TRAIN_POOL_START)
    parser.add_argument("--train-pool-end", type=float, default=v062.DEFAULT_TRAIN_POOL_END)
    parser.add_argument("--train-window", type=float, default=v062.DEFAULT_TRAIN_WINDOW_S)
    parser.add_argument("--validation-start", type=float, default=v062.DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", type=float, default=v062.DEFAULT_VALIDATION_END)
    parser.add_argument("--sight-start", type=float, default=v062.DEFAULT_SIGHT_START)
    parser.add_argument("--sight-end", type=float, default=v062.DEFAULT_SIGHT_END)
    parser.add_argument("--bootstrap-segments", type=int, default=v062.DEFAULT_BOOTSTRAP_SEGMENTS)
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
    parser.add_argument("--keep-round-checkpoints", action="store_true")
    parser.add_argument("--cross-hand", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0 or args.round_epochs <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("round counts and epoch counts must be positive")
    if args.bootstrap_segments <= 0 or args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("bootstrap-segments/hidden/chunk-steps must be positive")
    if args.press_recovery_cap <= 0:
        raise SystemExit("press-recovery-cap must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand
    chart_path = str(Path(args.chart).resolve())
    compiled = load_compiled_adofai(args.chart)
    duration = compiled.duration_s

    train_pool = v062.SegmentWindow(args.train_pool_start, min(args.train_pool_end, duration))
    validation_window = v062.SegmentWindow(args.validation_start, min(args.validation_end, duration))
    sight_window = v062.SegmentWindow(args.sight_start, min(args.sight_end, duration))
    try:
        v062._validate_disjoint_ranges(train_pool, validation_window, sight_window)
        bootstrap_windows = v062._even_windows(
            train_pool.start_s,
            train_pool.end_s,
            args.train_window,
            args.bootstrap_segments,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    validation_segment = build_playable_segment(
        compiled,
        start_s=validation_window.start_s,
        end_s=validation_window.end_s,
    )
    sight_segment = build_playable_segment(
        compiled,
        start_s=sight_window.start_s,
        end_s=sight_window.end_s,
    )
    anchor_segments = [
        build_playable_segment(compiled, start_s=window.start_s, end_s=window.end_s)
        for window in bootstrap_windows
    ]
    if any(not segment.targets for segment in anchor_segments):
        raise SystemExit("one of the bootstrap/anchor training segments contains no playable targets")
    if not validation_segment.targets or not sight_segment.targets:
        raise SystemExit("validation/sight segment contains no playable targets")

    calibration = calibrate_single_press_lead(control_dt_s=args.control_dt, same_hand=same_hand)
    signature = _run_signature(
        args,
        chart_path=chart_path,
        train_pool=train_pool,
        validation_window=validation_window,
        sight_window=sight_window,
    )

    print("=== DMDOD / Real Chart Student v0.6.4 Anchor Guard ===")
    print(f"chart={args.chart}")
    print(
        f"train-pool={train_pool.start_s:g}..{train_pool.end_s:g}s window={args.train_window:g}s "
        f"anchors=" + ",".join(f"{w.start_s:g}..{w.end_s:g}" for w in bootstrap_windows)
    )
    print(
        f"validation={validation_window.start_s:g}..{validation_window.end_s:g}s "
        f"targets={len(validation_segment.targets)} | sight={sight_window.start_s:g}..{sight_window.end_s:g}s "
        f"targets={len(sight_segment.targets)} FINAL-ONLY"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D hidden={args.hidden} lead={calibration.lead_s * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs} | candidate guard=train+validation+{len(anchor_segments)} anchors"
    )

    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    expert_stable: list[v056.StableSequence] = []
    for index, segment in enumerate(anchor_segments, 1):
        x, y, teacher_eval = v060._collect_expert_sequence(
            segment,
            lead_s=calibration.lead_s,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        expert_pairs.append((x, y))
        expert_stable.append(
            v056._make_stable_sequence(
                v060.v055.DAggerSequence(x, y, f"anchor-expert-{index}"),
                press_recovery_cap=args.press_recovery_cap,
                expert=True,
            )
        )
        print(
            f"teacher {index}: {bootstrap_windows[index - 1].start_s:g}..{bootstrap_windows[index - 1].end_s:g}s "
            f"H={teacher_eval.stats.hits}/{teacher_eval.stats.targets} "
            f"X={teacher_eval.stats.x_accuracy_percent:.2f}% over={teacher_eval.stats.overloaded}"
        )

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    rng = random.Random(args.seed * 1000003 + 64)

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        payload, validation_reference, anchor_references = _load_progress(
            model,
            checkpoint_path,
            current_signature=signature,
            anchor_count=len(anchor_segments),
            device=device,
        )
        best_state = copy.deepcopy(model.state_dict())
        bootstrap_history = list(payload.get("bootstrap_history", []))
        round_history = list(payload.get("round_history", []))
        completed_round = int(payload.get("completed_round", 0))
        rng.setstate(payload["rng_state"])
        if completed_round > args.rounds:
            raise SystemExit(
                f"checkpoint already completed round {completed_round}, greater than requested --rounds={args.rounds}"
            )
        print(
            f"resume={checkpoint_path} completed-round={completed_round}/{args.rounds} "
            f"V=H{validation_reference.stats.hits}/{validation_reference.stats.targets} "
            f"X{validation_reference.stats.x_accuracy_percent:.1f}% | "
            f"anchor-floors={_anchor_summary(anchor_references)}"
        )
        if payload.get("finalized") and completed_round < args.rounds:
            print("warning: extending a finalized run means the old sight result is no longer pristine to the experimenter")
    else:
        print(
            f"bootstrap=fresh multi-segment BC epochs={args.bootstrap_epochs} trajectories={len(expert_pairs)}"
        )
        best_state, bootstrap_train_evals, validation_reference, bootstrap_history = (
            v062._train_multisegment_bootstrap(
                model,
                expert_pairs,
                anchor_segments,
                validation_segment,
                epochs=args.bootstrap_epochs,
                learning_rate=args.bootstrap_lr,
                chunk_steps=args.chunk_steps,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                verbose=args.verbose,
            )
        )
        anchor_references = tuple(bootstrap_train_evals)
        model.load_state_dict(best_state)
        completed_round = 0
        round_history: list[dict] = []
        _save_progress(
            checkpoint_path,
            model_state=best_state,
            args=args,
            chart_path=chart_path,
            signature=signature,
            completed_round=0,
            rng_state=rng.getstate(),
            validation_reference=validation_reference,
            anchor_references=anchor_references,
            bootstrap_history=bootstrap_history,
            round_history=round_history,
        )
        print(f"checkpoint bootstrap: {checkpoint_path}")

    model.load_state_dict(best_state)
    print(
        f"floors: V=H{validation_reference.stats.hits}/{validation_reference.stats.targets} "
        f"X{validation_reference.stats.x_accuracy_percent:.1f}% | "
        f"anchors={_anchor_summary(anchor_references)}"
    )

    for round_index in range(completed_round + 1, args.rounds + 1):
        window = v062._sample_window(rng, train_pool.start_s, train_pool.end_s, args.train_window)
        train_segment = build_playable_segment(compiled, start_s=window.start_s, end_s=window.end_s)
        if not train_segment.targets:
            raise RuntimeError("sampled training segment contains no playable targets")

        model.load_state_dict(best_state)
        base_train_eval = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        beta = v056._mixture_beta(round_index)
        current_x, current_y, _ = v060._collect_expert_sequence(
            train_segment,
            lead_s=calibration.lead_s,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        current_expert = v056._make_stable_sequence(
            v060.v055.DAggerSequence(current_x, current_y, f"round-{round_index}-expert"),
            press_recovery_cap=args.press_recovery_cap,
            expert=True,
        )
        rollout = v060._collect_mixture_rollout(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            teacher_fraction=beta,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"anchor-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        print(
            f"round {round_index:03d} train={window.start_s:.2f}..{window.end_s:.2f}s "
            f"base=H{base_train_eval.stats.hits}/{base_train_eval.stats.targets} "
            f"X{base_train_eval.stats.x_accuracy_percent:.1f}% | "
            f"mix{beta * 100:.0f}=H{rollout.evaluation.stats.hits}/{rollout.evaluation.stats.targets} "
            f"X{rollout.evaluation.stats.x_accuracy_percent:.1f}%"
        )

        sequences = [*expert_stable, current_expert, rollout.sequence]
        accepted_epochs = 0
        epoch_history: list[dict] = []
        for epoch_index in range(1, args.round_epochs + 1):
            model.load_state_dict(best_state)
            base_train_eval = v054._evaluate_student(
                model,
                train_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
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

            choice, candidates, chosen_state = _evaluate_anchor_line_search(
                model,
                base_state=base_state,
                proposal_state=proposal_state,
                base_train_eval=base_train_eval,
                validation_reference=validation_reference,
                anchor_references=anchor_references,
                alphas=v058.DEFAULT_TRUST_ALPHAS,
                train_segment=train_segment,
                validation_segment=validation_segment,
                anchor_segments=anchor_segments,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                label_prefix=f"round {round_index:03d} e{epoch_index:02d}",
                verbose=args.verbose,
            )

            if (
                choice.accepted
                and chosen_state is not None
                and choice.train_eval is not None
                and choice.validation_eval is not None
                and choice.anchor_evals is not None
            ):
                accepted_epochs += 1
                best_state = copy.deepcopy(chosen_state)
                model.load_state_dict(best_state)
                validation_reference = _update_reference(validation_reference, choice.validation_eval)
                anchor_references = tuple(
                    _update_reference(reference, evaluation)
                    for reference, evaluation in zip(anchor_references, choice.anchor_evals)
                )
                print(
                    f"round {round_index:03d} e{epoch_index:02d}: ACCEPT a={choice.alpha:g} "
                    f"loss={loss:.4f} T=H{choice.train_eval.stats.hits}/{choice.train_eval.stats.targets} "
                    f"X{choice.train_eval.stats.x_accuracy_percent:.1f}% "
                    f"V=H{choice.validation_eval.stats.hits}/{choice.validation_eval.stats.targets} "
                    f"X{choice.validation_eval.stats.x_accuracy_percent:.1f}% "
                    f"A={_anchor_summary(choice.anchor_evals)}"
                )
            else:
                model.load_state_dict(best_state)
                print(f"round {round_index:03d} e{epoch_index:02d}: ROLLBACK loss={loss:.4f}")

            epoch_history.append(
                {
                    "epoch": epoch_index,
                    "loss": loss,
                    "accepted": choice.accepted,
                    "alpha": choice.alpha,
                    "candidates": [
                        {
                            "alpha": candidate.alpha,
                            "train_hits": candidate.train_eval.stats.hits,
                            "train_xacc": candidate.train_eval.stats.x_accuracy_percent,
                            "train_overloaded": candidate.train_eval.stats.overloaded,
                            "validation_hits": candidate.validation_eval.stats.hits,
                            "validation_xacc": candidate.validation_eval.stats.x_accuracy_percent,
                            "validation_overloaded": candidate.validation_eval.stats.overloaded,
                            "anchor_hits": [result.stats.hits for result in candidate.anchor_evals],
                            "anchor_xacc": [result.stats.x_accuracy_percent for result in candidate.anchor_evals],
                            "anchor_overloaded": [result.stats.overloaded for result in candidate.anchor_evals],
                            "train_guard": candidate.train_decision.reason,
                            "validation_guard": candidate.validation_decision.reason,
                            "anchor_guards": [decision.reason for decision in candidate.anchor_decisions],
                            "accepted": candidate.accepted,
                        }
                        for candidate in candidates
                    ],
                }
            )

        model.load_state_dict(best_state)
        final_train = v054._evaluate_student(
            model,
            train_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        current_validation = v054._evaluate_student(
            model,
            validation_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        current_anchors = tuple(
            v062._evaluate_many(
                model,
                anchor_segments,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
        )
        print(
            f"round {round_index:03d} summary: accepted={accepted_epochs}/{args.round_epochs} "
            f"T=H{final_train.stats.hits}/{final_train.stats.targets} X{final_train.stats.x_accuracy_percent:.1f}% | "
            f"V=H{current_validation.stats.hits}/{current_validation.stats.targets} "
            f"X{current_validation.stats.x_accuracy_percent:.1f}% over={current_validation.stats.overloaded} | "
            f"A={_anchor_summary(current_anchors)}"
        )
        round_history.append(
            {
                "round": round_index,
                "train_start": window.start_s,
                "train_end": window.end_s,
                "teacher_fraction": beta,
                "accepted_epochs": accepted_epochs,
                "anchors": [v063._eval_to_payload(result) for result in current_anchors],
                "epochs": epoch_history,
            }
        )
        completed_round = round_index
        _save_progress(
            checkpoint_path,
            model_state=best_state,
            args=args,
            chart_path=chart_path,
            signature=signature,
            completed_round=completed_round,
            rng_state=rng.getstate(),
            validation_reference=validation_reference,
            anchor_references=anchor_references,
            bootstrap_history=bootstrap_history,
            round_history=round_history,
        )
        print(f"checkpoint round {round_index:03d}: {checkpoint_path}")

    model.load_state_dict(best_state)
    anchor_final = tuple(
        v062._evaluate_many(
            model,
            anchor_segments,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
    )
    validation_final = v054._evaluate_student(
        model,
        validation_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    sight_final = v054._evaluate_student(
        model,
        sight_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    train_hits, train_targets, train_xacc, train_over = v063._summary_metrics(list(anchor_final))
    print(
        f"final train-anchors: H={train_hits}/{train_targets} meanX={train_xacc:.2f}% over={train_over}"
    )
    print(v054._format_eval("final validation", validation_final))
    print(v054._format_eval("FINAL sight-read", sight_final))

    final_metrics = {
        "train_hits": train_hits,
        "train_targets": train_targets,
        "train_mean_xacc": train_xacc,
        "train_overloaded": train_over,
        "anchors": [v063._eval_to_payload(result) for result in anchor_final],
        "validation": v063._eval_to_payload(validation_final),
        "sight": v063._eval_to_payload(sight_final),
    }
    _save_progress(
        checkpoint_path,
        model_state=best_state,
        args=args,
        chart_path=chart_path,
        signature=signature,
        completed_round=completed_round,
        rng_state=rng.getstate(),
        validation_reference=validation_reference,
        anchor_references=anchor_references,
        bootstrap_history=bootstrap_history,
        round_history=round_history,
        finalized=True,
        final_metrics=final_metrics,
    )
    print(f"checkpoint final: {checkpoint_path}")


if __name__ == "__main__":
    main()
