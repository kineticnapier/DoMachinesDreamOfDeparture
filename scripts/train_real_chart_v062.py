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
import train_real_chart_v061 as v061
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.6.2-multisegment-validation"
CHECKPOINT_FORMAT_VERSION = 10
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v062_multisegment.pt"
DEFAULT_TRAIN_POOL_START = 0.0
DEFAULT_TRAIN_POOL_END = 90.0
DEFAULT_TRAIN_WINDOW_S = 30.0
DEFAULT_VALIDATION_START = 90.0
DEFAULT_VALIDATION_END = 110.0
DEFAULT_SIGHT_START = 110.0
DEFAULT_SIGHT_END = 130.0
DEFAULT_BOOTSTRAP_SEGMENTS = 3
DEFAULT_VALIDATION_XACC_DROP = 2.0
DEFAULT_VALIDATION_EARLY_INCREASE = 4
DEFAULT_VALIDATION_HIT_DROP_FRACTION = 0.03
DEFAULT_VALIDATION_HIT_DROP_ABSOLUTE = 3


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    start_s: float
    end_s: float

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class DualCandidate:
    alpha: float
    train_eval: v054.StudentEvalResult
    validation_eval: v054.StudentEvalResult
    train_decision: v057.ConservativeDecision
    validation_decision: v057.ConservativeDecision

    @property
    def accepted(self) -> bool:
        return self.train_decision.accepted and self.validation_decision.accepted


@dataclass(frozen=True, slots=True)
class DualChoice:
    accepted: bool
    alpha: float | None
    train_eval: v054.StudentEvalResult | None
    validation_eval: v054.StudentEvalResult | None
    reason: str


def _even_windows(
    start_s: float,
    end_s: float,
    window_s: float,
    count: int,
) -> tuple[SegmentWindow, ...]:
    if end_s <= start_s:
        raise ValueError("training pool end must be after start")
    if window_s <= 0.0 or window_s > end_s - start_s + 1e-12:
        raise ValueError("training window must fit inside the training pool")
    if count <= 0:
        raise ValueError("count must be positive")

    latest = end_s - window_s
    if count == 1 or latest <= start_s + 1e-12:
        return (SegmentWindow(start_s, start_s + window_s),)
    step = (latest - start_s) / (count - 1)
    return tuple(
        SegmentWindow(start_s + step * i, start_s + step * i + window_s)
        for i in range(count)
    )


def _sample_window(
    rng: random.Random,
    start_s: float,
    end_s: float,
    window_s: float,
) -> SegmentWindow:
    if window_s <= 0.0 or end_s - start_s < window_s - 1e-12:
        raise ValueError("training window must fit inside the training pool")
    latest = end_s - window_s
    sampled = start_s if latest <= start_s else rng.uniform(start_s, latest)
    return SegmentWindow(sampled, sampled + window_s)


def _validate_disjoint_ranges(
    train_pool: SegmentWindow,
    validation: SegmentWindow,
    sight: SegmentWindow,
) -> None:
    ranges = [
        ("train pool", train_pool),
        ("validation", validation),
        ("sight", sight),
    ]
    for name, window in ranges:
        if window.end_s <= window.start_s:
            raise ValueError(f"{name} range must have positive length")
    for i, (name_a, a) in enumerate(ranges):
        for name_b, b in ranges[i + 1 :]:
            if max(a.start_s, b.start_s) < min(a.end_s, b.end_s) - 1e-12:
                raise ValueError(f"{name_a} and {name_b} ranges must not overlap")


def _validation_guard(
    reference: v054.StudentEvalResult,
    candidate: v054.StudentEvalResult,
) -> v057.ConservativeDecision:
    """Preserve a fixed validation floor while training on changing segments.

    Unlike the training guard, validation does not need to improve every step.
    It only prevents cumulative forgetting.  Once a safe validation policy is
    found, overloaded candidates are permanently rejected.  While validation is
    still overloaded, non-worsening overloaded candidates may pass so training
    can continue toward an eventual safety escape.
    """

    ref_over = bool(reference.stats.overloaded)
    cand_over = bool(candidate.stats.overloaded)
    if ref_over and not cand_over:
        return v057.ConservativeDecision(True, "validation escaped overload")
    if not ref_over and cand_over:
        return v057.ConservativeDecision(False, "validation safe->overload")

    hit_tolerance = max(
        DEFAULT_VALIDATION_HIT_DROP_ABSOLUTE,
        round(reference.stats.targets * DEFAULT_VALIDATION_HIT_DROP_FRACTION),
    )
    if candidate.stats.hits < reference.stats.hits - hit_tolerance:
        return v057.ConservativeDecision(False, f"validation hit regression>{hit_tolerance}")
    if (
        candidate.stats.x_accuracy_percent
        < reference.stats.x_accuracy_percent - DEFAULT_VALIDATION_XACC_DROP
    ):
        return v057.ConservativeDecision(False, "validation XAcc regression>2pt")
    if (
        candidate.stats.too_early_presses
        > reference.stats.too_early_presses + DEFAULT_VALIDATION_EARLY_INCREASE
    ):
        return v057.ConservativeDecision(False, "validation early regression>4")

    if ref_over and cand_over:
        return v057.ConservativeDecision(True, "validation overloaded non-regression")
    return v057.ConservativeDecision(True, "validation preserved")


def _validation_reference_key(result: v054.StudentEvalResult) -> tuple:
    stats = result.stats
    return (
        0 if stats.overloaded else 1,
        int(stats.hits),
        float(stats.x_accuracy_percent),
        float(stats.perfect_rate),
        -int(stats.too_early_presses),
        -int(stats.misses),
    )


def _choose_dual_candidate(
    base_train: v054.StudentEvalResult,
    candidates: list[DualCandidate],
) -> DualChoice:
    passing = [candidate for candidate in candidates if candidate.accepted]
    if not passing:
        reasons = ";".join(
            f"a={candidate.alpha:g}:{candidate.train_decision.reason}/{candidate.validation_decision.reason}"
            for candidate in candidates
        )
        return DualChoice(False, None, None, None, reasons or "no candidates")

    # Validation is a preservation constraint, not the optimization target.
    # Among candidates that preserve it, continue the v0.6.1 training ranking.
    chosen = max(
        passing,
        key=lambda item: v061._candidate_key(base_train, item.train_eval),
    )
    return DualChoice(
        True,
        chosen.alpha,
        chosen.train_eval,
        chosen.validation_eval,
        f"{chosen.train_decision.reason}; {chosen.validation_decision.reason}",
    )


def _candidate_brief(candidate: DualCandidate) -> str:
    t = candidate.train_eval.stats
    v = candidate.validation_eval.stats
    mark = "+" if candidate.accepted else "-"
    return (
        f"{candidate.alpha:g}{mark} "
        f"T{t.hits}/{t.targets} X{t.x_accuracy_percent:.1f} "
        f"V{v.hits}/{v.targets} X{v.x_accuracy_percent:.1f}"
        f"{'!' if v.overloaded else ''}"
    )


def _evaluate_dual_line_search(
    model: RecurrentActorCritic,
    *,
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    base_train_eval: v054.StudentEvalResult,
    validation_reference: v054.StudentEvalResult,
    alphas: tuple[float, ...],
    train_segment,
    validation_segment,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    label_prefix: str,
    verbose: bool,
) -> tuple[DualChoice, list[DualCandidate], dict[str, torch.Tensor] | None]:
    candidates: list[DualCandidate] = []
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
        train_decision = v061._safety_guard(base_train_eval, train_eval)
        validation_decision = _validation_guard(validation_reference, validation_eval)
        candidate = DualCandidate(
            alpha,
            train_eval,
            validation_eval,
            train_decision,
            validation_decision,
        )
        candidates.append(candidate)
        states[alpha] = state
        if verbose:
            print(
                f"{label_prefix} a={alpha:g}: "
                f"T H={train_eval.stats.hits}/{train_eval.stats.targets} "
                f"X={train_eval.stats.x_accuracy_percent:.2f}% early={train_eval.stats.too_early_presses} "
                f"over={train_eval.stats.overloaded} guard={train_decision.reason} | "
                f"V H={validation_eval.stats.hits}/{validation_eval.stats.targets} "
                f"X={validation_eval.stats.x_accuracy_percent:.2f}% early={validation_eval.stats.too_early_presses} "
                f"over={validation_eval.stats.overloaded} guard={validation_decision.reason}"
            )

    choice = _choose_dual_candidate(base_train_eval, candidates)
    if not verbose:
        print(f"{label_prefix}: " + " | ".join(_candidate_brief(c) for c in candidates))

    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None
    chosen_state = states[choice.alpha]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def _bootstrap_multi_epoch(
    model: RecurrentActorCritic,
    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    parameters = v057._policy_parameters(model)
    pairs = list(reversed(expert_pairs)) if reverse_order else expert_pairs
    loss_sum = 0.0
    frame_count = 0

    for observations, actions in pairs:
        state = model.initial_state(observations.device)
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


def _evaluate_many(
    model: RecurrentActorCritic,
    segments: list,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> list[v054.StudentEvalResult]:
    return [
        v054._evaluate_student(
            model,
            segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        for segment in segments
    ]


def _bootstrap_selection_key(
    train_evals: list[v054.StudentEvalResult],
    validation_eval: v054.StudentEvalResult,
) -> tuple:
    evaluations = [*train_evals, validation_eval]
    all_safe = all(not result.stats.overloaded for result in evaluations)
    total_hits = sum(result.stats.hits for result in evaluations)
    total_targets = sum(result.stats.targets for result in evaluations)
    mean_xacc = sum(result.stats.x_accuracy_percent for result in evaluations) / len(evaluations)
    mean_pp = sum(result.stats.perfect_rate for result in evaluations) / len(evaluations)
    total_early = sum(result.stats.too_early_presses for result in evaluations)
    return (
        int(all_safe),
        total_hits / max(1, total_targets),
        mean_xacc,
        mean_pp,
        -total_early,
    )


def _train_multisegment_bootstrap(
    model: RecurrentActorCritic,
    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    train_segments: list,
    validation_segment,
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    verbose: bool,
) -> tuple[dict[str, torch.Tensor], list[v054.StudentEvalResult], v054.StudentEvalResult, list[dict]]:
    optimizer = torch.optim.Adam(v057._policy_parameters(model), lr=learning_rate)

    train_evals = _evaluate_many(
        model,
        train_segments,
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
    best_key = _bootstrap_selection_key(train_evals, validation_eval)
    best_state = copy.deepcopy(model.state_dict())
    best_train = train_evals
    best_validation = validation_eval
    best_epoch = 0
    history: list[dict] = []

    print(
        f"bootstrap 00: safe={bool(best_key[0])} "
        f"completion={best_key[1] * 100.0:.1f}% meanX={best_key[2]:.1f}% "
        f"val=H{validation_eval.stats.hits}/{validation_eval.stats.targets} "
        f"X{validation_eval.stats.x_accuracy_percent:.1f}%"
    )

    for epoch in range(1, epochs + 1):
        loss = _bootstrap_multi_epoch(
            model,
            expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=not bool(epoch & 1),
        )
        train_evals = _evaluate_many(
            model,
            train_segments,
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
        key = _bootstrap_selection_key(train_evals, validation_eval)
        kept = key > best_key
        if kept:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_train = train_evals
            best_validation = validation_eval
            best_epoch = epoch

        if verbose or kept or epoch == 1 or epoch == epochs or epoch % 4 == 0:
            print(
                f"bootstrap {epoch:02d}: loss={loss:.6f} safe={bool(key[0])} "
                f"completion={key[1] * 100.0:.1f}% meanX={key[2]:.1f}% "
                f"val=H{validation_eval.stats.hits}/{validation_eval.stats.targets} "
                f"X{validation_eval.stats.x_accuracy_percent:.1f}% "
                f"over={validation_eval.stats.overloaded}"
                + (" KEEP" if kept else "")
            )
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "all_safe": bool(key[0]),
                "completion": key[1],
                "mean_xacc": key[2],
                "validation_hits": validation_eval.stats.hits,
                "validation_xacc": validation_eval.stats.x_accuracy_percent,
                "validation_overloaded": validation_eval.stats.overloaded,
                "kept": kept,
            }
        )

    model.load_state_dict(best_state)
    print(
        f"bootstrap selected: epoch={best_epoch} safe={bool(best_key[0])} "
        f"completion={best_key[1] * 100.0:.1f}% meanX={best_key[2]:.1f}% "
        f"val=H{best_validation.stats.hits}/{best_validation.stats.targets} "
        f"X{best_validation.stats.x_accuracy_percent:.1f}%"
    )
    return best_state, best_train, best_validation, history


def _load_resume(
    model: RecurrentActorCritic,
    path: Path,
    *,
    hidden_dim: int,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location=device)
    format_version = int(payload.get("format_version", -1))
    if format_version not in (v061.CHECKPOINT_FORMAT_VERSION, CHECKPOINT_FORMAT_VERSION):
        raise SystemExit("v0.6.2 resumes only from v0.6.1/v0.6.2 checkpoints")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit("checkpoint input dimension does not match current encoder")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit("checkpoint hidden size does not match --hidden")
    model.load_state_dict(payload["model_state"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-segment finger-agnostic DAgger with a fixed validation guard and a "
            "final held-out sight segment that is never used for model selection."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-pool-start", type=float, default=DEFAULT_TRAIN_POOL_START)
    parser.add_argument("--train-pool-end", type=float, default=DEFAULT_TRAIN_POOL_END)
    parser.add_argument("--train-window", type=float, default=DEFAULT_TRAIN_WINDOW_S)
    parser.add_argument("--validation-start", type=float, default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", type=float, default=DEFAULT_VALIDATION_END)
    parser.add_argument("--sight-start", type=float, default=DEFAULT_SIGHT_START)
    parser.add_argument("--sight-end", type=float, default=DEFAULT_SIGHT_END)
    parser.add_argument("--bootstrap-segments", type=int, default=DEFAULT_BOOTSTRAP_SEGMENTS)
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0 or args.round_epochs <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("round counts and epoch counts must be positive")
    if args.bootstrap_segments <= 0 or args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("bootstrap-segments/hidden/chunk-steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand
    compiled = load_compiled_adofai(args.chart)
    duration = compiled.duration_s

    train_pool = SegmentWindow(args.train_pool_start, min(args.train_pool_end, duration))
    validation_window = SegmentWindow(args.validation_start, min(args.validation_end, duration))
    sight_window = SegmentWindow(args.sight_start, min(args.sight_end, duration))
    try:
        _validate_disjoint_ranges(train_pool, validation_window, sight_window)
        bootstrap_windows = _even_windows(
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
    bootstrap_segments = [
        build_playable_segment(compiled, start_s=w.start_s, end_s=w.end_s)
        for w in bootstrap_windows
    ]
    if any(not segment.targets for segment in bootstrap_segments):
        raise SystemExit("one of the bootstrap training segments contains no playable targets")
    if not validation_segment.targets or not sight_segment.targets:
        raise SystemExit("validation/sight segment contains no playable targets")

    calibration = calibrate_single_press_lead(
        control_dt_s=args.control_dt,
        same_hand=same_hand,
    )
    print("=== DMDOD / Real Chart Student v0.6.2 Multi-Segment Validation ===")
    print(f"chart={args.chart}")
    print(
        f"train-pool={train_pool.start_s:g}..{train_pool.end_s:g}s window={args.train_window:g}s "
        f"bootstrap=" + ",".join(f"{w.start_s:g}..{w.end_s:g}" for w in bootstrap_windows)
    )
    print(
        f"validation={validation_window.start_s:g}..{validation_window.end_s:g}s "
        f"targets={len(validation_segment.targets)} | "
        f"sight={sight_window.start_s:g}..{sight_window.end_s:g}s "
        f"targets={len(sight_segment.targets)} FINAL-ONLY"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D hidden={args.hidden} lead={calibration.lead_s * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs} | validation guard fixed | compact output"
    )

    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    expert_stable: list[v056.StableSequence] = []
    for index, segment in enumerate(bootstrap_segments, 1):
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
                v060.v055.DAggerSequence(x, y, f"bootstrap-expert-{index}"),
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
    bootstrap_history: list[dict] = []

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        payload = _load_resume(model, checkpoint_path, hidden_dim=args.hidden, device=device)
        print(f"resume={checkpoint_path} format={payload.get('format_version')}")
        bootstrap_train_evals = _evaluate_many(
            model,
            bootstrap_segments,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        validation_reference = v054._evaluate_student(
            model,
            validation_segment,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        best_state = copy.deepcopy(model.state_dict())
    else:
        print(
            f"bootstrap=fresh multi-segment BC epochs={args.bootstrap_epochs} "
            f"trajectories={len(expert_pairs)}"
        )
        best_state, bootstrap_train_evals, validation_reference, bootstrap_history = (
            _train_multisegment_bootstrap(
                model,
                expert_pairs,
                bootstrap_segments,
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

    model.load_state_dict(best_state)
    print(
        f"validation baseline: H={validation_reference.stats.hits}/{validation_reference.stats.targets} "
        f"X={validation_reference.stats.x_accuracy_percent:.2f}% "
        f"early={validation_reference.stats.too_early_presses} over={validation_reference.stats.overloaded}"
    )

    rng = random.Random(args.seed * 1000003 + 62)
    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        window = _sample_window(
            rng,
            train_pool.start_s,
            train_pool.end_s,
            args.train_window,
        )
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
            source=f"multi-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        print(
            f"round {round_index:02d} train={window.start_s:.2f}..{window.end_s:.2f}s "
            f"base=H{base_train_eval.stats.hits}/{base_train_eval.stats.targets} "
            f"X{base_train_eval.stats.x_accuracy_percent:.1f}% | "
            f"mix{beta * 100:.0f}=H{rollout.evaluation.stats.hits}/{rollout.evaluation.stats.targets} "
            f"X{rollout.evaluation.stats.x_accuracy_percent:.1f}%"
        )

        # Fixed anchor experts prevent one random segment from erasing the rest
        # of the training pool.  The current expert and current DAgger trajectory
        # teach the newly sampled local pattern.
        sequences = [*expert_stable, current_expert, rollout.sequence]
        accepted_epochs = 0
        epoch_history: list[dict] = []

        for epoch_index in range(1, args.round_epochs + 1):
            model.load_state_dict(best_state)
            # Re-evaluate the trusted state on this round's segment.  Comparing
            # metrics across different random segments would be meaningless.
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

            choice, candidates, chosen_state = _evaluate_dual_line_search(
                model,
                base_state=base_state,
                proposal_state=proposal_state,
                base_train_eval=base_train_eval,
                validation_reference=validation_reference,
                alphas=v058.DEFAULT_TRUST_ALPHAS,
                train_segment=train_segment,
                validation_segment=validation_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                label_prefix=f"round {round_index:02d} e{epoch_index:02d}",
                verbose=args.verbose,
            )

            if choice.accepted and chosen_state is not None and choice.train_eval is not None and choice.validation_eval is not None:
                accepted_epochs += 1
                best_state = copy.deepcopy(chosen_state)
                model.load_state_dict(best_state)
                if _validation_reference_key(choice.validation_eval) > _validation_reference_key(validation_reference):
                    validation_reference = choice.validation_eval
                print(
                    f"round {round_index:02d} e{epoch_index:02d}: ACCEPT a={choice.alpha:g} "
                    f"loss={loss:.4f} T=H{choice.train_eval.stats.hits}/{choice.train_eval.stats.targets} "
                    f"X{choice.train_eval.stats.x_accuracy_percent:.1f}% "
                    f"V=H{choice.validation_eval.stats.hits}/{choice.validation_eval.stats.targets} "
                    f"X{choice.validation_eval.stats.x_accuracy_percent:.1f}%"
                )
            else:
                model.load_state_dict(best_state)
                print(f"round {round_index:02d} e{epoch_index:02d}: ROLLBACK loss={loss:.4f}")

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
                            "train_guard": candidate.train_decision.reason,
                            "validation_guard": candidate.validation_decision.reason,
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
        print(
            f"round {round_index:02d} summary: accepted={accepted_epochs}/{args.round_epochs} "
            f"T=H{final_train.stats.hits}/{final_train.stats.targets} X{final_train.stats.x_accuracy_percent:.1f}% | "
            f"V=H{current_validation.stats.hits}/{current_validation.stats.targets} "
            f"X{current_validation.stats.x_accuracy_percent:.1f}% over={current_validation.stats.overloaded}"
        )
        round_history.append(
            {
                "round": round_index,
                "train_start": window.start_s,
                "train_end": window.end_s,
                "teacher_fraction": beta,
                "accepted_epochs": accepted_epochs,
                "epochs": epoch_history,
            }
        )

    model.load_state_dict(best_state)
    bootstrap_final = _evaluate_many(
        model,
        bootstrap_segments,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    validation_final = v054._evaluate_student(
        model,
        validation_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    # This is intentionally the first and only sight evaluation in the entire
    # script.  Its result never participates in selection or rollback.
    sight_final = v054._evaluate_student(
        model,
        sight_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    train_hits = sum(result.stats.hits for result in bootstrap_final)
    train_targets = sum(result.stats.targets for result in bootstrap_final)
    train_xacc = sum(result.stats.x_accuracy_percent for result in bootstrap_final) / len(bootstrap_final)
    print(
        f"final train-anchors: H={train_hits}/{train_targets} meanX={train_xacc:.2f}% "
        f"over={any(result.stats.overloaded for result in bootstrap_final)}"
    )
    print(v054._format_eval("final validation", validation_final))
    print(v054._format_eval("FINAL sight-read", sight_final))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "input_dim": REAL_CHART_INPUT_DIM,
            "hidden_dim": args.hidden,
            "model_state": best_state,
            "chart": str(args.chart),
            "ranges": {
                "train_pool": [train_pool.start_s, train_pool.end_s],
                "train_window_s": args.train_window,
                "bootstrap_windows": [[w.start_s, w.end_s] for w in bootstrap_windows],
                "validation": [validation_window.start_s, validation_window.end_s],
                "sight": [sight_window.start_s, sight_window.end_s],
            },
            "multisegment_validation": {
                "finger_agnostic": True,
                "sight_used_for_selection": False,
                "bootstrap_history": bootstrap_history,
                "validation_reference": {
                    "hits": validation_reference.stats.hits,
                    "xacc": validation_reference.stats.x_accuracy_percent,
                    "overloaded": validation_reference.stats.overloaded,
                },
                "trust_alphas": list(v058.DEFAULT_TRUST_ALPHAS),
                "mixture_betas": list(v056.DEFAULT_MIXTURE_BETAS),
                "round_history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
