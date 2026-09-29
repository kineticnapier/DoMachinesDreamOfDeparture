from __future__ import annotations

"""v0.8.0: chart-level multi-chart training with human-visible HUD input.

Train, Validation, and Final are separated by chart, not by time slices from one
chart.  Training samples charts uniformly, keeps per-chart anti-forgetting
anchors, uses fixed validation windows for selection, and never evaluates Final
charts until finalization.
"""

import argparse
import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor
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
import train_real_chart_v064 as v064
import train_real_chart_v065 as v065
import train_real_chart_v070 as v070
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.flat_hud_eval import evaluate_hud_state_on_segment
from dmdod.multichart_dataset import DatasetChart, MultiChartDataset, discover_multichart_dataset
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.8.0-multichart-hud"
CHECKPOINT_FORMAT_VERSION = 15
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v080_multichart.pt"
DEFAULT_TRAIN_WINDOW_S = 30.0
DEFAULT_VALIDATION_WINDOW_S = 30.0
DEFAULT_ANCHORS_PER_CHART = 2
DEFAULT_ROUNDS = 48
DEFAULT_ROUND_EPOCHS = 12
DEFAULT_BOOTSTRAP_EPOCHS = 64
MAX_DEFAULT_WORKERS = 12


@dataclass(frozen=True, slots=True)
class ChartRuntime:
    spec: DatasetChart
    compiled: object

    @property
    def duration_s(self) -> float:
        return float(self.compiled.duration_s)


@dataclass(frozen=True, slots=True)
class NamedSegment:
    role: str
    chart_name: str
    chart_sha256: str
    start_s: float
    end_s: float
    segment: object

    @property
    def key(self) -> tuple:
        return (
            self.role,
            self.chart_sha256,
            round(float(self.start_s), 9),
            round(float(self.end_s), 9),
        )


@dataclass(frozen=True, slots=True)
class MultiCandidate:
    alpha: float
    train_eval: v054.StudentEvalResult
    train_decision: v057.ConservativeDecision
    validation_evals: tuple[v054.StudentEvalResult, ...]
    validation_decisions: tuple[v057.ConservativeDecision, ...]
    anchor_evals: tuple[v054.StudentEvalResult, ...]
    anchor_decisions: tuple[v057.ConservativeDecision, ...]

    @property
    def accepted(self) -> bool:
        return (
            self.train_decision.accepted
            and all(decision.accepted for decision in self.validation_decisions)
            and all(decision.accepted for decision in self.anchor_decisions)
        )


_EVAL_POOL: ProcessPoolExecutor | None = None
_EVAL_CACHE: dict[tuple, v054.StudentEvalResult] = {}
_EVAL_CACHE_HITS = 0
_EVAL_CACHE_MISSES = 0


def _configured_workers() -> int:
    raw = os.environ.get("DMDOD_EVAL_WORKERS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise SystemExit("DMDOD_EVAL_WORKERS must be a positive integer") from exc
    return min(MAX_DEFAULT_WORKERS, max(1, int(os.cpu_count() or 1)))


def _get_pool() -> ProcessPoolExecutor:
    global _EVAL_POOL
    if _EVAL_POOL is None:
        _EVAL_POOL = ProcessPoolExecutor(max_workers=_configured_workers())
    return _EVAL_POOL


def _shutdown_pool() -> None:
    global _EVAL_POOL
    if _EVAL_POOL is not None:
        _EVAL_POOL.shutdown(wait=True, cancel_futures=True)
        _EVAL_POOL = None


def _compile_role(items: tuple[DatasetChart, ...]) -> list[ChartRuntime]:
    return [ChartRuntime(item, load_compiled_adofai(item.resolved_path)) for item in items]


def _window(duration_s: float, start_s: float, length_s: float) -> tuple[float, float]:
    if duration_s <= 0.0:
        raise ValueError("chart duration must be positive")
    length = min(float(length_s), duration_s)
    start = min(max(0.0, float(start_s)), max(0.0, duration_s - length))
    return start, start + length


def _anchor_windows(duration_s: float, window_s: float, count: int) -> tuple[tuple[float, float], ...]:
    if count <= 0:
        raise ValueError("anchor count must be positive")
    length = min(window_s, duration_s)
    max_start = max(0.0, duration_s - length)
    if count == 1 or max_start <= 1e-12:
        return ((0.0, length),)
    starts = [max_start * index / (count - 1) for index in range(count)]
    result: list[tuple[float, float]] = []
    for start in starts:
        pair = _window(duration_s, start, length)
        if not result or abs(pair[0] - result[-1][0]) > 1e-9:
            result.append(pair)
    return tuple(result)


def _validation_window(duration_s: float, window_s: float) -> tuple[float, float]:
    length = min(window_s, duration_s)
    return _window(duration_s, (duration_s - length) * 0.5, length)


def _sample_window(rng: random.Random, duration_s: float, window_s: float) -> tuple[float, float]:
    length = min(window_s, duration_s)
    max_start = max(0.0, duration_s - length)
    start = 0.0 if max_start <= 1e-12 else rng.uniform(0.0, max_start)
    return _window(duration_s, start, length)


def _named_segment(runtime: ChartRuntime, role: str, start_s: float, end_s: float) -> NamedSegment:
    segment = build_playable_segment(runtime.compiled, start_s=start_s, end_s=end_s)
    if not segment.targets:
        raise ValueError(
            f"segment contains no playable targets: {runtime.spec.name} {start_s:.3f}..{end_s:.3f}s"
        )
    return NamedSegment(
        role=role,
        chart_name=runtime.spec.name,
        chart_sha256=runtime.spec.content_sha256,
        start_s=start_s,
        end_s=end_s,
        segment=segment,
    )


def _build_anchor_segments(
    train_charts: list[ChartRuntime],
    *,
    window_s: float,
    anchors_per_chart: int,
) -> list[NamedSegment]:
    result: list[NamedSegment] = []
    for chart in train_charts:
        for index, (start, end) in enumerate(
            _anchor_windows(chart.duration_s, window_s, anchors_per_chart), 1
        ):
            result.append(_named_segment(chart, f"anchor-{index}", start, end))
    return result


def _build_validation_segments(
    validation_charts: list[ChartRuntime], *, window_s: float
) -> list[NamedSegment]:
    result: list[NamedSegment] = []
    for chart in validation_charts:
        start, end = _validation_window(chart.duration_s, window_s)
        result.append(_named_segment(chart, "validation", start, end))
    return result


def _full_segments(charts: list[ChartRuntime], role: str) -> list[NamedSegment]:
    return [_named_segment(chart, role, 0.0, chart.duration_s) for chart in charts]


def _eval_cache_key(state_digest: str, named: NamedSegment) -> tuple:
    return state_digest, named.key


def _evaluate_states_on_segments(
    model: RecurrentActorCritic,
    states: dict[float, dict[str, torch.Tensor]],
    segments: list[NamedSegment],
    *,
    same_hand: bool,
    control_dt_s: float,
) -> dict[tuple[float, tuple], v054.StudentEvalResult]:
    global _EVAL_CACHE_HITS, _EVAL_CACHE_MISSES
    if not segments:
        return {}

    result: dict[tuple[float, tuple], v054.StudentEvalResult] = {}
    pending = {}
    digests = {alpha: v065._state_digest(state) for alpha, state in states.items()}

    for alpha, state in states.items():
        digest = digests[alpha]
        for named in segments:
            cache_key = _eval_cache_key(digest, named)
            cached = _EVAL_CACHE.get(cache_key)
            if cached is not None:
                result[(alpha, named.key)] = cached
                _EVAL_CACHE_HITS += 1
                continue
            pending[(alpha, named.key)] = (
                state,
                named,
                cache_key,
            )

    if pending:
        if _configured_workers() <= 1:
            for key, (state, named, cache_key) in pending.items():
                raw = evaluate_hud_state_on_segment(
                    state,
                    int(model.hidden_dim),
                    named.segment,
                    bool(same_hand),
                    float(control_dt_s),
                )
                evaluated = v054.StudentEvalResult(raw[0], raw[1])
                result[key] = evaluated
                _EVAL_CACHE[cache_key] = evaluated
                _EVAL_CACHE_MISSES += 1
        else:
            pool = _get_pool()
            futures = {
                key: (
                    pool.submit(
                        evaluate_hud_state_on_segment,
                        state,
                        int(model.hidden_dim),
                        named.segment,
                        bool(same_hand),
                        float(control_dt_s),
                    ),
                    cache_key,
                )
                for key, (state, named, cache_key) in pending.items()
            }
            for key, (future, cache_key) in futures.items():
                raw = future.result()
                evaluated = v054.StudentEvalResult(raw[0], raw[1])
                result[key] = evaluated
                _EVAL_CACHE[cache_key] = evaluated
                _EVAL_CACHE_MISSES += 1

    return result


def _evaluate_one_state_many(
    model: RecurrentActorCritic,
    state: dict[str, torch.Tensor],
    segments: list[NamedSegment],
    *,
    same_hand: bool,
    control_dt_s: float,
) -> list[v054.StudentEvalResult]:
    marker = 0.0
    raw = _evaluate_states_on_segments(
        model,
        {marker: state},
        segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    return [raw[(marker, segment.key)] for segment in segments]


def _bootstrap_key(
    anchors: list[v054.StudentEvalResult],
    validations: list[v054.StudentEvalResult],
) -> tuple:
    evaluations = [*anchors, *validations]
    safe = all(not item.stats.overloaded for item in evaluations)
    hits = sum(item.stats.hits for item in evaluations)
    targets = sum(item.stats.targets for item in evaluations)
    mean_x = sum(item.stats.x_accuracy_percent for item in evaluations) / max(1, len(evaluations))
    mean_pp = sum(item.stats.perfect_rate for item in evaluations) / max(1, len(evaluations))
    early = sum(item.stats.too_early_presses for item in evaluations)
    return int(safe), hits / max(1, targets), mean_x, mean_pp, -early


def _bootstrap(
    model: RecurrentActorCritic,
    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    anchor_segments: list[NamedSegment],
    validation_segments: list[NamedSegment],
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
    same_hand: bool,
    control_dt_s: float,
) -> tuple[
    dict[str, torch.Tensor],
    list[v054.StudentEvalResult],
    list[v054.StudentEvalResult],
    list[dict],
]:
    optimizer = torch.optim.Adam(v057._policy_parameters(model), lr=learning_rate)

    state = copy.deepcopy(model.state_dict())
    anchor_evals = _evaluate_one_state_many(
        model, state, anchor_segments, same_hand=same_hand, control_dt_s=control_dt_s
    )
    validation_evals = _evaluate_one_state_many(
        model, state, validation_segments, same_hand=same_hand, control_dt_s=control_dt_s
    )
    best_state = state
    best_anchors = anchor_evals
    best_validations = validation_evals
    best_key = _bootstrap_key(anchor_evals, validation_evals)
    best_epoch = 0
    history: list[dict] = []
    print(
        f"bootstrap 00: safe={bool(best_key[0])} completion={best_key[1] * 100.0:.1f}% "
        f"meanX={best_key[2]:.1f}%"
    )

    for epoch in range(1, epochs + 1):
        loss = v062._bootstrap_multi_epoch(
            model,
            expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=not bool(epoch & 1),
        )
        state = copy.deepcopy(model.state_dict())
        anchor_evals = _evaluate_one_state_many(
            model, state, anchor_segments, same_hand=same_hand, control_dt_s=control_dt_s
        )
        validation_evals = _evaluate_one_state_many(
            model, state, validation_segments, same_hand=same_hand, control_dt_s=control_dt_s
        )
        key = _bootstrap_key(anchor_evals, validation_evals)
        keep = key > best_key
        if keep:
            best_key = key
            best_state = state
            best_anchors = anchor_evals
            best_validations = validation_evals
            best_epoch = epoch
        if keep or epoch == 1 or epoch == epochs or epoch % 4 == 0:
            print(
                f"bootstrap {epoch:02d}: loss={loss:.6f} safe={bool(key[0])} "
                f"completion={key[1] * 100.0:.1f}% meanX={key[2]:.1f}%"
                + (" KEEP" if keep else "")
            )
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "safe": bool(key[0]),
                "completion": key[1],
                "mean_xacc": key[2],
                "kept": keep,
            }
        )

    model.load_state_dict(best_state)
    print(
        f"bootstrap selected: epoch={best_epoch} safe={bool(best_key[0])} "
        f"completion={best_key[1] * 100.0:.1f}% meanX={best_key[2]:.1f}%"
    )
    return best_state, best_anchors, best_validations, history


def _aggregate_summary(results: list[v054.StudentEvalResult]) -> str:
    hits, targets, mean_xacc, overloaded = v063._summary_metrics(results)
    return f"H{hits}/{targets} meanX{mean_xacc:.1f}% over={overloaded}"


def _mean_x(results: tuple[v054.StudentEvalResult, ...]) -> float:
    return sum(item.stats.x_accuracy_percent for item in results) / max(1, len(results))


def _candidate_brief(candidate: MultiCandidate) -> str:
    train = candidate.train_eval.stats
    vp = sum(decision.accepted for decision in candidate.validation_decisions)
    ap = sum(decision.accepted for decision in candidate.anchor_decisions)
    mark = "+" if candidate.accepted else "-"
    return (
        f"{candidate.alpha:g}{mark} T{train.hits}/{train.targets} X{train.x_accuracy_percent:.1f} "
        f"V{vp}/{len(candidate.validation_decisions)} X{_mean_x(candidate.validation_evals):.1f} "
        f"A{ap}/{len(candidate.anchor_decisions)}"
    )


def _line_search(
    model: RecurrentActorCritic,
    *,
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    base_train_eval: v054.StudentEvalResult,
    validation_references: tuple[v054.StudentEvalResult, ...],
    anchor_references: tuple[v054.StudentEvalResult, ...],
    train_segment: NamedSegment,
    validation_segments: list[NamedSegment],
    anchor_segments: list[NamedSegment],
    same_hand: bool,
    control_dt_s: float,
    label: str,
) -> tuple[MultiCandidate | None, list[MultiCandidate], dict[str, torch.Tensor] | None]:
    states = {
        float(alpha): v058._interpolate_state(base_state, proposal_state, float(alpha))
        for alpha in v058.DEFAULT_TRUST_ALPHAS
    }

    # Stage 1: current training window. Failed candidates cannot become accepted
    # later, so do not spend validation/anchor simulation on them.
    train_raw = _evaluate_states_on_segments(
        model,
        states,
        [train_segment],
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    train_evals = {
        alpha: train_raw[(alpha, train_segment.key)]
        for alpha in states
    }
    train_decisions = {
        alpha: v063.v062.v061._safety_guard(base_train_eval, train_evals[alpha])
        for alpha in states
    }
    validation_survivors = [alpha for alpha in states if train_decisions[alpha].accepted]

    validation_evals_by_alpha: dict[float, tuple[v054.StudentEvalResult, ...]] = {}
    validation_decisions_by_alpha: dict[float, tuple[v057.ConservativeDecision, ...]] = {}
    anchor_survivors: list[float] = []
    if validation_survivors:
        val_raw = _evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in validation_survivors},
            validation_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        for alpha in validation_survivors:
            evaluations = tuple(val_raw[(alpha, segment.key)] for segment in validation_segments)
            decisions = tuple(
                v062._validation_guard(reference, evaluation)
                for reference, evaluation in zip(validation_references, evaluations)
            )
            validation_evals_by_alpha[alpha] = evaluations
            validation_decisions_by_alpha[alpha] = decisions
            if all(decision.accepted for decision in decisions):
                anchor_survivors.append(alpha)

    anchor_evals_by_alpha: dict[float, tuple[v054.StudentEvalResult, ...]] = {}
    anchor_decisions_by_alpha: dict[float, tuple[v057.ConservativeDecision, ...]] = {}
    if anchor_survivors:
        anchor_raw = _evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in anchor_survivors},
            anchor_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        for alpha in anchor_survivors:
            evaluations = tuple(anchor_raw[(alpha, segment.key)] for segment in anchor_segments)
            decisions = tuple(
                v064._anchor_guard(reference, evaluation)
                for reference, evaluation in zip(anchor_references, evaluations)
            )
            anchor_evals_by_alpha[alpha] = evaluations
            anchor_decisions_by_alpha[alpha] = decisions

    candidates: list[MultiCandidate] = []
    for alpha in states:
        validations = validation_evals_by_alpha.get(alpha, ())
        validation_decisions = validation_decisions_by_alpha.get(alpha, ())
        anchors = anchor_evals_by_alpha.get(alpha, ())
        anchor_decisions = anchor_decisions_by_alpha.get(alpha, ())
        candidate = MultiCandidate(
            alpha=alpha,
            train_eval=train_evals[alpha],
            train_decision=train_decisions[alpha],
            validation_evals=validations,
            validation_decisions=validation_decisions,
            anchor_evals=anchors,
            anchor_decisions=anchor_decisions,
        )
        candidates.append(candidate)

    complete = [
        candidate
        for candidate in candidates
        if candidate.train_decision.accepted
        and len(candidate.validation_decisions) == len(validation_segments)
        and len(candidate.anchor_decisions) == len(anchor_segments)
        and candidate.accepted
    ]
    print(f"{label}: " + " | ".join(_candidate_brief(candidate) for candidate in candidates))

    if not complete:
        model.load_state_dict(base_state)
        return None, candidates, None

    chosen = max(
        complete,
        key=lambda item: v064.v061_candidate_key(base_train_eval, item.train_eval),
    )
    chosen_state = states[chosen.alpha]
    model.load_state_dict(chosen_state)
    return chosen, candidates, chosen_state


def _signature(args, dataset: MultiChartDataset) -> dict:
    def role(items: tuple[DatasetChart, ...]) -> list[dict[str, str]]:
        return [{"name": item.name, "sha256": item.content_sha256} for item in items]

    return {
        "trainer": TRAINER_VERSION,
        "input_dim": v070.HUD_REAL_CHART_INPUT_DIM,
        "train": role(dataset.train),
        "validation": role(dataset.validation),
        "final": role(dataset.final),
        "train_window": float(args.train_window),
        "validation_window": float(args.validation_window),
        "anchors_per_chart": int(args.anchors_per_chart),
        "round_epochs": int(args.round_epochs),
        "hidden": int(args.hidden),
        "lr": float(args.lr),
        "bootstrap_lr": float(args.bootstrap_lr),
        "chunk_steps": int(args.chunk_steps),
        "control_dt": float(args.control_dt),
        "seed": int(args.seed),
        "press_recovery_cap": int(args.press_recovery_cap),
        "same_hand": not bool(args.cross_hand),
    }


def _save_checkpoint(
    path: Path,
    *,
    model_state,
    args,
    dataset: MultiChartDataset,
    signature: dict,
    completed_round: int,
    rng_state,
    validation_references: tuple[v054.StudentEvalResult, ...],
    anchor_references: tuple[v054.StudentEvalResult, ...],
    bootstrap_history: list[dict],
    round_history: list[dict],
    finalized: bool,
    final_metrics: dict | None = None,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_version": TRAINER_VERSION,
        "input_dim": v070.HUD_REAL_CHART_INPUT_DIM,
        "hidden_dim": int(args.hidden),
        "model_state": model_state,
        "dataset_root": dataset.root,
        "dataset_signature": dataset.signature(),
        "signature": signature,
        "completed_round": int(completed_round),
        "rng_state": rng_state,
        "validation_references": [v063._eval_to_payload(item) for item in validation_references],
        "anchor_references": [v063._eval_to_payload(item) for item in anchor_references],
        "bootstrap_history": bootstrap_history,
        "round_history": round_history,
        "requested_rounds": int(args.rounds),
        "finalized": bool(finalized),
        "final_metrics": final_metrics,
        "final_used_for_selection": False,
        "hud_observation": v070.HUD_OBSERVATION_VERSION,
    }
    v063._atomic_torch_save(payload, path)
    if args.keep_round_checkpoints and completed_round > 0 and not finalized:
        v063._atomic_torch_save(payload, v063._round_snapshot_path(path, completed_round))


def _load_checkpoint(
    model: RecurrentActorCritic,
    path: Path,
    *,
    signature: dict,
    validation_count: int,
    anchor_count: int,
    device: torch.device,
):
    payload = torch.load(path, map_location=device)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise SystemExit("v0.8.0 --resume requires a format-15 multi-chart checkpoint")
    if int(payload.get("input_dim", -1)) != v070.HUD_REAL_CHART_INPUT_DIM:
        raise SystemExit("checkpoint input dimension does not match v0.8.0 HUD encoder")
    if int(payload.get("hidden_dim", -1)) != model.hidden_dim:
        raise SystemExit("checkpoint hidden size does not match --hidden")
    if payload.get("signature") != signature:
        raise SystemExit("resume configuration or dataset content differs from checkpoint")
    validation_payload = payload.get("validation_references")
    anchor_payload = payload.get("anchor_references")
    if not isinstance(validation_payload, list) or len(validation_payload) != validation_count:
        raise SystemExit("checkpoint validation reference count mismatch")
    if not isinstance(anchor_payload, list) or len(anchor_payload) != anchor_count:
        raise SystemExit("checkpoint anchor reference count mismatch")
    model.load_state_dict(payload["model_state"])
    validations = tuple(v063._eval_from_payload(item) for item in validation_payload)
    anchors = tuple(v063._eval_from_payload(item) for item in anchor_payload)
    return payload, validations, anchors


def _print_dataset(dataset: MultiChartDataset) -> None:
    print(
        f"dataset: train={len(dataset.train)} validation={len(dataset.validation)} "
        f"final={len(dataset.final)}"
    )
    for role, items in (
        ("TRAIN", dataset.train),
        ("VALID", dataset.validation),
        ("FINAL", dataset.final),
    ):
        print(f"{role}: " + " | ".join(item.name for item in items))


def _eval_payload(result: v054.StudentEvalResult) -> dict:
    return v063._eval_to_payload(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DMDOD v0.8.0 multi-chart HUD training. Dataset root must contain "
            "Train/Validation/Final, or be a zip bundle with those folders."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("--train-window", type=float, default=DEFAULT_TRAIN_WINDOW_S)
    parser.add_argument("--validation-window", type=float, default=DEFAULT_VALIDATION_WINDOW_S)
    parser.add_argument("--anchors-per-chart", type=int, default=DEFAULT_ANCHORS_PER_CHART)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--round-epochs", type=int, default=DEFAULT_ROUND_EPOCHS)
    parser.add_argument("--bootstrap-epochs", type=int, default=DEFAULT_BOOTSTRAP_EPOCHS)
    parser.add_argument("--hidden", type=int, default=128)
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
    args = parser.parse_args()

    if args.train_window <= 0 or args.validation_window <= 0:
        raise SystemExit("train/validation windows must be positive")
    if args.anchors_per_chart <= 0 or args.rounds <= 0 or args.round_epochs <= 0:
        raise SystemExit("anchors/round counts must be positive")
    if args.bootstrap_epochs <= 0 or args.hidden <= 0 or args.chunk_steps <= 0:
        raise SystemExit("bootstrap-epochs/hidden/chunk-steps must be positive")

    # Install the proven v0.7.0 245D HUD observation and exact repeated-BC cache.
    v070._install_v070()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand

    try:
        dataset = discover_multichart_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _print_dataset(dataset)

    train_charts = _compile_role(dataset.train)
    validation_charts = _compile_role(dataset.validation)
    final_charts = _compile_role(dataset.final)
    anchor_segments = _build_anchor_segments(
        train_charts,
        window_s=args.train_window,
        anchors_per_chart=args.anchors_per_chart,
    )
    validation_segments = _build_validation_segments(
        validation_charts,
        window_s=args.validation_window,
    )
    calibration = calibrate_single_press_lead(control_dt_s=args.control_dt, same_hand=same_hand)

    print("=== DMDOD v0.8.0 Multi-Chart HUD ===")
    print(
        f"input={v070.HUD_REAL_CHART_INPUT_DIM}D hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms"
    )
    print(
        f"train-window={args.train_window:g}s anchors={len(anchor_segments)} "
        f"validation-windows={len(validation_segments)} rounds={args.rounds}x{args.round_epochs}"
    )
    print(
        f"flat-eval={_configured_workers()} workers | staged guard=train -> validation -> anchors | "
        "FINAL charts are not evaluated during selection"
    )

    expert_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    expert_stable: list[v056.StableSequence] = []
    for index, named in enumerate(anchor_segments, 1):
        x, y, teacher_eval = v060._collect_expert_sequence(
            named.segment,
            lead_s=calibration.lead_s,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
        )
        expert_pairs.append((x, y))
        expert_stable.append(
            v056._make_stable_sequence(
                v060.v055.DAggerSequence(x, y, f"mc-anchor-{index}-{named.chart_name}"),
                press_recovery_cap=args.press_recovery_cap,
                expert=True,
            )
        )
        print(
            f"teacher {index:02d} {named.chart_name} {named.start_s:.1f}..{named.end_s:.1f}s "
            f"H={teacher_eval.stats.hits}/{teacher_eval.stats.targets} "
            f"X={teacher_eval.stats.x_accuracy_percent:.1f}% over={teacher_eval.stats.overloaded}"
        )

    model = RecurrentActorCritic(
        input_dim=v070.HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    signature = _signature(args, dataset)
    rng = random.Random(args.seed * 1000003 + 80)

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        payload, validation_references, anchor_references = _load_checkpoint(
            model,
            checkpoint_path,
            signature=signature,
            validation_count=len(validation_segments),
            anchor_count=len(anchor_segments),
            device=device,
        )
        best_state = copy.deepcopy(model.state_dict())
        completed_round = int(payload.get("completed_round", 0))
        if completed_round > args.rounds:
            raise SystemExit("checkpoint has already completed more rounds than requested")
        rng.setstate(payload["rng_state"])
        bootstrap_history = list(payload.get("bootstrap_history", []))
        round_history = list(payload.get("round_history", []))
        print(
            f"resume={checkpoint_path} completed-round={completed_round}/{args.rounds} "
            f"V={_aggregate_summary(list(validation_references))} "
            f"A={_aggregate_summary(list(anchor_references))}"
        )
        if payload.get("finalized") and completed_round < args.rounds:
            print("warning: extending a finalized run means its previous Final results are no longer pristine")
    else:
        print(
            f"bootstrap=fresh multi-chart BC epochs={args.bootstrap_epochs} "
            f"trajectories={len(expert_pairs)}"
        )
        best_state, bootstrap_anchors, bootstrap_validations, bootstrap_history = _bootstrap(
            model,
            expert_pairs,
            anchor_segments,
            validation_segments,
            epochs=args.bootstrap_epochs,
            learning_rate=args.bootstrap_lr,
            chunk_steps=args.chunk_steps,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
        )
        anchor_references = tuple(bootstrap_anchors)
        validation_references = tuple(bootstrap_validations)
        completed_round = 0
        round_history: list[dict] = []
        _save_checkpoint(
            checkpoint_path,
            model_state=best_state,
            args=args,
            dataset=dataset,
            signature=signature,
            completed_round=0,
            rng_state=rng.getstate(),
            validation_references=validation_references,
            anchor_references=anchor_references,
            bootstrap_history=bootstrap_history,
            round_history=round_history,
            finalized=False,
        )
        print(f"checkpoint bootstrap: {checkpoint_path}")

    print(
        f"floors: V={_aggregate_summary(list(validation_references))} | "
        f"A={_aggregate_summary(list(anchor_references))}"
    )

    try:
        for round_index in range(completed_round + 1, args.rounds + 1):
            chart = rng.choice(train_charts)  # chart-uniform, not note-count weighted
            start_s, end_s = _sample_window(rng, chart.duration_s, args.train_window)
            train_named = _named_segment(chart, "round-train", start_s, end_s)

            model.load_state_dict(best_state)
            base_train_eval = v054._evaluate_student(
                model,
                train_named.segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
            beta = v056._mixture_beta(round_index)
            current_x, current_y, _ = v060._collect_expert_sequence(
                train_named.segment,
                lead_s=calibration.lead_s,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
            current_expert = v056._make_stable_sequence(
                v060.v055.DAggerSequence(
                    current_x, current_y, f"mc-round-{round_index}-expert-{chart.spec.name}"
                ),
                press_recovery_cap=args.press_recovery_cap,
                expert=True,
            )
            rollout = v060._collect_mixture_rollout(
                model,
                train_named.segment,
                lead_s=calibration.lead_s,
                teacher_fraction=beta,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                source=f"mc-round-{round_index}-{chart.spec.name}",
                seed=args.seed * 1000 + round_index,
                press_recovery_cap=args.press_recovery_cap,
            )
            print(
                f"round {round_index:03d} chart={chart.spec.name} "
                f"train={start_s:.2f}..{end_s:.2f}s "
                f"base=H{base_train_eval.stats.hits}/{base_train_eval.stats.targets} "
                f"X{base_train_eval.stats.x_accuracy_percent:.1f}% | "
                f"mix{beta * 100:.0f}=H{rollout.evaluation.stats.hits}/{rollout.evaluation.stats.targets} "
                f"X{rollout.evaluation.stats.x_accuracy_percent:.1f}%"
            )

            sequences = [*expert_stable, current_expert, rollout.sequence]
            accepted_epochs = 0
            epoch_history: list[dict] = []
            rejected_parities: dict[str, set[bool]] = {}

            for epoch_index in range(1, args.round_epochs + 1):
                model.load_state_dict(best_state)
                base_state = copy.deepcopy(best_state)
                base_digest = v065._state_digest(base_state)
                base_train_eval = v054._evaluate_student(
                    model,
                    train_named.segment,
                    same_hand=same_hand,
                    control_dt_s=args.control_dt,
                    device=device,
                )
                reverse_order = not bool(epoch_index & 1)
                optimizer = v057._new_optimizer(model, args.lr)
                loss = v057._train_one_epoch(
                    model,
                    sequences,
                    optimizer=optimizer,
                    chunk_steps=args.chunk_steps,
                    reverse_order=reverse_order,
                )
                proposal_state = copy.deepcopy(model.state_dict())
                chosen, candidates, chosen_state = _line_search(
                    model,
                    base_state=base_state,
                    proposal_state=proposal_state,
                    base_train_eval=base_train_eval,
                    validation_references=validation_references,
                    anchor_references=anchor_references,
                    train_segment=train_named,
                    validation_segments=validation_segments,
                    anchor_segments=anchor_segments,
                    same_hand=same_hand,
                    control_dt_s=args.control_dt,
                    label=f"round {round_index:03d} e{epoch_index:02d}",
                )

                if chosen is not None and chosen_state is not None:
                    accepted_epochs += 1
                    best_state = copy.deepcopy(chosen_state)
                    model.load_state_dict(best_state)
                    validation_references = tuple(
                        v064._update_reference(reference, evaluation)
                        for reference, evaluation in zip(
                            validation_references, chosen.validation_evals
                        )
                    )
                    anchor_references = tuple(
                        v064._update_reference(reference, evaluation)
                        for reference, evaluation in zip(anchor_references, chosen.anchor_evals)
                    )
                    rejected_parities.clear()
                    print(
                        f"round {round_index:03d} e{epoch_index:02d}: ACCEPT a={chosen.alpha:g} "
                        f"loss={loss:.4f} T=H{chosen.train_eval.stats.hits}/{chosen.train_eval.stats.targets} "
                        f"X{chosen.train_eval.stats.x_accuracy_percent:.1f}% "
                        f"V={_aggregate_summary(list(chosen.validation_evals))} "
                        f"A={_aggregate_summary(list(chosen.anchor_evals))}"
                    )
                else:
                    model.load_state_dict(best_state)
                    rejected_parities.setdefault(base_digest, set()).add(reverse_order)
                    print(f"round {round_index:03d} e{epoch_index:02d}: ROLLBACK loss={loss:.4f}")

                epoch_history.append(
                    {
                        "epoch": epoch_index,
                        "loss": loss,
                        "accepted": chosen is not None,
                        "alpha": None if chosen is None else chosen.alpha,
                    }
                )

                # Fresh Adam + two deterministic sequence orders means that once
                # both parities are rejected from the exact same trusted state,
                # every remaining epoch in this round is an exact repeat.
                if len(rejected_parities.get(v065._state_digest(best_state), set())) >= 2:
                    print(
                        f"round {round_index:03d}: exact odd/even proposal fixed point; "
                        f"skip remaining {args.round_epochs - epoch_index} epochs"
                    )
                    break

            model.load_state_dict(best_state)
            final_train = v054._evaluate_student(
                model,
                train_named.segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
            )
            current_validations = _evaluate_one_state_many(
                model,
                best_state,
                validation_segments,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
            )
            current_anchors = _evaluate_one_state_many(
                model,
                best_state,
                anchor_segments,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
            )
            print(
                f"round {round_index:03d} summary: accepted={accepted_epochs} "
                f"T=H{final_train.stats.hits}/{final_train.stats.targets} "
                f"X{final_train.stats.x_accuracy_percent:.1f}% | "
                f"V={_aggregate_summary(current_validations)} | "
                f"A={_aggregate_summary(current_anchors)}"
            )
            round_history.append(
                {
                    "round": round_index,
                    "chart": chart.spec.name,
                    "chart_sha256": chart.spec.content_sha256,
                    "train_start": start_s,
                    "train_end": end_s,
                    "teacher_fraction": beta,
                    "accepted_epochs": accepted_epochs,
                    "epochs": epoch_history,
                }
            )
            completed_round = round_index
            _save_checkpoint(
                checkpoint_path,
                model_state=best_state,
                args=args,
                dataset=dataset,
                signature=signature,
                completed_round=completed_round,
                rng_state=rng.getstate(),
                validation_references=validation_references,
                anchor_references=anchor_references,
                bootstrap_history=bootstrap_history,
                round_history=round_history,
                finalized=False,
            )
            print(f"checkpoint round {round_index:03d}: {checkpoint_path}")

        # Final charts are materialized/evaluated only here. They never enter
        # bootstrap, DAgger data, candidate guards, or reference floors.
        model.load_state_dict(best_state)
        anchor_final = _evaluate_one_state_many(
            model,
            best_state,
            anchor_segments,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
        )
        validation_full_segments = _full_segments(validation_charts, "validation-full")
        final_full_segments = _full_segments(final_charts, "final-holdout")
        validation_final = _evaluate_one_state_many(
            model,
            best_state,
            validation_full_segments,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
        )
        final_holdout = _evaluate_one_state_many(
            model,
            best_state,
            final_full_segments,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
        )

        print(f"final train-anchors: {_aggregate_summary(anchor_final)}")
        for named, result in zip(validation_full_segments, validation_final):
            print(v054._format_eval(f"final validation [{named.chart_name}]", result))
        for named, result in zip(final_full_segments, final_holdout):
            print(v054._format_eval(f"FINAL holdout [{named.chart_name}]", result))
        print(f"final validation aggregate: {_aggregate_summary(validation_final)}")
        print(f"FINAL holdout aggregate: {_aggregate_summary(final_holdout)}")

        final_metrics = {
            "train_anchors": [_eval_payload(item) for item in anchor_final],
            "validation_full": [
                {"chart": named.chart_name, "eval": _eval_payload(result)}
                for named, result in zip(validation_full_segments, validation_final)
            ],
            "final_holdout": [
                {"chart": named.chart_name, "eval": _eval_payload(result)}
                for named, result in zip(final_full_segments, final_holdout)
            ],
        }
        _save_checkpoint(
            checkpoint_path,
            model_state=best_state,
            args=args,
            dataset=dataset,
            signature=signature,
            completed_round=completed_round,
            rng_state=rng.getstate(),
            validation_references=validation_references,
            anchor_references=anchor_references,
            bootstrap_history=bootstrap_history,
            round_history=round_history,
            finalized=True,
            final_metrics=final_metrics,
        )
        print(f"checkpoint final: {checkpoint_path}")
        print(
            f"eval-cache: hits={_EVAL_CACHE_HITS} misses={_EVAL_CACHE_MISSES} | "
            f"bc-proposal-cache: hits={v070._TRAIN_PROPOSAL_CACHE_HITS} "
            f"misses={v070._TRAIN_PROPOSAL_CACHE_MISSES}"
        )
    finally:
        _shutdown_pool()
        v070._shutdown_eval_pool()


if __name__ == "__main__":
    main()
