from __future__ import annotations

"""v1.2: allow non-regressive round proposals past the sampled train gate.

v1.1 requires every interpolation candidate to improve the currently sampled
30-second training window before validation/anchor preservation is even checked.
On larger multi-chart datasets this makes a single random window an optimization
target rather than a safety guard, and can reject globally useful DAgger steps as
"not better" before the anti-forgetting surfaces are evaluated.

v1.2 changes only round acceptance semantics:

* the sampled train window becomes a non-regression guard (safety, hits, XAcc,
  TooEarly) instead of a must-improve gate;
* validation and anchor guards remain unchanged;
* among candidates that pass every guard, select only an update whose aggregate
  guard-surface quality beats the current trusted policy;
* aggregate ranking preserves the v1.1 start-clean priority before completion
  and accuracy, so round updates cannot buy quality by reintroducing dirty starts.
"""

import sys

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090
import train_real_chart_v090_fast as v090_fast
import train_real_chart_v090_turbo as turbo
import train_real_chart_v100_start_micro as v100
import train_real_chart_v110_start_gate as v110


TRAINER_VERSION = "1.2.0-train-preserve"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v120_train_preserve.pt"
TRAIN_PRESERVE_VERSION = "v120-round-train-preserve-aggregate-v1"

_ORIGINAL_FAST_LINE_SEARCH = None
_INSTALLED = False
_AGGREGATE_KEY_FIELDS = (
    "safety",
    "start-clean",
    "start-early",
    "completion",
    "XAcc",
    "PP",
    "early",
    "miss",
)


def _train_preservation_guard(best, candidate):
    """Keep a round candidate if the sampled train window is non-regressive.

    This deliberately mirrors the existing validation-floor tolerances but does
    not require the candidate to outrank the current policy on this one window.
    Escaping overload remains immediately eligible; safe -> overload is always
    rejected.  If both policies are overloaded, the ordinary non-regression
    checks still apply instead of rejecting purely because both are overloaded.
    """

    best_failed = bool(best.stats.overloaded)
    candidate_failed = bool(candidate.stats.overloaded)
    if best_failed and not candidate_failed:
        return v080.v057.ConservativeDecision(True, "train escaped overload")
    if not best_failed and candidate_failed:
        return v080.v057.ConservativeDecision(False, "train safe->overload")

    hit_tolerance = max(
        int(v080.v062.DEFAULT_VALIDATION_HIT_DROP_ABSOLUTE),
        round(best.stats.targets * v080.v062.DEFAULT_VALIDATION_HIT_DROP_FRACTION),
    )
    if candidate.stats.hits < best.stats.hits - hit_tolerance:
        return v080.v057.ConservativeDecision(
            False, f"train hit regression>{hit_tolerance}"
        )
    if (
        candidate.stats.x_accuracy_percent
        < best.stats.x_accuracy_percent - v080.v062.DEFAULT_VALIDATION_XACC_DROP
    ):
        return v080.v057.ConservativeDecision(
            False,
            f"train XAcc regression>{v080.v062.DEFAULT_VALIDATION_XACC_DROP:g}pt",
        )
    if (
        candidate.stats.too_early_presses
        > best.stats.too_early_presses + v080.v062.DEFAULT_VALIDATION_EARLY_INCREASE
    ):
        return v080.v057.ConservativeDecision(
            False,
            f"train early regression>{v080.v062.DEFAULT_VALIDATION_EARLY_INCREASE}",
        )

    suffix = " overloaded" if best_failed and candidate_failed else ""
    return v080.v057.ConservativeDecision(True, f"train preserved{suffix}")


def _is_start_micro(named) -> bool:
    return str(getattr(named, "role", "")).startswith("start-micro-")


def _aggregate_key(train_eval, validation_evals, anchor_evals, anchor_segments) -> tuple:
    """Rank one policy over the full round guard surface.

    The ordering is intentionally completion-first after safety/start cleanliness.
    All candidates are measured on exactly the same segments, so summed hits and
    target-weighted accuracy provide a stable global objective instead of letting
    whichever 30-second train window happened to be sampled dominate selection.
    """

    validations = tuple(validation_evals)
    anchors = tuple(anchor_evals)
    anchor_segments = tuple(anchor_segments)
    if len(anchors) != len(anchor_segments):
        raise ValueError("anchor evaluation count must match anchor segment count")

    evaluations = (train_eval, *validations, *anchors)
    safe = all(not item.stats.overloaded for item in evaluations)
    start_early = sum(
        int(evaluation.stats.too_early_presses)
        for segment, evaluation in zip(anchor_segments, anchors)
        if _is_start_micro(segment)
    )
    hits = sum(int(item.stats.hits) for item in evaluations)
    targets = sum(int(item.stats.targets) for item in evaluations)
    weighted_x = sum(
        float(item.stats.x_accuracy_percent) * max(1, int(item.stats.targets))
        for item in evaluations
    ) / max(1, sum(max(1, int(item.stats.targets)) for item in evaluations))
    weighted_pp = sum(
        float(item.stats.perfect_rate) * max(1, int(item.stats.targets))
        for item in evaluations
    ) / max(1, sum(max(1, int(item.stats.targets)) for item in evaluations))
    early = sum(int(item.stats.too_early_presses) for item in evaluations)
    misses = sum(int(item.stats.misses) for item in evaluations)

    return (
        int(safe),
        int(start_early == 0),
        -int(start_early),
        hits / max(1, targets),
        weighted_x,
        weighted_pp,
        -early,
        -misses,
    )


def _aggregate_key_delta(base_key: tuple, candidate_key: tuple) -> dict:
    """Describe a candidate-vs-base aggregate comparison without new evaluation."""

    if len(base_key) != len(_AGGREGATE_KEY_FIELDS) or len(candidate_key) != len(base_key):
        raise ValueError("unexpected aggregate key shape")

    first = "tie"
    relation = "TIE"
    for index, field in enumerate(_AGGREGATE_KEY_FIELDS):
        base_value = base_key[index]
        candidate_value = candidate_key[index]
        if candidate_value == base_value:
            continue
        first = field
        relation = "WIN" if candidate_value > base_value else "LOSS"
        break

    # Expose human-direction deltas. start-early/early/miss are stored negated in
    # the ranking tuple, so positive numbers below mean the candidate has more.
    return {
        "first": first,
        "relation": relation,
        "safe": int(candidate_key[0]) - int(base_key[0]),
        "start_clean": int(candidate_key[1]) - int(base_key[1]),
        "start_early": (-int(candidate_key[2])) - (-int(base_key[2])),
        "completion_pp": (float(candidate_key[3]) - float(base_key[3])) * 100.0,
        "xacc_pt": float(candidate_key[4]) - float(base_key[4]),
        "pp_pt": float(candidate_key[5]) - float(base_key[5]),
        "early": (-int(candidate_key[6])) - (-int(base_key[6])),
        "miss": (-int(candidate_key[7])) - (-int(base_key[7])),
    }


def _format_aggregate_delta(alpha: float, base_key: tuple, candidate_key: tuple) -> str:
    delta = _aggregate_key_delta(base_key, candidate_key)
    return (
        f"a={float(alpha):g} first={delta['first']}:{delta['relation']} "
        f"dSafe={delta['safe']:+d} dStartClean={delta['start_clean']:+d} "
        f"dStartEarly={delta['start_early']:+d} "
        f"dCompletion={delta['completion_pp']:+.3f}pp "
        f"dX={delta['xacc_pt']:+.3f}pt dPP={delta['pp_pt']:+.3f}pt "
        f"dEarly={delta['early']:+d} dMiss={delta['miss']:+d}"
    )


def _complete_candidates(candidates, validation_count: int, anchor_count: int):
    return [
        candidate
        for candidate in candidates
        if candidate.train_decision.accepted
        and len(candidate.validation_decisions) == int(validation_count)
        and len(candidate.anchor_decisions) == int(anchor_count)
        and candidate.accepted
    ]


def _evaluate_base_surface(
    model,
    base_state,
    validation_segments,
    anchor_segments,
    *,
    same_hand: bool,
    control_dt_s: float,
):
    marker = 0.0
    combined = [*validation_segments, *anchor_segments]
    if not combined:
        return (), ()
    digest = v080.v065._state_digest(base_state)
    raw = v080_fast._evaluate_state_segment_groups(
        model,
        {marker: base_state},
        combined,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        digests={marker: digest},
    )
    ordered = [raw[(marker, segment.key)] for segment in combined]
    split = len(validation_segments)
    return tuple(ordered[:split]), tuple(ordered[split:])


def _train_preserve_line_search(
    model,
    *,
    base_state,
    proposal_state,
    base_train_eval,
    validation_references,
    anchor_references,
    train_segment,
    validation_segments,
    anchor_segments,
    same_hand: bool,
    control_dt_s: float,
    label: str,
):
    """Run the existing staged guards, then select by aggregate improvement."""

    assert _ORIGINAL_FAST_LINE_SEARCH is not None
    original_result = _ORIGINAL_FAST_LINE_SEARCH(
        model,
        base_state=base_state,
        proposal_state=proposal_state,
        base_train_eval=base_train_eval,
        validation_references=validation_references,
        anchor_references=anchor_references,
        train_segment=train_segment,
        validation_segments=validation_segments,
        anchor_segments=anchor_segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        label=label,
    )
    _legacy_chosen, candidates, _legacy_state = original_result

    complete = _complete_candidates(
        candidates,
        validation_count=len(validation_segments),
        anchor_count=len(anchor_segments),
    )
    if not complete:
        return original_result

    base_validations, base_anchors = _evaluate_base_surface(
        model,
        base_state,
        validation_segments,
        anchor_segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    base_key = _aggregate_key(
        base_train_eval,
        base_validations,
        base_anchors,
        anchor_segments,
    )

    improving = []
    compared = []
    for candidate in complete:
        key = _aggregate_key(
            candidate.train_eval,
            candidate.validation_evals,
            candidate.anchor_evals,
            anchor_segments,
        )
        compared.append((key, candidate))
        if key > base_key:
            improving.append((key, candidate))

    # Diagnostic-only: these lines reuse keys that selection already computed.
    # They trigger no extra gameplay evaluation and make a rollback explainable.
    for key, candidate in compared:
        print(
            f"{label}: aggregate-delta "
            + _format_aggregate_delta(candidate.alpha, base_key, key)
        )

    if not improving:
        model.load_state_dict(base_state)
        print(
            f"{label}: train-preserve aggregate ROLLBACK "
            f"complete={len(complete)} no-global-improvement"
        )
        return None, candidates, None

    # If multiple alphas produce the same discrete gameplay metrics, prefer the
    # smaller parameter move as the safer trust-region tie-breaker.
    _chosen_key, chosen = max(
        improving,
        key=lambda item: (*item[0], -float(item[1].alpha)),
    )
    chosen_state = v080.v058._interpolate_state(
        base_state,
        proposal_state,
        float(chosen.alpha),
    )
    model.load_state_dict(chosen_state)
    print(
        f"{label}: train-preserve aggregate ACCEPT a={chosen.alpha:g} "
        f"complete={len(complete)} improving={len(improving)}"
    )
    return chosen, candidates, chosen_state


def install_train_preserve_gate() -> None:
    """Install v1.2 round semantics on top of the currently active fast path."""

    global _ORIGINAL_FAST_LINE_SEARCH, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_FAST_LINE_SEARCH = v080_fast._fast_line_search
    v080.v063.v062.v061._safety_guard = _train_preservation_guard
    v080_fast._fast_line_search = _train_preserve_line_search
    # Non-modern entry points may already have installed v080._line_search.
    v080._line_search = _train_preserve_line_search
    _INSTALLED = True
    print(f"train-preserve-gate={TRAIN_PRESERVE_VERSION}")


def main() -> None:
    v090._configure_console_output()
    start_args, after_v100 = v100._consume_v100_args(sys.argv[1:])
    press_args, remaining = v090._consume_v090_args(after_v100)

    v090_fast._install_v090_fast_path(
        coef=press_args.press_persistence_coef,
        lookahead_frames=press_args.press_persistence_lookahead,
        commit_threshold=press_args.press_commit_threshold,
        hold_margin=press_args.press_hold_margin,
    )
    v100.install_start_micro(target_count=start_args.start_micro_targets)
    v080.TRAINER_VERSION = TRAINER_VERSION
    v080.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    if start_args.warm_start is not None:
        v100.install_warm_start(start_args.warm_start)
    v110.install_start_gate()
    turbo._install_turbo_path()
    install_train_preserve_gate()

    print("=== DMDOD v1.2.0 Train Preserve ===")
    print(
        f"train-preserve={TRAIN_PRESERVE_VERSION} | "
        "round train=non-regression | selection=aggregate guard-surface improvement"
    )
    print(
        f"start-gate={v110.START_GATE_VERSION} | start-micro targets={start_args.start_micro_targets} | "
        "aggregate priority=safety -> start-clean -> completion -> accuracy"
    )
    print(
        f"press-persistence coef={press_args.press_persistence_coef:g} "
        f"lookahead={press_args.press_persistence_lookahead}f "
        f"commit>={press_args.press_commit_threshold:+.2f} "
        f"hold>={press_args.press_hold_margin:+.2f}"
    )

    sys.argv = [sys.argv[0], *remaining]
    v080.main()
    print(
        f"turbo-stats: teacher-cache-hit={turbo._TEACHER_CACHE_HITS} "
        f"teacher-generated={turbo._TEACHER_CACHE_MISSES} "
        f"bootstrap-prunes={turbo._BOOTSTRAP_PRUNES}"
    )


if __name__ == "__main__":
    main()
