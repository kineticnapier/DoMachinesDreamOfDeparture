from __future__ import annotations

"""v1.3: move Validation/Anchor XAcc preservation to role aggregates.

v1.2 proved that the sampled-train must-improve gate was one fixed point.  Once
that was removed, the dominant rejection became the per-segment 2-point XAcc
floor across 16 Validation windows and ~190 training anchors.  With hundreds of
independent windows, a globally useful proposal is very likely to regress at
least one window by >2 points even when the role as a whole is preserved.

v1.3 changes only the XAcc part of the Validation/Anchor guard:

* overload remains a per-segment hard guard;
* hit regression remains a per-segment hard guard;
* TooEarly regression remains a per-segment hard guard;
* XAcc is checked once over the complete Validation role and once over the
  complete Anchor role, target-weighted, with the same 2-point tolerance;
* segments where the reference is overloaded and the candidate escapes overload
  are excluded from the aggregate XAcc comparison, matching the legacy guard's
  immediate safety-escape semantics.

Adaptive worker-wave scheduling is retained.  Validation aggregate XAcc is
checked before any anchors are evaluated, so candidates that fail the role-level
Validation floor still short-circuit early.
"""

import sys
import time

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090
import train_real_chart_v090_fast as v090_fast
import train_real_chart_v090_turbo as turbo
import train_real_chart_v100_start_micro as v100
import train_real_chart_v110_rejection_telemetry as telemetry
import train_real_chart_v110_start_gate as v110
import train_real_chart_v120_guard_wave_accel as guard_wave
import train_real_chart_v120_train_preserve as v120


TRAINER_VERSION = "1.3.0-aggregate-xacc"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v130_aggregate_xacc.pt"
AGGREGATE_XACC_VERSION = "v130-role-aggregate-xacc-v1"

_INSTALLED = False


def _validation_guard_without_xacc(reference, candidate):
    """Legacy validation floor with only the per-segment XAcc check removed."""

    ref_over = bool(reference.stats.overloaded)
    cand_over = bool(candidate.stats.overloaded)
    if ref_over and not cand_over:
        return v080.v057.ConservativeDecision(True, "validation escaped overload")
    if not ref_over and cand_over:
        return v080.v057.ConservativeDecision(False, "validation safe->overload")

    hit_tolerance = max(
        int(v080.v062.DEFAULT_VALIDATION_HIT_DROP_ABSOLUTE),
        round(
            int(reference.stats.targets)
            * float(v080.v062.DEFAULT_VALIDATION_HIT_DROP_FRACTION)
        ),
    )
    if int(candidate.stats.hits) < int(reference.stats.hits) - hit_tolerance:
        return v080.v057.ConservativeDecision(
            False, f"validation hit regression>{hit_tolerance}"
        )
    if (
        int(candidate.stats.too_early_presses)
        > int(reference.stats.too_early_presses)
        + int(v080.v062.DEFAULT_VALIDATION_EARLY_INCREASE)
    ):
        return v080.v057.ConservativeDecision(
            False,
            f"validation early regression>{v080.v062.DEFAULT_VALIDATION_EARLY_INCREASE}",
        )

    if ref_over and cand_over:
        return v080.v057.ConservativeDecision(True, "validation overloaded non-regression")
    return v080.v057.ConservativeDecision(True, "validation preserved; XAcc deferred")


def _anchor_guard_without_xacc(reference, candidate):
    decision = _validation_guard_without_xacc(reference, candidate)
    return v080.v057.ConservativeDecision(
        decision.accepted,
        decision.reason.replace("validation", "anchor", 1),
    )


def _weighted_xacc_pairs(references, candidates) -> tuple[float, float, int]:
    """Return target-weighted reference/candidate XAcc over comparable pairs.

    A reference-overloaded -> candidate-safe pair is omitted because the legacy
    per-segment guard accepted that safety escape immediately, without applying
    its hit/XAcc/TooEarly floors to that segment.
    """

    references = tuple(references)
    candidates = tuple(candidates)
    if len(references) != len(candidates):
        raise ValueError("aggregate XAcc reference/candidate count mismatch")

    reference_points = 0.0
    candidate_points = 0.0
    weight_total = 0
    for reference, candidate in zip(references, candidates):
        if bool(reference.stats.overloaded) and not bool(candidate.stats.overloaded):
            continue
        weight = max(0, int(reference.stats.targets))
        if weight <= 0:
            continue
        reference_points += float(reference.stats.x_accuracy_percent) * weight
        candidate_points += float(candidate.stats.x_accuracy_percent) * weight
        weight_total += weight

    if weight_total <= 0:
        return 100.0, 100.0, 0
    return (
        reference_points / weight_total,
        candidate_points / weight_total,
        weight_total,
    )


def _aggregate_xacc_guard(references, candidates, *, stage: str):
    reference_x, candidate_x, weight = _weighted_xacc_pairs(references, candidates)
    drop = float(v080.v062.DEFAULT_VALIDATION_XACC_DROP)
    if weight > 0 and candidate_x < reference_x - drop:
        return v080.v057.ConservativeDecision(
            False,
            f"{stage} aggregate XAcc regression>{drop:g}pt "
            f"({candidate_x:.2f}<{reference_x:.2f}-{drop:g})",
        )
    return v080.v057.ConservativeDecision(
        True,
        f"{stage} aggregate XAcc preserved "
        f"({candidate_x:.2f}>={reference_x - drop:.2f})",
    )


def _record_local_failure(stage: str, decision) -> None:
    if not decision.accepted:
        telemetry._record_failure(stage, decision.reason)


def _canonical(values: dict[int, object], count: int) -> tuple:
    return tuple(values[index] for index in range(count))


def _aggregate_xacc_line_search(
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
    """Adaptive v1.2 guard scheduling with role-level XAcc preservation."""

    started = time.perf_counter()
    workers = v080._configured_workers()
    states = {
        float(alpha): v080.v058._interpolate_state(base_state, proposal_state, float(alpha))
        for alpha in v080.v058.DEFAULT_TRUST_ALPHAS
    }
    state_digests = {
        alpha: v080.v065._state_digest(state)
        for alpha, state in states.items()
    }

    # Stage 1: v1.2 sampled-train preservation guard.
    train_raw = v080_fast._evaluate_state_segment_groups(
        model,
        states,
        [train_segment],
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        digests=state_digests,
    )
    train_evals = {alpha: train_raw[(alpha, train_segment.key)] for alpha in states}
    train_decisions = {
        alpha: v080.v063.v062.v061._safety_guard(base_train_eval, train_evals[alpha])
        for alpha in states
    }
    validation_alive = [alpha for alpha in states if train_decisions[alpha].accepted]

    # Stage 2a: Validation per-segment safety/hit/TooEarly only.
    validation_eval_maps = {alpha: {} for alpha in validation_alive}
    validation_decision_maps = {alpha: {} for alpha in validation_alive}
    validation_order = v080_fast._ordered_indices(
        validation_segments, v080_fast._VALIDATION_FAILURE_COUNTS
    )
    validation_cursor = 0
    validation_waves = 0
    while validation_alive and validation_cursor < len(validation_order):
        wave_size = guard_wave._adaptive_wave_size(
            len(validation_alive),
            len(validation_order) - validation_cursor,
            workers=workers,
        )
        batch_indices = validation_order[validation_cursor : validation_cursor + wave_size]
        validation_cursor += wave_size
        batch = [validation_segments[index] for index in batch_indices]
        raw = v080_fast._evaluate_state_segment_groups(
            model,
            {alpha: states[alpha] for alpha in validation_alive},
            batch,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            digests={alpha: state_digests[alpha] for alpha in validation_alive},
        )
        validation_waves += 1

        next_alive: list[float] = []
        for alpha in validation_alive:
            survived = True
            for index, segment in zip(batch_indices, batch):
                evaluation = raw[(alpha, segment.key)]
                decision = _validation_guard_without_xacc(
                    validation_references[index], evaluation
                )
                validation_eval_maps[alpha][index] = evaluation
                validation_decision_maps[alpha][index] = decision
                _record_local_failure("validation", decision)
                if not decision.accepted:
                    v080_fast._VALIDATION_FAILURE_COUNTS[segment.key] = (
                        v080_fast._VALIDATION_FAILURE_COUNTS.get(segment.key, 0) + 1
                    )
                    survived = False
                    break
            if survived:
                next_alive.append(alpha)
        validation_alive = next_alive

    # Stage 2b: only candidates with every Validation segment evaluated get the
    # role-level target-weighted XAcc floor.  Rejected candidates never enter the
    # much larger Anchor surface.
    validation_aggregate_alive: list[float] = []
    for alpha in validation_alive:
        evaluations = _canonical(validation_eval_maps[alpha], len(validation_segments))
        decision = _aggregate_xacc_guard(
            validation_references,
            evaluations,
            stage="validation",
        )
        if not decision.accepted:
            telemetry._record_failure("validation-aggregate", decision.reason)
            if validation_decision_maps[alpha]:
                first = min(validation_decision_maps[alpha])
                validation_decision_maps[alpha][first] = decision
            continue
        validation_aggregate_alive.append(alpha)
    validation_alive = validation_aggregate_alive

    # Stage 3a: Anchor per-segment safety/hit/TooEarly only.
    anchor_eval_maps = {alpha: {} for alpha in validation_alive}
    anchor_decision_maps = {alpha: {} for alpha in validation_alive}
    alive = list(validation_alive)
    anchor_order = v080_fast._ordered_indices(
        anchor_segments, v080_fast._ANCHOR_FAILURE_COUNTS
    )
    anchor_cursor = 0
    anchor_waves = 0
    while alive and anchor_cursor < len(anchor_order):
        wave_size = guard_wave._adaptive_wave_size(
            len(alive),
            len(anchor_order) - anchor_cursor,
            workers=workers,
            max_segments=v080_fast.DEFAULT_ANCHOR_BATCH_SIZE,
        )
        batch_indices = anchor_order[anchor_cursor : anchor_cursor + wave_size]
        anchor_cursor += wave_size
        batch = [anchor_segments[index] for index in batch_indices]
        raw = v080_fast._evaluate_state_segment_groups(
            model,
            {alpha: states[alpha] for alpha in alive},
            batch,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            digests={alpha: state_digests[alpha] for alpha in alive},
        )
        anchor_waves += 1

        next_alive: list[float] = []
        for alpha in alive:
            survived_batch = True
            for index, segment in zip(batch_indices, batch):
                evaluation = raw[(alpha, segment.key)]
                decision = _anchor_guard_without_xacc(
                    anchor_references[index], evaluation
                )
                anchor_eval_maps[alpha][index] = evaluation
                anchor_decision_maps[alpha][index] = decision
                stage = telemetry._anchor_stage(segment)
                _record_local_failure(stage, decision)
                if not decision.accepted:
                    v080_fast._ANCHOR_FAILURE_COUNTS[segment.key] = (
                        v080_fast._ANCHOR_FAILURE_COUNTS.get(segment.key, 0) + 1
                    )
                    survived_batch = False
            if survived_batch:
                next_alive.append(alpha)
        alive = next_alive

    # Stage 3b: exact aggregate Anchor XAcc floor after all anchors survive their
    # local hard guards.
    aggregate_alive: list[float] = []
    for alpha in alive:
        evaluations = _canonical(anchor_eval_maps[alpha], len(anchor_segments))
        decision = _aggregate_xacc_guard(
            anchor_references,
            evaluations,
            stage="anchor",
        )
        if not decision.accepted:
            telemetry._record_failure("anchor-aggregate", decision.reason)
            if anchor_decision_maps[alpha]:
                first = min(anchor_decision_maps[alpha])
                anchor_decision_maps[alpha][first] = decision
            continue
        aggregate_alive.append(alpha)
    alive = aggregate_alive

    candidates = []
    for alpha in states:
        candidates.append(
            v080.MultiCandidate(
                alpha=alpha,
                train_eval=train_evals[alpha],
                train_decision=train_decisions[alpha],
                validation_evals=v080_fast._indexed_tuple(
                    validation_eval_maps.get(alpha, {})
                ),
                validation_decisions=v080_fast._indexed_tuple(
                    validation_decision_maps.get(alpha, {})
                ),
                anchor_evals=v080_fast._indexed_tuple(anchor_eval_maps.get(alpha, {})),
                anchor_decisions=v080_fast._indexed_tuple(
                    anchor_decision_maps.get(alpha, {})
                ),
            )
        )

    complete = [
        candidate
        for candidate in candidates
        if candidate.train_decision.accepted
        and len(candidate.validation_decisions) == len(validation_segments)
        and len(candidate.anchor_decisions) == len(anchor_segments)
        and candidate.accepted
    ]
    print(f"{label}: " + " | ".join(v080._candidate_brief(candidate) for candidate in candidates))
    print(
        f"guard-perf: time={time.perf_counter() - started:.2f}s "
        f"adaptive-wave=on aggregate-X=on Vwaves={validation_waves} Awaves={anchor_waves}"
    )

    if not complete:
        model.load_state_dict(base_state)
        return None, candidates, None

    chosen = max(
        complete,
        key=lambda item: v080.v064.v061_candidate_key(base_train_eval, item.train_eval),
    )
    chosen_state = states[chosen.alpha]
    model.load_state_dict(chosen_state)
    return chosen, candidates, chosen_state


def install_aggregate_xacc_line_search() -> None:
    """Install the v1.3 guard before v1.2 train-preserve wraps line search."""

    global _INSTALLED
    if _INSTALLED:
        return
    v080_fast._fast_line_search = _aggregate_xacc_line_search
    _INSTALLED = True
    print(
        f"aggregate-xacc-guard={AGGREGATE_XACC_VERSION} "
        f"drop<={v080.v062.DEFAULT_VALIDATION_XACC_DROP:g}pt "
        "scope=Validation-role+Anchor-role local-hard=overload+hit+early"
    )


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
    install_aggregate_xacc_line_search()
    v120.install_train_preserve_gate()

    print("=== DMDOD v1.3.0 Aggregate XAcc ===")
    print(
        f"aggregate-xacc={AGGREGATE_XACC_VERSION} | "
        "Validation/Anchor XAcc=target-weighted role floor | "
        "per-segment hard=overload+hit+TooEarly"
    )
    print(
        f"train-preserve={v120.TRAIN_PRESERVE_VERSION} | "
        "round train=non-regression | selection=aggregate guard-surface improvement"
    )
    print(
        f"start-gate={v110.START_GATE_VERSION} | start-micro targets={start_args.start_micro_targets}"
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
