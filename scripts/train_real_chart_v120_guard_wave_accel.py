from __future__ import annotations

"""Exact-semantics adaptive scheduling for v1.2 validation/anchor guards.

The fast evaluator already short-circuits failed candidates, but its fixed guard
waves can badly over-evaluate on a 12-worker machine.  In particular, evaluating
12 anchors for 6 live interpolation candidates creates 72 state x segment jobs
before any of those candidates can be pruned.

This module keeps the same segment order, guards, trust alphas, candidate
ranking, and acceptance semantics.  It only sizes each validation/anchor wave so
roughly one worker-wave of state x segment jobs is launched at a time:

    segments_per_wave ~= workers // live_candidates

That preserves CPU occupancy while allowing failures to prune candidates before
another large batch of unnecessary segment simulations is scheduled.
"""

import time

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast


GUARD_WAVE_VERSION = "v120-adaptive-guard-wave-v1"
_INSTALLED = False


def _adaptive_wave_size(
    live_candidates: int,
    remaining_segments: int,
    *,
    workers: int,
    max_segments: int | None = None,
) -> int:
    """Choose a segment batch that fills about one state x segment worker wave."""

    live = max(1, int(live_candidates))
    remaining = max(0, int(remaining_segments))
    worker_count = max(1, int(workers))
    if remaining <= 0:
        return 0
    cap = worker_count if max_segments is None else max(1, int(max_segments))
    by_occupancy = max(1, worker_count // live)
    return min(remaining, cap, by_occupancy)


def _adaptive_fast_line_search(
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
    """v0.8 fast line-search semantics with adaptive guard wave sizes."""

    workers = v080._configured_workers()
    states = {
        float(alpha): v080.v058._interpolate_state(base_state, proposal_state, float(alpha))
        for alpha in v080.v058.DEFAULT_TRUST_ALPHAS
    }
    state_digests = {
        alpha: v080.v065._state_digest(state)
        for alpha, state in states.items()
    }

    # Stage 1: sampled training window, unchanged.
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

    # Stage 2: preserve the same historical selectivity order, but evaluate as
    # many validation segments at once as can fill roughly one worker wave.
    validation_eval_maps = {alpha: {} for alpha in validation_alive}
    validation_decision_maps = {alpha: {} for alpha in validation_alive}
    validation_order = v080_fast._ordered_indices(
        validation_segments, v080_fast._VALIDATION_FAILURE_COUNTS
    )
    validation_cursor = 0
    validation_waves = 0
    while validation_alive and validation_cursor < len(validation_order):
        wave_size = _adaptive_wave_size(
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
            # Sequentially consume the already-computed wave.  Once this
            # candidate fails, later results in the wave are deliberately ignored
            # so failure accounting/partial candidate data matches short-circuit
            # semantics.
            for index, segment in zip(batch_indices, batch):
                evaluation = raw[(alpha, segment.key)]
                decision = v080.v062._validation_guard(
                    validation_references[index], evaluation
                )
                validation_eval_maps[alpha][index] = evaluation
                validation_decision_maps[alpha][index] = decision
                if not decision.accepted:
                    v080_fast._VALIDATION_FAILURE_COUNTS[segment.key] = (
                        v080_fast._VALIDATION_FAILURE_COUNTS.get(segment.key, 0) + 1
                    )
                    survived = False
                    break
            if survived:
                next_alive.append(alpha)
        validation_alive = next_alive

    anchor_survivors = list(validation_alive)

    # Stage 3: same anchor order/guards, but avoid the old fixed 12-segment wave
    # when several candidates are still alive.  E.g. 6 candidates on 12 workers
    # now evaluate 2 anchors (12 jobs), not 12 anchors (72 jobs), before pruning.
    anchor_eval_maps = {alpha: {} for alpha in anchor_survivors}
    anchor_decision_maps = {alpha: {} for alpha in anchor_survivors}
    alive = list(anchor_survivors)
    anchor_order = v080_fast._ordered_indices(
        anchor_segments, v080_fast._ANCHOR_FAILURE_COUNTS
    )
    anchor_cursor = 0
    anchor_waves = 0
    while alive and anchor_cursor < len(anchor_order):
        wave_size = _adaptive_wave_size(
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
            # Keep the existing anchor-batch behavior: all decisions in the
            # scheduled wave are recorded even if one fails.
            for index, segment in zip(batch_indices, batch):
                evaluation = raw[(alpha, segment.key)]
                decision = v080.v064._anchor_guard(anchor_references[index], evaluation)
                anchor_eval_maps[alpha][index] = evaluation
                anchor_decision_maps[alpha][index] = decision
                if not decision.accepted:
                    v080_fast._ANCHOR_FAILURE_COUNTS[segment.key] = (
                        v080_fast._ANCHOR_FAILURE_COUNTS.get(segment.key, 0) + 1
                    )
                    survived_batch = False
            if survived_batch:
                next_alive.append(alpha)
        alive = next_alive

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

    if not complete:
        model.load_state_dict(base_state)
        return None, candidates, None, validation_waves, anchor_waves

    chosen = max(
        complete,
        key=lambda item: v080.v064.v061_candidate_key(base_train_eval, item.train_eval),
    )
    chosen_state = states[chosen.alpha]
    model.load_state_dict(chosen_state)
    return chosen, candidates, chosen_state, validation_waves, anchor_waves


def _timed_adaptive_line_search(*args, **kwargs):
    started = time.perf_counter()
    chosen, candidates, chosen_state, validation_waves, anchor_waves = (
        _adaptive_fast_line_search(*args, **kwargs)
    )
    print(
        f"guard-perf: time={time.perf_counter() - started:.2f}s "
        f"adaptive-wave=on Vwaves={validation_waves} Awaves={anchor_waves}"
    )
    return chosen, candidates, chosen_state


def install_adaptive_guard_waves() -> None:
    """Install adaptive guard scheduling after round acceleration is active."""

    global _INSTALLED
    if _INSTALLED:
        return
    v080_fast._fast_line_search = _timed_adaptive_line_search
    _INSTALLED = True
    print(
        f"guard-wave-accel={GUARD_WAVE_VERSION} workers={v080._configured_workers()} "
        "semantics=unchanged target=one-worker-wave"
    )
