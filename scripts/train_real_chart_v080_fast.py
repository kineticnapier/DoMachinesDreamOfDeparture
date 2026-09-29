from __future__ import annotations

"""Execution-only acceleration for the v0.8.0 multi-chart HUD trainer.

The checkpoint format, run signature, model, optimizer updates, trust alphas,
guards, candidate ranking, and Final split semantics stay exactly v0.8.0.
Only evaluation scheduling changes:

* bootstrap evaluates anchors + validation in one shared process-pool wave;
* direct serial student evaluation is cached by exact state + segment identity;
* validation guards are evaluated one segment at a time and stop dead candidates;
* anchor guards are evaluated in small batches and stop dead candidates;
* anchors that historically reject more candidates are tried earlier, while a
  fully accepted candidate is restored to the canonical anchor order.

This wrapper is intentionally resume-compatible with format-15 checkpoints
created by ``train_real_chart_v080.py``.
"""

import copy
from typing import Iterable

import torch

import train_real_chart_v080 as v080


FAST_EVAL_VERSION = "v080-exact-eval-pruning-v2"
DEFAULT_ANCHOR_BATCH_SIZE = 2

_ORIGINAL_EVALUATE_STUDENT = v080.v054._evaluate_student
_DIRECT_EVAL_CACHE: dict[tuple, tuple[object, object]] = {}
_DIRECT_EVAL_CACHE_HITS = 0
_DIRECT_EVAL_CACHE_MISSES = 0
_ANCHOR_FAILURE_COUNTS: dict[tuple, int] = {}


def _anchor_batches(items: list, size: int = DEFAULT_ANCHOR_BATCH_SIZE) -> Iterable[tuple[int, list]]:
    if size <= 0:
        raise ValueError("anchor batch size must be positive")
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _cached_direct_evaluate_student(
    model,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device,
):
    """Cache exact serial evaluations used for round base/final train metrics.

    The segment object itself is retained in the cache value so Python cannot
    recycle its id and accidentally produce a false cache hit later in the run.
    HUD globals installed by v0.7 are still resolved by the original evaluator
    at call time, so this wrapper changes only whether an identical simulation
    is repeated.
    """

    global _DIRECT_EVAL_CACHE_HITS, _DIRECT_EVAL_CACHE_MISSES

    key = (
        v080.v065._state_digest(model.state_dict()),
        id(segment),
        bool(same_hand),
        float(control_dt_s),
        str(device),
        int(model.hidden_dim),
    )
    cached = _DIRECT_EVAL_CACHE.get(key)
    if cached is not None and cached[0] is segment:
        _DIRECT_EVAL_CACHE_HITS += 1
        return cached[1]

    evaluated = _ORIGINAL_EVALUATE_STUDENT(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
    )
    _DIRECT_EVAL_CACHE[key] = (segment, evaluated)
    _DIRECT_EVAL_CACHE_MISSES += 1
    return evaluated


def _evaluate_bootstrap_state(
    model,
    state,
    anchor_segments,
    validation_segments,
    *,
    same_hand: bool,
    control_dt_s: float,
):
    """Evaluate one bootstrap state in a single pool wave and split the result."""

    combined = [*anchor_segments, *validation_segments]
    evaluations = v080._evaluate_one_state_many(
        model,
        state,
        combined,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    split = len(anchor_segments)
    return evaluations[:split], evaluations[split:]


def _fast_bootstrap(
    model,
    expert_pairs,
    anchor_segments,
    validation_segments,
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
    same_hand: bool,
    control_dt_s: float,
):
    """Exact v0.8.0 bootstrap with anchors and validation scheduled together."""

    optimizer = torch.optim.Adam(v080.v057._policy_parameters(model), lr=learning_rate)

    state = copy.deepcopy(model.state_dict())
    anchor_evals, validation_evals = _evaluate_bootstrap_state(
        model,
        state,
        anchor_segments,
        validation_segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    best_state = state
    best_anchors = anchor_evals
    best_validations = validation_evals
    best_key = v080._bootstrap_key(anchor_evals, validation_evals)
    best_epoch = 0
    history: list[dict] = []
    print(
        f"bootstrap 00: safe={bool(best_key[0])} completion={best_key[1] * 100.0:.1f}% "
        f"meanX={best_key[2]:.1f}%"
    )

    for epoch in range(1, epochs + 1):
        loss = v080.v062._bootstrap_multi_epoch(
            model,
            expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=not bool(epoch & 1),
        )
        state = copy.deepcopy(model.state_dict())
        anchor_evals, validation_evals = _evaluate_bootstrap_state(
            model,
            state,
            anchor_segments,
            validation_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        key = v080._bootstrap_key(anchor_evals, validation_evals)
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


def _ordered_anchor_indices(anchor_segments) -> list[int]:
    """Try historically selective anchors first, keeping deterministic ties."""

    return sorted(
        range(len(anchor_segments)),
        key=lambda index: (-_ANCHOR_FAILURE_COUNTS.get(anchor_segments[index].key, 0), index),
    )


def _indexed_tuple(values: dict[int, object]) -> tuple:
    return tuple(values[index] for index in sorted(values))


def _fast_line_search(
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
    """Exact v0.8.0 line search with incremental validation/anchor pruning."""

    states = {
        float(alpha): v080.v058._interpolate_state(base_state, proposal_state, float(alpha))
        for alpha in v080.v058.DEFAULT_TRUST_ALPHAS
    }

    # Stage 1: current train window.
    train_raw = v080._evaluate_states_on_segments(
        model,
        states,
        [train_segment],
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    train_evals = {alpha: train_raw[(alpha, train_segment.key)] for alpha in states}
    train_decisions = {
        alpha: v080.v063.v062.v061._safety_guard(base_train_eval, train_evals[alpha])
        for alpha in states
    }
    validation_alive = [alpha for alpha in states if train_decisions[alpha].accepted]

    # Stage 2: validation one segment at a time. A failed guard is a conjunctive
    # rejection, so evaluating later validation segments cannot revive it.
    validation_eval_maps = {alpha: {} for alpha in validation_alive}
    validation_decision_maps = {alpha: {} for alpha in validation_alive}
    for index, (segment, reference) in enumerate(zip(validation_segments, validation_references)):
        if not validation_alive:
            break
        raw = v080._evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in validation_alive},
            [segment],
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        next_alive: list[float] = []
        for alpha in validation_alive:
            evaluation = raw[(alpha, segment.key)]
            decision = v080.v062._validation_guard(reference, evaluation)
            validation_eval_maps[alpha][index] = evaluation
            validation_decision_maps[alpha][index] = decision
            if decision.accepted:
                next_alive.append(alpha)
        validation_alive = next_alive

    anchor_survivors = list(validation_alive)

    # Stage 3: evaluate the most failure-prone anchors first. The adaptive order
    # affects execution only. For a fully surviving candidate, results are put
    # back into canonical anchor order before reference updates/candidate ranking.
    anchor_eval_maps = {alpha: {} for alpha in anchor_survivors}
    anchor_decision_maps = {alpha: {} for alpha in anchor_survivors}
    alive = list(anchor_survivors)
    ordered_indices = _ordered_anchor_indices(anchor_segments)
    for batch_start in range(0, len(ordered_indices), DEFAULT_ANCHOR_BATCH_SIZE):
        if not alive:
            break
        batch_indices = ordered_indices[batch_start : batch_start + DEFAULT_ANCHOR_BATCH_SIZE]
        batch = [anchor_segments[index] for index in batch_indices]
        raw = v080._evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in alive},
            batch,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        next_alive: list[float] = []
        for alpha in alive:
            survived_batch = True
            for index, segment in zip(batch_indices, batch):
                evaluation = raw[(alpha, segment.key)]
                decision = v080.v064._anchor_guard(anchor_references[index], evaluation)
                anchor_eval_maps[alpha][index] = evaluation
                anchor_decision_maps[alpha][index] = decision
                if not decision.accepted:
                    _ANCHOR_FAILURE_COUNTS[segment.key] = (
                        _ANCHOR_FAILURE_COUNTS.get(segment.key, 0) + 1
                    )
                    survived_batch = False
            if survived_batch:
                next_alive.append(alpha)
        alive = next_alive

    candidates = []
    for alpha in states:
        candidate = v080.MultiCandidate(
            alpha=alpha,
            train_eval=train_evals[alpha],
            train_decision=train_decisions[alpha],
            validation_evals=_indexed_tuple(validation_eval_maps.get(alpha, {})),
            validation_decisions=_indexed_tuple(validation_decision_maps.get(alpha, {})),
            anchor_evals=_indexed_tuple(anchor_eval_maps.get(alpha, {})),
            anchor_decisions=_indexed_tuple(anchor_decision_maps.get(alpha, {})),
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
    print(f"{label}: " + " | ".join(v080._candidate_brief(candidate) for candidate in candidates))

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


def _install_fast_path() -> None:
    v080._bootstrap = _fast_bootstrap
    v080._line_search = _fast_line_search
    v080.v054._evaluate_student = _cached_direct_evaluate_student


def main() -> None:
    _install_fast_path()
    print("=== DMDOD v0.8.0 Fast Eval Wrapper ===")
    print(
        f"fast-eval={FAST_EVAL_VERSION} | bootstrap=combined anchors+validation | "
        f"validation=segment-short-circuit | anchor-guard batches={DEFAULT_ANCHOR_BATCH_SIZE} adaptive-order"
    )
    print("checkpoint/signature/model/training semantics=v0.8.0 unchanged")
    v080.main()
    print(
        f"direct-eval-cache: hits={_DIRECT_EVAL_CACHE_HITS} "
        f"misses={_DIRECT_EVAL_CACHE_MISSES}"
    )


if __name__ == "__main__":
    main()
