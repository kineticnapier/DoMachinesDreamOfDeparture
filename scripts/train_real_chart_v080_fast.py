from __future__ import annotations

"""Execution-only acceleration for the v0.8.0 multi-chart HUD trainer.

The checkpoint format, run signature, model, optimizer updates, trust alphas,
guards, candidate ranking, and Final split semantics stay exactly v0.8.0.
Only evaluation scheduling changes:

* bootstrap evaluates anchors + validation in one shared process-pool wave;
* anchor guards are evaluated in small batches and stop evaluating a candidate
  after its first failed batch, because a failed conjunct can never recover.

This wrapper is intentionally resume-compatible with format-15 checkpoints
created by ``train_real_chart_v080.py``.
"""

import copy
from typing import Iterable

import torch

import train_real_chart_v080 as v080


FAST_EVAL_VERSION = "v080-combined-bootstrap-anchor-batches-v1"
DEFAULT_ANCHOR_BATCH_SIZE = 2


def _anchor_batches(items: list, size: int = DEFAULT_ANCHOR_BATCH_SIZE) -> Iterable[tuple[int, list]]:
    if size <= 0:
        raise ValueError("anchor batch size must be positive")
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


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
    """Exact v0.8.0 line search with incremental anchor-guard pruning."""

    states = {
        float(alpha): v080.v058._interpolate_state(base_state, proposal_state, float(alpha))
        for alpha in v080.v058.DEFAULT_TRUST_ALPHAS
    }

    # Stage 1: current train window. This is identical to v0.8.0.
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
    validation_survivors = [alpha for alpha in states if train_decisions[alpha].accepted]

    # Stage 2: both validation charts. There are only two in the intended v0.8
    # dataset, so one wave already fills the pool efficiently.
    validation_evals_by_alpha = {}
    validation_decisions_by_alpha = {}
    anchor_survivors: list[float] = []
    if validation_survivors:
        val_raw = v080._evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in validation_survivors},
            validation_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        for alpha in validation_survivors:
            evaluations = tuple(val_raw[(alpha, segment.key)] for segment in validation_segments)
            decisions = tuple(
                v080.v062._validation_guard(reference, evaluation)
                for reference, evaluation in zip(validation_references, evaluations)
            )
            validation_evals_by_alpha[alpha] = evaluations
            validation_decisions_by_alpha[alpha] = decisions
            if all(decision.accepted for decision in decisions):
                anchor_survivors.append(alpha)

    # Stage 3: anchors in two-segment waves. With six trust alphas this exposes
    # up to 12 worker tasks per wave. A candidate that fails any anchor guard is
    # permanently dead, so later anchors for it are provably unnecessary.
    anchor_evals_lists = {alpha: [] for alpha in anchor_survivors}
    anchor_decision_lists = {alpha: [] for alpha in anchor_survivors}
    alive = list(anchor_survivors)
    for start, batch in _anchor_batches(anchor_segments):
        if not alive:
            break
        raw = v080._evaluate_states_on_segments(
            model,
            {alpha: states[alpha] for alpha in alive},
            batch,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        batch_references = anchor_references[start : start + len(batch)]
        next_alive: list[float] = []
        for alpha in alive:
            evaluations = [raw[(alpha, segment.key)] for segment in batch]
            decisions = [
                v080.v064._anchor_guard(reference, evaluation)
                for reference, evaluation in zip(batch_references, evaluations)
            ]
            anchor_evals_lists[alpha].extend(evaluations)
            anchor_decision_lists[alpha].extend(decisions)
            if all(decision.accepted for decision in decisions):
                next_alive.append(alpha)
        alive = next_alive

    anchor_evals_by_alpha = {
        alpha: tuple(evaluations) for alpha, evaluations in anchor_evals_lists.items()
    }
    anchor_decisions_by_alpha = {
        alpha: tuple(decisions) for alpha, decisions in anchor_decision_lists.items()
    }

    candidates = []
    for alpha in states:
        candidate = v080.MultiCandidate(
            alpha=alpha,
            train_eval=train_evals[alpha],
            train_decision=train_decisions[alpha],
            validation_evals=validation_evals_by_alpha.get(alpha, ()),
            validation_decisions=validation_decisions_by_alpha.get(alpha, ()),
            anchor_evals=anchor_evals_by_alpha.get(alpha, ()),
            anchor_decisions=anchor_decisions_by_alpha.get(alpha, ()),
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


def main() -> None:
    _install_fast_path()
    print("=== DMDOD v0.8.0 Fast Eval Wrapper ===")
    print(
        f"fast-eval={FAST_EVAL_VERSION} | bootstrap=combined anchors+validation | "
        f"anchor-guard batches={DEFAULT_ANCHOR_BATCH_SIZE}"
    )
    print("checkpoint/signature/model/training semantics=v0.8.0 unchanged")
    v080.main()


if __name__ == "__main__":
    main()
