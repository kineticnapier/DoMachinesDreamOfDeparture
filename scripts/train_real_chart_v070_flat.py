from __future__ import annotations

"""v0.7.0 HUD trainer with finer-grained exact CPU parallel evaluation.

This is execution-only acceleration for the existing v0.7.0 checkpoint format.
It keeps the same model, training data, trust alphas, guards, run signature, and
checkpoint compatibility.  A line-search miss is flattened from six alpha tasks
into alpha x segment tasks, so a 6-alpha / 6-anchor run exposes 48 independent
policy evaluations to the process pool.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import train_real_chart_v054 as v054
import train_real_chart_v058 as v058
import train_real_chart_v062 as v062
import train_real_chart_v063 as v063
import train_real_chart_v064 as v064
import train_real_chart_v065 as v065
import train_real_chart_v070 as v070

from dmdod.flat_hud_eval import evaluate_hud_state_on_segment


FLAT_PARALLEL_EVAL_VERSION = "alpha-segment-flat-pool-v2"
_MAX_DEFAULT_WORKERS = 12
_FLAT_EVAL_POOL: ProcessPoolExecutor | None = None
_INSTALLED = False
_PARENT_CHECKPOINT_PAYLOAD = None


def _configured_flat_eval_workers() -> int:
    raw = os.environ.get("DMDOD_EVAL_WORKERS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise SystemExit("DMDOD_EVAL_WORKERS must be a positive integer") from exc

    logical = max(1, int(os.cpu_count() or 1))
    # Fine-grained tasks can use SMT too.  Keep the automatic default bounded so
    # large-core machines do not spawn an excessive number of Python simulators.
    return min(_MAX_DEFAULT_WORKERS, logical)


def _get_flat_eval_pool() -> ProcessPoolExecutor:
    global _FLAT_EVAL_POOL
    if _FLAT_EVAL_POOL is None:
        _FLAT_EVAL_POOL = ProcessPoolExecutor(max_workers=_configured_flat_eval_workers())
    return _FLAT_EVAL_POOL


def _shutdown_flat_eval_pool() -> None:
    global _FLAT_EVAL_POOL
    if _FLAT_EVAL_POOL is not None:
        _FLAT_EVAL_POOL.shutdown(wait=True, cancel_futures=True)
        _FLAT_EVAL_POOL = None


def _flat_parallel_evaluate_anchor_line_search(
    model,
    *,
    base_state,
    proposal_state,
    base_train_eval,
    validation_reference,
    anchor_references,
    alphas,
    train_segment,
    validation_segment,
    anchor_segments,
    same_hand,
    control_dt_s,
    device,
    label_prefix,
    verbose,
):
    """Evaluate every (alpha, segment) pair independently in one shared pool."""

    serial = v070._SERIAL_ANCHOR_LINE_SEARCH
    if serial is None:
        raise RuntimeError("v0.7.0 serial line-search backend is not installed")

    workers = _configured_flat_eval_workers()
    if workers <= 1 or getattr(device, "type", str(device)) != "cpu":
        return serial(
            model,
            base_state=base_state,
            proposal_state=proposal_state,
            base_train_eval=base_train_eval,
            validation_reference=validation_reference,
            anchor_references=anchor_references,
            alphas=alphas,
            train_segment=train_segment,
            validation_segment=validation_segment,
            anchor_segments=anchor_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
            label_prefix=label_prefix,
            verbose=verbose,
        )
    if len(anchor_references) != len(anchor_segments):
        raise ValueError("anchor reference count must match anchor segment count")

    segments = (train_segment, validation_segment, *anchor_segments)
    states = {
        float(alpha): v058._interpolate_state(base_state, proposal_state, alpha)
        for alpha in alphas
    }

    try:
        pool = _get_flat_eval_pool()
        futures = {}
        for alpha in alphas:
            alpha = float(alpha)
            for segment_index, segment in enumerate(segments):
                futures[(alpha, segment_index)] = pool.submit(
                    evaluate_hud_state_on_segment,
                    states[alpha],
                    int(model.hidden_dim),
                    segment,
                    bool(same_hand),
                    float(control_dt_s),
                )

        raw_by_alpha = {
            float(alpha): [
                futures[(float(alpha), segment_index)].result()
                for segment_index in range(len(segments))
            ]
            for alpha in alphas
        }
    except Exception as exc:
        # This optimization must never change correctness.  On process/IPC
        # failure, discard the pool and run the original serial line search.
        print(f"{label_prefix}: flat-parallel fallback ({type(exc).__name__}: {exc})")
        _shutdown_flat_eval_pool()
        return serial(
            model,
            base_state=base_state,
            proposal_state=proposal_state,
            base_train_eval=base_train_eval,
            validation_reference=validation_reference,
            anchor_references=anchor_references,
            alphas=alphas,
            train_segment=train_segment,
            validation_segment=validation_segment,
            anchor_segments=anchor_segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
            label_prefix=label_prefix,
            verbose=verbose,
        )

    candidates = []
    for alpha in alphas:
        alpha = float(alpha)
        raw = raw_by_alpha[alpha]
        expected = 2 + len(anchor_segments)
        if len(raw) != expected:
            raise RuntimeError(
                f"flat parallel evaluator returned {len(raw)} results; expected {expected}"
            )

        evaluations = [v054.StudentEvalResult(stats, keydowns) for stats, keydowns in raw]
        train_eval = evaluations[0]
        validation_eval = evaluations[1]
        anchor_evals = tuple(evaluations[2:])

        train_decision = v063.v062.v061._safety_guard(base_train_eval, train_eval)
        validation_decision = v062._validation_guard(validation_reference, validation_eval)
        anchor_decisions = tuple(
            v064._anchor_guard(reference, evaluation)
            for reference, evaluation in zip(anchor_references, anchor_evals)
        )
        candidate = v064.AnchorCandidate(
            alpha,
            train_eval,
            validation_eval,
            anchor_evals,
            train_decision,
            validation_decision,
            anchor_decisions,
        )
        candidates.append(candidate)

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

    choice = v064._choose_anchor_candidate(base_train_eval, candidates)
    if not verbose:
        print(
            f"{label_prefix}: "
            + " | ".join(v064._candidate_brief(candidate) for candidate in candidates)
        )

    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None

    chosen_state = states[float(choice.alpha)]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def _flat_checkpoint_payload(**kwargs) -> dict:
    if _PARENT_CHECKPOINT_PAYLOAD is None:
        raise RuntimeError("flat checkpoint backend is not installed")
    payload = _PARENT_CHECKPOINT_PAYLOAD(**kwargs)
    payload["parallel_eval"] = FLAT_PARALLEL_EVAL_VERSION
    return payload


def _install_flat_parallel() -> None:
    global _INSTALLED, _PARENT_CHECKPOINT_PAYLOAD
    if _INSTALLED:
        return

    # First install the complete v0.7.0 HUD stack: 245D observation, exact BC
    # proposal cache, exact line-search cache, and format-14 resume semantics.
    v070._install_v070()

    _PARENT_CHECKPOINT_PAYLOAD = v064._checkpoint_payload
    v064._checkpoint_payload = _flat_checkpoint_payload

    # v0.6.5's memoizer calls this function only on an exact cache miss.
    # Replace the six coarse alpha workers with the finer alpha x segment queue.
    v065._ORIGINAL_ANCHOR_LINE_SEARCH = _flat_parallel_evaluate_anchor_line_search
    _INSTALLED = True


def main() -> None:
    _install_flat_parallel()
    workers = _configured_flat_eval_workers()
    print("=== DMDOD v0.7.0 Human-Visible HUD / Flat Parallel Eval ===")
    print(
        f"observation={v070.HUD_REAL_CHART_INPUT_DIM}D | "
        f"flat-eval={workers} workers | tasks=alpha x (train+validation+anchors)"
    )
    print(
        "semantics=unchanged format-14 checkpoint | "
        "exact line-search cache + exact repeated-BC proposal cache"
    )
    try:
        v064.main()
    finally:
        _shutdown_flat_eval_pool()
        v070._shutdown_eval_pool()

    print(
        f"proposal-cache: hits={v065._PROPOSAL_CACHE.hits} "
        f"misses={v065._PROPOSAL_CACHE.misses} "
        f"saved-line-searches={v065._PROPOSAL_CACHE.hits}"
    )
    print(
        f"train-proposal-cache: hits={v070._TRAIN_PROPOSAL_CACHE_HITS} "
        f"misses={v070._TRAIN_PROPOSAL_CACHE_MISSES} "
        f"saved-bc-epochs={v070._TRAIN_PROPOSAL_CACHE_HITS}"
    )


if __name__ == "__main__":
    main()
