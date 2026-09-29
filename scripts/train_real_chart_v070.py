from __future__ import annotations

"""v0.7.0: train the existing anchor-guard pipeline with player-visible HUD input.

The training/guard semantics remain v0.6.4 plus v0.6.5's exact proposal cache.
Only the observation interface changes: Tile BPM, Real BPM, and transient timing
feedback are appended to the previous 233D visible geometry/motor vector.
"""

import copy
import os
from concurrent.futures import ProcessPoolExecutor

import train_real_chart_v054 as v054
import train_real_chart_v055 as v055
import train_real_chart_v057 as v057
import train_real_chart_v058 as v058
import train_real_chart_v060 as v060
import train_real_chart_v062 as v062
import train_real_chart_v063 as v063
import train_real_chart_v064 as v064
import train_real_chart_v065 as v065

from dmdod.parallel_hud_eval import evaluate_hud_state_on_segments
from dmdod.real_chart_hud import DEFAULT_FEEDBACK_HOLD_S, DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import (
    HUD_FEATURE_VERSION,
    HUD_REAL_CHART_INPUT_DIM,
    encode_hud_real_chart_observation,
)


TRAINER_VERSION = "0.7.0-human-visible-hud"
CHECKPOINT_FORMAT_VERSION = 14
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v070_hud.pt"
HUD_OBSERVATION_VERSION = HUD_FEATURE_VERSION
TRAIN_PROPOSAL_CACHE_VERSION = "exact-fresh-adam-v1"
PARALLEL_EVAL_VERSION = "alpha-process-pool-v1"

_INSTALLED = False
_PARENT_RUN_SIGNATURE = None
_PARENT_CHECKPOINT_PAYLOAD = None
_PARENT_LOAD_PROGRESS = None
_ORIGINAL_TRAIN_ONE_EPOCH = None
_SERIAL_ANCHOR_LINE_SEARCH = None
_TRAIN_PROPOSAL_CACHE: dict[tuple, tuple[float, dict]] = {}
_SEQUENCE_SIGNATURE_CACHE: dict[int, tuple[object, tuple]] = {}
_TRAIN_PROPOSAL_CACHE_HITS = 0
_TRAIN_PROPOSAL_CACHE_MISSES = 0
_EVAL_POOL: ProcessPoolExecutor | None = None


def _configured_eval_workers() -> int:
    raw = os.environ.get("DMDOD_EVAL_WORKERS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise SystemExit("DMDOD_EVAL_WORKERS must be a positive integer") from exc

    logical = max(1, int(os.cpu_count() or 1))
    # The current target machine is 6C/12T. Physics evaluation is Python/CPU
    # heavy, so default to roughly one worker per physical core instead of one
    # per SMT thread. Cap at the six trust-region alphas because extra workers
    # cannot help this layer.
    physical_guess = max(1, logical // 2) if logical > 1 else 1
    return min(6, physical_guess)


def _get_eval_pool() -> ProcessPoolExecutor:
    global _EVAL_POOL
    if _EVAL_POOL is None:
        _EVAL_POOL = ProcessPoolExecutor(max_workers=_configured_eval_workers())
    return _EVAL_POOL


def _shutdown_eval_pool() -> None:
    global _EVAL_POOL
    if _EVAL_POOL is not None:
        _EVAL_POOL.shutdown(wait=True, cancel_futures=True)
        _EVAL_POOL = None


def _install_dagger_input_dimension() -> None:
    """Make legacy DAggerSequence validate the v0.7 HUD vector width.

    DAggerSequence.__post_init__ reads v0.5.5's module-global
    REAL_CHART_INPUT_DIM at runtime. v0.7 changes the encoder from 233D to 245D,
    so that legacy validator must be updated in the v0.7 training process too.
    """

    v055.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM


def _optimizer_signature(optimizer) -> tuple:
    """Return the optimizer settings that affect one fresh Adam proposal."""

    groups = []
    for group in optimizer.param_groups:
        groups.append(
            (
                float(group.get("lr", 0.0)),
                tuple(float(value) for value in group.get("betas", (0.9, 0.999))),
                float(group.get("eps", 1e-8)),
                float(group.get("weight_decay", 0.0)),
                bool(group.get("amsgrad", False)),
                bool(group.get("maximize", False)),
            )
        )
    return tuple(groups)


def _stable_sequence_signature(stable) -> tuple:
    """Hash one immutable StableSequence once per process.

    Exact tensor digests avoid relying only on object identity. Holding the
    object in the cache entry also prevents Python id reuse from producing a
    false match later in a long training run.
    """

    cache_key = id(stable)
    cached = _SEQUENCE_SIGNATURE_CACHE.get(cache_key)
    if cached is not None and cached[0] is stable:
        return cached[1]

    sequence = stable.sequence
    signature = (
        str(sequence.source),
        int(sequence.frames),
        v065._tensor_digest(sequence.observations),
        v065._tensor_digest(sequence.teacher_actions),
        v065._tensor_digest(stable.loss_weights),
    )
    _SEQUENCE_SIGNATURE_CACHE[cache_key] = (stable, signature)
    return signature


def _cached_train_one_epoch(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Reuse an exactly repeated one-epoch BC proposal.

    v0.6.4 intentionally creates a brand-new Adam optimizer for every epoch.
    After a rollback, the trusted model is restored. With the same trajectory
    order, the next odd/even epoch therefore computes byte-identical weights.
    Reusing that final proposal state is execution-only caching: accepted model
    states, guards, and checkpoint semantics are unchanged.
    """

    global _TRAIN_PROPOSAL_CACHE_HITS, _TRAIN_PROPOSAL_CACHE_MISSES
    assert _ORIGINAL_TRAIN_ONE_EPOCH is not None

    key = (
        TRAIN_PROPOSAL_CACHE_VERSION,
        v065._state_digest(model.state_dict()),
        tuple(_stable_sequence_signature(stable) for stable in sequences),
        _optimizer_signature(optimizer),
        int(chunk_steps),
        bool(reverse_order),
    )
    cached = _TRAIN_PROPOSAL_CACHE.get(key)
    if cached is not None:
        loss, state = cached
        model.load_state_dict(state)
        _TRAIN_PROPOSAL_CACHE_HITS += 1
        return loss

    loss = _ORIGINAL_TRAIN_ONE_EPOCH(
        model,
        sequences,
        optimizer=optimizer,
        chunk_steps=chunk_steps,
        reverse_order=reverse_order,
    )
    _TRAIN_PROPOSAL_CACHE[key] = (float(loss), copy.deepcopy(model.state_dict()))
    _TRAIN_PROPOSAL_CACHE_MISSES += 1
    return float(loss)


def _parallel_evaluate_anchor_line_search(
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
    """Evaluate independent trust-region alphas concurrently on CPU processes.

    Each alpha still runs train, validation, and all anchors serially inside one
    worker. This keeps every alpha's gameplay evaluation exactly isolated while
    using up to six CPU cores across the six independent policy states.
    """

    assert _SERIAL_ANCHOR_LINE_SEARCH is not None
    workers = _configured_eval_workers()
    if workers <= 1 or getattr(device, "type", str(device)) != "cpu":
        return _SERIAL_ANCHOR_LINE_SEARCH(
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
        pool = _get_eval_pool()
        futures = {
            float(alpha): pool.submit(
                evaluate_hud_state_on_segments,
                states[float(alpha)],
                int(model.hidden_dim),
                segments,
                bool(same_hand),
                float(control_dt_s),
            )
            for alpha in alphas
        }
        raw_by_alpha = {float(alpha): futures[float(alpha)].result() for alpha in alphas}
    except Exception as exc:
        # Parallelism is an execution optimization only. If process startup or
        # IPC fails on a machine, retain correctness by falling back to the
        # pre-existing serial implementation for this line search.
        print(f"{label_prefix}: parallel-eval fallback ({type(exc).__name__}: {exc})")
        _shutdown_eval_pool()
        return _SERIAL_ANCHOR_LINE_SEARCH(
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
            raise RuntimeError(f"parallel evaluator returned {len(raw)} results; expected {expected}")

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
        print(f"{label_prefix}: " + " | ".join(v064._candidate_brief(candidate) for candidate in candidates))

    if not choice.accepted or choice.alpha is None:
        model.load_state_dict(base_state)
        return choice, candidates, None

    chosen_state = states[float(choice.alpha)]
    model.load_state_dict(chosen_state)
    return choice, candidates, chosen_state


def _install_v070() -> None:
    global _INSTALLED, _PARENT_RUN_SIGNATURE, _PARENT_CHECKPOINT_PAYLOAD, _PARENT_LOAD_PROGRESS
    global _ORIGINAL_TRAIN_ONE_EPOCH, _SERIAL_ANCHOR_LINE_SEARCH
    if _INSTALLED:
        return

    # Install v0.6.5 first so deterministic duplicate line searches remain
    # cached. Then replace only the observation/checkpoint identity surface.
    v065._install_v065()

    _PARENT_RUN_SIGNATURE = v064._run_signature
    _PARENT_CHECKPOINT_PAYLOAD = v064._checkpoint_payload
    _PARENT_LOAD_PROGRESS = v064._load_progress
    _ORIGINAL_TRAIN_ONE_EPOCH = v057._train_one_epoch
    _SERIAL_ANCHOR_LINE_SEARCH = v065._ORIGINAL_ANCHOR_LINE_SEARCH

    # Keep v0.6.5's exact line-search memoization, but make its cache misses call
    # the parallel alpha evaluator instead of the original serial evaluator.
    v065._ORIGINAL_ANCHOR_LINE_SEARCH = _parallel_evaluate_anchor_line_search

    # v0.5.4 owns the shared evaluator. v0.6.0 owns expert/DAgger collection.
    # Both resolve these module globals at runtime, so swapping them here keeps
    # the mature training/guard pipeline while changing only what the policy sees.
    v054.DiagnosticRealChartMotorEnv = DiagnosticHudRealChartMotorEnv
    v054.encode_real_chart_observation = encode_hud_real_chart_observation
    v054.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    # v0.6.0 constructs v0.5.5 DAggerSequence objects. Its shape validator
    # still uses the v0.5.5 module-global input dimension, so patch that runtime
    # constant as part of the HUD process as well.
    _install_dagger_input_dimension()

    v060.encode_real_chart_observation = encode_hud_real_chart_observation
    v060.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    # These constants are used for model construction/checkpoint validation in
    # the multi-segment/resume/anchor-guard layers.
    v062.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM
    v063.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM
    v064.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    # v0.6.4 creates a fresh Adam proposal on every epoch. Cache exact repeats
    # before v0.6.5's line-search cache so repeated odd/even rollback epochs skip
    # both the BC pass and the expensive gameplay evaluations.
    v057._train_one_epoch = _cached_train_one_epoch

    v064.TRAINER_VERSION = TRAINER_VERSION
    v064.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v064.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    v064._run_signature = _hud_run_signature
    v064._checkpoint_payload = _hud_checkpoint_payload
    v064._load_progress = _hud_load_progress
    _INSTALLED = True


def _hud_run_signature(args, *, chart_path: str, train_pool, validation_window, sight_window) -> dict:
    assert _PARENT_RUN_SIGNATURE is not None
    signature = _PARENT_RUN_SIGNATURE(
        args,
        chart_path=chart_path,
        train_pool=train_pool,
        validation_window=validation_window,
        sight_window=sight_window,
    )
    signature["hud_observation"] = HUD_OBSERVATION_VERSION
    signature["hud_feedback_hold_s"] = DEFAULT_FEEDBACK_HOLD_S
    return signature


def _hud_checkpoint_payload(**kwargs) -> dict:
    assert _PARENT_CHECKPOINT_PAYLOAD is not None
    payload = _PARENT_CHECKPOINT_PAYLOAD(**kwargs)
    payload["trainer_version"] = TRAINER_VERSION
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION
    payload["input_dim"] = HUD_REAL_CHART_INPUT_DIM
    payload["hud_observation"] = HUD_OBSERVATION_VERSION
    payload["hud_feedback_hold_s"] = DEFAULT_FEEDBACK_HOLD_S
    payload["hud_semantics"] = {
        "always_visible": ["tile_bpm", "real_bpm"],
        "transient": ["last_judgement", "signed_timing_error_ms"],
        "excluded": [
            "target_timestamp",
            "time_to_next_target",
            "chart_time",
            "absolute_floor_index",
            "x_accuracy",
            "progress",
            "attempt_count",
            "kv",
            "overload_counter",
            "fatigue",
        ],
    }
    # Caches and process-level parallel evaluation are execution optimizations.
    # They are not part of the run signature, so checkpoints created before
    # either optimization remain exactly resumable.
    payload["train_proposal_cache"] = TRAIN_PROPOSAL_CACHE_VERSION
    payload["parallel_eval"] = PARALLEL_EVAL_VERSION
    return payload


def _hud_load_progress(model, path, *, current_signature: dict, anchor_count: int, device):
    assert _PARENT_LOAD_PROGRESS is not None
    try:
        return _PARENT_LOAD_PROGRESS(
            model,
            path,
            current_signature=current_signature,
            anchor_count=anchor_count,
            device=device,
        )
    except SystemExit as exc:
        message = str(exc).replace("v0.6.5", "v0.7.0").replace("v0.6.4", "v0.7.0")
        raise SystemExit(message) from None


def main() -> None:
    _install_v070()
    workers = _configured_eval_workers()
    print("=== DMDOD v0.7.0 Human-Visible HUD ===")
    print(
        f"observation={HUD_REAL_CHART_INPUT_DIM}D = old 233D + "
        "Tile BPM + Real BPM + transient judgement/error HUD"
    )
    print(
        f"feedback-hold={DEFAULT_FEEDBACK_HOLD_S:.2f}s | "
        "XAcc/progress/KV/attempt/internal timing truth remain hidden"
    )
    print(
        "backend=v0.6.4 anchor guard + v0.6.5 exact line-search cache + "
        f"exact repeated-BC proposal cache + alpha-parallel-eval({workers} workers)"
    )
    try:
        v064.main()
    finally:
        _shutdown_eval_pool()
    print(
        f"proposal-cache: hits={v065._PROPOSAL_CACHE.hits} misses={v065._PROPOSAL_CACHE.misses} "
        f"saved-line-searches={v065._PROPOSAL_CACHE.hits}"
    )
    print(
        f"train-proposal-cache: hits={_TRAIN_PROPOSAL_CACHE_HITS} "
        f"misses={_TRAIN_PROPOSAL_CACHE_MISSES} saved-bc-epochs={_TRAIN_PROPOSAL_CACHE_HITS}"
    )


if __name__ == "__main__":
    main()
