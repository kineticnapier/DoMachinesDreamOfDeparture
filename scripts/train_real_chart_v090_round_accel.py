from __future__ import annotations

"""Execution-only round acceleration for v0.9 multi-chart training.

Three optimizations are installed before the v0.9 fast/turbo stack starts:

* Hybrid guard evaluation: grouped state jobs are retained when there are enough
  surviving policy states to occupy the worker pool; when only a few states are
  alive, evaluation is flattened across state x segment so spare CPU cores work
  on different chart segments.
* Whole-proposal BC cache: repeated proposals from the exact same trusted state
  still bypass training entirely.
* Bidirectional CUDA BC dispatch: both forward and reverse round proposals are
  trained on a GPU copy when CUDA is available. The older CPU fixed-prefix cache
  remains only as a fallback when CUDA is disabled or unavailable.

Sequence order, chunk boundaries, optimizer-step count, trust alphas, guards,
and checkpoint/run signatures are unchanged.
"""

import copy
import time

import train_real_chart_v065 as v065
import train_real_chart_v070 as v070
import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090_cuda_bc as cuda_bc


ROUND_ACCEL_VERSION = "v090-hybrid-guard-bidir-bc-cuda-v3"
PREFIX_CACHE_VERSION = "forward-fixed-expert-prefix-v1"
ROUND_LOCAL_SEQUENCE_COUNT = 2

_ORIGINAL_GROUPED_EVAL = v080_fast._evaluate_state_segment_groups
_ORIGINAL_FAST_LINE_SEARCH = v080_fast._fast_line_search
_ORIGINAL_CACHED_TRAIN_ONE_EPOCH = v070._cached_train_one_epoch

_PREFIX_CACHE: dict[tuple, tuple[dict, dict, float, float]] = {}
_PREFIX_CACHE_HITS = 0
_PREFIX_CACHE_MISSES = 0
_HYBRID_FLAT_CALLS = 0
_HYBRID_GROUPED_CALLS = 0
_INSTALLED = False


def _sequence_weight(stable, chunk_steps: int) -> float:
    """Return the exact denominator used by v0.5.7's weighted BC average."""

    weights = stable.loss_weights
    total = 0.0
    for start in range(0, stable.sequence.frames, int(chunk_steps)):
        end = min(stable.sequence.frames, start + int(chunk_steps))
        total += max(float(weights[start:end].sum().item()), 1.0)
    return total


def _should_flatten(states, segments, workers: int) -> bool:
    """Use segment-level parallelism only when grouped jobs leave cores idle."""

    return int(workers) > 1 and len(states) < int(workers) and len(segments) > 1


def _hybrid_evaluate_state_segment_groups(
    model,
    states,
    segments,
    *,
    same_hand: bool,
    control_dt_s: float,
    digests=None,
):
    """Switch between grouped IPC and flat state x segment tasks adaptively."""

    global _HYBRID_FLAT_CALLS, _HYBRID_GROUPED_CALLS

    workers = v080._configured_workers()
    if _should_flatten(states, segments, workers):
        _HYBRID_FLAT_CALLS += 1
        return v080._evaluate_states_on_segments(
            model,
            states,
            segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )

    _HYBRID_GROUPED_CALLS += 1
    return _ORIGINAL_GROUPED_EVAL(
        model,
        states,
        segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        digests=digests,
    )


def _whole_proposal_key(model, sequences, optimizer, chunk_steps: int, reverse_order: bool) -> tuple:
    return (
        v070.TRAIN_PROPOSAL_CACHE_VERSION,
        v065._state_digest(model.state_dict()),
        tuple(v070._stable_sequence_signature(stable) for stable in sequences),
        v070._optimizer_signature(optimizer),
        int(chunk_steps),
        bool(reverse_order),
    )


def _prefix_key(model, prefix, optimizer, chunk_steps: int) -> tuple:
    return (
        PREFIX_CACHE_VERSION,
        v065._state_digest(model.state_dict()),
        tuple(v070._stable_sequence_signature(stable) for stable in prefix),
        v070._optimizer_signature(optimizer),
        int(chunk_steps),
    )


def _prefix_cached_train_one_epoch(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Whole-proposal cache, bidirectional CUDA, then CPU fallback paths."""

    global _PREFIX_CACHE_HITS, _PREFIX_CACHE_MISSES

    whole_key = _whole_proposal_key(model, sequences, optimizer, chunk_steps, reverse_order)
    cached = v070._TRAIN_PROPOSAL_CACHE.get(whole_key)
    if cached is not None:
        loss, state = cached
        model.load_state_dict(state)
        v070._TRAIN_PROPOSAL_CACHE_HITS += 1
        print(f"bc-perf: full-cache hit loss={float(loss):.6f}")
        return float(loss)

    # Both parities are expensive when the trusted model changes after an ACCEPT.
    # The old forward-prefix cache only helps when the exact prefix start state
    # repeats, so prefer CUDA for every uncached proposal when available.
    if cuda_bc.cuda_requested():
        started = time.perf_counter()
        try:
            loss = cuda_bc.train_on_cuda(
                model,
                sequences,
                optimizer=optimizer,
                chunk_steps=chunk_steps,
                reverse_order=reverse_order,
            )
        except Exception as exc:
            parity = "rev" if reverse_order else "fwd"
            print(f"bc-cuda: parity={parity} fallback CPU ({type(exc).__name__}: {exc})")
        else:
            v070._TRAIN_PROPOSAL_CACHE[whole_key] = (
                float(loss),
                copy.deepcopy(model.state_dict()),
            )
            v070._TRAIN_PROPOSAL_CACHE_MISSES += 1
            parity = "rev" if reverse_order else "fwd"
            print(f"bc-perf: parity={parity} cuda={time.perf_counter() - started:.2f}s")
            return float(loss)

    # CPU fallback retains the existing forward-prefix optimization.
    if reverse_order or len(sequences) <= ROUND_LOCAL_SEQUENCE_COUNT:
        started = time.perf_counter()
        loss = _ORIGINAL_CACHED_TRAIN_ONE_EPOCH(
            model,
            sequences,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=reverse_order,
        )
        print(f"bc-perf: parity={'rev' if reverse_order else 'fwd'} full={time.perf_counter() - started:.2f}s")
        return float(loss)

    original_train = v070._ORIGINAL_TRAIN_ONE_EPOCH
    if original_train is None:
        return _ORIGINAL_CACHED_TRAIN_ONE_EPOCH(
            model,
            sequences,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=reverse_order,
        )

    prefix = list(sequences[:-ROUND_LOCAL_SEQUENCE_COUNT])
    suffix = list(sequences[-ROUND_LOCAL_SEQUENCE_COUNT:])
    key = _prefix_key(model, prefix, optimizer, chunk_steps)
    started = time.perf_counter()
    prefix_cached = _PREFIX_CACHE.get(key)

    if prefix_cached is None:
        prefix_loss = float(
            original_train(
                model,
                prefix,
                optimizer=optimizer,
                chunk_steps=chunk_steps,
                reverse_order=False,
            )
        )
        prefix_weight = sum(_sequence_weight(stable, chunk_steps) for stable in prefix)
        _PREFIX_CACHE[key] = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
            prefix_loss,
            prefix_weight,
        )
        _PREFIX_CACHE_MISSES += 1
        prefix_mode = "miss"
    else:
        prefix_state, optimizer_state, prefix_loss, prefix_weight = prefix_cached
        model.load_state_dict(prefix_state)
        optimizer.load_state_dict(optimizer_state)
        _PREFIX_CACHE_HITS += 1
        prefix_mode = "hit"

    suffix_loss = float(
        original_train(
            model,
            suffix,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=False,
        )
    )
    suffix_weight = sum(_sequence_weight(stable, chunk_steps) for stable in suffix)
    total_weight = prefix_weight + suffix_weight
    loss = (
        prefix_loss * prefix_weight + suffix_loss * suffix_weight
    ) / max(1.0, total_weight)

    v070._TRAIN_PROPOSAL_CACHE[whole_key] = (float(loss), copy.deepcopy(model.state_dict()))
    v070._TRAIN_PROPOSAL_CACHE_MISSES += 1
    elapsed = time.perf_counter() - started
    print(
        f"bc-perf: parity=fwd prefix={prefix_mode} fixed={len(prefix)} "
        f"local={len(suffix)} time={elapsed:.2f}s"
    )
    return float(loss)


def _timed_fast_line_search(*args, **kwargs):
    started = time.perf_counter()
    result = _ORIGINAL_FAST_LINE_SEARCH(*args, **kwargs)
    print(
        f"guard-perf: time={time.perf_counter() - started:.2f}s "
        f"flat={_HYBRID_FLAT_CALLS} grouped={_HYBRID_GROUPED_CALLS}"
    )
    return result


def install_round_acceleration() -> None:
    """Install round scheduling/cache/CUDA optimizations before trainer setup."""

    global _INSTALLED
    if _INSTALLED:
        return

    workers = v080._configured_workers()
    v080_fast.DEFAULT_ANCHOR_BATCH_SIZE = max(2, int(workers))
    v080_fast._evaluate_state_segment_groups = _hybrid_evaluate_state_segment_groups
    v080_fast._fast_line_search = _timed_fast_line_search

    v070._cached_train_one_epoch = _prefix_cached_train_one_epoch
    _INSTALLED = True

    print(
        f"round-accel={ROUND_ACCEL_VERSION} workers={workers} "
        f"anchor-wave={v080_fast.DEFAULT_ANCHOR_BATCH_SIZE} "
        f"bc-cuda-bidir={'on' if cuda_bc.cuda_requested() else 'off'} "
        "cpu-prefix-fallback=on hybrid-guard=on"
    )


def stats() -> dict[str, int]:
    return {
        "prefix_hits": _PREFIX_CACHE_HITS,
        "prefix_misses": _PREFIX_CACHE_MISSES,
        "hybrid_flat_calls": _HYBRID_FLAT_CALLS,
        "hybrid_grouped_calls": _HYBRID_GROUPED_CALLS,
    }
