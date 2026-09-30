from __future__ import annotations

"""Execution-only round acceleration for v0.9 multi-chart training.

Two exact optimizations are installed before the v0.9 fast/turbo stack starts:

* Hybrid guard evaluation: grouped state jobs are retained when there are enough
  surviving policy states to occupy the worker pool; when only a few states are
  alive, evaluation is flattened across state x segment so spare CPU cores work
  on different chart segments.
* Forward-BC prefix cache: round training always consists of the immutable
  anchor expert set followed by the current round expert and rollout.  With a
  fresh Adam optimizer, the immutable prefix is byte-for-byte repeatable from an
  identical trusted model state.  Cache both model and Adam state after that
  prefix, then train only the two round-local suffix sequences on a cache hit.

The loss, sequence order, optimizer steps, trust alphas, guards, accepted states,
and checkpoint/run signatures are unchanged.
"""

import copy
import time

import train_real_chart_v065 as v065
import train_real_chart_v070 as v070
import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast


ROUND_ACCEL_VERSION = "v090-hybrid-guard-prefix-bc-v1"
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
        # v0.8's flat evaluator uses the same exact state/segment cache and the
        # same deterministic worker implementation; only task scheduling differs.
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
    """Exact v0.7 proposal cache plus reusable immutable forward prefix."""

    global _PREFIX_CACHE_HITS, _PREFIX_CACHE_MISSES

    # Preserve v0.7's complete-proposal cache first.  This is even cheaper than
    # prefix reuse when the entire round proposal repeats exactly.
    whole_key = _whole_proposal_key(model, sequences, optimizer, chunk_steps, reverse_order)
    cached = v070._TRAIN_PROPOSAL_CACHE.get(whole_key)
    if cached is not None:
        loss, state = cached
        model.load_state_dict(state)
        v070._TRAIN_PROPOSAL_CACHE_HITS += 1
        print(f"bc-perf: full-cache hit loss={float(loss):.6f}")
        return float(loss)

    # Reverse order starts with the two round-local sequences, so the immutable
    # anchor corpus is a suffix and cannot be reused independently of them.
    # Fall back to the proven v0.7 implementation for that parity.
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
        # Defensive fallback for direct unit use before v0.7 installation.
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

    # Populate the original whole-proposal cache exactly as v0.7 would.
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
    """Install exact round scheduling/cache optimizations before trainer setup."""

    global _INSTALLED
    if _INSTALLED:
        return

    workers = v080._configured_workers()
    v080_fast.DEFAULT_ANCHOR_BATCH_SIZE = max(2, int(workers))
    v080_fast._evaluate_state_segment_groups = _hybrid_evaluate_state_segment_groups
    v080_fast._fast_line_search = _timed_fast_line_search

    # v0.7's installer later assigns v057._train_one_epoch from this module
    # global, so replacing it now makes the prefix cache compose naturally with
    # the existing exact whole-proposal cache.
    v070._cached_train_one_epoch = _prefix_cached_train_one_epoch
    _INSTALLED = True

    print(
        f"round-accel={ROUND_ACCEL_VERSION} workers={workers} "
        f"anchor-wave={v080_fast.DEFAULT_ANCHOR_BATCH_SIZE} "
        "bc-prefix-cache=on hybrid-guard=on"
    )


def stats() -> dict[str, int]:
    return {
        "prefix_hits": _PREFIX_CACHE_HITS,
        "prefix_misses": _PREFIX_CACHE_MISSES,
        "hybrid_flat_calls": _HYBRID_FLAT_CALLS,
        "hybrid_grouped_calls": _HYBRID_GROUPED_CALLS,
    }
