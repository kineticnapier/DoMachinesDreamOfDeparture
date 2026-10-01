from __future__ import annotations

"""Execution-only overlap of reverse CUDA BC with forward CPU guard evaluation.

Each round starts from one trusted CPU policy and tries two deterministic fresh-
Adam BC proposals: forward sequence order, then reverse sequence order.  When the
forward proposal is rejected, the reverse proposal starts from exactly the same
trusted state and is independent of the forward guard result.  That makes it safe
to train the reverse proposal on CUDA in a background thread while the CPU process
pool evaluates the forward proposal.

If the forward proposal is accepted, the trusted state changes.  In that case the
prefetched reverse proposal is discarded and the ordinary path recomputes reverse
BC from the new trusted state.  Exact proposal keys include state, trajectories,
optimizer settings, chunk size, and parity, so a prefetch can never be applied to
a different training request.

This module changes scheduling only.  Losses, optimizer steps, proposal states,
guards, trust alphas, selection, and checkpoint semantics are unchanged.
"""

import builtins
import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

import train_real_chart_v057 as v057
import train_real_chart_v070 as v070
import train_real_chart_v090_cuda_bc as cuda_bc
import train_real_chart_v090_round_accel as round_accel


BC_GUARD_OVERLAP_VERSION = "v120-forward-guard-reverse-bc-overlap-v1"

_ORIGINAL_CACHED_TRAIN = None
_ORIGINAL_CUDA_EMIT = None
_ORIGINAL_CUDA_PRINT = None
_INSTALLED = False
_EXECUTOR: ThreadPoolExecutor | None = None
_THREAD_STATE = threading.local()

_PREFETCH_STARTED = 0
_PREFETCH_CONSUMED = 0
_PREFETCH_DISCARDED = 0
_PREFETCH_FAILED = 0
_PREFETCH_WAIT_SECONDS = 0.0
_PREFETCH_HIDDEN_SECONDS = 0.0


@dataclass(frozen=True, slots=True)
class _PrefetchResult:
    loss: float
    state: dict[str, torch.Tensor]
    elapsed_s: float


@dataclass(slots=True)
class _PendingPrefetch:
    key: tuple
    future: Future
    launched_at: float


_PENDING: _PendingPrefetch | None = None


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dmdod-bc-prefetch")
    return _EXECUTOR


def _quiet_cuda_emit(event: str, **payload) -> None:
    assert _ORIGINAL_CUDA_EMIT is not None
    if getattr(_THREAD_STATE, "quiet_cuda", False):
        return
    _ORIGINAL_CUDA_EMIT(event, **payload)


def _quiet_cuda_print(*args, **kwargs) -> None:
    assert _ORIGINAL_CUDA_PRINT is not None
    if getattr(_THREAD_STATE, "quiet_cuda", False):
        return
    _ORIGINAL_CUDA_PRINT(*args, **kwargs)


def _fresh_optimizer_like(source_optimizer, model) -> torch.optim.Optimizer:
    """Create the exact fresh one-group Adam used by the foreground proposal."""

    if not isinstance(source_optimizer, torch.optim.Adam):
        raise TypeError("BC overlap currently requires Adam")
    if source_optimizer.state:
        raise ValueError("BC overlap requires a fresh optimizer with no state")
    if len(source_optimizer.param_groups) != 1:
        raise ValueError("BC overlap currently requires one Adam parameter group")

    group = source_optimizer.param_groups[0]
    return torch.optim.Adam(
        v057._policy_parameters(model),
        lr=float(group["lr"]),
        betas=tuple(float(value) for value in group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=bool(group["amsgrad"]),
        maximize=bool(group.get("maximize", False)),
    )


def _prefetch_worker(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
) -> _PrefetchResult:
    """Train one reverse proposal silently on CUDA and return its CPU state."""

    started = time.perf_counter()
    _THREAD_STATE.quiet_cuda = True
    try:
        loss = cuda_bc.train_on_cuda(
            model,
            sequences,
            optimizer=optimizer,
            chunk_steps=int(chunk_steps),
            reverse_order=True,
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        return _PrefetchResult(
            loss=float(loss),
            state=state,
            elapsed_s=time.perf_counter() - started,
        )
    finally:
        _THREAD_STATE.quiet_cuda = False


def _launch_reverse_prefetch(
    base_model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
) -> None:
    global _PENDING, _PREFETCH_STARTED

    key = round_accel._whole_proposal_key(
        base_model,
        sequences,
        optimizer,
        int(chunk_steps),
        True,
    )
    if key in v070._TRAIN_PROPOSAL_CACHE:
        return

    prefetch_optimizer = _fresh_optimizer_like(optimizer, base_model)
    future = _executor().submit(
        _prefetch_worker,
        base_model,
        list(sequences),
        optimizer=prefetch_optimizer,
        chunk_steps=int(chunk_steps),
    )
    _PENDING = _PendingPrefetch(key=key, future=future, launched_at=time.perf_counter())
    _PREFETCH_STARTED += 1
    print("bc-overlap: reverse prefetch started during forward guard")


def _drain_pending(*, reason: str) -> None:
    """Wait for and discard a stale prefetch before another CUDA BC may start."""

    global _PENDING, _PREFETCH_DISCARDED, _PREFETCH_FAILED
    pending = _PENDING
    if pending is None:
        return

    wait_started = time.perf_counter()
    try:
        pending.future.result()
    except Exception as exc:
        _PREFETCH_FAILED += 1
        print(f"bc-overlap: stale prefetch failed ({type(exc).__name__}: {exc})")
    else:
        _PREFETCH_DISCARDED += 1
        waited = time.perf_counter() - wait_started
        print(f"bc-overlap: discard stale reverse prefetch reason={reason} wait={waited:.2f}s")
    finally:
        _PENDING = None


def _consume_matching_prefetch(model, key: tuple) -> float | None:
    """Apply an exact-key prefetch; return None when no matching prefetch exists."""

    global _PENDING, _PREFETCH_CONSUMED, _PREFETCH_FAILED
    global _PREFETCH_WAIT_SECONDS, _PREFETCH_HIDDEN_SECONDS

    pending = _PENDING
    if pending is None or pending.key != key:
        return None

    wait_started = time.perf_counter()
    try:
        result = pending.future.result()
    except Exception as exc:
        _PREFETCH_FAILED += 1
        _PENDING = None
        print(f"bc-overlap: reverse prefetch failed ({type(exc).__name__}: {exc}); fallback")
        return None

    wait_s = time.perf_counter() - wait_started
    hidden_s = max(0.0, float(result.elapsed_s) - wait_s)
    model.load_state_dict(result.state)

    # Match the ordinary whole-proposal cache bookkeeping.  The prefetched work
    # becomes a real proposal only when the exact reverse request consumes it.
    if key not in v070._TRAIN_PROPOSAL_CACHE:
        v070._TRAIN_PROPOSAL_CACHE[key] = (
            float(result.loss),
            copy.deepcopy(result.state),
        )
        v070._TRAIN_PROPOSAL_CACHE_MISSES += 1

    _PREFETCH_CONSUMED += 1
    _PREFETCH_WAIT_SECONDS += wait_s
    _PREFETCH_HIDDEN_SECONDS += hidden_s
    _PENDING = None
    print(
        f"bc-overlap: reverse prefetched total={result.elapsed_s:.2f}s "
        f"wait={wait_s:.2f}s hidden={hidden_s:.2f}s"
    )
    return float(result.loss)


def _overlapped_train_one_epoch(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Foreground BC wrapper that prefetches the opposite parity when safe."""

    assert _ORIGINAL_CACHED_TRAIN is not None

    request_key = round_accel._whole_proposal_key(
        model,
        sequences,
        optimizer,
        int(chunk_steps),
        bool(reverse_order),
    )

    if reverse_order:
        prefetched = _consume_matching_prefetch(model, request_key)
        if prefetched is not None:
            return float(prefetched)
        # A different key means the forward proposal was accepted or some other
        # part of the request changed.  Do not overlap two CUDA trainers.
        if _PENDING is not None:
            _drain_pending(reason="trusted-state-changed")
        return float(
            _ORIGINAL_CACHED_TRAIN(
                model,
                sequences,
                optimizer=optimizer,
                chunk_steps=int(chunk_steps),
                reverse_order=True,
            )
        )

    # There should normally be no pending work at the next forward parity.  If a
    # prior prefetch became stale, finish it before starting another CUDA job.
    if _PENDING is not None:
        _drain_pending(reason="next-forward")

    can_prefetch = cuda_bc.cuda_requested()
    base_model = copy.deepcopy(model) if can_prefetch else None
    loss = float(
        _ORIGINAL_CACHED_TRAIN(
            model,
            sequences,
            optimizer=optimizer,
            chunk_steps=int(chunk_steps),
            reverse_order=False,
        )
    )

    if can_prefetch and base_model is not None:
        try:
            _launch_reverse_prefetch(
                base_model,
                sequences,
                optimizer=optimizer,
                chunk_steps=int(chunk_steps),
            )
        except Exception as exc:
            # Prefetch is execution-only.  Any setup failure leaves the ordinary
            # reverse call untouched.
            print(f"bc-overlap: prefetch setup skipped ({type(exc).__name__}: {exc})")
    return loss


def install_bc_guard_overlap() -> None:
    """Install after round acceleration and before v0.7 captures BC training."""

    global _ORIGINAL_CACHED_TRAIN, _ORIGINAL_CUDA_EMIT, _ORIGINAL_CUDA_PRINT, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_CACHED_TRAIN = v070._cached_train_one_epoch
    _ORIGINAL_CUDA_EMIT = cuda_bc.emit_progress
    _ORIGINAL_CUDA_PRINT = getattr(cuda_bc, "print", builtins.print)
    cuda_bc.emit_progress = _quiet_cuda_emit
    cuda_bc.print = _quiet_cuda_print
    v070._cached_train_one_epoch = _overlapped_train_one_epoch
    _INSTALLED = True
    print(
        f"bc-guard-overlap={BC_GUARD_OVERLAP_VERSION} "
        f"cuda={'on' if cuda_bc.cuda_requested() else 'off'} semantics=unchanged"
    )


def overlap_stats() -> dict[str, float | int]:
    return {
        "started": int(_PREFETCH_STARTED),
        "consumed": int(_PREFETCH_CONSUMED),
        "discarded": int(_PREFETCH_DISCARDED),
        "failed": int(_PREFETCH_FAILED),
        "wait_seconds": float(_PREFETCH_WAIT_SECONDS),
        "hidden_seconds": float(_PREFETCH_HIDDEN_SECONDS),
    }


def print_overlap_stats() -> None:
    stats = overlap_stats()
    print(
        "bc-overlap-stats: "
        f"started={stats['started']} consumed={stats['consumed']} "
        f"discarded={stats['discarded']} failed={stats['failed']} "
        f"wait={stats['wait_seconds']:.2f}s hidden={stats['hidden_seconds']:.2f}s"
    )
