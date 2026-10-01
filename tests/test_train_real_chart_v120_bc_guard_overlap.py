from __future__ import annotations

import copy
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v070 as v070
import train_real_chart_v090_cuda_bc as cuda_bc
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v120_bc_guard_overlap as overlap


def _model(value: float) -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(float(value))
    return model


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([[float(value)]], dtype=torch.float32)}


def _done_future(result) -> Future:
    future = Future()
    future.set_result(result)
    return future


def test_matching_prefetch_applies_exact_state_and_populates_cache(monkeypatch) -> None:
    key = ("reverse", "exact")
    model = _model(1.0)
    result = overlap._PrefetchResult(loss=0.125, state=_state(7.0), elapsed_s=3.0)
    pending = overlap._PendingPrefetch(key=key, future=_done_future(result), launched_at=0.0)

    monkeypatch.setattr(overlap, "_PENDING", pending)
    monkeypatch.setattr(overlap, "_PREFETCH_CONSUMED", 0)
    monkeypatch.setattr(overlap, "_PREFETCH_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(overlap, "_PREFETCH_HIDDEN_SECONDS", 0.0)
    monkeypatch.setattr(v070, "_TRAIN_PROPOSAL_CACHE", {})
    monkeypatch.setattr(v070, "_TRAIN_PROPOSAL_CACHE_MISSES", 0)

    loss = overlap._consume_matching_prefetch(model, key)

    assert loss == pytest.approx(0.125)
    assert model.weight.item() == pytest.approx(7.0)
    assert key in v070._TRAIN_PROPOSAL_CACHE
    assert v070._TRAIN_PROPOSAL_CACHE_MISSES == 1
    assert overlap._PENDING is None
    assert overlap._PREFETCH_CONSUMED == 1


def test_mismatched_prefetch_is_never_applied(monkeypatch) -> None:
    model = _model(1.0)
    pending = overlap._PendingPrefetch(
        key=("old-base", True),
        future=_done_future(
            overlap._PrefetchResult(loss=1.0, state=_state(99.0), elapsed_s=1.0)
        ),
        launched_at=0.0,
    )
    monkeypatch.setattr(overlap, "_PENDING", pending)

    assert overlap._consume_matching_prefetch(model, ("new-base", True)) is None
    assert model.weight.item() == pytest.approx(1.0)
    assert overlap._PENDING is pending


def test_forward_prefetch_is_consumed_by_reverse_from_same_base(monkeypatch) -> None:
    model = _model(0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    calls: list[bool] = []

    def proposal_key(model_arg, sequences, optimizer_arg, chunk_steps, reverse_order):
        del sequences, optimizer_arg, chunk_steps
        return (round(float(model_arg.weight.item()), 6), bool(reverse_order))

    def foreground_train(
        model_arg,
        sequences,
        *,
        optimizer,
        chunk_steps,
        reverse_order,
    ):
        del sequences, optimizer, chunk_steps
        calls.append(bool(reverse_order))
        with torch.no_grad():
            model_arg.weight.add_(20.0 if reverse_order else 10.0)
        return 2.0 if reverse_order else 1.0

    def launch(base_model, sequences, *, optimizer, chunk_steps):
        del sequences, optimizer, chunk_steps
        key = (round(float(base_model.weight.item()), 6), True)
        result = overlap._PrefetchResult(
            loss=2.0,
            state=_state(float(base_model.weight.item()) + 20.0),
            elapsed_s=5.0,
        )
        overlap._PENDING = overlap._PendingPrefetch(
            key=key,
            future=_done_future(result),
            launched_at=0.0,
        )

    monkeypatch.setattr(round_accel, "_whole_proposal_key", proposal_key)
    monkeypatch.setattr(cuda_bc, "cuda_requested", lambda: True)
    monkeypatch.setattr(overlap, "_ORIGINAL_CACHED_TRAIN", foreground_train)
    monkeypatch.setattr(overlap, "_launch_reverse_prefetch", launch)
    monkeypatch.setattr(overlap, "_PENDING", None)
    monkeypatch.setattr(v070, "_TRAIN_PROPOSAL_CACHE", {})
    monkeypatch.setattr(v070, "_TRAIN_PROPOSAL_CACHE_MISSES", 0)

    forward_loss = overlap._overlapped_train_one_epoch(
        model,
        [],
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=False,
    )
    assert forward_loss == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(10.0)

    # A rejected forward proposal restores the same trusted base before e02.
    model.load_state_dict(_state(0.0))
    reverse_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reverse_loss = overlap._overlapped_train_one_epoch(
        model,
        [],
        optimizer=reverse_optimizer,
        chunk_steps=192,
        reverse_order=True,
    )

    assert reverse_loss == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(20.0)
    # Reverse was supplied by the prefetch, not recomputed by foreground_train.
    assert calls == [False]


def test_changed_base_discards_prefetch_and_recomputes_reverse(monkeypatch) -> None:
    model = _model(5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    calls: list[bool] = []

    def proposal_key(model_arg, sequences, optimizer_arg, chunk_steps, reverse_order):
        del sequences, optimizer_arg, chunk_steps
        return (round(float(model_arg.weight.item()), 6), bool(reverse_order))

    def foreground_train(
        model_arg,
        sequences,
        *,
        optimizer,
        chunk_steps,
        reverse_order,
    ):
        del sequences, optimizer, chunk_steps
        calls.append(bool(reverse_order))
        with torch.no_grad():
            model_arg.weight.add_(3.0)
        return 3.0

    stale = overlap._PendingPrefetch(
        key=(0.0, True),
        future=_done_future(
            overlap._PrefetchResult(loss=2.0, state=_state(20.0), elapsed_s=1.0)
        ),
        launched_at=0.0,
    )
    monkeypatch.setattr(round_accel, "_whole_proposal_key", proposal_key)
    monkeypatch.setattr(overlap, "_ORIGINAL_CACHED_TRAIN", foreground_train)
    monkeypatch.setattr(overlap, "_PENDING", stale)

    loss = overlap._overlapped_train_one_epoch(
        model,
        [],
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=True,
    )

    assert loss == pytest.approx(3.0)
    assert calls == [True]
    assert model.weight.item() == pytest.approx(8.0)
    assert overlap._PENDING is None
