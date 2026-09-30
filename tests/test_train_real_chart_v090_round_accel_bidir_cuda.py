from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v090_round_accel as accel


class _Model:
    def __init__(self) -> None:
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state) -> None:
        self.value = int(state["value"])


@pytest.mark.parametrize("reverse_order", [False, True])
def test_uncached_round_proposals_use_cuda_for_both_parities(monkeypatch, reverse_order: bool) -> None:
    model = _Model()
    optimizer = object()
    sequences = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    calls = []

    monkeypatch.setattr(accel, "_whole_proposal_key", lambda *args, **kwargs: ("proposal", reverse_order))
    monkeypatch.setattr(accel.v070, "_TRAIN_PROPOSAL_CACHE", {})
    monkeypatch.setattr(accel.v070, "_TRAIN_PROPOSAL_CACHE_HITS", 0)
    monkeypatch.setattr(accel.v070, "_TRAIN_PROPOSAL_CACHE_MISSES", 0)
    monkeypatch.setattr(accel.cuda_bc, "cuda_requested", lambda: True)

    def fake_cuda(model_arg, sequences_arg, *, optimizer, chunk_steps, reverse_order):
        calls.append((list(sequences_arg), chunk_steps, reverse_order))
        model_arg.value = 7
        return 0.125

    monkeypatch.setattr(accel.cuda_bc, "train_on_cuda", fake_cuda)
    monkeypatch.setattr(
        accel,
        "_ORIGINAL_CACHED_TRAIN_ONE_EPOCH",
        lambda *args, **kwargs: pytest.fail("CPU fallback should not run when CUDA succeeds"),
    )

    loss = accel._prefix_cached_train_one_epoch(
        model,
        sequences,
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=reverse_order,
    )

    assert loss == pytest.approx(0.125)
    assert model.value == 7
    assert calls == [(sequences, 192, reverse_order)]
    assert accel.v070._TRAIN_PROPOSAL_CACHE[("proposal", reverse_order)][0] == pytest.approx(0.125)
    assert accel.v070._TRAIN_PROPOSAL_CACHE_MISSES == 1


def test_whole_proposal_cache_still_precedes_cuda(monkeypatch) -> None:
    model = _Model()
    monkeypatch.setattr(accel, "_whole_proposal_key", lambda *args, **kwargs: ("cached",))
    monkeypatch.setattr(accel.v070, "_TRAIN_PROPOSAL_CACHE", {("cached",): (0.25, {"value": 9})})
    monkeypatch.setattr(accel.v070, "_TRAIN_PROPOSAL_CACHE_HITS", 0)
    monkeypatch.setattr(
        accel.cuda_bc,
        "cuda_requested",
        lambda: pytest.fail("CUDA should not be queried on a whole-proposal cache hit"),
    )

    loss = accel._prefix_cached_train_one_epoch(
        model,
        [SimpleNamespace(name="a")],
        optimizer=object(),
        chunk_steps=192,
        reverse_order=False,
    )

    assert loss == pytest.approx(0.25)
    assert model.value == 9
    assert accel.v070._TRAIN_PROPOSAL_CACHE_HITS == 1
