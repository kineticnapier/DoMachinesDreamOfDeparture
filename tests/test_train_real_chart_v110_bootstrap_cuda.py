from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v062 as v062
import train_real_chart_v110_bootstrap_cuda as bc


def test_installer_replaces_bootstrap_multi_epoch(monkeypatch) -> None:
    monkeypatch.setattr(bc, "_INSTALLED", False)
    monkeypatch.setattr(v062, "_bootstrap_multi_epoch", bc._BASE_BOOTSTRAP_MULTI_EPOCH)

    bc.install_bootstrap_cuda_acceleration()

    assert v062._bootstrap_multi_epoch is bc._bootstrap_multi_epoch_cuda


def test_cpu_fallback_preserves_bootstrap_call(monkeypatch) -> None:
    seen = {}

    def fake_base(model, expert_pairs, *, optimizer, chunk_steps, reverse_order):
        seen.update(
            model=model,
            expert_pairs=expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=reverse_order,
        )
        return 1.25

    monkeypatch.setattr(bc.cuda_bc, "cuda_requested", lambda: False)
    monkeypatch.setattr(bc, "_BASE_BOOTSTRAP_MULTI_EPOCH", fake_base)

    model = object()
    pairs = [(torch.zeros((1, 2)), torch.zeros((1, 2)))]
    optimizer = object()
    result = bc._bootstrap_multi_epoch_cuda(
        model,
        pairs,
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=True,
    )

    assert result == 1.25
    assert seen == {
        "model": model,
        "expert_pairs": pairs,
        "optimizer": optimizer,
        "chunk_steps": 192,
        "reverse_order": True,
    }
