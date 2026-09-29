from __future__ import annotations

import io
import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v090 as v090


def test_console_stream_escapes_characters_missing_from_cp932() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp932", errors="strict")

    v090._configure_text_stream(stream)
    stream.write("집")
    stream.flush()

    assert stream.errors == "backslashreplace"
    assert raw.getvalue() == b"\\uc9d1"


def test_imminent_press_mask_looks_back_from_teacher_press() -> None:
    target = torch.zeros((6, 2), dtype=torch.float32)
    target[4, 1] = 1.0
    mask = v090._imminent_press_mask(target, 3)

    assert not bool(mask[0, 1])
    assert bool(mask[1, 1])
    assert bool(mask[2, 1])
    assert bool(mask[3, 1])
    assert bool(mask[4, 1])
    assert not bool(mask[5, 1])
    assert not bool(mask[:, 0].any())


def test_press_persistence_penalizes_immediate_cancellation() -> None:
    target = torch.zeros((5, 2), dtype=torch.float32)
    target[2:4, 1] = 1.0

    held = torch.zeros_like(target)
    held[0, 1] = 0.55
    held[1, 1] = 0.45
    held[2, 1] = 0.80
    held[3, 1] = 0.80

    cancelled = held.clone()
    cancelled[1, 1] = -0.05

    held_loss = v090._press_persistence_loss(held, target)
    cancelled_loss = v090._press_persistence_loss(cancelled, target)

    assert cancelled_loss.item() > held_loss.item()


def test_zero_persistence_coefficient_matches_legacy_loss() -> None:
    target = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    predicted = torch.tensor(
        [[0.0, 0.5], [0.0, -0.1], [0.0, 0.8]], dtype=torch.float32
    )

    legacy = v090._ORIGINAL_ACTUATION_LOSS(predicted, target)
    v090_loss = v090._press_persistence_loss(predicted, target, coef=0.0)

    assert torch.allclose(v090_loss, legacy)
