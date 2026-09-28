from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v054 as trainer  # noqa: E402


def test_actuation_loss_prefers_commands_that_cross_physical_margins():
    target = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    good = torch.tensor(
        [
            [0.85, 0.00],
            [-0.60, 0.00],
            [0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    weak = torch.tensor(
        [
            [0.25, 0.00],
            [-0.10, 0.00],
            [0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    assert trainer._actuation_loss(good, target) < trainer._actuation_loss(weak, target)


def test_actuation_loss_penalizes_positive_neutral_force_asymmetrically():
    target = torch.zeros((2, 2), dtype=torch.float32)
    safe_lift = torch.full((2, 2), -0.10, dtype=torch.float32)
    unsafe_push = torch.full((2, 2), 0.10, dtype=torch.float32)

    safe_loss = trainer._actuation_loss(safe_lift, target)
    unsafe_loss = trainer._actuation_loss(unsafe_push, target)

    assert unsafe_loss > safe_loss
    assert unsafe_loss > trainer.MSE_COEF * unsafe_push.square().mean()


def test_actuation_loss_has_explicit_press_release_and_neutral_thresholds():
    assert trainer.PRESS_MARGIN >= 0.7
    assert trainer.RELEASE_MARGIN <= -0.3
    assert trainer.NEUTRAL_PUSH_LIMIT <= 0.05


def test_actuation_loss_rejects_wrong_shape():
    with pytest.raises(ValueError):
        trainer._actuation_loss(torch.zeros(3), torch.zeros(3))
