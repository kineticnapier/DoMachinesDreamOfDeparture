from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v053 as trainer  # noqa: E402


def test_action_frame_counts_separate_press_release_and_neutral():
    actions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [0.1, -0.1],
        ],
        dtype=torch.float32,
    )

    press, release, neutral = trainer._action_frame_counts(actions)
    assert (press, release, neutral) == (2, 1, 2)


def test_neutral_positive_push_gets_extra_safety_penalty():
    target = torch.zeros((1, 2), dtype=torch.float32)
    safe = torch.tensor([[0.04, 0.0]], dtype=torch.float32)
    unsafe = torch.tensor([[0.20, 0.0]], dtype=torch.float32)
    negative = torch.tensor([[-0.20, 0.0]], dtype=torch.float32)

    safe_loss = trainer._bc_loss(safe, target)
    unsafe_loss = trainer._bc_loss(unsafe, target)
    negative_loss = trainer._bc_loss(negative, target)

    assert unsafe_loss > negative_loss
    assert unsafe_loss > safe_loss * 10.0


def test_press_error_is_weighted_more_than_release_and_neutral_error():
    prediction = torch.zeros((1, 2), dtype=torch.float32)

    press_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    release_target = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
    neutral_target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    press_loss = trainer._bc_loss(prediction, press_target)
    release_loss = trainer._bc_loss(prediction, release_target)
    neutral_loss = trainer._bc_loss(prediction, neutral_target)

    assert press_loss > release_loss > neutral_loss


def test_v053_checkpoint_is_not_silently_compatible_with_v052():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 2
    assert trainer.TRAINER_VERSION == "0.5.3-neutral-safe-bc"
