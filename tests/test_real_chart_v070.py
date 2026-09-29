from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v070 as trainer  # noqa: E402
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM  # noqa: E402
from dmdod.real_chart_hud_features import HUD_FEATURE_DIM, HUD_REAL_CHART_INPUT_DIM  # noqa: E402


def test_v070_uses_separate_hud_checkpoint_format_and_dimension():
    assert trainer.TRAINER_VERSION == "0.7.0-human-visible-hud"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 14
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v070_hud.pt")
    assert HUD_REAL_CHART_INPUT_DIM == REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM
    assert HUD_REAL_CHART_INPUT_DIM == 245


def test_v070_import_does_not_mutate_legacy_encoder_before_main():
    # v0.6.x checkpoints/evaluator remain usable unless the v0.7.0 trainer is
    # explicitly installed inside its own training process.
    assert REAL_CHART_INPUT_DIM == 233


def test_v070_installs_hud_width_for_legacy_dagger_validator(monkeypatch):
    # v0.6.0 creates v0.5.5 DAggerSequence objects. Their __post_init__ reads
    # the v0.5.5 module-global width at runtime, so v0.7 must update it to 245D.
    monkeypatch.setattr(trainer.v055, "REAL_CHART_INPUT_DIM", REAL_CHART_INPUT_DIM)
    trainer._install_dagger_input_dimension()
    assert trainer.v055.REAL_CHART_INPUT_DIM == HUD_REAL_CHART_INPUT_DIM

    observations = __import__("torch").zeros((3, HUD_REAL_CHART_INPUT_DIM))
    actions = __import__("torch").zeros((3, 2))
    sequence = trainer.v055.DAggerSequence(observations, actions, "hud-test")
    assert sequence.frames == 3
