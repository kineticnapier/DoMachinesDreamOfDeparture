from __future__ import annotations

import sys
from pathlib import Path

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
