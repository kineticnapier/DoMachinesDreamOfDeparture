from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_real_chart as evaluator  # noqa: E402


def test_resolve_eval_config_uses_checkpoint_signature_by_default():
    payload = {"signature": {"same_hand": False, "control_dt": 0.0125}}
    same_hand, control_dt, hand_source, control_source = evaluator._resolve_eval_config(
        payload,
        same_hand_override=None,
        control_dt_override=None,
    )
    assert same_hand is False
    assert control_dt == pytest.approx(0.0125)
    assert hand_source == "checkpoint"
    assert control_source == "checkpoint"


def test_resolve_eval_config_allows_explicit_overrides():
    payload = {"signature": {"same_hand": False, "control_dt": 0.0125}}
    same_hand, control_dt, hand_source, control_source = evaluator._resolve_eval_config(
        payload,
        same_hand_override=True,
        control_dt_override=0.02,
    )
    assert same_hand is True
    assert control_dt == pytest.approx(0.02)
    assert hand_source == "override"
    assert control_source == "override"


def test_resolve_eval_config_has_historical_safe_defaults():
    same_hand, control_dt, hand_source, control_source = evaluator._resolve_eval_config(
        {},
        same_hand_override=None,
        control_dt_override=None,
    )
    assert same_hand is True
    assert control_dt == pytest.approx(evaluator.DEFAULT_CONTROL_DT_S)
    assert hand_source == "default"
    assert control_source == "default"


def test_resolve_range_clamps_end_to_chart_duration():
    start, end = evaluator._resolve_range(12.5, 2.0, 99.0)
    assert start == pytest.approx(2.0)
    assert end == pytest.approx(12.5)


def test_resolve_range_defaults_to_full_remaining_chart():
    start, end = evaluator._resolve_range(12.5, 2.0, None)
    assert start == pytest.approx(2.0)
    assert end == pytest.approx(12.5)


def test_resolve_range_rejects_invalid_window():
    with pytest.raises(SystemExit, match="--end must be greater"):
        evaluator._resolve_range(12.5, 5.0, 4.0)


def test_checkpoint_metadata_accepts_current_encoder_shape():
    payload = {
        "input_dim": evaluator.REAL_CHART_INPUT_DIM,
        "hidden_dim": 128,
        "model_state": {},
    }
    input_dim, hidden_dim = evaluator._checkpoint_metadata(payload)
    assert input_dim == evaluator.REAL_CHART_INPUT_DIM
    assert hidden_dim == 128


def test_checkpoint_metadata_rejects_wrong_encoder_dimension():
    payload = {"input_dim": 999, "hidden_dim": 128, "model_state": {}}
    with pytest.raises(SystemExit, match="input dimension"):
        evaluator._checkpoint_metadata(payload)
