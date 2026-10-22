from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_phase22_stock_estimator_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase22", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase22_conservative_replay_never_false_identifies_stock_floor() -> None:
    result = _load().run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "reports/phase20c-frame-stock-state-20261022-r1.json",
        ROOT / "reports/phase21-stock-visible-feedback-20261022-r1.json",
    )
    conservative = result["conservative_failmiss_unknown"]
    assert conservative["first_alignment_risk_visible_at_s"] == 3.28
    assert conservative["false_unique_identification_count"] == 0
    assert conservative["oracle_posthoc"]["unknown_remainder_retained"]
    assert "STATE_NONIDENTIFIABLE" in result["verdicts"]


def test_phase22_does_not_claim_safe_motor_intervention() -> None:
    result = _load().run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "reports/phase20c-frame-stock-state-20261022-r1.json",
        ROOT / "reports/phase21-stock-visible-feedback-20261022-r1.json",
    )
    intervention = result["event89_prevention"]
    assert intervention["available_margin_ms"] < intervention["required_motor_lead_ms"]
    assert not intervention["safe_intervention"]
    assert result["stock_runtime_verified"] is False
