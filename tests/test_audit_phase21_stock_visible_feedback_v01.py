from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_phase21_stock_visible_feedback_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase21", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase21_keeps_unverified_automatic_failmiss_out_of_conservative_surface() -> None:
    result = _load().run(ROOT / "reports/phase20c-frame-stock-state-20261022-r1.json")
    invisible = result["conservative_failmiss_unknown"]["invisible_events"]
    assert len(invisible) == 4
    assert {row["category"] for row in invisible} == {"fail_miss"}
    assert result["verdicts"] == ["STOCK_VISIBLE_FEEDBACK_PARTIAL"]


def test_phase21_observation_times_do_not_claim_hidden_occurrence_time() -> None:
    result = _load().run(ROOT / "reports/phase20c-frame-stock-state-20261022-r1.json")
    milestone = result["phase16_milestones"]
    assert milestone["fixture_failmiss_observed_s"] == pytest.approx(3.26)
    assert milestone["id38_very_late_observed_s"] == pytest.approx(3.28)
    assert milestone["warning_before_id38_physical_ms_if_failmiss_visible"] < 1.001
    assert not result["causal_conclusions"]["safe_44ms_intervention"]
