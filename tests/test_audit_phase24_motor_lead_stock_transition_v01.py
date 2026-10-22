from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_phase24_motor_lead_stock_transition_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase24", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result() -> dict:
    return _load().run(
        phase16_path=ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        phase23_path=ROOT / "reports/phase23-predictive-timing-observability-20261022-r3.json",
        train_dir=ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
    )


def test_phase24_preserves_causal_and_production_boundaries() -> None:
    report = result()
    assert report["scope"]["production_changed"] is False
    assert report["student_boundary"]["production_modules_modified"] == []
    assert report["event37"]["counterfactual_pre_44ms"]["maximum_hud_difference"] == 0.0
    assert report["student_boundary"]["oracle_exact_due_at_3_180"].startswith("explicit hypothetical")


def test_phase24_lead_matrix_keeps_four_dimensions_separate() -> None:
    report = result()
    assert report["lead_origin"]["observed_cold_axis_range_ms"] == pytest.approx([39.0, 43.0])
    assert report["lead_origin"]["reconstructed_nominal_lead_ms"] == pytest.approx(44.0)
    current = report["event37"]["lead_cases"]["existing_point_forecast_3_190"]
    by_lead = {round(row["lead_s"] * 1000): row for row in current}
    assert not by_lead[44]["nominal_lead_temporally_available"]
    assert by_lead[30]["nominal_lead_temporally_available"]
    assert not by_lead[30]["cold_keydown_by_due"]
    assert by_lead[30]["cold_keydown_in_stock_window"]
    assert not any(row["all_counted_hit_conditions"] for row in current)


def test_nofail_candidate_is_phase_conditional_and_does_not_save_event89() -> None:
    report = result()
    runs = [
        row
        for row in report["nofail_delayed_candidate"]["phase_and_microtime_runs"]
        if row["candidate_time_s"] == 3.251
    ]
    assert len(runs) == 10
    assert all(row["candidate_transition"]["target_ordinal"] == 37 for row in runs)
    assert all(row["candidate_transition"]["judgement"] == "very_late" for row in runs)
    assert all(row["selected_events"]["89"][0]["automatic_miss"] for row in runs)
    assert all(row["base_input_mismatch_count"] == 46 for row in runs)


def test_boundary_keydown_exposes_update_phase_sensitivity() -> None:
    report = result()
    runs = [
        row
        for row in report["nofail_delayed_candidate"]["phase_and_microtime_runs"]
        if row["candidate_time_s"] == 3.249
    ]
    assert {row["candidate_transition"]["target_ordinal"] for row in runs} == {36, 37}
    phase9 = next(row for row in runs if row["phase_ms"] == 9.0)
    assert phase9["candidate_transition"]["target_ordinal"] == 36
    assert phase9["candidate_transition"]["judgement"] == "too_late"


def test_normal_fail_never_promotes_3251_candidate_to_event37() -> None:
    report = result()
    runs = report["nofail_delayed_candidate"]["normal_fail_runs_at_3_251"]
    assert len(runs) == 10
    assert all(row["candidate_transition"]["target_ordinal"] == 36 for row in runs)
    assert all(row["candidate_transition"]["judgement"] == "too_late" for row in runs)
    assert all(row["failed"] for row in runs)
