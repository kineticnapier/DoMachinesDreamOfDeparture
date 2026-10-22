from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_phase29_integrated_feasibility_v01.py"


def result():
    spec = importlib.util.spec_from_file_location("phase29", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {
        "phase16": "phase16-firsttown12-event89-causal-20261021-r2.json",
        "phase23": "phase23-predictive-timing-observability-20261022-r3.json",
        "phase25": "phase25-stock-runtime-parity-20261022-r1.json",
        "phase26": "phase26-state-aware-motor-latency-20261022-r1.json",
        "phase27": "phase27-robust-scheduler-20261022-r1.json",
        "phase28": "phase28-counterfactual-interventions-20261022-r1.json",
    }
    return module.run({key: ROOT / "reports" / value for key, value in names.items()})


def test_integrated_audit_never_promotes_virtual_results_to_physical_hits() -> None:
    report = result()
    virtual = report["research_candidates_not_physically_executed"]
    assert virtual["state_aware_latency"]["hit"] is None
    assert virtual["robust_scheduler"]["hit"] is None
    assert not report["scope"]["new_motor_ab_executed"]


def test_event89_remains_oracle_only_and_baseline_regression_is_visible() -> None:
    report = result()
    interventions = report["research_candidates_not_physically_executed"]["counterfactual_interventions"]
    assert interventions["event89_oracle_recovered"]
    assert not interventions["event89_hud_recovered"]
    assert report["physical_results_already_executed"]["lost_ordinals_posthoc"] == [89]
    assert not report["physical_results_already_executed"]["existing_hit_preservation"]


def test_no_false_safe_or_oracle_leakage() -> None:
    report = result()
    virtual = report["research_candidates_not_physically_executed"]
    assert virtual["robust_scheduler"]["false_safe_decisions"] == 0
    assert virtual["counterfactual_interventions"]["false_safe_decisions"] == 0
    assert not report["causality_and_leakage"]["oracle_leakage_detected"]
    assert report["causality_and_leakage"]["counterfactual_prefix_max_difference"] == 0.0


def test_runtime_uncertainty_blocks_adoption() -> None:
    report = result()
    assert report["stock_state"]["runtime_parity"] == "RUNTIME_PARITY_UNVERIFIED"
    assert not report["adoption"]["ready_for_production"]
    assert "NO_SAFE_INTERVENTION_ESTABLISHED" in report["verdicts"]
