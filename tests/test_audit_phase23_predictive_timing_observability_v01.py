from __future__ import annotations

import importlib.util
import ast
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_phase23_predictive_timing_observability_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase23", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run():
    return _load().run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "reports/phase20c-frame-stock-state-20261022-r1.json",
        ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
        [
            ROOT / "reports/phase16-firsttown12-motor-ab-20261020-r1.json",
            ROOT / "reports/phase16-bool5-motor-ab-20261020-r1.json",
        ],
    )


def test_event37_counterfactual_is_identical_before_44ms_deadline() -> None:
    result = _run()
    counterfactual = result["observability"]["counterfactual"]
    assert counterfactual["status"] == "early_due_nonidentifiable"
    assert counterfactual["maximum_hud_difference"] == 0.0
    assert counterfactual["countdown_equal"]
    assert counterfactual["last_compared_at_s"] == 3.17
    assert counterfactual["first_difference_at_s"] == 3.18
    assert counterfactual["due_gap_ms_posthoc"] > 300.0


def test_event37_compatible_worlds_have_disjoint_safe_strike_windows() -> None:
    result = _run()
    assert result["lead"]["predeadline_shared_assessment"]["status"] == "lead_44ms_infeasible"
    left = result["lead"]["original_stock_counted_window_s"]
    right = result["lead"]["alternate_stock_counted_window_s"]
    assert left[1] < right[0]
    assert result["lead"]["existing_timing_lead_ms"] < 44.0


def test_event37_only_intervention_does_not_magically_target_event37_or_fix_event89() -> None:
    result = _run()
    oracle = result["reference_interventions"]["oracle_event37_due_only"]["no_fail"]
    assert oracle["selected_intervention_rows"][0]["target_ordinal"] == 36
    assert not oracle["event89_input_hit"]
    candidate = result["reference_interventions"]["hud_candidate_after_virtual_nofail_auto"]
    assert candidate["no_fail"]["selected_intervention_rows"][0]["target_ordinal"] == 37
    assert candidate["normal_fail_anchored_after_event35"]["selected_intervention_rows"][0]["judgement"] == "too_late"
    assert not candidate["no_fail"]["event89_input_hit"]
    oracle_upper_bound = result["reference_interventions"]["oracle_all_pre89_lead_abstentions"]
    assert oracle_upper_bound["no_fail"]["event89_input_hit"]


def test_generalization_is_bounded_to_saved_reports() -> None:
    result = _run()
    general = result["generalization"]
    assert general["sample_count"] == 14
    assert general["lead_ms"]["all_below_44ms"]
    assert sum(general["classification_counts"].values()) == 14
    assert "saved two-chart" in general["claim_limit"]


def test_student_primitive_has_no_chart_or_scorer_identity_api() -> None:
    import dmdod.planning.predictive_timing_observability_v01 as model

    source = inspect.getsource(model)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("adofai" in name or "envs" in name for name in imported)
    audit_source = inspect.getsource(_load()._run_reference)
    assert "RealChartMotorEnv" not in audit_source
