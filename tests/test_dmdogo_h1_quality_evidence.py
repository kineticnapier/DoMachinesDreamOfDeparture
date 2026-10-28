from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_dmdogo_h1_quality_evidence import (
    FORMAT, H06_BLOB_SHA, TERMS, git_blob_sha,
    compile_register, main,
)


@pytest.fixture(scope="module")
def saved_source() -> dict:
    return json.loads((ROOT / "reports" /
        "dmdogo-v09-h06-improved-suffix-r2.json").read_text(encoding="utf-8"))


def test_pinned_saved_h06_source_digest_and_evidence_inventory(saved_source):
    file = ROOT / "reports" / "dmdogo-v09-h06-improved-suffix-r2.json"
    assert git_blob_sha(file) == H06_BLOB_SHA
    report = compile_register(saved_source, input_blob_sha=H06_BLOB_SHA)
    assert report["format"] == FORMAT
    assert report["cohort"]["chart_count"] == 3
    assert report["cohort"]["total_events"] == 10231
    assert report["cohort"]["independent_counterfactual_targets"] == 22
    assert report["cohort"]["lower_frozen_heuristic_cost"] == 7
    assert report["cohort"]["no_lower_frozen_heuristic_cost"] == 15
    assert len(report["candidates"]) == 22
    assert len({x["candidate_id"] for x in report["candidates"]}) == 22
    assert sum(x["abstract_score"]["lower_frozen_heuristic_cost"]
               for x in report["candidates"]) == 7


def test_human_preference_and_alternate_motor_not_fabricated(saved_source):
    report = compile_register(saved_source, input_blob_sha=H06_BLOB_SHA)
    gate = report["quality_evidence_gate"]
    assert gate["overall_human_quality_verified"] is False
    assert gate["motor_alternate_success_verified"] is False
    assert gate["alternate_physical_motor"] == "not_observed"
    assert gate["human_fingering_preference"] == "not_observed"
    assert gate["blind_human_ab"].startswith("not_review_ready")
    assert report["motor_executed"] is False
    assert report["human_demonstrations_used"] is False
    assert report["validation20_or_final_accessed"] is False
    assert all(x["evidence"]["alternate_motor_feasibility"] == "not_tested"
               and x["evidence"]["human_preference"] == "not_measured"
               and not x["evidence"]["blind_review_ready"]
               for x in report["candidates"])


def test_full_six_term_cost_differences_reconcile(saved_source):
    report = compile_register(saved_source, input_blob_sha=H06_BLOB_SHA)
    for case in report["candidates"]:
        costs = case["abstract_score"]
        assert set(costs["six_term_deltas"]) == set(TERMS)
        assert sum(costs["six_term_deltas"].values()) == pytest.approx(
            costs["forced_minus_original"], abs=2e-6)
        assert costs["lower_frozen_heuristic_cost"] == (
            costs["forced_minus_original"] < -1e-7)
        assert (case["abstract_exploratory_proxies"]
                ["forced_suffix"]["abstract_reuse_under_80ms_count"] == 0)


@pytest.mark.parametrize("field,new", [
    ("human_likeness_verified", True),
    ("physical_motor_executed", True),
    ("causal_effect_established", True),
    ("planner_motor_timing_runner_changed", True),
])
def test_rejects_unsupported_source_claims(saved_source, field, new):
    changed = deepcopy(saved_source)
    changed[field] = new
    with pytest.raises(ValueError, match="unrecognized or repurposed"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)


def test_rejects_changed_objective_components(saved_source):
    changed = deepcopy(saved_source)
    changed["cases"][0]["evidence"][0]["cost"]["six_component_deltas"][
        "hand_exposure"] += 0.1
    with pytest.raises(ValueError, match="six-term"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)


def test_rejects_missing_target_and_duplicate_seam(saved_source):
    changed = deepcopy(saved_source)
    changed["cases"][0]["evidence"].pop()
    with pytest.raises(ValueError, match="per-chart"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)
    changed = deepcopy(saved_source)
    changed["cases"][0]["evidence"][1]["flag_pair"] = changed[
        "cases"][0]["evidence"][0]["flag_pair"]
    with pytest.raises(ValueError, match="repeated comparison candidate ID"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)


def test_rejects_misreported_quick_flag_count(saved_source):
    changed = deepcopy(saved_source)
    changed["cases"][0]["evidence"][0]["flag_effects"]["quick"][
        "removed_count"] += 1
    with pytest.raises(ValueError, match="inconsistent quick"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)


def test_rejects_forged_reuse_violation(saved_source):
    changed = deepcopy(saved_source)
    changed["cases"][0]["evidence"][0]["forced_assignment_metrics"][
        "abstract_reuse_under_80ms_count"] = 1
    with pytest.raises(ValueError, match="violates 80ms"):
        compile_register(changed, input_blob_sha=H06_BLOB_SHA)


def test_create_once_output_does_not_overwrite(tmp_path, monkeypatch):
    source = ROOT / "reports" / "dmdogo-v09-h06-improved-suffix-r2.json"
    output = tmp_path / "h1.json"
    monkeypatch.setattr(sys, "argv", [
        "h1", "--h06-report", str(source), "--output", str(output)])
    main()
    before = output.read_bytes()
    assert json.loads(before)["cohort"][
        "independent_counterfactual_targets"] == 22
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == before


def test_source_pinning_rejects_edited_report_file(tmp_path, monkeypatch):
    source = ROOT / "reports" / "dmdogo-v09-h06-improved-suffix-r2.json"
    edited = tmp_path / "h06-edited.json"
    edited.write_bytes(source.read_bytes().replace(
        b'"case_count": 3', b'"case_count": 4', 1))
    assert git_blob_sha(edited) != H06_BLOB_SHA
    monkeypatch.setattr(sys, "argv", [
        "h1", "--h06-report", str(edited),
        "--output", str(tmp_path / "h1.json")])
    with pytest.raises(ValueError, match="pinned original Git blob"):
        main()
