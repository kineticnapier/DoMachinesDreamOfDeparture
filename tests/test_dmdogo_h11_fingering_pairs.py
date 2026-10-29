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

from dmdod.planning.gesture_chunk_planner_v08 import GestureEvent
from audit_dmdogo_v09_h06_improved_suffix import (
    _assignment_metrics, _quick_flags, _rapid_flags, _flag_delta,
)
from audit_dmdogo_h1_quality_evidence import compile_register, H06_BLOB_SHA
from reconstruct_dmdogo_h11_fingering_pairs import (
    FORMAT, reconstruction, compile_reconstruction, main,
)


def fixture_pair():
    events = tuple(GestureEvent(i, 1.0 + .3*i) for i in range(5))
    original = [{
        "start_event_index": i, "end_event_index": i,
        "event_count": 1, "rhythm_kind": "single",
        "start_reason": "test", "gesture_family": "single",
        "axes": [[("right_1" if i%2 else "left_1")]],
    } for i in range(5)]
    alternate = deepcopy(original)
    alternate[1]["axes"] = [["right_2"]]
    old_q = _quick_flags(events, original)
    new_q = _quick_flags(events, alternate)
    old_r = _rapid_flags(events, original)
    new_r = _rapid_flags(events, alternate)
    saved = {
        "flag_pair": [0, 1],
        "window_index": 1,
        "assignment_effects": {
            "changed_chunk_count": 1,
            "changed_chunk_indices": [1],
            "changed_event_assignment_count": 1,
            "first_changed_chunk": 1,
            "last_changed_chunk": 1,
            "permanent_rejoin_chunk": 2,
        },
        "original_assignment_metrics": _assignment_metrics(events, original),
        "forced_assignment_metrics": _assignment_metrics(events, alternate),
        "flag_effects": {
            "quick": _flag_delta(old_q,new_q),
            "rapid": _flag_delta(old_r,new_r),
            "target_quick_flag_removed": (0,1) not in new_q,
        },
    }
    return events, original, alternate, saved


def test_reconstruction_keeps_entire_event_axis_vector_and_exact_delta():
    events, old, new, saved = fixture_pair()
    result = reconstruction(
        events, old, new, saved, candidate_id="test-0-1")
    assert result["candidate_id"] == "test-0-1"
    assert len(result["alternative_axes_by_event_order"]) == 5
    assert result["alternative_axes_by_event_order"][1] == ["right_2"]
    assert result["alternative_axes_by_event_order"][0] == ["left_1"]
    assert len(result["changed_events"]) == 1
    assert result["changed_events"][0]["event_index"] == 1
    assert result["changed_chunk_indices"] == [1]
    assert result["alternate_motor_result"] == "not_executed"
    assert result["human_preference"] == "unobserved"
    assert result["independent_intervention_only"] is True
    assert result["alternative_chunks"] == new


@pytest.mark.parametrize("field,wrong", [
    ("changed_event_assignment_count", 0),
    ("changed_chunk_count", 2),
    ("first_changed_chunk", 0),
    ("last_changed_chunk", 3),
])
def test_rejects_h06_changed_event_or_chunk_mismatch(field, wrong):
    events, old, new, saved = fixture_pair()
    saved["assignment_effects"][field] = wrong
    with pytest.raises(ValueError, match="changed chunk/event"):
        reconstruction(events, old, new, saved, candidate_id="x")


def test_rejects_changed_chronological_event_coverage():
    events, old, new, saved = fixture_pair()
    with pytest.raises(ValueError, match="event coverage"):
        reconstruction(events[:-1], old, new, saved, candidate_id="x")


def test_rejects_tampered_original_and_forced_axis_metrics():
    events, old, new, saved = fixture_pair()
    saved["forced_assignment_metrics"]["axis_use_counts"]["right_2"] = 50
    with pytest.raises(ValueError, match="metrics/reuse"):
        reconstruction(events, old, new, saved, candidate_id="x")


def test_rejects_changed_flag_identity():
    events, old, new, saved = fixture_pair()
    saved["flag_effects"]["quick"]["added_count"] += 1
    with pytest.raises(ValueError, match="flag change identities"):
        reconstruction(events, old, new, saved, candidate_id="x")


def test_rejects_changed_frozen_segmentation():
    events, old, new, saved = fixture_pair()
    new[1]["rhythm_kind"] = "modified"
    with pytest.raises(ValueError, match="changed frozen segmentation"):
        reconstruction(events, old, new, saved, candidate_id="x")


def test_rejects_wrong_rejoin_claim():
    events, old, new, saved = fixture_pair()
    saved["assignment_effects"]["permanent_rejoin_chunk"] = 1
    with pytest.raises(ValueError, match="permanent rejoin"):
        reconstruction(events, old, new, saved, candidate_id="x")


def test_h11_requires_original_register_and_sources_before_execution():
    report = json.loads((ROOT / "reports" /
        "dmdogo-v09-h06-improved-suffix-r2.json").read_text(encoding="utf-8"))
    compiled = compile_register(report, input_blob_sha=H06_BLOB_SHA)
    assert compiled["cohort"]["independent_counterfactual_targets"] == 22
    assert compiled["cohort"]["lower_frozen_heuristic_cost"] == 7
    assert FORMAT == "dmdogo-h11-paired-fingering-reconstruction-v1"


def test_create_once_refuses_existing_file_without_reading_any_inputs(
        tmp_path, monkeypatch):
    output = tmp_path / "h11.json"
    output.write_text("do not overwrite", encoding="utf-8")
    names = [
        "--manifest", "--verified-report", "--h0-report",
        "--h01-report", "--h04-report", "--h05-report",
        "--h06-report", "--h1-report", "--results-root",
    ]
    args = ["h11"]
    for name in names:
        args.extend([name, str(tmp_path / "not-here.json")])
    args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        main()
    assert output.read_text(encoding="utf-8") == "do not overwrite"
