from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dmdogo_v09_h01_conditions import analyze_conditions
from analyze_dmdogo_v09_h02_boundary_cf import (
    _counterfactual_at_flag, _local_replan,
)
from analyze_dmdogo_v09_h03_boundary_controls import analyze_flag, analyze_case
from test_dmdogo_v09_style_proxies import sample


def synthetic(*, original_chunk_sizes=(4, 4)):
    size = sum(original_chunk_sizes)
    if size == 8:
        axes = ["left_1", "left_2", "left_3", "left_4"] * 2
    elif size == 6:
        axes = ["left_1", "left_2", "left_3"] * 2
    else:
        raise ValueError("unsupported synthetic sample")
    summary, telemetry, replay = sample(axes=axes, gaps=[50.] * (size - 1))
    chunks = []
    cursor = 0
    for length in original_chunk_sizes:
        chunks.append({
            "start_event_index": cursor,
            "end_event_index": cursor + length - 1,
            "event_count": length,
            "gesture_family": "same_hand_roll",
            "axes": [[axis] for axis in axes[cursor:cursor+length]],
        })
        cursor += length
    summary["abstract_schedule"]["chunks"] = chunks
    rows = [
        {"event_index": i, "due_ms": d["due_ms"],
         "axes": tuple(d["axes"]), "desired_axes": tuple(d["desired_axes"])}
        for i, d in enumerate(telemetry["decisions"])
    ]
    join = original_chunk_sizes[0]
    h02 = _counterfactual_at_flag(rows, chunks, join)
    return summary, telemetry, replay, rows, h02


def test_same_boundary_control_and_moved_partitions_are_separated():
    s, t, r, rows, saved = synthetic()
    compared = analyze_flag(rows, s["abstract_schedule"]["chunks"], saved)
    assert compared["original_quick_flag"] is True
    assert compared["same_boundary"]["abstract_feasible"] is True
    assert compared["feasible_alternative_count"] == saved[
        "alternative_feasible_count"]
    assert compared["alternatives_retaining_quick"] == saved[
        "alternative_quick_flag_still_present"]
    assert compared["alternatives_removing_quick"] == saved[
        "alternative_quick_flag_absent"]
    assert compared["classification"] in {
        "replanning_alone_suffices_for_this_flag",
        "partition_associated_under_same_surrogate",
    }
    assert compared["feasible_same_chunk_count_partitions"] == 0
    assert compared["feasible_extra_chunk_partitions"] > 0


def test_same_surrogate_baseline_matches_fresh_replan():
    s, t, r, rows, saved = synthetic()
    expected = _local_replan(rows, 0, 8, (4, 4))
    assert saved["same_boundary_surrogate"] == expected
    actual = analyze_flag(rows, s["abstract_schedule"]["chunks"], saved)
    assert actual["same_boundary"]["surrogate_cost"] == expected["cost"]


def test_six_note_region_has_equal_chunk_count_counterfactuals():
    s, t, r, rows, saved = synthetic(original_chunk_sizes=(3, 3))
    compared = analyze_flag(rows, s["abstract_schedule"]["chunks"], saved)
    assert compared["feasible_same_chunk_count_partitions"] > 0
    assert compared["feasible_extra_chunk_partitions"] >= 0


def test_grammar_enumeration_explains_absence_without_human_claim():
    s, t, r, rows, saved = synthetic()
    compared = analyze_flag(rows, s["abstract_schedule"]["chunks"], saved)
    for alt in compared["alternatives"]:
        if alt["abstract_feasible"]:
            g = alt["grammar_audit"]
            assert g["candidate_sequences_at_join"] > 0
            assert g["sequences_allowing_quick_jump_at_join"] == 0
            assert g["grammar_structurally_excludes_jump"]
    assert compared["all_feasible_alternatives_grammar_exclude_quick"]


def test_case_alignment_with_h01_detected_pair():
    s, t, r, rows, saved = synthetic()
    h01 = analyze_conditions(s, t, r)
    h02 = {"chart_sha256": s["chart_sha256"],
           "event_count": 8, "cases": [saved]}
    actual = analyze_case(s, t, r, h01, h02)
    assert actual["flag_count"] == 1
    assert actual["feasible_moved_seam_variants"] == saved[
        "alternative_feasible_count"]
    assert actual["same_boundary_still_flag_count"] + actual[
        "same_boundary_removed_flag_count"] == 1


def test_tampered_same_boundary_baseline_refused():
    s, t, r, rows, saved = synthetic()
    wrong = copy.deepcopy(saved)
    wrong["same_boundary_surrogate"]["cost"] -= 10.
    with pytest.raises(ValueError, match="same-boundary control"):
        analyze_flag(rows, s["abstract_schedule"]["chunks"], wrong)


def test_tampered_alternative_assignment_refused():
    s, t, r, rows, saved = synthetic()
    wrong = copy.deepcopy(saved)
    alt = next(p for p in wrong["alternatives"] if p["feasible"])
    alt["axes"] = [["left_1"] for _ in alt["axes"]]
    with pytest.raises(ValueError):
        analyze_flag(rows, s["abstract_schedule"]["chunks"], wrong)


def test_duplicate_alternative_partition_refused():
    s, t, r, rows, saved = synthetic()
    wrong = copy.deepcopy(saved)
    wrong["alternatives"].append(copy.deepcopy(wrong["alternatives"][0]))
    with pytest.raises(ValueError, match="duplicated"):
        analyze_flag(rows, s["abstract_schedule"]["chunks"], wrong)


def test_cross_chart_and_flag_mismatch_refused():
    s, t, r, rows, saved = synthetic()
    h01 = analyze_conditions(s, t, r)
    h02 = {"chart_sha256": "b" * 64, "event_count": 8, "cases": [saved]}
    with pytest.raises(ValueError, match="identity mismatch"):
        analyze_case(s, t, r, h01, h02)
    h02["chart_sha256"] = s["chart_sha256"]
    h02["cases"][0]["flag_pair"] = [1, 2]
    with pytest.raises(ValueError, match="flags do not match"):
        analyze_case(s, t, r, h01, h02)
