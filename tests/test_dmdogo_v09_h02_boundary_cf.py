from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dmdogo_v09_h02_boundary_cf import (
    _quick, _rapid_runs, _split_lengths, _counterfactual_at_flag,
    _local_replan, analyze_case,
)
from analyze_dmdogo_v09_h01_conditions import analyze_conditions
from test_dmdogo_v09_style_proxies import sample


def fixture():
    axes = ["left_1", "left_2", "left_3", "left_4"] * 2
    s, t, r = sample(axes=axes, gaps=[50.] * 7)
    rows = [dict(event_index=x["event_index"], due_ms=x["due_ms"],
                 axes=tuple(x["axes"]), desired_axes=tuple(x["desired_axes"]))
            for x in t["decisions"]]
    return s, t, r, rows


def test_split_options_respect_frozen_four_note_cap():
    splits = _split_lengths(8)
    assert (4, 4) in splits
    assert (2, 4, 2) in splits
    assert all(len(x) in (2, 3) and max(x) <= 4 and sum(x) == 8
               for x in splits)


def test_fixed_axis_labels_do_not_alter_quick_predicate():
    s, t, r, rows = fixture()
    assert _quick(rows, 4) is True
    assert _rapid_runs(rows) == [(0, 7)]
    output = _counterfactual_at_flag(rows, s["abstract_schedule"]["chunks"], 4)
    assert output["original_quick_flag"] is True
    assert output["fixed_axes_quick_flag_under_all_relabellings"] is True
    assert output["boundary_relabels_tested"] >= 1
    assert output["boundary_relabels_tested"] == output[
        "fixed_axes_relabels_making_original_pair_within_chunk"]
    assert output["fixed_axes_original_rapid_crossing_runs"] == 1
    assert all(x["fixed_axes_rapid_crossing_runs"] in (0, 1)
               for x in output["alternatives"])


def test_local_surrogate_is_not_automatically_a_physical_run():
    s, t, r, rows = fixture()
    result = _local_replan(rows, 0, 8, (4, 4))
    assert result["feasible"] is True
    assert result["different_from_v08"] >= 0
    assert len(result["axes"]) == 8
    assert result["candidates_evaluated"] > 0


def test_case_tracks_verified_h01_flags_and_no_new_training():
    s, t, r, _ = fixture()
    checked = analyze_conditions(s, t, r)
    report = analyze_case(s, t, r, checked)
    assert report["original_quick_count"] == 1
    assert report["original_rapid_count"] == 1
    assert report["rollup"]["flags_unchanged_when_only_labels_change"] == 1
    assert report["rollup"]["targeted_flags"] == 1
    assert report["original_abstract_reuse"]["violations"] == 0


def test_mismatched_h01_flag_locations_fail_closed():
    s, t, r, _ = fixture()
    checked = analyze_conditions(s, t, r)
    changed = copy.deepcopy(checked)
    changed["quick_same_hand_depth_jump"]["detected"][0]["end_event_index"] += 1
    with pytest.raises(ValueError, match="H0.1 detected regions differ"):
        analyze_case(s, t, r, changed)


def test_original_abstract_reuse_violation_refused():
    s, t, r, rows = fixture()
    rows[4]["axes"] = ("left_4",)
    with pytest.raises(ValueError, match="frozen input violates"):
        _counterfactual_at_flag(rows, s["abstract_schedule"]["chunks"], 4)


def test_duplicate_chunk_coverage_refused():
    s, t, r, rows = fixture()
    chunks = copy.deepcopy(s["abstract_schedule"]["chunks"])
    chunks[1]["start_event_index"] = 3
    with pytest.raises(ValueError, match="overlapping chunks"):
        _counterfactual_at_flag(rows, chunks, 4)
