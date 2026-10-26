from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08,
    GestureEvent,
    _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import (
    _cost_terms,
    _force_suffix,
    _next,
    _seed_state,
)
from audit_dmdogo_v09_h06_improved_suffix import (
    FORMAT,
    _assignment_metrics,
    _fixed_original_suffix,
    _force_suffix_detailed,
    _quick_flags,
)


def _events(gaps: list[float]) -> tuple[GestureEvent, ...]:
    due = .5
    output = [GestureEvent(0, due)]
    for gap in gaps:
        due += gap
        output.append(GestureEvent(len(output), due))
    return tuple(output)


def test_detailed_forced_suffix_preserves_frozen_search_result():
    events = _events([.03125] * 11)
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    candidate = _candidate_gestures(windows[0])[0]
    seed = _next(_seed_state(events[0].due_s), windows[0], candidate, None,
                 frozen_candidate=False)
    assert seed is not None

    old = _force_suffix(seed, windows, 0)
    detailed = _force_suffix_detailed(seed, windows, 0, candidate)

    assert detailed["reached_terminal"] == old["reached_terminal"]
    assert detailed["final_cost"] == old["final_cost"]
    assert detailed["final_cost_terms"] == old["final_cost_terms"]
    assert detailed["suffix_candidates_evaluated"] == old[
        "suffix_candidates_evaluated"]
    assert len(detailed["chunks"]) == len(windows)


def test_fixed_original_suffix_is_an_independent_single_seam_control():
    events = _events([.03125] * 11)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    windows = planner._windows(events)
    original = [_chunk_record(chunk) for chunk in plan.chunks]
    candidate = _candidate_gestures(windows[0])[0]
    seed = _next(_seed_state(events[0].due_s), windows[0], candidate, None,
                 frozen_candidate=False)
    assert seed is not None

    result = _fixed_original_suffix(seed, windows, original, 0)
    assert result["reached_terminal"] is True
    assert result["final_cost"] is not None
    assert set(result["final_cost_terms"]) == {
        "gesture_preference", "hand_exposure", "axis_reuse_margin",
        "intra_gesture_hand_changes", "intra_gesture_depth_distance",
        "chunk_join_transition",
    }


def test_quick_flags_report_new_and_removed_pairs_without_claiming_identity():
    events = _events([.05, .05, .05])
    original = [
        {"start_event_index": 0, "end_event_index": 1,
         "axes": [["left_1"], ["left_2"]]},
        {"start_event_index": 2, "end_event_index": 3,
         "axes": [["left_4"], ["left_3"]]},
    ]
    alternate = [
        {"start_event_index": 0, "end_event_index": 1,
         "axes": [["left_1"], ["left_2"]]},
        {"start_event_index": 2, "end_event_index": 3,
         "axes": [["right_1"], ["right_4"]]},
    ]
    before = _quick_flags(events, original)
    after = _quick_flags(events, alternate)
    assert set(before) == {(1, 2)}
    assert set(after) == {(2, 3)}


def test_assignment_metrics_measure_reuse_and_hand_load_only():
    events = _events([.1, .1, .1])
    chunks = [{
        "start_event_index": 0,
        "end_event_index": 3,
        "axes": [["left_1"], ["left_2"], ["left_1"], ["right_1"]],
    }]
    metrics = _assignment_metrics(events, chunks)
    assert metrics["axis_use_counts"] == {
        "left_1": 2, "left_2": 1, "right_1": 1}
    assert metrics["minimum_axis_reuse_ms"] == pytest.approx(200.)
    assert metrics["axis_reuse_under_150ms_count"] == 0
    assert metrics["hand_axis_uses"] == {"left": 3, "right": 1}
    assert metrics["longest_single_hand_event_run"] == 3


def test_contract_is_descriptive_and_h06_specific():
    assert "h06" in FORMAT
