from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h08_protected_quota import trace_quota
from audit_dmdogo_v09_h09_quota_mechanism import (
    TARGET_QUOTA, _first, _permanent_from, summarize_windows,
    trace_window_mechanism, compare_exact_h08,
)


def synthetic():
    events = tuple(GestureEvent(i, 1. + .03125 * i) for i in range(16))
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    windows = planner._windows(events)
    chunks = [_chunk_record(chunk) for chunk in plan.chunks]
    assert len(windows) == 4
    # Reproduce the frozen prefix and generate all globally-ranked
    # candidates at the next chunk. Select a non-frozen candidate.
    first = []
    for candidate in _candidate_gestures(windows[0]):
        branch = _next(
            _seed_state(events[0].due_s), windows[0], candidate, None,
            frozen_candidate=(
                [list(x) for x in candidate.axes] == chunks[0]["axes"]
                and candidate.family == chunks[0]["gesture_family"]),
        )
        if branch is not None:
            first.append(branch)
    first.sort(key=lambda state: state.score)
    first = first[:24]
    assert sum(s.frozen_prefix for s in first) == 1
    expansions = []
    for parent_rank, parent in enumerate(first, 1):
        for ci, candidate in enumerate(_candidate_gestures(windows[1])):
            nxt = _next(parent, windows[1], candidate,
                        windows[0].events[-1].due_s,
                        frozen_candidate=False)
            if nxt is not None:
                expansions.append((nxt.score, parent.frozen_prefix, ci, candidate))
    expansions.sort(key=lambda x: x[0])
    rank, picked = next(
        (rank, item) for rank, item in enumerate(expansions, 1)
        if item[1] and [list(row) for row in item[3].axes] != chunks[1]["axes"]
    )
    target = {
        "flag_pair": [3, 4],
        "window_index": 1,
        "nonquick_candidate_index": picked[2],
        "nonquick_join_expanded_rank": rank,
        "nonquick_axes": [list(row) for row in picked[3].axes],
    }
    return events, chunks, plan, target


def record(i, natural, retained, *, replaced=0, best_rank=None,
           untagged=1):
    return {
        "window_index": i,
        "natural_tagged_count": natural,
        "retained_tagged_count": retained,
        "replaced_slots": replaced,
        "best_tagged_expanded_rank": best_rank,
        "best_tagged_minus_best_untagged": (
            None if not untagged else -1.0 if best_rank == 1 else 2.0),
        "feasible_untagged_count": untagged,
    }


def test_first_and_permanent_window_semantics():
    rows = [record(3, 1, 1), record(4, 8, 8),
            record(5, 2, 8, replaced=6), record(6, 8, 8)]
    result = summarize_windows(rows)
    assert result["first_natural_quota_window"] == 4
    assert result["first_permanently_natural_quota_window"] == 6
    assert result["rescue_windows"] == 1
    assert result["displaced_untagged_slots"] == 6
    assert result["natural_at_least_quota_windows"] == 2
    assert result["quota_underfilled_windows"] == 1


def test_terminal_not_naturally_full_means_no_permanent_quota_window():
    rows = [record(10, 8, 8), record(11, 2, 8, replaced=6)]
    assert summarize_windows(rows)["first_permanently_natural_quota_window"] is None
    assert _first(rows, lambda x: x["window_index"] == 11) == 11
    assert _permanent_from(rows, lambda x: x["natural_tagged_count"] == 8) is None


def test_undercount_and_discontinuous_windows_fail_closed():
    with pytest.raises(ValueError, match="not contiguous"):
        summarize_windows([record(2, 1, 1), record(4, 2, 2)])
    with pytest.raises(ValueError, match="impossible protected replacement"):
        summarize_windows([record(2, 0, 1, replaced=2)])
    with pytest.raises(ValueError, match="no post-join"):
        summarize_windows([])


def test_h09_quota8_exactly_matches_h08_on_synthetic_frozen_chart():
    events, chunks, plan, target = synthetic()
    saved = trace_quota(
        events, chunks, target, quota=TARGET_QUOTA,
        expected_cost=plan.metrics["beam_cost"],
        expected_evaluated=plan.candidates_evaluated,
        reference_final_cost=plan.metrics["beam_cost"],
    )
    diagnostic = trace_window_mechanism(events, chunks, target)
    compare_exact_h08(diagnostic, saved)
    assert diagnostic["admission_rank"] == target[
        "nonquick_join_expanded_rank"]
    assert diagnostic["summary"]["windows_after_and_including_join"] == 3
    assert diagnostic["summary"]["rescue_windows"] == len(
        saved["slot_replacement_windows"])
    assert diagnostic["terminal_best_is_tagged"] == saved[
        "beam_winner"]["is_from_protected_lineage"]


def test_mismatching_saved_rescue_windows_rejected():
    events, chunks, plan, target = synthetic()
    saved = trace_quota(
        events, chunks, target, quota=TARGET_QUOTA,
        expected_cost=plan.metrics["beam_cost"],
        expected_evaluated=plan.candidates_evaluated,
        reference_final_cost=plan.metrics["beam_cost"],
    )
    diagnostic = trace_window_mechanism(events, chunks, target)
    saved["total_displaced_nonprotected_slots"] += 1
    with pytest.raises(ValueError, match="diverge"):
        compare_exact_h08(diagnostic, saved)


def test_mismatching_saved_cost_rejected():
    events, chunks, plan, target = synthetic()
    saved = trace_quota(
        events, chunks, target, quota=TARGET_QUOTA,
        expected_cost=plan.metrics["beam_cost"],
        expected_evaluated=plan.candidates_evaluated,
        reference_final_cost=plan.metrics["beam_cost"],
    )
    diagnostic = trace_window_mechanism(events, chunks, target)
    saved["beam_winner"]["final_cost"] += 1
    with pytest.raises(ValueError, match="diverge"):
        compare_exact_h08(diagnostic, saved)


def test_original_nonquick_rank_tampering_rejected():
    events, chunks, plan, target = synthetic()
    target["nonquick_join_expanded_rank"] += 1
    with pytest.raises(ValueError, match="alternative mismatch"):
        trace_window_mechanism(events, chunks, target)


def test_v08_frozen_width_remains_24():
    assert GestureChunkPlannerV08().beam_width == 24
    assert TARGET_QUOTA == 8
