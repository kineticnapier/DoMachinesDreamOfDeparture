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
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h07_shadow_lane import (
    BEAM_WIDTH, Proposal, choose_beam, _delta_flags, trace_one_slot,
)


def proposal(cost, protected=False):
    state = _seed_state(0.)
    state.score = float(cost)
    state.tags = frozenset({0}) if protected else frozenset()
    return Proposal(state, None, {}, 1, 0)


def test_one_slot_replaces_worst_regular_when_tagged_rank_is_outside_beam():
    proposals = [proposal(i) for i in range(5)] + [proposal(5, True)]
    selected, status = choose_beam(proposals, width=4)
    assert [item.state.score for item in selected] == [0, 1, 2, 5]
    assert len(selected) == 4
    assert status["best_protected_expanded_rank"] == 6
    assert status["displaced_rank"] == 4
    assert status["slot_replacement"] is True
    assert status["protected_natural_survival"] is False


def test_no_intervention_when_tag_is_already_in_top_width():
    proposals = [proposal(1, True)] + [proposal(i) for i in (0, 2, 3, 4)]
    selected, status = choose_beam(proposals, width=4)
    assert [item.state.score for item in selected] == [0, 1, 2, 3]
    assert status["slot_replacement"] is False
    assert status["protected_natural_survival"] is True
    assert len(selected) == 4


def test_abstract_dead_end_is_not_misreported_as_retention():
    proposals = [proposal(i) for i in range(6)]
    selected, status = choose_beam(proposals, width=4)
    assert [p.state.score for p in selected] == [0, 1, 2, 3]
    assert status["protected_feasible"] is False
    assert status["slot_replacement"] is False


def test_score_ties_stable_and_width_preserved():
    proposals = [proposal(1), proposal(1), proposal(1), proposal(1),
                 proposal(1, True)]
    selected, status = choose_beam(proposals, width=4)
    assert selected == [proposals[0], proposals[1], proposals[2], proposals[4]]
    assert status["slot_replacement"] is True
    assert len(selected) == 4


def test_bad_width_and_empty_proposals_fail_closed():
    with pytest.raises(ValueError):
        choose_beam([proposal(0)], width=1)
    with pytest.raises(ValueError):
        choose_beam([], width=24)


def test_quick_flag_comparison_uses_locations_not_pretend_identity():
    before = {(1, 2): {"event_pair": [1, 2]}}
    after = {(4, 5): {"event_pair": [4, 5]}}
    change = _delta_flags(before, after)
    assert change["added_count"] == 1
    assert change["removed_count"] == 1
    assert change["added"][0]["event_pair"] == [4, 5]


def _synthetic_target():
    events = tuple(GestureEvent(i, .5 + i * .03125)
                   for i in range(16))
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    windows = planner._windows(events)
    assert len(windows) == 4
    frozen = [_chunk_record(c) for c in plan.chunks]

    # Independent exact computation of the first beam step and second-step
    # expanded rank. No hand-selected rank is supplied to the audit.
    first = []
    for candidate in _candidate_gestures(windows[0]):
        branch = _next(_seed_state(events[0].due_s), windows[0],
                       candidate, None,
                       frozen_candidate=(
                           [list(x) for x in candidate.axes] == frozen[0]["axes"]
                           and candidate.family == frozen[0]["gesture_family"]))
        if branch is not None:
            first.append(branch)
    first.sort(key=lambda item: item.score)
    first = first[:BEAM_WIDTH]
    assert sum(s.frozen_prefix for s in first) == 1
    options = _candidate_gestures(windows[1])
    expanded = []
    for parent_index, state in enumerate(first, 1):
        for candidate_index, candidate in enumerate(options):
            branch = _next(state, windows[1], candidate,
                           windows[0].events[-1].due_s,
                           frozen_candidate=False)
            if branch is not None:
                expanded.append((branch.score, parent_index,
                                 candidate_index, candidate))
    expanded.sort(key=lambda t: t[0])
    chosen = next((rank, ci, candidate)
                  for rank, (_, parent_index, ci, candidate)
                  in enumerate(expanded, 1)
                  if first[parent_index - 1].frozen_prefix
                  and [list(x) for x in candidate.axes] != frozen[1]["axes"])
    rank, ci, candidate = chosen
    target = {
        "flag_pair": [3, 4],
        "window_index": 1,
        "nonquick_candidate_index": ci,
        "nonquick_join_expanded_rank": rank,
        "nonquick_axes": [list(x) for x in candidate.axes],
    }
    return events, plan, frozen, target


def test_shadow_candidate_preserved_without_widening_original_beam():
    events, plan, frozen, target = _synthetic_target()
    output = trace_one_slot(
        events, frozen, target,
        expected_cost=plan.metrics["beam_cost"],
        expected_evaluated=plan.candidates_evaluated)
    assert output["width"] == BEAM_WIDTH
    assert output["admission_expanded_rank"] == target[
        "nonquick_join_expanded_rank"]
    assert output["protected_descendants_reached_terminal"]
    assert output["beam_winner"]["axis_reuse_under_80ms"] == 0
    assert output["best_protected_terminal"]["axis_reuse_under_80ms"] == 0


def test_wrong_target_rank_rejected_before_reporting():
    events, plan, frozen, target = _synthetic_target()
    target["nonquick_join_expanded_rank"] += 1
    with pytest.raises(ValueError, match="candidate rank/identity mismatch"):
        trace_one_slot(
            events, frozen, target,
            expected_cost=plan.metrics["beam_cost"],
            expected_evaluated=plan.candidates_evaluated)


def test_wrong_source_beam_cost_rejected():
    events, plan, frozen, target = _synthetic_target()
    with pytest.raises(ValueError, match="Planner/source mismatch"):
        trace_one_slot(
            events, frozen, target,
            expected_cost=plan.metrics["beam_cost"] + 1.,
            expected_evaluated=plan.candidates_evaluated)
