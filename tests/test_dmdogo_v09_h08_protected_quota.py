from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureEvent, GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h07_shadow_lane import (
    Proposal, choose_beam, trace_one_slot,
)
from audit_dmdogo_v09_h08_protected_quota import (
    QUOTAS, BEAM_WIDTH, _candidate_target,
    _h07_projection, choose_quota, trace_quota,
)


def proposal(cost: float, *, tagged=False):
    branch = _seed_state(0.0)
    branch.score = float(cost)
    if tagged:
        branch.tags = frozenset({0})
    return Proposal(branch, None, {}, 1, 0)


def test_quota_one_matches_h07_selection_for_natural_and_rescued_cases():
    candidates = ([proposal(n) for n in range(12)]
                  + [proposal(12, tagged=True), proposal(13, tagged=True)])
    for offset in range(14):
        rows = candidates[offset:] + candidates[:offset]
        expected, old = choose_beam(rows, width=8)
        actual, new = choose_quota(rows, quota=1, width=8)
        assert actual == expected
        assert new["displaced_nonprotected_count"] == int(old["slot_replacement"])
        assert new["feasible_tagged_expansions"] == 2


def test_quota_protects_multiple_descendants_without_expanding_beam():
    candidates = ([proposal(n) for n in range(10)]
                  + [proposal(n, tagged=True) for n in (10, 11, 12, 13, 14)])
    winners, status = choose_quota(candidates, quota=4, width=8)
    assert [row.state.score for row in winners] == [0, 1, 2, 3, 10, 11, 12, 13]
    assert len(winners) == 8
    assert status["tagged_kept"] == 4
    assert status["displaced_nonprotected_count"] == 4
    assert status["requested_quota_satisfied"] is True


def test_quota_less_than_available_respects_natural_top24():
    candidates = [proposal(0, tagged=True), proposal(1, tagged=True),
                  proposal(2, tagged=True), proposal(3)]
    chosen, status = choose_quota(candidates, quota=2, width=4)
    assert chosen == candidates
    assert status["tagged_kept"] == 3
    assert status["displaced_nonprotected_count"] == 0


def test_fewer_feasible_tagged_than_quota_does_not_fabricate_paths():
    candidates = [proposal(n) for n in range(8)] + [proposal(8, tagged=True)]
    chosen, status = choose_quota(candidates, quota=4, width=8)
    assert len(chosen) == 8
    assert sum(p.protected for p in chosen) == 1
    assert status["desired_tagged_count"] == 1
    assert status["requested_quota_satisfied"] is False


def test_no_protected_candidate_is_explicit():
    chosen, status = choose_quota([proposal(i) for i in range(8)],
                                  quota=8, width=8)
    assert len(chosen) == 8
    assert status["feasible_tagged_expansions"] == 0
    assert status["tagged_kept"] == 0
    assert status["displaced_nonprotected_count"] == 0


def test_stable_tie_order_is_not_modified():
    candidates = [proposal(1) for _ in range(6)]
    candidates += [proposal(1, tagged=True) for _ in range(3)]
    chosen, status = choose_quota(candidates, quota=2, width=6)
    assert chosen == candidates[:4] + candidates[6:8]
    assert status["tagged_kept"] == 2


@pytest.mark.parametrize("quota", [0, -1, 9])
def test_invalid_quota_fail_closed(quota):
    with pytest.raises(ValueError):
        choose_quota([proposal(0)], quota=quota, width=8)


def _example():
    events = tuple(GestureEvent(i, 1.0 + i * .03125) for i in range(16))
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    frozen = [_chunk_record(c) for c in plan.chunks]
    windows = planner._windows(events)
    states = [_seed_state(events[0].due_s)]
    for i, window in enumerate(windows[:2]):
        new = []
        for state in states:
            for ci, candidate in enumerate(_candidate_gestures(window)):
                nxt = _next(
                    state, window, candidate,
                    windows[i - 1].events[-1].due_s if i else None,
                    frozen_candidate=(
                        [list(x) for x in candidate.axes] == frozen[i]["axes"]
                        and candidate.family == frozen[i]["gesture_family"]),
                )
                if nxt is not None:
                    new.append((nxt.score, state, nxt, ci, candidate))
        new.sort(key=lambda row: row[0])
        if i == 0:
            states = [x[2] for x in new[:24]]
            continue
        available = [
            (rank, x) for rank, x in enumerate(new, 1)
            if x[1].frozen_prefix
            and [list(axis) for axis in x[4].axes] != frozen[1]["axes"]
        ]
        rank, row = available[0]
        target = {
            "flag_pair": [3, 4],
            "window_index": 1,
            "nonquick_candidate_index": row[3],
            "nonquick_join_expanded_rank": rank,
            "nonquick_axes": [list(axis) for axis in row[4].axes],
        }
        return events, plan, frozen, target
    raise AssertionError("no test target")


def test_full_quota_one_reproduces_original_h07_on_synthetic_chart():
    events, plan, frozen, target = _example()
    kwargs = {
        "expected_cost": plan.metrics["beam_cost"],
        "expected_evaluated": plan.candidates_evaluated,
        "reference_final_cost": plan.metrics["beam_cost"],
    }
    old = trace_one_slot(events, frozen, target, **kwargs)
    new = trace_quota(events, frozen, target, quota=1, **kwargs)
    assert _h07_projection(new) == _h07_projection(old)
    assert new["width"] == 24


@pytest.mark.parametrize("quota", QUOTAS)
def test_synthetic_sweep_returns_valid_abstract_paths(quota):
    events, plan, frozen, target = _example()
    actual = trace_quota(
        events, frozen, target, quota=quota,
        expected_cost=plan.metrics["beam_cost"],
        expected_evaluated=plan.candidates_evaluated,
        reference_final_cost=plan.metrics["beam_cost"],
    )
    assert actual["width"] == BEAM_WIDTH
    assert actual["quota"] == quota
    assert actual["protected_descendants_reached_terminal"]
    assert actual["best_protected_terminal"]["axis_reuse_under_80ms"] == 0
    assert actual["beam_winner"]["axis_reuse_under_80ms"] == 0
    assert actual["peak_retained_tagged_hypotheses"] <= quota or (
        actual["peak_retained_tagged_hypotheses"] > quota
        and actual["total_displaced_nonprotected_slots"] >= 0)


def test_tampered_candidate_rank_rejected():
    events, plan, frozen, target = _example()
    target["nonquick_join_expanded_rank"] += 1
    with pytest.raises(ValueError, match="original nonquick target/rank changed"):
        trace_quota(events, frozen, target, quota=2,
                    expected_cost=plan.metrics["beam_cost"],
                    expected_evaluated=plan.candidates_evaluated,
                    reference_final_cost=plan.metrics["beam_cost"])


def test_frozen_score_difference_rejected_before_running_sweep():
    events, plan, frozen, target = _example()
    with pytest.raises(ValueError, match="source/beam does not reproduce"):
        trace_quota(events, frozen, target, quota=4,
                    expected_cost=plan.metrics["beam_cost"] + 0.01,
                    expected_evaluated=plan.candidates_evaluated,
                    reference_final_cost=plan.metrics["beam_cost"])


def test_selection_matches_committed_h06_seven_cases():
    report = json.loads((Path(__file__).resolve().parents[1] /
        "reports/dmdogo-v09-h06-improved-suffix-r2.json").read_text(encoding="utf-8"))
    selected = [x for case in report["cases"] for x in case["evidence"]
                if x["improved_final_cost"]]
    assert len(selected) == 7
    for entry in selected:
        assert entry["cost"]["final_delta"] < 0
        assert entry["beam_lineage"]["join_expanded_rank"] >= 1
