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
from audit_dmdogo_v09_h014_regret_autopsy import replay_with_snapshots
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    generate, select_alternative,
)
from audit_dmdogo_v09_h015_admissible_suffix_bound import (
    certificate, suffix_lower_bounds,
)
from audit_dmdogo_v09_h016_cold_block_lower_bounds import (
    FORMAT, FUTURE_BLOCK_MAX, GRACE, cold_block_lower_bound,
    cold_suffix_bounds, inspect_frontier, summarize, _source_trial_map,
    reconcile_h015_targets,
)


def test_invalid_or_empty_cold_block_rejected():
    events = tuple(GestureEvent(i, 1.0+i*.04) for i in range(12))
    windows = GestureChunkPlannerV08()._windows(events)
    with pytest.raises(ValueError, match="one or two"):
        cold_block_lower_bound(windows, 0, 0)
    with pytest.raises(ValueError, match="one or two"):
        cold_block_lower_bound(windows, 0, 3)
    with pytest.raises(ValueError, match="one or two"):
        cold_block_lower_bound(windows, -1, 1)
    assert FUTURE_BLOCK_MAX == 2


@pytest.mark.parametrize("trial_id", [0, 5, 4115])
def test_cold_block_suffix_is_admissible_for_real_frozen_path(trial_id):
    events, _ = generate(trial_id)
    planner = GestureChunkPlannerV08()
    original = planner.plan(events)
    assert original.feasible
    windows = planner._windows(events)
    frozen = [_chunk_record(c) for c in original.chunks]
    scores = [0.0]
    branch = _seed_state(events[0].due_s)
    for wi, win in enumerate(windows):
        choices = [
            c for c in _candidate_gestures(win)
            if c.family == frozen[wi]["gesture_family"]
            and [list(x) for x in c.axes] == frozen[wi]["axes"]]
        assert len(choices) == 1
        nxt = _next(
            branch, win, choices[0],
            windows[wi-1].events[-1].due_s if wi else None,
            frozen_candidate=True)
        assert nxt is not None
        assert nxt.score >= branch.score - 1e-9
        branch = nxt
        scores.append(branch.score)
    assert round(scores[-1], 6) == original.metrics["beam_cost"]
    floors = cold_suffix_bounds(windows)
    assert len(floors) == 4
    for method, suffix in floors.items():
        assert len(suffix) == len(windows)+1, method
        assert suffix[-1] == 0.0, method
        for wi, optimistic in enumerate(suffix):
            actual_remaining = scores[-1] - scores[wi]
            assert optimistic <= actual_remaining + 1e-7, (
                trial_id, method, wi, optimistic, actual_remaining)
    assert all(
        floors["maximum"][wi] + 1e-7 >= floors["cold_two"][wi]
        and floors["cold_two"][wi] + 1e-7 >= floors["cold_one"][wi]
        and floors["cold_one"][wi] + 1e-7 >= floors["intrinsic_h015"][wi]
        for wi in range(len(floors["maximum"])))


def test_block_retains_intra_chunk_hand_exposure():
    # A single 4-event tight burst has nonzero own-hand exposure,
    # even with zero inherited state.
    events = tuple(GestureEvent(i, 1. + i * .028) for i in range(4))
    win = GestureChunkPlannerV08()._windows(events)
    assert len(win) == 1
    cold = cold_block_lower_bound(win, 0, 1)
    old = suffix_lower_bounds(win)[0]
    assert cold > old + 1e-6


def test_pair_linkage_keeps_cross_chunk_score_or_equal():
    # Two forced chunks from a small synthetic level: the two-chunk
    # exact minimum cannot be smaller than separately minimized cold
    # chunk costs, since all inherited effects are nonnegative.
    events, _ = generate(1)
    windows = GestureChunkPlannerV08()._windows(events)
    assert len(windows) >= 2
    individual = (
        cold_block_lower_bound(windows, 0, 1)
        + cold_block_lower_bound(windows, 1, 2))
    together = cold_block_lower_bound(windows, 0, 2)
    assert together + 1e-7 >= individual


def test_false_certificate_with_winning_tagged_terminal_is_detected():
    events, _ = generate(4115)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    windows = planner._windows(events)
    frozen = [_chunk_record(c) for c in plan.chunks]
    join = 8
    target = select_alternative(events, frozen, windows, join)
    trace = replay_with_snapshots(
        events, frozen, target, policy="full_quota8")
    rows = trace["snapshots"]
    vals = cold_suffix_bounds(windows)["maximum"]
    # The known protected outcome is below the frozen incumbent;
    # hence no honest bound can certify a loss of its entire frontier.
    known_v08 = float(plan.metrics["beam_cost"])
    assert trace["terminal_best_tagged"] is not None
    assert trace["terminal_best_tagged"]["final_cost"] < known_v08 - 1e-6
    for snap in rows:
        tagged = snap["retained_best_tagged"]
        if tagged is not None:
            outcome = certificate(
                tagged["score"], vals[snap["window_index"]+1],
                known_v08)
            assert not outcome["certified"]


def test_inspect_frontier_rejects_tampered_saved_baseline():
    events, _ = generate(1)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    windows = planner._windows(events)
    frozen = [_chunk_record(c) for c in plan.chunks]
    target = select_alternative(events, frozen, windows, 1)
    snapshots = replay_with_snapshots(
        events, frozen, target, policy="full_quota8")["snapshots"]
    previous = suffix_lower_bounds(windows)
    baseline = []
    for row in snapshots:
        wi = row["window_index"]
        tagged = row["retained_best_tagged"]
        item = certificate(
            tagged["score"] if tagged is not None else None,
            previous[wi+1],
            float(plan.metrics["beam_cost"]))
        baseline.append({
            "window_index": wi,
            "best_tagged_current_score": (
                round(tagged["score"], 9)
                if tagged is not None else None),
            **item,
        })
    # First verify exact original certification before corruption.
    summary = inspect_frontier(
        snapshots, windows, join=1,
        incumbent_cost=float(plan.metrics["beam_cost"]),
        known_tagged_final_cost=None,
        baseline_records=baseline)
    assert summary["maximum"]["certified_window_count"] >= (
        summary["intrinsic_h015"]["certified_window_count"])
    baseline[0]["margin_lb_minus_incumbent"] = 1e6
    with pytest.raises(ValueError, match="baseline certificate differs"):
        inspect_frontier(
            snapshots, windows, join=1,
            incumbent_cost=float(plan.metrics["beam_cost"]),
            known_tagged_final_cost=None, baseline_records=baseline)


def test_summary_keeps_methods_and_source_groups_separate():
    fake = [{
        "trial_index": 4115,
        "targets": [
            {"five_chunk_rule_regret": True,
             "tagged_final_beats_incumbent": True,
             "methods": {
                 label: {"certified_target": False,
                         "early_certificate": False,
                         "certified_window_count": 0,
                         "first_certificate_offset": None}
                 for label in ("intrinsic_h015", "cold_one",
                               "cold_two", "maximum")
             }}
        ],
    }]
    result = summarize(fake)
    assert result["trial_count"] == 1
    assert result["target_count"] == 1
    assert result["tagged_final_beats_incumbent"] == 1
    assert result["known_regret_target_count"] == 1
    assert result["methods"]["maximum"]["certifiable_targets"] == 0


def test_fixed_h015_group_census_is_strict():
    fake = {
        "known_regret_cases": {"trials": [
            {"trial_index": 4115}, {"trial_index": 4127}]},
    }
    result = _source_trial_map(fake, "known_regret_cases")
    assert list(result) == [4115, 4127]
    fake["known_regret_cases"]["trials"] = [
        {"trial_index": 4115}, {"trial_index": 4115}]
    with pytest.raises(ValueError, match="census"):
        _source_trial_map(fake, "known_regret_cases")


def test_not_a_new_pruning_policy():
    assert GRACE == 5
    assert FORMAT == "dmdogo-v09-h016-cold-block-lower-bounds-v1"
    assert GestureChunkPlannerV08().beam_width == 24


def test_h015_historical_single_target_subset_is_not_expanded():
    # Two H0.12 join candidates exist but H0.15 intentionally recorded
    # only the known regret target for each historical failure.
    original = {"targets": [
        {"window_index": 8, "kind": "regret"},
        {"window_index": 14, "kind": "other"},
    ]}
    saved = {"target_count": 1, "targets": [
        {"window_index": 8, "kind": "regret"}]}
    triples = reconcile_h015_targets(saved, original)
    assert triples == [(8, saved["targets"][0], original["targets"][0])]
    saved["target_count"] = 2
    with pytest.raises(ValueError, match="target count"):
        reconcile_h015_targets(saved, original)
    saved = {"target_count": 1, "targets": [
        {"window_index": 30, "kind": "wrong"}]}
    with pytest.raises(ValueError, match="targeted join"):
        reconcile_h015_targets(saved, original)
