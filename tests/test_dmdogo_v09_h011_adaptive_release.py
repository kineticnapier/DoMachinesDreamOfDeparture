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
from audit_dmdogo_v09_h09_quota_mechanism import trace_window_mechanism
from audit_dmdogo_v09_h011_adaptive_release import (
    QUOTA, OBSERVE_CHUNKS, _aggregate, _difference,
    quota8_diagnostic, stop_after_grace, trace_adaptive,
)


def source_fixture():
    events = tuple(GestureEvent(i, 0.5 + i * .03125) for i in range(24))
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    chunks = [_chunk_record(c) for c in plan.chunks]
    windows = planner._windows(events)
    assert len(windows) == 6
    first_candidates = []
    for candidate in _candidate_gestures(windows[0]):
        nxt = _next(
            _seed_state(events[0].due_s), windows[0], candidate, None,
            frozen_candidate=(
                [list(x) for x in candidate.axes] == chunks[0]["axes"]
                and candidate.family == chunks[0]["gesture_family"]),
        )
        if nxt is not None:
            first_candidates.append(nxt)
    first_candidates.sort(key=lambda item: item.score)
    first_candidates = first_candidates[:24]
    assert sum(item.frozen_prefix for item in first_candidates) == 1
    expansions = []
    for item in first_candidates:
        for ci, candidate in enumerate(_candidate_gestures(windows[1])):
            nxt = _next(
                item, windows[1], candidate,
                windows[0].events[-1].due_s,
                frozen_candidate=False)
            if nxt is not None:
                expansions.append((
                    nxt.score, item.frozen_prefix, ci, candidate))
    expansions.sort(key=lambda item: item[0])
    rank, item = next(
        (rank, item) for rank, item in enumerate(expansions, 1)
        if item[1]
        and [list(x) for x in item[3].axes] != chunks[1]["axes"])
    target = {
        "flag_pair": [3, 4],
        "window_index": 1,
        "nonquick_candidate_index": item[2],
        "nonquick_join_expanded_rank": rank,
        "nonquick_axes": [list(x) for x in item[3].axes],
    }
    kwargs = {
        "expected_cost": plan.metrics["beam_cost"],
        "expected_evaluated": plan.candidates_evaluated,
        "reference_final_cost": plan.metrics["beam_cost"],
    }
    return events, chunks, target, kwargs


def test_grace_rule_only_fires_on_fifth_observed_chunk_without_rank1():
    for n in range(5):
        assert not stop_after_grace(
            chunks_seen=n, ever_protected_rank1=False)
    assert stop_after_grace(
        chunks_seen=5, ever_protected_rank1=False)
    assert not stop_after_grace(
        chunks_seen=5, ever_protected_rank1=True)
    assert not stop_after_grace(
        chunks_seen=5, ever_protected_rank1=False,
        already_released=True)
    assert not stop_after_grace(
        chunks_seen=6, ever_protected_rank1=False)
    with pytest.raises(ValueError):
        stop_after_grace(chunks_seen=-1, ever_protected_rank1=False)


def test_cost_components_difference_is_exact_and_fail_closed():
    assert _difference({"a": 1.25, "b": 2}, {"a": 1, "b": 4}) == {
        "a": .25, "b": -2}
    with pytest.raises(ValueError, match="different frozen cost"):
        _difference({"a": 1}, {"b": 1})


def test_full_quota8_diagnostic_replays_exact_h08_on_synthetic_chart():
    events, chunks, target, kwargs = source_fixture()
    reference = trace_quota(events, chunks, target, quota=QUOTA, **kwargs)
    traced = trace_window_mechanism(events, chunks, target, quota=QUOTA)
    evidence = quota8_diagnostic(reference, traced)
    assert evidence["full_quota8_rescue_windows"] == (
        reference["slot_replacement_count"])
    assert evidence["full_quota8_displaced_slots"] == (
        reference["total_displaced_nonprotected_slots"])
    assert evidence["grace_chunk_count"] == min(
        OBSERVE_CHUNKS, len(traced["windows"]))
    assert sum(evidence["final_term_differences_protected_minus_winner"].values()
               ) == pytest.approx(
                   evidence["final_best_protected_minus_winner"], abs=1e-6)


def test_quota8_diagnostic_rejects_tampered_source():
    events, chunks, target, kwargs = source_fixture()
    reference = trace_quota(events, chunks, target, quota=QUOTA, **kwargs)
    traced = trace_window_mechanism(events, chunks, target, quota=QUOTA)
    reference["total_displaced_nonprotected_slots"] += 1
    with pytest.raises(ValueError, match="diverge"):
        quota8_diagnostic(reference, traced)


def test_adaptive_result_agrees_with_quota8_until_grace():
    events, chunks, target, kwargs = source_fixture()
    reference = trace_quota(events, chunks, target, quota=QUOTA, **kwargs)
    traced = trace_window_mechanism(events, chunks, target, quota=QUOTA)
    evidence = quota8_diagnostic(reference, traced)
    adaptive = trace_adaptive(events, chunks, target, **kwargs)
    assert adaptive["width"] == 24
    assert adaptive["quota_until_release"] == 8
    assert adaptive["admission_expanded_rank"] == target[
        "nonquick_join_expanded_rank"]
    assert adaptive["ever_protected_rank1_before_decision"] == (
        evidence["ever_rank1_by_grace"])
    if adaptive["quota_released"]:
        assert adaptive["release_decision_window"] == (
            target["window_index"] + OBSERVE_CHUNKS - 1)
        assert adaptive["first_post_release_window"] == (
            target["window_index"] + OBSERVE_CHUNKS)
        assert adaptive["rescue_windows"] == (
            reference["slot_replacement_windows"][:len(
                adaptive["rescue_windows"])])
    else:
        assert adaptive["beam_winner"] == reference["beam_winner"]
        assert adaptive["best_protected_terminal"] == (
            reference["best_protected_terminal"])
        assert adaptive["rescue_windows"] == (
            reference["slot_replacement_windows"])


def test_wrong_frozen_source_fails_closed_before_adaptive_results():
    events, chunks, target, kwargs = source_fixture()
    kwargs["expected_cost"] += 1
    with pytest.raises(ValueError, match="planning source mismatch"):
        trace_adaptive(events, chunks, target, **kwargs)


def test_adaptive_cohort_aggregate_tracks_savings_without_aggregating_charts():
    def sample(*, released, baseline_rescues, adaptive_rescues,
               baseline_cost, final_cost, cohort):
        return {
            "flag_pair": [3, 4],
            "h06_cohort": cohort,
            "quota8_reference": {
                "beam_winner": {
                    "final_cost": baseline_cost,
                    "delta_cost_vs_frozen_v08": baseline_cost - 10},
                "slot_replacement_count": baseline_rescues,
                "total_displaced_nonprotected_slots": 8*baseline_rescues,
            },
            "adaptive": {
                "quota_released": released,
                "beam_winner": {
                    "final_cost": final_cost,
                    "delta_cost_vs_frozen_v08": final_cost - 10,
                    "quick": {"removed": [], "added_count": 0},
                    "rapid": {"added_count": 0},
                },
                "rescue_window_count": adaptive_rescues,
                "displaced_untagged_slots": 8*adaptive_rescues,
            },
            "diagnostic": {
                "final_term_differences_protected_minus_winner": {
                    "hand_exposure": 1.0,
                    "gesture_preference": -0.5,
                }},
        }
    rows = [
        sample(released=False, baseline_rescues=2,
               adaptive_rescues=2, baseline_cost=9, final_cost=9,
               cohort="previously_improved_7"),
        sample(released=True, baseline_rescues=25,
               adaptive_rescues=3, baseline_cost=10, final_cost=10,
               cohort="previously_nonimproved_15"),
    ]
    results = _aggregate(rows)
    assert results["target_count"] == 2
    assert results["released"] == 1
    assert results["adaptive_winner_improved_vs_v08"] == 1
    assert results["quota8_winner_improved_vs_v08"] == 1
    assert results["quota8_rescue_windows"] == 27
    assert results["adaptive_rescue_windows"] == 5
    assert results["saved_rescue_windows"] == 22
    assert results["saved_displaced_slots"] == 176
    assert results["sum_best_protected_minus_winner_components"] == {
        "hand_exposure": 2.0, "gesture_preference": -1.0}
