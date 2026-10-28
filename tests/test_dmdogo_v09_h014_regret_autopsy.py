from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h08_protected_quota import trace_quota
from audit_dmdogo_v09_h011_adaptive_release import trace_adaptive
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    audit_one, generate, original_precise_cost, select_alternative,
)
from audit_dmdogo_v09_h014_regret_autopsy import (
    TARGET_IDS, QUOTA, OBSERVE_CHUNKS, BEAM_WIDTH, _terms_difference,
    replay_with_snapshots, verify_replay, explain_failure,
)


def test_component_difference_is_exhaustive_and_signed():
    assert _terms_difference({"a": 1, "b": 5}, {"a": 3, "b": 2}) == {
        "a": -2, "b": 3}
    assert _terms_difference(None, {"a": 5}) is None
    with pytest.raises(ValueError, match="cost term sets differ"):
        _terms_difference({"a": 1}, {"b": 1})


def test_replay_rejects_invalid_policy():
    with pytest.raises(ValueError, match="unknown replay policy"):
        replay_with_snapshots((), [], {"window_index": 1}, policy="invented")


def test_h014_freeze_and_target_identity():
    assert TARGET_IDS == (4115, 4127)
    assert QUOTA == 8
    assert OBSERVE_CHUNKS == 5
    assert BEAM_WIDTH == GestureChunkPlannerV08().beam_width == 24


@pytest.mark.parametrize("trial_id", TARGET_IDS)
def test_known_regret_autopsy_reproduces_both_original_searches(trial_id):
    events, spec = generate(trial_id)
    plan = GestureChunkPlannerV08().plan(events)
    assert plan.feasible
    windows = GestureChunkPlannerV08()._windows(events)
    chunks = [_chunk_record(x) for x in plan.chunks]
    exact = original_precise_cost(events, plan, windows, chunks)
    saved = audit_one(trial_id)
    regrets = [
        x for x in saved["targets"]
        if x["counterexample"]["rule_regret_after_release"]]
    assert len(regrets) == 1
    known = regrets[0]
    target = select_alternative(
        events, chunks, windows, known["window_index"])
    assert target == known["target"]
    args = {
        "expected_cost": plan.metrics["beam_cost"],
        "expected_evaluated": plan.candidates_evaluated,
        "reference_final_cost": exact,
    }
    original_full = trace_quota(
        events, chunks, target, quota=QUOTA, **args)
    original_adaptive = trace_adaptive(events, chunks, target, **args)
    full = replay_with_snapshots(
        events, chunks, target, policy="full_quota8")
    adapted = replay_with_snapshots(
        events, chunks, target, policy="five_chunk_release")
    verify_replay(full, original_full, adaptive=False)
    verify_replay(adapted, original_adaptive, adaptive=True)
    explanation = explain_failure(full, adapted, exact, known)
    assert explanation["regret_early_release_minus_full"] > 0
    assert (explanation["grace_ends_at"] ==
            explanation["target_window"] + OBSERVE_CHUNKS - 1)
    assert explanation["first_different_retained_population_window"] is not None
    assert (explanation["first_different_retained_population_window"] >
            explanation["grace_ends_at"])
    assert sum(explanation["regret_six_component_deltas"].values()
               ) == pytest.approx(
        explanation["regret_early_release_minus_full"], abs=1e-6)
    assert explanation["terminal_full_quota8_cost"] < (
        explanation["terminal_v08_cost"])
    assert explanation["terminal_early_release_cost"] > (
        explanation["terminal_full_quota8_cost"])
    assert (full["snapshots"][:OBSERVE_CHUNKS][0]["window_index"]
            == target["window_index"])


def test_h014_replay_fails_when_canonical_rescue_data_tampered():
    events, spec = generate(4115)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    windows = planner._windows(events)
    frozen = [_chunk_record(x) for x in plan.chunks]
    target = select_alternative(events, frozen, windows, 1)
    args = {
        "expected_cost": plan.metrics["beam_cost"],
        "expected_evaluated": plan.candidates_evaluated,
        "reference_final_cost": original_precise_cost(
            events, plan, windows, frozen),
    }
    ref = trace_quota(events, frozen, target, quota=QUOTA, **args)
    audit = replay_with_snapshots(
        events, frozen, target, policy="full_quota8")
    verify_replay(audit, ref, adaptive=False)
    ref["total_displaced_nonprotected_slots"] += 1
    with pytest.raises(ValueError, match="diverged"):
        verify_replay(audit, ref, adaptive=False)


def test_h014_grace_window_population_identity_is_stable():
    events, _ = generate(4115)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    windows = planner._windows(events)
    frozen = [_chunk_record(x) for x in plan.chunks]
    target = select_alternative(events, frozen, windows, 1)
    full = replay_with_snapshots(
        events, frozen, target, policy="full_quota8")
    adaptive = replay_with_snapshots(
        events, frozen, target, policy="five_chunk_release")
    for i in range(OBSERVE_CHUNKS):
        assert full["snapshots"][i]["retained_keys"] == (
            adaptive["snapshots"][i]["retained_keys"])
    assert all(len(row["retained_keys"]) <= BEAM_WIDTH
               for row in full["snapshots"])
    assert all(len(row["retained_keys"]) <= BEAM_WIDTH
               for row in adaptive["snapshots"])
