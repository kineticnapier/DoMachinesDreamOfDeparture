from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent, _candidate_gestures,
)
from audit_dmdogo_v09_h05_nonquick_suffix import _seed_state, _next
from audit_dmdogo_v09_h015_admissible_suffix_bound import (
    BOUND_SLACK_ABS, PINNED_GIT_BLOBS, TARGET_IDS,
    SURVEY_DISCOVERY_IDS, SURVEY_EVALUATION_IDS,
    git_blob_sha, verify_frozen_cost_source, intrinsic_cost,
    suffix_lower_bounds, certificate, evaluate_trace,
)


def test_source_pin_for_nonnegative_frozen_cost_formula():
    observed = verify_frozen_cost_source(ROOT)
    assert observed == PINNED_GIT_BLOBS
    assert len(observed) == 2


def test_git_blob_normalizes_windows_line_endings(tmp_path):
    p = tmp_path / "a.py"
    p.write_bytes(b"aaa\r\nbbb\r\n")
    q = tmp_path / "b.py"
    q.write_bytes(b"aaa\nbbb\n")
    assert git_blob_sha(p) == git_blob_sha(q)


def test_nonnegative_intrinsic_per_candidate_and_suffix_monotonic():
    events = tuple(GestureEvent(i, 1.0+i*.03125) for i in range(28))
    windows = GestureChunkPlannerV08()._windows(events)
    bounds = suffix_lower_bounds(windows)
    assert len(bounds) == len(windows)+1
    assert bounds[-1] == 0.0
    assert all(bounds[i] >= bounds[i+1]
               for i in range(len(windows)))
    for window in windows:
        assert all(intrinsic_cost(c) >= 0 for c in _candidate_gestures(window))


def test_intrinsic_is_lower_bound_on_actual_frozen_one_step_cost():
    events = tuple(GestureEvent(i, 1.0+i*.03125) for i in range(8))
    windows = GestureChunkPlannerV08()._windows(events)
    state = _seed_state(events[0].due_s)
    for wi, window in enumerate(windows):
        prev_due = windows[wi-1].events[-1].due_s if wi else None
        feasible = []
        for c in _candidate_gestures(window):
            next_state = _next(
                state, window, c, prev_due, frozen_candidate=True)
            if next_state is not None:
                assert next_state.score - state.score >= (
                    intrinsic_cost(c) - 1e-9)
                feasible.append(next_state)
        assert feasible
        state = min(feasible, key=lambda x: x.score)


def test_certificate_is_conservative_and_distinguishes_nonproof():
    proven = certificate(12.0, 0, 10.0)
    assert proven["certified"]
    assert proven["lower_bound"] == 12.0
    not_proven = certificate(8, 1, 10)
    assert not not_proven["certified"]
    tie = certificate(10, 0, 10)
    assert not tie["certified"]
    near = certificate(10+BOUND_SLACK_ABS/2, 0, 10)
    assert not near["certified"]
    none = certificate(None, 0, 10)
    assert not none["applicable"]
    assert not none["certified"]
    with pytest.raises(ValueError):
        certificate(-1, 0, 10)
    with pytest.raises(ValueError):
        certificate(1, -1, 10)
    with pytest.raises(ValueError):
        certificate(1, float("nan"), 10)
    with pytest.raises(ValueError):
        certificate(1, 0, float("inf"))


def test_outcome_condition_catches_false_certificate():
    events = tuple(GestureEvent(i, 1.0+i*.03125) for i in range(12))
    windows = GestureChunkPlannerV08()._windows(events)
    join = 1
    retained = [
        {"window_index": wi,
         "retained_best_tagged": {"score": 100.0},
         "tagged_retained_count": 1,
         "best_tagged_expanded_rank": 5}
        for wi in range(join, len(windows))
    ]
    with pytest.raises(ValueError, match="false optimistic suffix"):
        evaluate_trace(
            retained, windows, target_join=join,
            incumbent_cost=10.0,
            full_protected_terminal_cost=9.0)


def test_certificate_nonproof_for_known_improving_terminal():
    events = tuple(GestureEvent(i, 1.0+i*.03125) for i in range(12))
    windows = GestureChunkPlannerV08()._windows(events)
    join = 1
    retained = [
        {"window_index": wi,
         "retained_best_tagged": {"score": 0.25},
         "tagged_retained_count": 1,
         "best_tagged_expanded_rank": 5}
        for wi in range(join, len(windows))
    ]
    result = evaluate_trace(
        retained, windows, target_join=join,
        incumbent_cost=10.0,
        full_protected_terminal_cost=9.0)
    assert not result["certified_frontier_windows"]
    assert result["first_certified_window"] is None


def test_missing_frontier_is_not_marked_safe_to_prune():
    events = tuple(GestureEvent(i, 1.0+i*.03125) for i in range(12))
    windows = GestureChunkPlannerV08()._windows(events)
    rows = [
        {"window_index": wi,
         "retained_best_tagged": None,
         "tagged_retained_count": 0,
         "best_tagged_expanded_rank": None}
        for wi in range(1, len(windows))
    ]
    result = evaluate_trace(rows, windows, target_join=1,
                            incumbent_cost=10.0,
                            full_protected_terminal_cost=None)
    assert result["certified_frontier_windows"] == 0


def test_survey_is_fixed_and_nonoverlapping_with_known_regrets():
    assert TARGET_IDS == (4115, 4127)
    assert SURVEY_DISCOVERY_IDS == tuple(range(4096, 4104))
    assert SURVEY_EVALUATION_IDS == tuple(range(8192, 8200))
    assert not (set(TARGET_IDS) & set(SURVEY_DISCOVERY_IDS)
                or set(TARGET_IDS) & set(SURVEY_EVALUATION_IDS)
                or set(SURVEY_DISCOVERY_IDS) & set(SURVEY_EVALUATION_IDS))


def test_known_counterexample_cannot_be_certified_in_full_quota8():
    # This test is deliberately an expensive but small real abstract
    # replay; no real Train charts or Motor.
    from audit_dmdogo_v09_h015_admissible_suffix_bound import inspect_trial
    record = inspect_trial(4115, selected_join=8)
    assert len(record["targets"]) == 1
    target = record["targets"][0]
    assert target["release_rule_regret_in_existing_trial"]
    assert target["protected_terminal_beats_incumbent"]
    assert target["certificate"]["certified_frontier_windows"] == 0
