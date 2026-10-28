from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _seed_state
from audit_dmdogo_v09_h014_regret_autopsy import replay_with_snapshots
from audit_dmdogo_v09_h015_admissible_suffix_bound import (
    certificate, verify_frozen_cost_source,
)
from audit_dmdogo_v09_h016_cold_block_lower_bounds import cold_suffix_bounds
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    generate, select_alternative, original_precise_cost,
)
from audit_dmdogo_v09_h017_state_aware_lower_bound import (
    FORMAT, QUOTA, CHECK_CHUNKS,
    LOOKAHEAD_MAX, tagged_grace_states, exact_lookahead_bound,
    compare_early, summarize,
)


def source(trial, join):
    events, _ = generate(trial)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    windows = planner._windows(events)
    chunks = [_chunk_record(c) for c in plan.chunks]
    target = select_alternative(events, chunks, windows, join)
    assert target is not None
    return events, windows, chunks, target, plan


def test_frozen_heuristic_source_and_protocol_pinned():
    root = Path(__file__).resolve().parents[1]
    assert len(verify_frozen_cost_source(root)) == 2
    assert QUOTA == 8
    assert CHECK_CHUNKS == 5
    assert LOOKAHEAD_MAX == 2
    assert GestureChunkPlannerV08().beam_width == 24
    assert FORMAT == "dmdogo-v09-h017-state-aware-lower-bound-v1"


def test_exact_lookahead_is_admissible_against_frozen_complete_plan():
    events, windows, chunks, target, plan = source(4115, 8)
    root = _seed_state(events[0].due_s)
    incumbent = original_precise_cost(events, plan, windows, chunks)
    floors = cold_suffix_bounds(windows)["maximum"]
    for horizon in (1, 2):
        found = exact_lookahead_bound(
            [root], windows, next_window=0,
            lookahead=horizon, cold_suffix=floors)
        assert found["applicable"]
        assert found["feasible_continuations"] > 0
        assert found["optimistic_total_cost"] <= incumbent + 1e-6
        assert found["candidate_evaluations"] > 0


def test_lookahead_uses_all_tagged_states_not_only_first():
    events, windows, chunks, target, plan = source(4115, 8)
    start = _seed_state(events[0].due_s)
    expensive = replace(start, score=10.0)
    floors = cold_suffix_bounds(windows)["maximum"]
    one = exact_lookahead_bound(
        [start], windows, next_window=0,
        lookahead=1, cold_suffix=floors)
    two = exact_lookahead_bound(
        [expensive, start], windows, next_window=0,
        lookahead=1, cold_suffix=floors)
    assert one["applicable"] and two["applicable"]
    assert two["optimistic_total_cost"] == pytest.approx(
        one["optimistic_total_cost"], abs=1e-9)
    assert two["feasible_continuations"] == (
        2 * one["feasible_continuations"])


def test_empty_states_never_produce_certificate():
    events, windows, *_ = source(4115, 8)
    floors = cold_suffix_bounds(windows)["maximum"]
    missing = exact_lookahead_bound(
        [], windows, next_window=4,
        lookahead=2, cold_suffix=floors)
    assert not missing["applicable"]
    assert missing["optimistic_total_cost"] is None


@pytest.mark.parametrize("bad_horizon", [0, 3, -1])
def test_invalid_lookahead_fails_closed(bad_horizon):
    events, windows, *_ = source(4115, 8)
    floors = cold_suffix_bounds(windows)["maximum"]
    with pytest.raises(ValueError, match="invalid state-aware"):
        exact_lookahead_bound(
            [_seed_state(events[0].due_s)],
            windows, next_window=0,
            lookahead=bad_horizon, cold_suffix=floors)


@pytest.mark.parametrize("trial,join", [(4115, 8), (4127, 10)])
def test_independent_tagged_grace_replays_exact_original_quota8(trial, join):
    events, windows, chunks, target, plan = source(trial, join)
    traced = tagged_grace_states(events, chunks, target)
    known = replay_with_snapshots(
        events, chunks, target, policy="full_quota8")
    assert len(traced) == CHECK_CHUNKS
    for source_row, expected in zip(
            traced, known["snapshots"][:CHECK_CHUNKS]):
        assert source_row["window_index"] == expected["window_index"]
        assert source_row["retained_keys"] == expected["retained_keys"]
        assert source_row["tagged_count"] == expected["tagged_retained_count"]
        assert source_row["best_tagged_score"] == (
            expected["retained_best_tagged"]["score"])
        assert source_row["tagged_count"] > 0


def test_known_regret_paths_do_not_receive_false_early_certificate():
    trial, join = 4115, 8
    events, windows, chunks, target, plan = source(trial, join)
    source_rows = tagged_grace_states(events, chunks, target)
    full = replay_with_snapshots(
        events, chunks, target, policy="full_quota8")
    incumbent = original_precise_cost(events, plan, windows, chunks)
    assert full["terminal_best_tagged"]["final_cost"] < incumbent - 1e-7
    cold = cold_suffix_bounds(windows)["maximum"]
    baseline = []
    for item in full["snapshots"][:CHECK_CHUNKS]:
        wi = item["window_index"]
        tagged = item["retained_best_tagged"]
        score = tagged["score"] if tagged is not None else None
        baseline.append({
            "window_index": wi,
            **certificate(score, cold[wi+1], incumbent),
        })
    result = compare_early(
        source_rows, full["snapshots"], windows,
        saved_maximum=baseline,
        incumbent=incumbent,
        tagged_terminal=full["terminal_best_tagged"]["final_cost"])
    assert result["checked_windows"] == CHECK_CHUNKS
    assert result["early_certificates_h017"] == 0
    assert result["new_certificates"] == 0
    assert result["two_step_candidate_evaluations"] > 0
    baseline[0]["lower_bound"] = 1e8
    with pytest.raises(ValueError, match="H0.16 saved maximum"):
        compare_early(
            source_rows, full["snapshots"], windows,
            saved_maximum=baseline,
            incumbent=incumbent,
            tagged_terminal=full["terminal_best_tagged"]["final_cost"])


def test_summary_keeps_full_candidate_work_count():
    result = summarize([{
        "trial_index": 4115,
        "targets": [{
            "known_regret": True,
            "tagged_beats_v08": True,
            "new_certificates": 0,
            "early": {
                "early_certificates_h016": 0,
                "early_certificates_h017": 0,
                "two_step_candidate_evaluations": 1400,
            }},
        ],
    }])
    assert result["trial_count"] == 1
    assert result["target_count"] == 1
    assert result["known_regret_target_count"] == 1
    assert result["new_certified_windows"] == 0
    assert result["two_step_candidate_evaluations"] == 1400
