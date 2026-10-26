from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureEvent, GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import (
    FORMAT, TERMS, _seed_state, _next, _force_suffix,
    _cost_terms, trace_unrestricted,
)


def events(gaps):
    t = 0.5
    result = [GestureEvent(0, t)]
    for gap in gaps:
        t += gap
        result.append(GestureEvent(len(result), t))
    return tuple(result)


def frozen(gaps):
    sequence = events(gaps)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(sequence)
    assert plan.feasible
    return sequence, planner._windows(sequence), [
        _chunk_record(chunk) for chunk in plan.chunks], plan


@pytest.mark.parametrize("gaps", [
    [.03125] * 11,
    [.02083, .229128] * 7,
    [.03125] * 11 + [.500] + [.04] * 7,
    [.02083, .229128] * 5 + [.03125] * 14,
])
def test_tag_free_global_trace_matches_original_frozen_beam(gaps):
    sequence, _, chunks, plan = frozen(gaps)
    flags, seeds, original = trace_unrestricted(
        sequence, chunks, [],
        plan.metrics["beam_cost"], plan.candidates_evaluated)
    assert flags == []
    assert seeds == {}
    assert round(original.score, 6) == plan.metrics["beam_cost"]
    assert original.frozen_prefix
    assert original.tags == frozenset()


def test_cost_terms_are_exact_accumulated_objective():
    sequence, _, chunks, plan = frozen([.03125] * 11)
    _, _, winner = trace_unrestricted(
        sequence, chunks, [],
        plan.metrics["beam_cost"], plan.candidates_evaluated)
    assert set(_cost_terms(winner)) == set(TERMS)
    assert sum(winner.terms) == pytest.approx(winner.score, abs=1e-9)


def test_forced_branch_is_evaluated_with_unmodified_suffix_grammar():
    sequence, windows, _, _ = frozen([.03125] * 11)
    state = _seed_state(sequence[0].due_s)
    chosen = _candidate_gestures(windows[0])[0]
    seed = _next(state, windows[0], chosen, None, frozen_candidate=False)
    assert seed is not None
    outcome = _force_suffix(seed, windows, 0)
    assert outcome["suffix_candidates_evaluated"] > 0
    if outcome["reached_terminal"]:
        assert outcome["final_cost"] is not None
        assert set(outcome["final_cost_terms"]) == set(TERMS)
    else:
        assert outcome["failure_reason"] == "no_reuse_safe_beam_continuation"
        assert outcome["first_failed_chunk_index"] is not None


def test_constrained_suffix_with_no_future_chunks_returns_seed_exactly():
    sequence, windows, _, _ = frozen([.03125] * 11)
    window = windows[0]
    seed = _next(_seed_state(sequence[0].due_s),
                 window, _candidate_gestures(window)[0],
                 None, frozen_candidate=False)
    assert seed is not None
    end = _force_suffix(seed, (window,), 0)
    assert end["reached_terminal"] is True
    assert end["final_cost"] == round(seed.score, 9)
    assert end["suffix_candidates_evaluated"] == 0


def test_global_trace_rejects_wrong_original_cost():
    sequence, _, chunks, plan = frozen([.03125] * 11)
    with pytest.raises(ValueError, match="beam cost differs"):
        trace_unrestricted(
            sequence, chunks, [],
            plan.metrics["beam_cost"] + 10., plan.candidates_evaluated)


def test_global_trace_rejects_wrong_candidate_count():
    sequence, _, chunks, plan = frozen([.03125] * 11)
    with pytest.raises(ValueError, match="evaluated count"):
        trace_unrestricted(
            sequence, chunks, [],
            plan.metrics["beam_cost"], plan.candidates_evaluated + 1)


def test_global_trace_rejects_unmatched_h04_seam():
    sequence, _, chunks, plan = frozen([.03125] * 11)
    with pytest.raises(ValueError, match="unique chunk seam"):
        trace_unrestricted(
            sequence, chunks, [{"flag_pair": [0, 2]}],
            plan.metrics["beam_cost"], plan.candidates_evaluated)


def test_no_implicit_human_likeness_claim_in_result_contract():
    assert "nonquick" in FORMAT
    assert len(TERMS) == 6
