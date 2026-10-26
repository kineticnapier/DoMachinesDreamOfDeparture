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
from audit_dmdogo_v09_h04_beam_decisions import (
    TERMS, State, _expand, _quick_at_join, _chunk_record, audit_beam,
)


def _events(gaps):
    times = [1.]
    for gap in gaps:
        times.append(times[-1] + gap)
    return tuple(GestureEvent(i, t, 1) for i, t in enumerate(times))


def _baseline(gaps):
    events = _events(gaps)
    planned = GestureChunkPlannerV08().plan(events)
    assert planned.feasible
    frozen = [_chunk_record(c) for c in planned.chunks]
    return events, frozen, planned


@pytest.mark.parametrize("gaps", [
    [.03125] * 15,
    [.02083, .229128] * 7,
    [.02083, .229128] * 6 + [.03125] * 8,
    [.03125] * 11 + [.5] + [.04] * 7,
])
def test_instrumented_beam_exactly_reproduces_frozen_planner(gaps):
    events, chunks, planned = _baseline(gaps)
    actual = audit_beam(
        events, chunks, planned.metrics["beam_cost"],
        planned.candidates_evaluated, [],
    )
    assert actual["events"] == len(events)
    assert actual["exact_frozen_chunk_path_matched"] is True
    assert actual["exact_frozen_candidate_count_matched"] is True
    assert actual["exact_frozen_beam_cost_matched"] is True
    assert actual["canonical_frozen_planner_independently_matched"] is True
    assert actual["flag_evidence"] == []


def test_cost_decomposition_preserves_exact_incremental_score():
    events = _events([.05, .05, .05])
    planner = GestureChunkPlannerV08()
    window = planner._windows(events)[0]
    candidate = _candidate_gestures(window)[0]
    state = State(0., {}, None, (0., 0.), events[0].due_s, None)
    evaluated = _expand(state, window, candidate, None)
    assert evaluated is not None
    assert set(evaluated[5]) == set(TERMS)
    assert sum(evaluated[5].values()) == pytest.approx(evaluated[7], abs=1e-12)
    assert evaluated[0] == evaluated[7]


def test_reuse_invalid_candidate_is_rejected():
    events = _events([.010, .010, .010])
    window = GestureChunkPlannerV08()._windows(events)[0]
    candidate = next(c for c in _candidate_gestures(window)
                     if len(set(axis for group in c.axes for axis in group))
                     < len(c.axes))
    state = State(0., {}, None, (0., 0.), events[0].due_s, None)
    assert _expand(state, window, candidate, None) is None


def test_review_flag_detected_at_same_hand_seam_only():
    assert _quick_at_join("left_4", "left_1", .08125)
    assert not _quick_at_join("left_4", "right_1", .08125)
    assert not _quick_at_join("left_2", "left_1", .08125)
    assert not _quick_at_join("left_4", "left_1", .121)
    assert not _quick_at_join(None, "left_1", .05)


def test_mutated_saved_cost_fails_closed():
    events, chunks, planned = _baseline([.03125] * 11)
    with pytest.raises(ValueError, match="objective differs"):
        audit_beam(
            events, chunks, planned.metrics["beam_cost"] + 1.,
            planned.candidates_evaluated, [], check_planner=False)


def test_mutated_candidate_count_fails_closed():
    events, chunks, planned = _baseline([.03125] * 11)
    with pytest.raises(ValueError, match="candidate count differs"):
        audit_beam(
            events, chunks, planned.metrics["beam_cost"],
            planned.candidates_evaluated + 1, [], check_planner=False)


def test_mutated_chunk_fails_closed():
    events, chunks, planned = _baseline([.03125] * 11)
    chunks[0]["axes"][0][0] = "not_an_axis"
    with pytest.raises(ValueError, match="diverged at chunk"):
        audit_beam(
            events, chunks, planned.metrics["beam_cost"],
            planned.candidates_evaluated, [], check_planner=False)


def test_h01_flag_must_reference_real_chunk_seam():
    events, chunks, planned = _baseline([.03125] * 11)
    with pytest.raises(ValueError, match="not a chunk start"):
        audit_beam(
            events, chunks, planned.metrics["beam_cost"],
            planned.candidates_evaluated, [(0, 2)], check_planner=False)
