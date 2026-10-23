from __future__ import annotations

import pytest
from dmdod.planning.human_phrase_planner_v07 import HumanPhrasePlannerV07, PhraseEvent


def make(gaps, chords=None):
    times = [0.]
    for gap in gaps:
        times.append(times[-1] + gap)
    chords = chords or [1] * len(times)
    return tuple(PhraseEvent(i, value, chords[i]) for i, value in enumerate(times))


def test_alternating_gaps_are_detected_as_pair_motif():
    seq = make([.020, .229] * 8)
    plan = HumanPhrasePlannerV07().plan(seq)
    assert plan.feasible
    assert plan.phrases[0].rhythm_class == "alternating_gaps"
    assert plan.phrases[0].template == "paired_hand"
    assert plan.metrics["abstract_reuse_violations"] == 0
    assert len(plan.assignments) == len(seq)
    assert plan.model_contract["human_likeness_verified"] is False


def test_periodic_three_and_five_gap_motifs_are_not_called_irregular():
    for period, gaps in ((3, [.021, .104, .125] * 5),
                         (5, [.021, .042, .062, .062, .062] * 3)):
        plan = HumanPhrasePlannerV07().plan(make(gaps))
        assert plan.feasible
        assert plan.phrases[0].rhythm_class == f"periodic_{period}"


def test_even_fast_note_window_uses_declared_roll_prior():
    plan = HumanPhrasePlannerV07().plan(make([.03125] * 15))
    assert plan.feasible
    assert plan.phrases[0].rhythm_class == "even_dense"
    assert plan.phrases[0].template == "hand_roll"
    assert plan.model_contract["motor_feasibility_verified"] is False


def test_pause_creates_boundary_and_reuse_across_windows():
    seq = make([.030] * 7 + [.800] + [.030] * 7)
    plan = HumanPhrasePlannerV07().plan(seq)
    assert plan.feasible
    assert len(plan.phrases) == 2
    assert plan.phrases[1].boundary_reason == "pause"


def test_reuse_constraint_applies_across_horizon_boundaries():
    seq = make([.03125] * 31)
    plan = HumanPhrasePlannerV07(max_phrase_events=16).plan(seq)
    assert plan.feasible
    assert len(plan.phrases) == 2
    used = {}
    for event, axes in zip(seq, plan.assignments):
        for axis in axes:
            if axis in used:
                assert event.due_s - used[axis] >= .080 - 1e-9
            used[axis] = event.due_s


def test_simultaneous_demand_fails_explicitly():
    seq = (PhraseEvent(0, 0., 8), PhraseEvent(1, .005, 8))
    result = HumanPhrasePlannerV07().plan(seq)
    assert not result.feasible
    assert result.failure_reason == "no_abstract_reuse_safe_gesture_in_beam"


def test_chords_use_distinct_physical_axes():
    plan = HumanPhrasePlannerV07().plan(make([.200] * 7, [2] * 8))
    assert plan.feasible
    assert all(len(set(axes)) == 2 for axes in plan.assignments)


def test_result_determinism_and_beam_honesty():
    seq = make([.021, .229, .021, .229, .021, .229])
    scheduler = HumanPhrasePlannerV07()
    assert scheduler.plan(seq) == scheduler.plan(seq)
    assert scheduler.plan(seq).model_contract["beam_optimality"] == "heuristic_not_globally_optimal"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_event_time_rejected(bad):
    assert HumanPhrasePlannerV07().plan([PhraseEvent(0, bad)]).failure_reason == "non_finite_time"


def test_empty_and_chord_validation():
    assert HumanPhrasePlannerV07().plan([]).feasible
    assert HumanPhrasePlannerV07().plan([PhraseEvent(0, 0, 9)]).failure_reason == "invalid_chord_size"
