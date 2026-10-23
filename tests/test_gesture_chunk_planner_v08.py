from __future__ import annotations

import pytest

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent,
)


def events(gaps, chord_sizes=None):
    values = [0.0]
    for gap in gaps:
        values.append(values[-1] + gap)
    chord_sizes = chord_sizes or [1] * len(values)
    return tuple(GestureEvent(i, t, c) for i, (t, c)
                 in enumerate(zip(values, chord_sizes)))


def test_doublet_gestures_have_adaptive_boundaries():
    data = events([.02083, .229128] * 8)
    plan = GestureChunkPlannerV08().plan(data)
    assert plan.feasible
    assert all(c.event_count == 2 for c in plan.chunks[:8])
    assert all(c.start_reason == "salient_rhythm_gap" for c in plan.chunks[1:8])
    assert plan.chunks[0].axes[0][0].startswith("left")
    assert plan.chunks[1].axes[0][0].startswith("right")
    assert plan.model_contract["human_likeness_verified"] is False
    assert plan.model_contract["physical_motor_evaluated"] is False


def test_stream_is_broken_into_local_gestures_without_global_key_cycle():
    plan = GestureChunkPlannerV08().plan(events([.03125] * 31))
    assert plan.feasible
    assert len(plan.chunks) == 8
    assert all(chunk.event_count == 4 for chunk in plan.chunks)
    assert any(chunk.start_reason == "stream_continuation" for chunk in plan.chunks)
    assert plan.metrics["abstract_reuse_violations"] == 0


def test_mixed_motif_produces_variable_chunk_sizes():
    plan = GestureChunkPlannerV08().plan(events(
        [.02083, .229128] * 10 + [.03125] * 32 +
        [.02083, .104149, .124979] * 6))
    assert plan.feasible
    assert len(set(chunk.event_count for chunk in plan.chunks)) >= 2
    assert plan.metrics["distinct_gesture_families"] >= 2


def test_reuse_is_safe_over_gesture_boundaries():
    sequence = events([.03125] * 79)
    plan = GestureChunkPlannerV08().plan(sequence)
    assert plan.feasible
    use = {}
    for event, axes in zip(sequence, plan.assignments):
        assert len(set(axes)) == len(axes)
        for axis in axes:
            if axis in use:
                assert event.due_s - use[axis] >= .08 - 1e-9
            use[axis] = event.due_s


def test_pause_and_chords_stand_alone():
    sequence = events([.02, .500, .100, .150, .200],
                      [1, 1, 2, 1, 1, 1])
    plan = GestureChunkPlannerV08().plan(sequence)
    assert plan.feasible
    chord = [c for c in plan.chunks if c.rhythm_kind == "chord"]
    assert len(chord) == 1 and chord[0].event_count == 1
    assert len(chord[0].axes[0]) == 2
    assert any(c.start_reason == "pause" for c in plan.chunks)


def test_infeasible_simultaneous_events_fail_closed():
    plan = GestureChunkPlannerV08().plan(events([0., 0.], [8, 8, 8]))
    assert not plan.feasible
    assert plan.failure_reason == "no_reuse_safe_gesture_in_beam"


def test_bad_inputs_are_rejected():
    planner = GestureChunkPlannerV08()
    assert planner.plan([GestureEvent(0, float("nan"))]).failure_reason == "non_finite_due_time"
    assert planner.plan([GestureEvent(0, 0., 9)]).failure_reason == "invalid_chord_size"
    assert planner.plan([GestureEvent(0, 1.), GestureEvent(1, 0.)]).failure_reason == "nonmonotonic_event_times"
    assert planner.plan([GestureEvent(0, 0.), GestureEvent(0, 1.)]).failure_reason == "duplicate_event_index"
    with pytest.raises(ValueError):
        GestureChunkPlannerV08(beam_width=0)
    with pytest.raises(ValueError):
        GestureChunkPlannerV08(minimum_reuse_s=float("inf"))


def test_deterministic_and_empty():
    planner = GestureChunkPlannerV08()
    sequence = events([.02083, .229128] * 4 + [.03125] * 12)
    assert planner.plan(sequence) == planner.plan(sequence)
    assert planner.plan([]).feasible
