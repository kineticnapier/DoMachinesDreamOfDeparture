from __future__ import annotations

from dmdod.planning.fingering_planner_v04 import (
    CandidateNode,
    EventCandidates,
    ShortHorizonFingeringPlanner,
)


def _event(index: int, due_ms: float, axes: list[str], *, eligible=True) -> EventCandidates:
    return EventCandidates(index, due_ms, tuple(
        CandidateNode(index, due_ms, f"{axis}-{rank}", (axis,), rank,
                      eligible if index == 0 else True)
        for rank, axis in enumerate(axes)
    ))


def test_planner_is_deterministic_and_bounded():
    events = tuple(_event(i, i * 31.0, ["left_1", "left_2", "right_1", "right_2"])
                   for i in range(6))
    planner = ShortHorizonFingeringPlanner(horizon=4, beam_width=8, candidates_per_event=3)
    first = planner.plan(events)
    second = ShortHorizonFingeringPlanner(
        horizon=4, beam_width=8, candidates_per_event=3).plan(events)
    assert first.allocations == second.allocations
    assert first.evaluated_candidates <= 4 * 8 * 3
    assert len(first.allocations) == 4


def test_planner_prefers_consistent_same_hand_roll_within_network_top_three():
    events = (
        _event(0, 0, ["left_1", "right_3", "left_4"]),
        _event(1, 31, ["left_2", "right_1", "left_4"]),
        _event(2, 62, ["left_3", "right_4", "left_1"]),
        _event(3, 93, ["left_4", "right_2", "left_1"]),
    )
    plan = ShortHorizonFingeringPlanner().plan(events)
    assert [item.axes for item in plan.allocations] == [
        ("left_1",), ("left_2",), ("left_3",), ("left_4",)
    ]


def test_planner_never_uses_ineligible_first_candidate_or_impossible_reuse():
    first = EventCandidates(0, 0, (
        CandidateNode(0, 0, "bad", ("left_1",), 0, False),
        CandidateNode(0, 0, "good", ("left_2",), 1, True),
    ))
    second = EventCandidates(1, 31, (
        CandidateNode(1, 31, "reuse", ("left_2",), 0, True),
        CandidateNode(1, 31, "other", ("left_3",), 1, True),
    ))
    plan = ShortHorizonFingeringPlanner(min_axis_reuse_ms=80).plan((first, second))
    assert plan.allocations[0].allocation_id == "good"
    assert plan.allocations[1].allocation_id == "other"


def test_committed_cache_is_reused_only_when_still_eligible_and_reset_clears_it():
    planner = ShortHorizonFingeringPlanner()
    events = (_event(0, 0, ["left_1", "left_2"]),
              _event(1, 31, ["left_2", "left_3"]))
    plan = planner.plan(events)
    planner.commit(plan)
    cached = planner.cached_choice(1, events[1].candidates)
    assert cached is not None
    invalid = tuple(CandidateNode(c.event_index, c.due_ms, c.allocation_id,
                                  c.axes, c.network_rank, False)
                    for c in events[1].candidates)
    assert planner.cached_choice(1, invalid) is None
    assert planner.discard_count == 1
    planner.reset()
    assert planner.cached_choice(1, events[1].candidates) is None
