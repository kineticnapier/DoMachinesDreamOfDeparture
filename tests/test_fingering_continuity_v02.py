from __future__ import annotations

from dataclasses import dataclass

import pytest

from dmdod.planning.fingering_continuity_v02 import (
    ContinuitySelector,
    continuity_metrics,
)


@dataclass(frozen=True)
class Finger:
    axis: str


def _single(*axes: str):
    return tuple((Finger(axis),) for axis in axes)


def test_free_mode_preserves_network_argmin_exactly():
    allocations = _single("left_1", "left_2", "right_1")
    selector = ContinuitySelector(mode="free")
    decision = selector.select(
        network_scores=(0.4, -0.7, -0.2),
        allocations=allocations,
        eligible=(True, True, True),
        event_index=10,
        due_s=1.0,
    )
    assert decision.selected_index == 1
    assert decision.free_index == 1
    assert decision.changed_from_free is False


def test_continuity_never_selects_an_ineligible_candidate():
    allocations = _single("left_1", "left_2", "left_3")
    selector = ContinuitySelector(mode="continuity")
    selector.select((0.0, 1.0, 2.0), allocations, (True, True, True), 0, 1.000)
    selector.select((2.0, 0.0, 1.0), allocations, (True, True, True), 1, 1.020)
    decision = selector.select(
        (2.0, 1.0, 0.0), allocations, (True, True, False), 2, 1.040
    )
    assert decision.selected_index is not None
    assert decision.selected_index != 2
    assert decision.eligible[decision.selected_index] is True


def test_regular_fast_sequence_keeps_a_stable_roll_when_network_wanders():
    allocations = _single("left_1", "left_2", "left_3", "right_4", "left_4")
    selector = ContinuitySelector(mode="continuity")
    chosen = []
    score_rows = (
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (2.0, 0.0, 1.0, 3.0, 4.0),
        (2.0, 1.0, 0.2, 0.0, 3.0),
        (3.0, 2.0, 1.0, 0.0, 0.2),
    )
    for event_index, scores in enumerate(score_rows):
        decision = selector.select(
            scores, allocations, (True,) * len(allocations),
            event_index, 1.0 + event_index * 0.02,
        )
        chosen.append(allocations[decision.selected_index][0].axis)
    assert chosen == ["left_1", "left_2", "left_3", "left_4"]


def test_roll_can_change_when_expected_finger_is_not_eligible():
    allocations = _single("left_1", "left_2", "left_3", "right_1")
    selector = ContinuitySelector(mode="continuity")
    selector.select((0.0, 1.0, 2.0, 3.0), allocations, (True,) * 4, 0, 1.000)
    selector.select((2.0, 0.0, 1.0, 3.0), allocations, (True,) * 4, 1, 1.020)
    decision = selector.select(
        (3.0, 2.0, 0.0, 1.0), allocations,
        (False, False, False, True), 2, 1.040,
    )
    assert allocations[decision.selected_index][0].axis == "right_1"


def test_simultaneous_input_remains_atomic_and_eligible():
    allocations = (
        (Finger("left_1"), Finger("right_1")),
        (Finger("left_2"), Finger("right_2")),
    )
    selector = ContinuitySelector(mode="continuity")
    decision = selector.select(
        (0.0, 1.0), allocations, (False, True), 0, 2.0
    )
    assert decision.selected_index == 1
    assert tuple(finger.axis for finger in allocations[decision.selected_index]) == (
        "left_2", "right_2"
    )


def test_metrics_measure_switches_reuse_auxiliary_use_and_changed_fraction():
    rows = [
        {"event_index": 0, "due_ms": 1000.0, "axes": ["left_1"],
         "different_from_free": False},
        {"event_index": 1, "due_ms": 1020.0, "axes": ["left_2"],
         "different_from_free": True},
        {"event_index": 2, "due_ms": 1040.0, "axes": ["right_1"],
         "different_from_free": True},
        {"event_index": 3, "due_ms": 1060.0, "axes": ["left_1"],
         "different_from_free": False},
        {"event_index": 4, "due_ms": 1080.0, "axes": ["left_5"],
         "different_from_free": False},
        {"event_index": 5, "due_ms": 1100.0, "axes": ["right_6"],
         "different_from_free": False},
    ]
    result = continuity_metrics(rows)
    assert result["left_right_switches"] == 3
    assert result["thumb_uses"] == 1
    assert result["side_uses"] == 1
    assert result["same_part_reinputs_le_35ms"] == 0
    assert result["changed_from_free_count"] == 2
    assert result["changed_from_free_rate"] == pytest.approx(2 / 6)
    assert result["axis_reuse_interval_ms"]["minimum"] == pytest.approx(60.0)


def test_invalid_mode_and_shape_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        ContinuitySelector(mode="rolling-only")
    selector = ContinuitySelector(mode="continuity")
    with pytest.raises(ValueError, match="length"):
        selector.select((0.0,), _single("left_1", "left_2"), (True, True), 0, 0.0)
