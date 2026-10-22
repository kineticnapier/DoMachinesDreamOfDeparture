from __future__ import annotations

import math

import pytest

from dmdod.adofai.stock_floor_state_machine import (
    StockFloorStateMachine,
    StockFloorTarget,
    StockRunMode,
    StockTransitionKind,
)
from dmdod.adofai.stock_frame_state_machine import (
    StockFrameStateMachine,
    StockQueuedKeyDown,
    first_frame_strictly_after,
)


def target(ordinal: int, due_s: float, bpm: float = 180.0) -> StockFloorTarget:
    return StockFloorTarget(ordinal, ordinal + 1, due_s, bpm)


def test_keydown_free_frame_can_force_no_fail_auto_miss() -> None:
    floor = StockFloorStateMachine((target(0, 1.0),), run_mode=StockRunMode.NO_FAIL)
    frames = StockFrameStateMachine(floor, frame_dt_s=0.010)

    update = frames.update(112)  # 1.120 > 1.000 + 60deg*1.01/540deg/s

    assert [row.kind for row in update.automatic_transitions] == [
        StockTransitionKind.AUTO_MISS_ADVANCE
    ]
    assert update.processed_keydowns == ()
    assert update.current_floor_after == 1


def test_fail_frame_stops_before_queued_input_is_processed() -> None:
    floor = StockFloorStateMachine((target(0, 1.0),), run_mode=StockRunMode.FAIL)
    frames = StockFrameStateMachine(floor, frame_dt_s=0.010)
    frames.queue_keydown(StockQueuedKeyDown("k", 1.339))

    update = frames.update(134)  # strict 180deg threshold crossed at 1.333...

    assert update.automatic_transitions[0].kind is StockTransitionKind.AUTO_MISS_FAIL
    assert update.processed_keydowns[0].input_transition is not None
    assert (
        update.processed_keydowns[0].input_transition.kind
        is StockTransitionKind.IGNORED_AFTER_FAIL
    )


def test_exact_threshold_waits_until_next_frame() -> None:
    floor = StockFloorStateMachine((target(0, 1.0),), run_mode=StockRunMode.FAIL)
    frames = StockFrameStateMachine(floor, frame_dt_s=1.0 / 3.0)

    exact = frames.update(4)  # 1.333333..., exactly 180 degrees late
    after = frames.update(5)

    assert exact.automatic_transitions == ()
    assert after.automatic_transitions[0].kind is StockTransitionKind.AUTO_MISS_FAIL


def test_update_precedes_same_frame_keydown() -> None:
    floor = StockFloorStateMachine(
        (target(0, 1.0), target(1, 1.20)), run_mode=StockRunMode.NO_FAIL
    )
    frames = StockFrameStateMachine(floor, frame_dt_s=0.010)
    frames.queue_keydown(StockQueuedKeyDown("k", 1.119))

    update = frames.update(112)

    assert update.automatic_transitions[0].target_ordinal == 0
    assert update.processed_keydowns[0].input_transition.target_ordinal == 1


def test_multiple_keydowns_keep_timestamp_then_insertion_order() -> None:
    floor = StockFloorStateMachine((target(0, 1.0), target(1, 1.0)))
    frames = StockFrameStateMachine(floor)
    frames.queue_keydown(StockQueuedKeyDown("a", 1.0, "left"))
    frames.queue_keydown(StockQueuedKeyDown("b", 1.0, "right"))

    update = frames.update(100)

    assert [row.keydown.input_id for row in update.processed_keydowns] == ["a", "b"]
    assert [row.input_transition.target_ordinal for row in update.processed_keydowns] == [0, 1]


def test_first_frame_helper_preserves_strict_boundary_and_phase() -> None:
    index, time_s = first_frame_strictly_after(
        3.249473567, frame_dt_s=0.010, phase_s=0.0
    )
    assert index == 325
    assert time_s == pytest.approx(3.25)

    exact_index, exact_time = first_frame_strictly_after(
        3.25, frame_dt_s=0.010, phase_s=0.0
    )
    assert exact_index == 326
    assert exact_time == pytest.approx(3.26)


def test_frame_contract_rejects_retroactive_or_duplicate_input() -> None:
    frames = StockFrameStateMachine(StockFloorStateMachine((target(0, 1.0),)))
    frames.update(10)
    with pytest.raises(ValueError):
        frames.queue_keydown(StockQueuedKeyDown("past", 0.1))
    frames.queue_keydown(StockQueuedKeyDown("future", 0.2))
    with pytest.raises(ValueError):
        frames.queue_keydown(StockQueuedKeyDown("future", 0.3))


def test_nonfinite_keydown_is_rejected() -> None:
    with pytest.raises(ValueError):
        StockQueuedKeyDown("bad", math.nan)
