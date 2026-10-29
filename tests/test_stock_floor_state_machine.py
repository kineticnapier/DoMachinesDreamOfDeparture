from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai

from dmdod.adofai.stock_floor_state_machine import (
    StockFloorStateMachine,
    StockFloorTarget,
    StockReproducibility,
    StockRunMode,
    StockTransitionKind,
    targets_from_playable_segment,
)
from dmdod.adofai.stock_judgement_reference import StockHitMargin


def target(ordinal: int, due_s: float, *, bpm: float = 180.0) -> StockFloorTarget:
    return StockFloorTarget(
        ordinal=ordinal,
        floor_index=ordinal + 1,
        due_s=due_s,
        approach_bpm=bpm,
    )


def input_rows(machine: StockFloorStateMachine):
    return [row for row in machine.trace if row.input_time_s is not None]


def test_perfect_keydown_advances_to_next_floor() -> None:
    machine = StockFloorStateMachine((target(0, 1.0), target(1, 2.0)))

    rows = machine.key_down(1.0)

    assert rows[-1].kind is StockTransitionKind.HIT_ADVANCE
    assert rows[-1].judgement is StockHitMargin.PERFECT
    assert rows[-1].current_floor_before == 0
    assert rows[-1].current_floor_after == 1
    assert machine.next_target == target(1, 2.0)


def test_too_early_and_too_late_inputs_do_not_advance() -> None:
    early = StockFloorStateMachine((target(0, 1.0),))
    late = StockFloorStateMachine((target(0, 1.0),))

    early_row = early.key_down(0.80)[-1]
    late_row = late.key_down(1.12)[-1]

    assert early_row.kind is StockTransitionKind.INVALID_INPUT
    assert early_row.judgement is StockHitMargin.TOO_EARLY
    assert late_row.kind is StockTransitionKind.INVALID_INPUT
    assert late_row.judgement is StockHitMargin.TOO_LATE
    assert early.next_index == late.next_index == 0


def test_fail_mode_auto_miss_stops_without_floor_transition() -> None:
    machine = StockFloorStateMachine((target(0, 1.0),), run_mode=StockRunMode.FAIL)

    # At 180 BPM the Counted boundary is 60 degrees; normal fail mode uses
    # max(180, 2*60) and the DLL comparison is strict `>`.
    assert machine.advance(1.0 + 180.0 / 540.0) == ()
    rows = machine.advance(math.nextafter(1.0 + 180.0 / 540.0, math.inf))

    assert rows[0].kind is StockTransitionKind.AUTO_MISS_FAIL
    assert rows[0].failed
    assert not rows[0].transitioned
    assert machine.current_floor == 0


def test_no_fail_auto_miss_forces_floor_transition_before_input() -> None:
    machine = StockFloorStateMachine(
        (target(0, 1.0), target(1, 1.5)), run_mode=StockRunMode.NO_FAIL
    )
    threshold_s = 60.0 * 1.01 / 540.0

    rows = machine.key_down(math.nextafter(1.0 + threshold_s, math.inf))

    assert [row.kind for row in rows] == [
        StockTransitionKind.AUTO_MISS_ADVANCE,
        StockTransitionKind.INVALID_INPUT,
    ]
    assert rows[0].judgement == "fail_miss"
    assert rows[0].current_floor_after == 1
    assert rows[1].target_ordinal == 1
    assert rows[1].judgement is StockHitMargin.TOO_EARLY


def test_input_before_auto_miss_targets_old_floor() -> None:
    machine = StockFloorStateMachine(
        (target(0, 1.0), target(1, 1.5)), run_mode=StockRunMode.NO_FAIL
    )

    row = machine.key_down(1.10)[-1]

    assert row.target_ordinal == 0
    assert row.kind is StockTransitionKind.HIT_ADVANCE
    assert machine.next_target == target(1, 1.5)


def test_multiple_keydowns_at_same_time_are_not_collapsed() -> None:
    machine = StockFloorStateMachine((target(0, 1.0), target(1, 1.0)))

    first = machine.key_down(1.0)[-1]
    second = machine.key_down(1.0)[-1]

    assert (first.target_ordinal, second.target_ordinal) == (0, 1)
    assert first.transitioned and second.transitioned
    assert len(input_rows(machine)) == 2


def test_unsupported_special_floor_stops_instead_of_guessing() -> None:
    special = StockFloorTarget(
        ordinal=0,
        floor_index=1,
        due_s=1.0,
        approach_bpm=180.0,
        supported=False,
        unsupported_reason="Hold branch is not implemented",
    )
    machine = StockFloorStateMachine((special,))

    rows = machine.key_down(1.0)

    assert rows[-1].kind is StockTransitionKind.UNSUPPORTED
    assert rows[-1].reproducibility is StockReproducibility.UNSUPPORTED
    assert machine.next_index == 0


def test_reset_restores_initial_state_after_fail() -> None:
    machine = StockFloorStateMachine((target(0, 1.0),))
    machine.advance(2.0)

    machine.reset()

    assert not machine.failed
    assert machine.current_floor == 0
    assert machine.next_index == 0
    assert machine.trace == []


def test_phase16_shape_differs_by_explicit_fail_mode() -> None:
    targets = (
        StockFloorTarget(36, 39, 3.183823567304636, 1360.0),
        StockFloorTarget(37, 40, 3.2205882724022707, 1360.0),
    )
    no_fail = StockFloorStateMachine(targets, run_mode=StockRunMode.NO_FAIL)
    fail = StockFloorStateMachine(targets, run_mode=StockRunMode.FAIL)

    no_fail_rows = no_fail.key_down(3.260999999999752)
    fail_rows = fail.key_down(3.260999999999752)

    assert [(row.kind, row.target_ordinal) for row in no_fail_rows] == [
        (StockTransitionKind.AUTO_MISS_ADVANCE, 36),
        (StockTransitionKind.HIT_ADVANCE, 37),
    ]
    assert no_fail_rows[-1].judgement is StockHitMargin.VERY_LATE
    assert [(row.kind, row.target_ordinal) for row in fail_rows] == [
        (StockTransitionKind.INVALID_INPUT, 36)
    ]
    assert fail_rows[-1].judgement is StockHitMargin.TOO_LATE


def test_fail_mode_dies_on_post_floor_check_before_later_input() -> None:
    machine = StockFloorStateMachine(
        (StockFloorTarget(36, 39, 3.183823567304636, 1360.0),),
        run_mode=StockRunMode.FAIL,
        initial_floor=37,
    )
    machine.key_down(3.260999999999752)
    machine.key_down(3.3109999999997464)

    rows = machine.key_down(3.3549999999997415)

    assert [row.kind for row in rows] == [
        StockTransitionKind.AUTO_MISS_FAIL,
        StockTransitionKind.IGNORED_AFTER_FAIL,
    ]
    assert rows[0].target_ordinal == 36
    assert rows[0].dll_flow == ("CheckPostHoldFail", "Die")


def _plain_compiled_chart():
    return compile_adofai(
        parse_adofai_text(
            '{"angleData":[0,0,0,0],"settings":{"bpm":120,'
            '"countdownTicks":0},"actions":[]}'
        )
    )


def test_sliced_stock_reference_uses_explicit_landed_floor() -> None:
    chart = _plain_compiled_chart()
    start_s = (chart.floors[2].target_time_s + chart.floors[3].target_time_s) / 2
    segment = build_playable_segment(chart, start_s=start_s)
    assert segment.targets[0].floor_index == 3

    # Legacy callers fail closed rather than silently assuming earlier hits.
    assert not targets_from_playable_segment(segment)[0].supported
    targets = targets_from_playable_segment(segment, initial_floor=2)
    assert targets[0].supported
    assert targets[0].unsupported_reason is None

    machine = StockFloorStateMachine(targets, initial_floor=2)
    row = machine.key_down(targets[0].due_s)[-1]
    assert row.kind is StockTransitionKind.HIT_ADVANCE
    assert row.current_floor_before == 2
    assert row.current_floor_after == 3


def test_sliced_stock_reference_rejects_wrong_or_future_landed_floor() -> None:
    chart = _plain_compiled_chart()
    start_s = (chart.floors[2].target_time_s + chart.floors[3].target_time_s) / 2
    segment = build_playable_segment(chart, start_s=start_s)
    for wrong in (0, 1, 3, -1, len(chart.floors), True):
        with pytest.raises(ValueError, match="initial_floor"):
            targets_from_playable_segment(segment, initial_floor=wrong)


def test_sliced_stock_reference_still_detects_omitted_in_segment_floor() -> None:
    chart = _plain_compiled_chart()
    start_s = (chart.floors[2].target_time_s + chart.floors[3].target_time_s) / 2
    segment = build_playable_segment(chart, start_s=start_s)
    assert len(segment.targets) >= 2
    omitted = replace(segment, targets=segment.targets[1:])
    first = targets_from_playable_segment(omitted, initial_floor=2)[0]
    assert not first.supported
    assert "non-midspin floor omitted" in first.unsupported_reason


def test_whole_chart_stock_reference_is_unchanged_with_floor_zero() -> None:
    segment = build_playable_segment(_plain_compiled_chart())
    assert targets_from_playable_segment(segment) == targets_from_playable_segment(
        segment, initial_floor=0
    )


def test_sliced_stock_reference_preserves_midspin_bridge() -> None:
    chart = compile_adofai(
        parse_adofai_text(
            '{"angleData":[0,0,999,0],"settings":{"bpm":120,'
            '"countdownTicks":0},"actions":[]}'
        )
    )
    assert chart.floors[2].midspin
    start_s = (chart.floors[1].target_time_s + chart.floors[3].target_time_s) / 2
    segment = build_playable_segment(chart, start_s=start_s)
    assert segment.targets[0].floor_index == 3
    first = targets_from_playable_segment(segment, initial_floor=1)[0]
    assert first.supported
    assert first.skipped_midspin_floors == (2,)
