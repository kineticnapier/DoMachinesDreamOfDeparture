from __future__ import annotations

import pytest

from dmdod.evaluation.hold_judgement_v01 import (
    HoldBehavior,
    HoldFailure,
    HoldInputFrame,
    HoldIntervalContract,
    HoldJudgementState,
    HoldJudgementStateMachine,
    HoldSignal,
    TimeWindow,
)


def contract(*, chained=False):
    return HoldIntervalContract(
        hold_id="hold:7",
        start_window=TimeWindow(.095, .105),
        release_window=TimeWindow(.995, 1.005),
        continues_to_hold=chained,
    )


def frame(time, *, down=(), up=()):
    return HoldInputFrame(time, frozenset(down), frozenset(up))


@pytest.mark.parametrize("behavior", tuple(HoldBehavior))
def test_all_modes_start_once_and_do_not_double_score_completion(behavior):
    machine = HoldJudgementStateMachine(behavior)
    machine.arm(contract())
    started = machine.observe(frame(.100, down=(3,)))
    completed = machine.observe(frame(1.000, up=(3,)))

    assert started.state is HoldJudgementState.HOLDING
    assert started.signals == (HoldSignal.STARTED,)
    assert started.ordinary_hit_credit == 1
    assert completed.state is HoldJudgementState.COMPLETED
    assert completed.hold_completion_credit == 1
    assert completed.ordinary_hit_credit == 0


def test_normal_early_release_fails_even_when_another_start_key_remains_held():
    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    machine.arm(contract())
    machine.observe(frame(.100, down=(1, 2)))
    result = machine.observe(frame(.500, up=(1,)))

    assert result.state is HoldJudgementState.FAILED
    assert result.failure is HoldFailure.EARLY_RELEASE
    assert result.captured_contacts == frozenset({2})


def test_normal_unrelated_key_does_not_complete_hold():
    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    machine.arm(contract())
    machine.observe(frame(.100, down=(1,)))
    result = machine.observe(frame(1.000, down=(9,)))

    assert result.state is HoldJudgementState.HOLDING
    assert result.signals == (HoldSignal.UNRELATED_KEY_DOWN,)


def test_can_hit_end_accepts_new_keydown_but_not_early_release():
    machine = HoldJudgementStateMachine(HoldBehavior.CAN_HIT_END)
    machine.arm(contract())
    machine.observe(frame(.100, down=(1,)))
    early = machine.observe(frame(.800, down=(9,)))
    done = machine.observe(frame(1.000, down=(9,)))

    assert early.state is HoldJudgementState.HOLDING
    assert early.signals == (HoldSignal.UNRELATED_KEY_DOWN,)
    assert done.state is HoldJudgementState.COMPLETED
    assert done.signals == (HoldSignal.COMPLETED_BY_KEY_DOWN,)

    failing = HoldJudgementStateMachine(HoldBehavior.CAN_HIT_END)
    failing.arm(contract())
    failing.observe(frame(.100, down=(1,)))
    assert failing.observe(frame(.500, up=(1,))).failure is HoldFailure.EARLY_RELEASE


def test_no_hold_needed_allows_release_then_requires_end_input():
    machine = HoldJudgementStateMachine(HoldBehavior.NO_HOLD_NEEDED)
    machine.arm(contract())
    machine.observe(frame(.100, down=(4,)))
    released = machine.observe(frame(.400, up=(4,)))
    completed = machine.observe(frame(1.000, down=(7,)))

    assert released.state is HoldJudgementState.WAITING_FOR_RELEASE
    assert released.signals == (HoldSignal.RELEASED_WITHOUT_FAILURE,)
    assert completed.state is HoldJudgementState.COMPLETED
    assert completed.signals == (HoldSignal.COMPLETED_BY_KEY_DOWN,)


def test_start_and_release_boundaries_are_inclusive_and_time_is_monotonic():
    for start in (.095, .105):
        machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
        machine.arm(contract())
        assert machine.observe(frame(start, down=(1,))).state is HoldJudgementState.HOLDING
        assert machine.observe(frame(.995, up=(1,))).state is HoldJudgementState.COMPLETED

    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    machine.arm(contract())
    machine.observe(frame(.100, down=(1,)))
    with pytest.raises(ValueError, match="monotonic"):
        machine.observe(frame(.090))


def test_missed_start_and_release_fail_deterministically():
    missing_start = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    missing_start.arm(contract())
    assert missing_start.observe(frame(.106)).failure is HoldFailure.START_MISSED

    missing_end = HoldJudgementStateMachine(HoldBehavior.NO_HOLD_NEEDED)
    missing_end.arm(contract())
    missing_end.observe(frame(.100, down=(1,)))
    assert missing_end.observe(frame(1.006)).failure is HoldFailure.RELEASE_MISSED


def test_too_early_start_is_reported_without_false_success():
    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    machine.arm(contract())
    early = machine.observe(frame(.050, down=(1,)))
    assert early.state is HoldJudgementState.WAITING_FOR_START
    assert early.signals == (HoldSignal.START_TOO_EARLY,)
    assert early.ordinary_hit_credit == 0


def test_reset_clears_keys_clock_and_terminal_state():
    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    machine.arm(contract())
    machine.observe(frame(.100, down=(1, 2)))
    machine.reset()
    assert machine.snapshot().state is HoldJudgementState.IDLE
    assert machine.snapshot().captured_contacts == frozenset()
    machine.arm(contract())
    assert machine.observe(frame(.095, down=(3,))).state is HoldJudgementState.HOLDING


def test_consecutive_hold_boundary_is_rejected_until_stock_runtime_is_verified():
    machine = HoldJudgementStateMachine(HoldBehavior.NORMAL)
    with pytest.raises(NotImplementedError, match="consecutive Hold"):
        machine.arm(contract(chained=True))


@pytest.mark.parametrize("bad", ((1,), (1, 1), (-1,)))
def test_contacts_must_be_unique_nonnegative_integers(bad):
    # Duplicate cannot survive a frozenset, so use invalid negative/type cases
    # to verify the public frame boundary instead of inventing key ordering.
    values = frozenset(bad)
    if values == frozenset({1}):
        assert HoldInputFrame(.1, values, frozenset()).down == values
    else:
        with pytest.raises(ValueError, match="contacts"):
            HoldInputFrame(.1, values, frozenset())
