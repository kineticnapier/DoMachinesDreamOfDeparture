from __future__ import annotations

"""Deterministic Hold input state machine for the H0-confirmed contract.

Timing windows are supplied by an evaluator/integration layer. This module
does not infer stock margins, chart time, Hold duration, or future events.
Consecutive-Hold boundary behavior remains fail-closed until runtime evidence
is available.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class HoldBehavior(str, Enum):
    NORMAL = "normal"
    CAN_HIT_END = "can_hit_end"
    NO_HOLD_NEEDED = "no_hold_needed"

    @property
    def strict(self) -> bool:
        return self is HoldBehavior.NORMAL

    @property
    def requires_holding(self) -> bool:
        return self is not HoldBehavior.NO_HOLD_NEEDED


class HoldJudgementState(str, Enum):
    IDLE = "idle"
    WAITING_FOR_START = "waiting_for_start"
    HOLDING = "holding"
    WAITING_FOR_RELEASE = "waiting_for_release"
    COMPLETED = "completed"
    FAILED = "failed"


class HoldSignal(str, Enum):
    START_TOO_EARLY = "start_too_early"
    STARTED = "started"
    UNRELATED_KEY_DOWN = "unrelated_key_down"
    RELEASED_WITHOUT_FAILURE = "released_without_failure"
    COMPLETED_BY_RELEASE = "completed_by_release"
    COMPLETED_BY_KEY_DOWN = "completed_by_key_down"


class HoldFailure(str, Enum):
    START_MISSED = "start_missed"
    EARLY_RELEASE = "early_release"
    RELEASE_MISSED = "release_missed"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    lower_s: float
    upper_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.lower_s) or not isfinite(self.upper_s):
            raise ValueError("Hold time window must be finite")
        if self.lower_s > self.upper_s:
            raise ValueError("Hold time window is reversed")

    def contains(self, value: float) -> bool:
        return self.lower_s <= value <= self.upper_s


@dataclass(frozen=True, slots=True)
class HoldIntervalContract:
    hold_id: str
    start_window: TimeWindow
    release_window: TimeWindow
    continues_to_hold: bool = False

    def __post_init__(self) -> None:
        if not self.hold_id:
            raise ValueError("Hold ID must be nonempty")
        if self.release_window.lower_s < self.start_window.upper_s:
            raise ValueError("Hold release window overlaps its start window")
        if type(self.continues_to_hold) is not bool:
            raise TypeError("continues_to_hold must be bool")


def _validate_contacts(values: frozenset[int]) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Hold contacts must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class HoldInputFrame:
    observed_at_s: float
    down: frozenset[int] = frozenset()
    up: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s):
            raise ValueError("Hold input time must be finite")
        _validate_contacts(self.down)
        _validate_contacts(self.up)
        if self.down & self.up:
            raise ValueError("one contact cannot be both down and up in a frame")


@dataclass(frozen=True, slots=True)
class HoldJudgementUpdate:
    hold_id: str | None
    observed_at_s: float | None
    state: HoldJudgementState
    behavior: HoldBehavior
    captured_contacts: frozenset[int]
    signals: tuple[HoldSignal, ...] = ()
    failure: HoldFailure | None = None
    ordinary_hit_credit: int = 0
    hold_completion_credit: int = 0


class HoldJudgementStateMachine:
    """One-Hold adjudicator with no chart or Motor dependencies."""

    def __init__(self, behavior: HoldBehavior) -> None:
        if not isinstance(behavior, HoldBehavior):
            raise TypeError("behavior must be HoldBehavior")
        self.behavior = behavior
        self.reset()

    def reset(self) -> None:
        self._contract: HoldIntervalContract | None = None
        self._state = HoldJudgementState.IDLE
        self._captured: set[int] = set()
        self._last_time: float | None = None
        self._failure: HoldFailure | None = None

    def arm(self, contract: HoldIntervalContract) -> HoldJudgementUpdate:
        if not isinstance(contract, HoldIntervalContract):
            raise TypeError("contract must be HoldIntervalContract")
        if contract.continues_to_hold:
            raise NotImplementedError(
                "consecutive Hold boundary semantics are runtime-unverified"
            )
        if self._state is not HoldJudgementState.IDLE:
            raise ValueError("reset Hold state machine before arming")
        self._contract = contract
        self._state = HoldJudgementState.WAITING_FOR_START
        return self.snapshot()

    def snapshot(self) -> HoldJudgementUpdate:
        return HoldJudgementUpdate(
            hold_id=self._contract.hold_id if self._contract else None,
            observed_at_s=self._last_time,
            state=self._state,
            behavior=self.behavior,
            captured_contacts=frozenset(self._captured),
            failure=self._failure,
        )

    def _update(self, at_s: float, *, signals=(), ordinary=0,
                completion=0) -> HoldJudgementUpdate:
        return HoldJudgementUpdate(
            hold_id=self._contract.hold_id if self._contract else None,
            observed_at_s=at_s,
            state=self._state,
            behavior=self.behavior,
            captured_contacts=frozenset(self._captured),
            signals=tuple(signals),
            failure=self._failure,
            ordinary_hit_credit=ordinary,
            hold_completion_credit=completion,
        )

    def _fail(self, at_s: float, reason: HoldFailure) -> HoldJudgementUpdate:
        self._state = HoldJudgementState.FAILED
        self._failure = reason
        return self._update(at_s)

    def _complete(self, at_s: float, signal: HoldSignal) -> HoldJudgementUpdate:
        self._state = HoldJudgementState.COMPLETED
        return self._update(at_s, signals=(signal,), completion=1)

    def observe(self, frame: HoldInputFrame) -> HoldJudgementUpdate:
        if not isinstance(frame, HoldInputFrame):
            raise TypeError("frame must be HoldInputFrame")
        if self._contract is None or self._state is HoldJudgementState.IDLE:
            raise ValueError("arm Hold state machine before observing input")
        if self._state in (HoldJudgementState.COMPLETED, HoldJudgementState.FAILED):
            raise ValueError("terminal Hold state requires reset")
        if self._last_time is not None and frame.observed_at_s < self._last_time:
            raise ValueError("Hold input time must be monotonic")
        self._last_time = frame.observed_at_s
        at_s = frame.observed_at_s
        contract = self._contract

        if self._state is HoldJudgementState.WAITING_FOR_START:
            if at_s > contract.start_window.upper_s:
                return self._fail(at_s, HoldFailure.START_MISSED)
            if not frame.down:
                return self._update(at_s)
            if at_s < contract.start_window.lower_s:
                return self._update(at_s, signals=(HoldSignal.START_TOO_EARLY,))
            self._captured = set(frame.down)
            self._state = HoldJudgementState.HOLDING
            return self._update(
                at_s, signals=(HoldSignal.STARTED,), ordinary=1
            )

        if at_s > contract.release_window.upper_s:
            return self._fail(at_s, HoldFailure.RELEASE_MISSED)

        released = self._captured & frame.up
        self._captured.difference_update(frame.up)
        if released:
            if at_s < contract.release_window.lower_s:
                if self.behavior.requires_holding:
                    return self._fail(at_s, HoldFailure.EARLY_RELEASE)
                self._state = HoldJudgementState.WAITING_FOR_RELEASE
                return self._update(
                    at_s, signals=(HoldSignal.RELEASED_WITHOUT_FAILURE,)
                )
            return self._complete(at_s, HoldSignal.COMPLETED_BY_RELEASE)

        unrelated = frame.down - self._captured
        if unrelated:
            if (
                not self.behavior.strict
                and contract.release_window.contains(at_s)
            ):
                return self._complete(at_s, HoldSignal.COMPLETED_BY_KEY_DOWN)
            return self._update(
                at_s, signals=(HoldSignal.UNRELATED_KEY_DOWN,)
            )
        return self._update(at_s)
