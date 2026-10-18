from __future__ import annotations

"""Versioned, truth-free Timing -> Fingering contract for the 12-axis Motor.

This is an interface and validator only. No chart targets, evaluator judgements,
hidden times, or future frames enter these data types. The legacy 8-key/263D
contract in causal_contract.py is unchanged.

Adapters emit one TimingFrameUpdate per 10 ms local policy step; local event
IDs are persistent and not chart/floor/target ordinals. A separate scheduler
owns motor reservations.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


CONTROL_DT_S = 0.010
MAX_CHORD_SIZE = 8
CONTRACT_VERSION = "timing-fingering-v0.2"
ELIGIBLE = "eligible"
ELIGIBLE_PARTIAL = "eligible_partial"
ABSTAIN = "abstain"
VALID_STATUSES = frozenset((ELIGIBLE, ELIGIBLE_PARTIAL, ABSTAIN))


def _nonnegative(value: object, field: str) -> None:
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")


def _ids(values: tuple[int, ...], field: str) -> None:
    if type(values) is not tuple or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{field} must be a tuple of nonnegative integer local IDs")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must contain sorted unique local IDs")


@dataclass(frozen=True, slots=True)
class TimingEventToken:
    """Local forecast, not a true target ID or event time.

    confidence=None means uncalibrated/unknown, suitable for deterministic
    Timing v1.4. uncertainty_s is optional and must not be fabricated.
    """

    local_id: int
    due_local_s: float
    chord_size: int
    confirmations: int = 1
    confirmed_now: bool = True
    confidence: float | None = None
    uncertainty_s: float | None = None

    def __post_init__(self) -> None:
        if type(self.local_id) is not int or self.local_id < 0:
            raise ValueError("local_id must be a nonnegative integer")
        _nonnegative(self.due_local_s, "due_local_s")
        if type(self.chord_size) is not int or not 1 <= self.chord_size <= MAX_CHORD_SIZE:
            raise ValueError("chord_size must be 1..8")
        if type(self.confirmations) is not int or self.confirmations < 0:
            raise ValueError("confirmations must be a nonnegative integer")
        if type(self.confirmed_now) is not bool:
            raise ValueError("confirmed_now must be bool")
        if self.confirmed_now and self.confirmations == 0:
            raise ValueError("confirmed_now requires a positive confirmation count")
        if self.confidence is not None:
            _nonnegative(self.confidence, "confidence")
            if self.confidence > 1:
                raise ValueError("confidence must be in [0, 1]")
        if self.uncertainty_s is not None:
            _nonnegative(self.uncertainty_s, "uncertainty_s")

    def remaining_s(self, step: int) -> float:
        """Signed local-time remaining, allowed to be negative after due."""
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return self.due_local_s - step * CONTROL_DT_S

    def remaining_lower_s(self, step: int) -> float:
        """Conservative remaining time from the due interval lower bound."""
        uncertainty = 0.0 if self.uncertainty_s is None else self.uncertainty_s
        return self.remaining_s(step) - uncertainty


@dataclass(frozen=True, slots=True)
class TimingFrameUpdate:
    """One causal update with explicit local forecast lifecycle.

    Pending tokens are sorted by due time. Exactly simultaneous predictions
    are one chord; distinct events even within one 10ms interval are separate.
    new_ids and reused_ids partition pending IDs. canceled_ids and elapsed_ids
    retire prior live forecasts, not evaluator outcomes.
    """

    step: int
    status: str
    covered_until_s: float
    pending: tuple[TimingEventToken, ...]
    source: str
    version: str = CONTRACT_VERSION
    new_ids: tuple[int, ...] = ()
    reused_ids: tuple[int, ...] = ()
    canceled_ids: tuple[int, ...] = ()
    elapsed_ids: tuple[int, ...] = ()
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self.status not in VALID_STATUSES:
            raise ValueError("unsupported Timing forecast status")
        _nonnegative(self.covered_until_s, "covered_until_s")
        if self.status != ABSTAIN and self.covered_until_s + 1e-9 < self.local_clock_s:
            raise ValueError("eligible coverage cannot end before current policy step")
        if type(self.source) is not str or not self.source.strip():
            raise ValueError("forecast source must be a nonempty identifier")
        if self.version != CONTRACT_VERSION:
            raise ValueError("unknown Timing-Fingering contract version")
        if type(self.pending) is not tuple or any(
            not isinstance(token, TimingEventToken) for token in self.pending
        ):
            raise ValueError("pending must be a tuple of TimingEventToken")
        previous_due = None
        seen: set[int] = set()
        for token in self.pending:
            if token.local_id in seen:
                raise ValueError("duplicate local event ID in pending")
            seen.add(token.local_id)
            if previous_due is not None and token.due_local_s <= previous_due:
                raise ValueError("pending must be sorted; merge exactly simultaneous chords")
            previous_due = token.due_local_s
        fields = {
            "new_ids": self.new_ids,
            "reused_ids": self.reused_ids,
            "canceled_ids": self.canceled_ids,
            "elapsed_ids": self.elapsed_ids,
        }
        for name, values in fields.items():
            _ids(values, name)
        groups = {name: set(values) for name, values in fields.items()}
        for name, ids in groups.items():
            for other, others in groups.items():
                if name < other and ids & others:
                    raise ValueError(f"overlapping lifecycle ID groups: {name}/{other}")
        if groups["new_ids"] | groups["reused_ids"] != seen:
            raise ValueError("new_ids and reused_ids must partition pending")
        if self.status == ABSTAIN:
            if type(self.abstention_reason) is not str or not self.abstention_reason.strip():
                raise ValueError("abstention requires an explicit reason")
            if self.new_ids or self.canceled_ids or self.elapsed_ids:
                raise ValueError("abstention cannot create, cancel, or expire forecasts")
            if any(token.confirmed_now for token in self.pending):
                raise ValueError("abstention cannot reconfirm events")
        elif self.status == ELIGIBLE and self.abstention_reason is not None:
            raise ValueError("fully eligible updates cannot carry an abstention reason")

    @property
    def local_clock_s(self) -> float:
        return self.step * CONTROL_DT_S

    @property
    def pending_ids(self) -> tuple[int, ...]:
        return tuple(token.local_id for token in self.pending)

    def remaining_s(self, local_id: int) -> float:
        """Compute remaining time on this frame, avoiding stale offsets."""
        for token in self.pending:
            if token.local_id == local_id:
                return token.remaining_s(self.step)
        raise KeyError(local_id)


class TimingFrameSequenceValidator:
    """Stateful, truth-free check for local ID lifecycle and causal step order."""

    def __init__(self) -> None:
        self._step: int | None = None
        self._source: str | None = None
        self._max_ever_id = -1
        self._live: frozenset[int] = frozenset()

    @property
    def last_step(self) -> int | None:
        return self._step

    @property
    def live_ids(self) -> frozenset[int]:
        return self._live

    def accept(self, frame: TimingFrameUpdate) -> None:
        if not isinstance(frame, TimingFrameUpdate):
            raise TypeError("expected TimingFrameUpdate")
        if self._step is not None:
            if frame.step != self._step + 1:
                raise ValueError("Timing updates must be consecutive local control steps")
            if frame.source != self._source:
                raise ValueError("Timing source cannot change in the middle of a stream")

        new_ids = set(frame.new_ids)
        reused_ids = set(frame.reused_ids)
        released_ids = set(frame.canceled_ids) | set(frame.elapsed_ids)
        if any(local_id <= self._max_ever_id for local_id in new_ids):
            raise ValueError("new local IDs must increase monotonically over the stream")
        if reused_ids - self._live:
            raise ValueError("reused local ID has no live prior forecast")
        if released_ids - self._live:
            raise ValueError("cannot retire a non-live local forecast")
        if self._live - reused_ids != released_ids:
            raise ValueError("forecast disappeared without explicit lifecycle update")
        if self._step is None and (reused_ids or released_ids):
            raise ValueError("first update cannot reuse or retire unseen local IDs")

        # Update only after all validation, so rejected frames do not mutate state.
        self._live = frozenset(frame.pending_ids)
        if new_ids:
            self._max_ever_id = max(self._max_ever_id, max(new_ids))
        self._step = frame.step
        self._source = frame.source


def validate_timing_frames(frames: Iterable[TimingFrameUpdate]) -> None:
    """Validate a finite stream without accessing chart truth."""
    validator = TimingFrameSequenceValidator()
    for frame in frames:
        validator.accept(frame)
