"""Frame-driven adapter for the audited Stock ordinary-floor reference.

The adapter fixes an explicit virtual Update grid.  It does not claim that the
grid equals a live Unity player's Update phase.  Each frame performs the DLL
order audited in Phase 20C: time/angle update, ``CheckPostHoldFail``, then the
queued KeyDowns in timestamp/insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import math

from .stock_floor_state_machine import (
    StockFloorStateMachine,
    StockStateTransition,
)


@dataclass(frozen=True, slots=True)
class StockQueuedKeyDown:
    input_id: str
    occurred_at_s: float
    contact: str = ""

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValueError("input_id must be non-empty")
        if not math.isfinite(self.occurred_at_s):
            raise ValueError("KeyDown time must be finite")


@dataclass(frozen=True, slots=True)
class StockProcessedKeyDown:
    keydown: StockQueuedKeyDown
    transitions: tuple[StockStateTransition, ...]

    @property
    def input_transition(self) -> StockStateTransition | None:
        rows = [row for row in self.transitions if row.input_time_s is not None]
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("one KeyDown produced multiple input transitions")
        return rows[0]


@dataclass(frozen=True, slots=True)
class StockFrameUpdate:
    frame_index: int
    frame_time_s: float
    prior_frame_time_s: float | None
    automatic_transitions: tuple[StockStateTransition, ...]
    processed_keydowns: tuple[StockProcessedKeyDown, ...]
    current_floor_before: int
    current_floor_after: int
    next_target_ordinal: int | None
    failed: bool

    @property
    def transitions(self) -> tuple[StockStateTransition, ...]:
        return self.automatic_transitions + tuple(
            row
            for processed in self.processed_keydowns
            for row in processed.transitions
        )


def first_frame_strictly_after(
    threshold_s: float,
    *,
    frame_dt_s: float,
    phase_s: float = 0.0,
) -> tuple[int, float]:
    """Return the first virtual frame whose time is strictly above a threshold."""

    if not all(math.isfinite(value) for value in (threshold_s, frame_dt_s, phase_s)):
        raise ValueError("frame values must be finite")
    if frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be positive")
    relative = (threshold_s - phase_s) / frame_dt_s
    index = math.floor(relative) + 1
    return index, phase_s + index * frame_dt_s


class StockFrameStateMachine:
    """Drive :class:`StockFloorStateMachine` from explicit Update frames."""

    def __init__(
        self,
        floor_machine: StockFloorStateMachine,
        *,
        frame_dt_s: float = 0.010,
        phase_s: float = 0.0,
    ) -> None:
        if not math.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be finite and positive")
        if not math.isfinite(phase_s):
            raise ValueError("phase_s must be finite")
        self.floor_machine = floor_machine
        self.frame_dt_s = float(frame_dt_s)
        self.phase_s = float(phase_s)
        self._queue: list[tuple[float, int, StockQueuedKeyDown]] = []
        self._seen_input_ids: set[str] = set()
        self._sequence = 0
        self._last_frame_index: int | None = None
        self._last_frame_time_s: float | None = None

    def queue_keydown(self, keydown: StockQueuedKeyDown) -> None:
        if keydown.input_id in self._seen_input_ids:
            raise ValueError("input_id must be unique")
        if (
            self._last_frame_time_s is not None
            and keydown.occurred_at_s <= self._last_frame_time_s + 1e-12
        ):
            raise ValueError("cannot queue a KeyDown at or before the last frame")
        self._seen_input_ids.add(keydown.input_id)
        self._sequence += 1
        heappush(self._queue, (keydown.occurred_at_s, self._sequence, keydown))

    def update(self, frame_index: int) -> StockFrameUpdate:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame indices must increase strictly")
        frame_time_s = self.phase_s + frame_index * self.frame_dt_s
        prior = self._last_frame_time_s
        current_before = self.floor_machine.current_floor

        automatic = self.floor_machine.advance(frame_time_s)
        ready: list[StockQueuedKeyDown] = []
        while self._queue and self._queue[0][0] <= frame_time_s + 1e-12:
            ready.append(heappop(self._queue)[2])
        processed = tuple(
            StockProcessedKeyDown(
                keydown=keydown,
                transitions=self.floor_machine.key_down_after_update(
                    keydown.occurred_at_s
                ),
            )
            for keydown in ready
        )
        self._last_frame_index = frame_index
        self._last_frame_time_s = frame_time_s
        next_target = self.floor_machine.next_target
        return StockFrameUpdate(
            frame_index=frame_index,
            frame_time_s=frame_time_s,
            prior_frame_time_s=prior,
            automatic_transitions=automatic,
            processed_keydowns=processed,
            current_floor_before=current_before,
            current_floor_after=self.floor_machine.current_floor,
            next_target_ordinal=None if next_target is None else next_target.ordinal,
            failed=self.floor_machine.failed,
        )

    @property
    def pending_keydowns(self) -> tuple[StockQueuedKeyDown, ...]:
        return tuple(row[2] for row in sorted(self._queue))
