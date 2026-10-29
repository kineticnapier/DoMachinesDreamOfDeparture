from __future__ import annotations

"""Time-indexed due-event selection for the research-only physical runner.

This is evaluator/runner bookkeeping, not a Student HUD input. The caller
still owns the selected set and may retry deferred events until expiry.
"""

from bisect import bisect_left, bisect_right
from math import isfinite
from typing import Generic, Protocol, Sequence, TypeVar


class TimedEvent(Protocol):
    event_index: int
    episode_time_s: float


EventT = TypeVar("EventT", bound=TimedEvent)


class DueEventWindow(Generic[EventT]):
    """Find due events in original order without rescanning future events."""

    def __init__(self, events: Sequence[EventT]):
        self._events = tuple(events)
        self._times = tuple(event.episode_time_s for event in self._events)
        if any(not isfinite(value) for value in self._times):
            raise ValueError("nonfinite due-event time")
        if any(left > right for left, right in zip(self._times, self._times[1:])):
            raise ValueError("due-event timeline must be sorted")

    def due(
        self, now_s: float, *, preview_tail_s: float,
        prediction_s: float, selected: set[int],
    ) -> tuple[EventT, ...]:
        """Match the old two-stage upcoming/due filters exactly.

        Inclusive expiry: event.time + preview_tail_s >= now_s.
        Inclusive prediction: event.time <= now_s + prediction_s + 1e-9.
        Selected events are omitted, but deferred events are not consumed.
        """
        start = bisect_left(self._times, now_s - preview_tail_s)
        # The old predicate adds the tail to the event timestamp. At an
        # exact boundary, (event.time + tail >= now) can differ by one ULP
        # from (event.time >= now - tail). Keep the original predicate.
        while start and self._times[start - 1] + preview_tail_s >= now_s:
            start -= 1
        stop = bisect_right(self._times, now_s + prediction_s + 1e-9)
        return tuple(
            event for event in self._events[start:stop]
            if event.event_index not in selected
            and event.episode_time_s + preview_tail_s >= now_s
        )
