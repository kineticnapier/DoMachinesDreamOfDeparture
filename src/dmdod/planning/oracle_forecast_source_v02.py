from __future__ import annotations

"""Privileged *comparison only* Oracle, emitting the identical public contract.

This is the ONLY component in the timing-source pair allowed to receive true
event timestamps. Never pass this instance, or chart targets, into the HUD
policy. The emitted TimingFrameUpdate contains local IDs and local deadlines.
"""

from dataclasses import dataclass
from math import isfinite

from dmdod.planning.timing_fingering_contract_v02 import (
    ELIGIBLE, TimingEventToken, TimingFrameSequenceValidator, TimingFrameUpdate,
)

SOURCE = "oracle-comparator-v0.2"


@dataclass(frozen=True)
class OracleDue:
    """Comparison-only input; not exposed to the Fingering planner."""

    due_local_s: float
    chord_size: int = 1

    def __post_init__(self) -> None:
        if not isfinite(self.due_local_s) or self.due_local_s < 0:
            raise ValueError("Oracle due times must be finite and nonnegative")
        if type(self.chord_size) is not int or not 1 <= self.chord_size <= 8:
            raise ValueError("invalid Oracle chord size")


class OracleForecastSource:
    def __init__(self, due_events: tuple[OracleDue, ...], *,
                 horizon_s: float = .080) -> None:
        if not 0 < horizon_s <= .080 or not isfinite(horizon_s):
            raise ValueError("Oracle horizon outside frozen 80ms limit")
        ordered = tuple(sorted(due_events, key=lambda x: x.due_local_s))
        # Real simultaneous targets are represented by one chord token.
        coalesced: list[OracleDue] = []
        for item in ordered:
            if coalesced and abs(coalesced[-1].due_local_s - item.due_local_s) <= 1e-9:
                current = coalesced[-1]
                coalesced[-1] = OracleDue(current.due_local_s,
                                          current.chord_size + item.chord_size)
            else:
                coalesced.append(item)
        if any(x.chord_size > 8 for x in coalesced):
            raise ValueError("chord exceeds eight logical contacts")
        self._events = tuple(enumerate(coalesced))
        self._horizon_s = horizon_s
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._live: frozenset[int] = frozenset()
        self._validator = TimingFrameSequenceValidator()

    def observe(self, step: int) -> TimingFrameUpdate:
        if type(step) is not int or step != self._next_step:
            raise ValueError("Oracle steps must be consecutive, starting at zero")
        now = step * .010
        pending = tuple(
            TimingEventToken(
                local_id=index, due_local_s=event.due_local_s,
                chord_size=event.chord_size,
            )
            for index, event in self._events
            if now < event.due_local_s <= now + self._horizon_s + 1e-9
        )
        ids = frozenset(item.local_id for item in pending)
        previously_due = dict((index, item.due_local_s) for index, item in self._events)
        retired = self._live - ids
        frame = TimingFrameUpdate(
            step=step, status=ELIGIBLE, covered_until_s=now + self._horizon_s,
            pending=pending, source=SOURCE,
            new_ids=tuple(sorted(ids - self._live)),
            reused_ids=tuple(sorted(ids & self._live)),
            elapsed_ids=tuple(sorted(x for x in retired
                                     if previously_due[x] <= now + 1e-9)),
            canceled_ids=tuple(sorted(x for x in retired
                                      if previously_due[x] > now + 1e-9)),
        )
        self._validator.accept(frame)
        self._live = ids
        self._next_step += 1
        return frame
