from __future__ import annotations

"""Causal differencing for player-visible cumulative judgement counters.

The observer deliberately knows neither ADOFAI's scorer nor event identities.
Its input is the already-rendered/captured counter surface.  A simulator may
use privileged state to render that surface, just as the game does, but that
privilege ends at :class:`JudgementCounterFrame`.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence


class JudgementDeltaStatus(str, Enum):
    BASELINE = "baseline"
    UNCHANGED = "unchanged"
    DELTA = "delta"
    RESET = "reset"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class JudgementCounterFrame:
    """One HUD capture of cumulative visible counters.

    ``displayed_at_s`` is the timestamp assigned to the rendered frame;
    ``observed_at_s`` is when the Student receives it.  Neither is asserted to
    be the hidden judgement's exact occurrence time.
    """

    observed_at_s: float
    displayed_at_s: float
    counters: tuple[tuple[str, object], ...]

    @classmethod
    def from_mapping(
        cls,
        *,
        observed_at_s: float,
        displayed_at_s: float,
        counters: Mapping[str, object],
    ) -> JudgementCounterFrame:
        return cls(
            observed_at_s=float(observed_at_s),
            displayed_at_s=float(displayed_at_s),
            counters=tuple((str(name), value) for name, value in counters.items()),
        )


@dataclass(frozen=True, slots=True)
class JudgementDelta:
    status: JudgementDeltaStatus
    observed_at_s: float
    displayed_at_s: float
    deltas: tuple[tuple[str, int], ...] = ()
    event_interval_s: tuple[float, float] | None = None
    delta_available: bool = False
    within_interval_order_known: bool = False
    reason: str = ""

    @property
    def available_at_s(self) -> float:
        return self.observed_at_s

    @property
    def delta_map(self) -> dict[str, int]:
        return dict(self.deltas)

    @property
    def total_increment(self) -> int:
        return sum(value for _, value in self.deltas)


class JudgementObserver:
    """Difference a fixed, explicitly configured visible counter schema.

    The labels are display-contract identifiers, not an implicit mapping to
    ``TimingJudgement``.  Unknown/missing labels, invalid values, or noncausal
    timestamps invalidate the baseline so a later frame cannot fabricate a
    delta across an unobserved discontinuity.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        normalized = tuple(str(label) for label in labels)
        if not normalized or any(not label for label in normalized):
            raise ValueError("counter labels must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("counter labels must be unique")
        self.labels = normalized
        self._baseline: tuple[int, ...] | None = None
        self._previous_observed_at_s: float | None = None
        self._previous_displayed_at_s: float | None = None

    def reset(self) -> None:
        self._baseline = None
        self._previous_observed_at_s = None
        self._previous_displayed_at_s = None

    def _invalid(self, frame: JudgementCounterFrame, reason: str) -> JudgementDelta:
        self.reset()
        return JudgementDelta(
            status=JudgementDeltaStatus.INVALID,
            observed_at_s=float(frame.observed_at_s),
            displayed_at_s=float(frame.displayed_at_s),
            reason=reason,
        )

    def observe(self, frame: JudgementCounterFrame) -> JudgementDelta:
        observed = float(frame.observed_at_s)
        displayed = float(frame.displayed_at_s)
        if not isfinite(observed) or not isfinite(displayed):
            return self._invalid(frame, "timestamps must be finite")
        if displayed > observed + 1e-12:
            return self._invalid(frame, "display timestamp is later than observation")
        if self._previous_observed_at_s is not None and observed < self._previous_observed_at_s:
            return self._invalid(frame, "observation timestamp moved backward")
        if self._previous_displayed_at_s is not None and displayed < self._previous_displayed_at_s:
            return self._invalid(frame, "display timestamp moved backward")

        raw: dict[str, object] = {}
        for name, value in frame.counters:
            if name in raw:
                return self._invalid(frame, f"duplicate counter label: {name}")
            raw[name] = value
        if tuple(sorted(raw)) != tuple(sorted(self.labels)):
            return self._invalid(frame, "visible counter schema is missing or changed")

        values: list[int] = []
        for label in self.labels:
            value = raw[label]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return self._invalid(frame, f"counter {label!r} must be a non-negative integer")
            values.append(value)
        current = tuple(values)

        if self._baseline is None:
            self._baseline = current
            self._previous_observed_at_s = observed
            self._previous_displayed_at_s = displayed
            return JudgementDelta(
                status=JudgementDeltaStatus.BASELINE,
                observed_at_s=observed,
                displayed_at_s=displayed,
                reason="first valid frame establishes the cumulative baseline",
            )

        previous = self._baseline
        previous_displayed = self._previous_displayed_at_s
        assert previous_displayed is not None
        decreases = [
            label for label, before, after in zip(self.labels, previous, current)
            if after < before
        ]
        self._baseline = current
        self._previous_observed_at_s = observed
        self._previous_displayed_at_s = displayed
        if decreases:
            return JudgementDelta(
                status=JudgementDeltaStatus.RESET,
                observed_at_s=observed,
                displayed_at_s=displayed,
                reason="cumulative counter decreased: " + ", ".join(decreases),
            )

        deltas = tuple(
            (label, after - before)
            for label, before, after in zip(self.labels, previous, current)
        )
        total = sum(value for _, value in deltas)
        return JudgementDelta(
            status=(JudgementDeltaStatus.DELTA if total else JudgementDeltaStatus.UNCHANGED),
            observed_at_s=observed,
            displayed_at_s=displayed,
            deltas=deltas,
            event_interval_s=(previous_displayed, displayed),
            delta_available=True,
            within_interval_order_known=False,
            reason=("visible cumulative counters increased" if total else "no visible counter change"),
        )
