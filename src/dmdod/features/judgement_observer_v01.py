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


def _finite_clock(value: object) -> float | None:
    """Accept only finite numeric HUD timestamps without implicit coercion."""
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) else None


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
        # Never promote a non-string key into a genuine visible HUD label.
        if any(type(name) is not str or not name for name in counters):
            raise ValueError("counter labels must be non-empty strings")
        observed = _finite_clock(observed_at_s)
        displayed = _finite_clock(displayed_at_s)
        if observed is None or displayed is None:
            raise ValueError("counter timestamps must be finite numbers")
        return cls(
            observed_at_s=observed,
            displayed_at_s=displayed,
            counters=tuple(counters.items()),
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
        # A bare string must not silently become a schema of characters.
        if isinstance(labels, str):
            raise ValueError("counter labels must be a sequence of non-empty strings")
        normalized = tuple(labels)
        if not normalized or any(type(label) is not str or not label for label in normalized):
            raise ValueError("counter labels must be non-empty strings")
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
        observed = _finite_clock(frame.observed_at_s)
        displayed = _finite_clock(frame.displayed_at_s)
        return JudgementDelta(
            status=JudgementDeltaStatus.INVALID,
            # An invalid direct frame may have a nonnumeric or overflowing
            # clock. Preserve INVALID without coercing it into a valid time.
            observed_at_s=observed if observed is not None else float("nan"),
            displayed_at_s=displayed if displayed is not None else float("nan"),
            reason=reason,
        )

    def observe(self, frame: JudgementCounterFrame) -> JudgementDelta:
        observed = _finite_clock(frame.observed_at_s)
        displayed = _finite_clock(frame.displayed_at_s)
        if observed is None or displayed is None:
            return self._invalid(frame, "timestamps must be finite numbers")
        if displayed > observed + 1e-12:
            return self._invalid(frame, "display timestamp is later than observation")
        if self._previous_observed_at_s is not None and observed < self._previous_observed_at_s:
            return self._invalid(frame, "observation timestamp moved backward")
        if self._previous_displayed_at_s is not None and displayed < self._previous_displayed_at_s:
            return self._invalid(frame, "display timestamp moved backward")

        # Direct dataclass construction can bypass the mapping builder and
        # supply None, a scalar, or malformed rows. Treat these as an invalid
        # HUD frame rather than letting unpacking/iteration raise an exception.
        try:
            counter_rows = tuple(frame.counters)
        except (TypeError, ValueError):
            return self._invalid(frame, "counter rows must be iterable label/value pairs")
        raw: dict[str, object] = {}
        for row in counter_rows:
            try:
                name, value = row
            except (TypeError, ValueError):
                return self._invalid(frame, "counter rows must contain label/value pairs")
            # Directly constructed frames bypass from_mapping(). Fail closed,
            # including for unhashable keys, instead of raising TypeError.
            if type(name) is not str or not name:
                return self._invalid(frame, "counter labels must be non-empty strings")
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
