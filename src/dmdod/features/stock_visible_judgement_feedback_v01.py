from __future__ import annotations

"""Research renderer for Stock-like visible cumulative judgement counters.

Privileged judgement events terminate at :class:`VisibleCounterRenderer`.
Consumers receive only ``JudgementCounterFrame`` instances, exactly as a
screen sensor would.  The module deliberately does not expose target/floor
identity, target time, or scorer cursor state.
"""

from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from math import isfinite
from typing import Mapping, Sequence

from dmdod.features.judgement_observer_v01 import JudgementCounterFrame


def _finite_timestamp(value: object, *, name: str) -> float:
    """Reject non-numeric, boolean and overflowing clock values."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class CounterVisibilityEvidence(str, Enum):
    USER_CONFIRMED_VISIBLE = "user_confirmed_visible"
    DLL_INTERNAL_COUNT_ONLY = "dll_internal_count_only"
    TEST_FIXTURE_ONLY = "test_fixture_only"


@dataclass(frozen=True, slots=True)
class VisibleCounterContract:
    labels: tuple[str, ...]
    category_to_label: tuple[tuple[str, str], ...]
    evidence: tuple[tuple[str, CounterVisibilityEvidence], ...]

    @classmethod
    def build(
        cls,
        *,
        labels: Sequence[str],
        category_to_label: Mapping[str, str],
        evidence: Mapping[str, CounterVisibilityEvidence],
    ) -> "VisibleCounterContract":
        normalized = tuple(str(label) for label in labels)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("counter labels must be non-empty and unique")
        mapping = tuple((str(category), str(label)) for category, label in category_to_label.items())
        if any(label not in normalized for _, label in mapping):
            raise ValueError("every mapped category must reference a visible label")
        if set(evidence) != {category for category, _ in mapping}:
            raise ValueError("visibility evidence must cover exactly the mapped categories")
        return cls(
            labels=normalized,
            category_to_label=mapping,
            evidence=tuple((str(category), evidence[str(category)]) for category, _ in mapping),
        )

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self.category_to_label)


@dataclass(frozen=True, slots=True)
class StockJudgementNotification:
    occurred_at_s: float
    category: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class CounterRenderDiagnostic:
    accepted: bool
    reason: str
    category: str
    occurred_at_s: float


class VisibleCounterRenderer:
    """Convert internal events to a delayed cumulative *visible* surface.

    This is a renderer boundary, not a Student API.  Unmapped or insufficiently
    evidenced categories remain invisible.  Captures can be omitted to model a
    missing frame; later captures contain only cumulative totals and therefore
    do not invent within-interval ordering.
    """

    def __init__(
        self,
        contract: VisibleCounterContract,
        *,
        display_latency_s: float = 0.0,
        capture_latency_s: float = 0.0,
    ) -> None:
        display_latency_s = _finite_timestamp(display_latency_s, name="display latency")
        capture_latency_s = _finite_timestamp(capture_latency_s, name="capture latency")
        if display_latency_s < 0.0:
            raise ValueError("display latency must be finite and non-negative")
        if capture_latency_s < 0.0:
            raise ValueError("capture latency must be finite and non-negative")
        self.contract = contract
        # Internal-only and fixture-only categories must never enter the HUD.
        self._confirmed_categories = frozenset(
            category for category, evidence in contract.evidence
            if evidence is CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE
        )
        self.display_latency_s = float(display_latency_s)
        self.capture_latency_s = float(capture_latency_s)
        self._counters = {label: 0 for label in contract.labels}
        # Min-heap by occurrence time, with insertion order for equal times.
        # This is internal to the renderer; no privileged ordering is exposed.
        self._pending: list[tuple[float, int, StockJudgementNotification]] = []
        self._pending_sequence = 0
        self._last_capture_s: float | None = None

    def notify(self, event: StockJudgementNotification) -> CounterRenderDiagnostic:
        occurred_at_s = _finite_timestamp(event.occurred_at_s, name="judgement occurrence time")
        if isinstance(event.count, bool) or not isinstance(event.count, int) or event.count <= 0:
            raise ValueError("judgement count must be a positive integer")
        # Only literal category strings may enter the pending heap. Coercing an
        # object with __str__ can pass the visibility check but later fail
        # when capture indexes the original, non-string event.category.
        if type(event.category) is not str:
            raise ValueError("judgement category must be a string")
        label = self.contract.mapping.get(event.category)
        if label is None:
            return CounterRenderDiagnostic(False, "category is not in the visible contract", str(event.category), float(event.occurred_at_s))
        if str(event.category) not in self._confirmed_categories:
            return CounterRenderDiagnostic(
                False,
                "category has no user-confirmed visibility evidence",
                str(event.category),
                float(event.occurred_at_s),
            )
        self._pending_sequence += 1
        heappush(self._pending, (occurred_at_s, self._pending_sequence, event))
        return CounterRenderDiagnostic(True, "queued for visible rendering", str(event.category), float(event.occurred_at_s))

    def reset_visible_counters(self) -> None:
        self._counters = {label: 0 for label in self.contract.labels}
        self._pending.clear()
        self._pending_sequence = 0
        # A new episode can restart its display clock at zero.
        # Captures within an episode still reject backward timestamps.
        self._last_capture_s = None

    def capture(self, displayed_at_s: float) -> JudgementCounterFrame:
        displayed = _finite_timestamp(displayed_at_s, name="display timestamp")
        if self._last_capture_s is not None and displayed < self._last_capture_s:
            raise ValueError("capture timestamp moved backward")
        # Finite inputs can still overflow when adding capture latency.
        # Validate before consuming pending events or advancing the clock.
        observed = displayed + self.capture_latency_s
        if not isfinite(observed):
            raise ValueError("observation timestamp must be finite")
        cutoff = displayed - self.display_latency_s
        mapping = self.contract.mapping
        # Leave future events in the heap without scanning them each frame.
        # Keep the original inclusive 1e-12 boundary unchanged.
        while self._pending and self._pending[0][0] <= cutoff + 1e-12:
            event = heappop(self._pending)[2]
            self._counters[mapping[event.category]] += event.count
        self._last_capture_s = displayed
        return JudgementCounterFrame.from_mapping(
            observed_at_s=observed,
            displayed_at_s=displayed,
            counters=self._counters,
        )

