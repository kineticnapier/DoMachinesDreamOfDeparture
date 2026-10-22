from __future__ import annotations

"""Research renderer for Stock-like visible cumulative judgement counters.

Privileged judgement events terminate at :class:`VisibleCounterRenderer`.
Consumers receive only ``JudgementCounterFrame`` instances, exactly as a
screen sensor would.  The module deliberately does not expose target/floor
identity, target time, or scorer cursor state.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence

from dmdod.features.judgement_observer_v01 import JudgementCounterFrame


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
        if not isfinite(display_latency_s) or display_latency_s < 0.0:
            raise ValueError("display latency must be finite and non-negative")
        if not isfinite(capture_latency_s) or capture_latency_s < 0.0:
            raise ValueError("capture latency must be finite and non-negative")
        self.contract = contract
        self.display_latency_s = float(display_latency_s)
        self.capture_latency_s = float(capture_latency_s)
        self._counters = {label: 0 for label in contract.labels}
        self._pending: list[StockJudgementNotification] = []
        self._last_capture_s: float | None = None

    def notify(self, event: StockJudgementNotification) -> CounterRenderDiagnostic:
        if not isfinite(event.occurred_at_s):
            raise ValueError("judgement occurrence time must be finite")
        if isinstance(event.count, bool) or not isinstance(event.count, int) or event.count <= 0:
            raise ValueError("judgement count must be a positive integer")
        label = self.contract.mapping.get(str(event.category))
        if label is None:
            return CounterRenderDiagnostic(False, "category is not in the visible contract", str(event.category), float(event.occurred_at_s))
        self._pending.append(event)
        self._pending.sort(key=lambda item: item.occurred_at_s)
        return CounterRenderDiagnostic(True, "queued for visible rendering", str(event.category), float(event.occurred_at_s))

    def reset_visible_counters(self) -> None:
        self._counters = {label: 0 for label in self.contract.labels}
        self._pending.clear()

    def capture(self, displayed_at_s: float) -> JudgementCounterFrame:
        displayed = float(displayed_at_s)
        if not isfinite(displayed):
            raise ValueError("display timestamp must be finite")
        if self._last_capture_s is not None and displayed < self._last_capture_s:
            raise ValueError("capture timestamp moved backward")
        cutoff = displayed - self.display_latency_s
        remaining: list[StockJudgementNotification] = []
        mapping = self.contract.mapping
        for event in self._pending:
            if event.occurred_at_s <= cutoff + 1e-12:
                self._counters[mapping[event.category]] += event.count
            else:
                remaining.append(event)
        self._pending = remaining
        self._last_capture_s = displayed
        return JudgementCounterFrame.from_mapping(
            observed_at_s=displayed + self.capture_latency_s,
            displayed_at_s=displayed,
            counters=self._counters,
        )

