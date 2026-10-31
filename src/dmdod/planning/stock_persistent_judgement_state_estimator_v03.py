from __future__ import annotations

"""Set-valued Stock-floor estimator using only causal, visible evidence.

Unlike the repository FIFO estimator, this model carries a possible Stock
floor cursor.  It never advances by hidden due truth.  If automatic FailMiss
visibility is not established, both an unchanged cursor and a possible hidden
automatic transition are retained; omitted transitions are represented by an
UNKNOWN remainder rather than guessed away.
"""

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite

from dmdod.features.judgement_observer_v01 import JudgementDelta, JudgementDeltaStatus
from dmdod.planning.judgement_label_orders import (
    MAX_ORDER_EXPANSIONS,
    MAX_VISIBLE_EVENTS,
    bounded_unique_orders,
)
from dmdod.planning.judgement_state_estimator_v01 import (
    MotorKeyDownEvidence,
    TimingEventEvidence,
    TimingEventStatus,
)


class StockEstimateStatus(str, Enum):
    NO_UPDATE = "no_update"
    IDENTIFIED_UNDER_MODEL = "identified_under_model"
    AMBIGUOUS = "ambiguous"
    OBSERVATION_GAP = "observation_gap"
    TRUNCATED = "truncated"


class StockReservationStatus(str, Enum):
    RESERVED = "reserved"
    CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class StockMotorReservationEvidence:
    reservation_id: str
    local_id: str
    observed_at_s: float
    status: StockReservationStatus

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.local_id or not isfinite(self.observed_at_s):
            raise ValueError("reservation IDs and finite timestamp are required")


@dataclass(frozen=True, slots=True)
class StockFloorAttribution:
    input_id: str | None
    intended_local_id: str | None
    judged_local_id: str | None
    label: str
    observed_at_s: float
    source: str  # keydown or visible_automatic
    advances_floor: bool


@dataclass(frozen=True, slots=True)
class StockFloorHypothesis:
    hypothesis_id: str
    cursor: int = 0
    attributions: tuple[StockFloorAttribution, ...] = ()
    used_input_ids: tuple[str, ...] = ()
    hidden_automatic_steps: int = 0
    unknown_search_remainder: bool = False
    observation_gaps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StockFloorStateEstimate:
    status: StockEstimateStatus
    observed_at_s: float
    hypotheses: tuple[StockFloorHypothesis, ...]
    possible_current_local_ids: tuple[str, ...]
    identifiable: bool
    stock_runtime_verified: bool
    reason: str


class StockPersistentJudgementStateEstimator:
    """Research-only sequential floor estimator.

    ``automatic_failmiss_visible`` is an observation-contract assertion, not a
    learned setting.  ``None`` retains both zero and one hidden automatic-step
    branches at a judgement update and adds UNKNOWN for any omitted remainder.
    """

    def __init__(
        self,
        *,
        automatic_failmiss_visible: bool | None,
        max_hypotheses: int = 128,
    ) -> None:
        if max_hypotheses < 2:
            raise ValueError("max_hypotheses must leave room for UNKNOWN")
        self.automatic_failmiss_visible = automatic_failmiss_visible
        self.max_hypotheses = int(max_hypotheses)
        self._events: list[TimingEventEvidence] = []
        self._event_positions: dict[str, int] = {}
        self._keydowns: list[MotorKeyDownEvidence] = []
        self._reservations: dict[str, StockMotorReservationEvidence] = {}
        self._hypotheses = (StockFloorHypothesis("stock-h0"),)
        self._last_time = float("-inf")
        self._serial = 0

    @property
    def hypotheses(self) -> tuple[StockFloorHypothesis, ...]:
        return self._hypotheses

    def _time(self, value: float) -> None:
        if not isfinite(value) or value + 1e-12 < self._last_time:
            raise ValueError("observations must use finite, monotonic timestamps")
        self._last_time = max(self._last_time, value)

    def _id(self, parent: str, suffix: str) -> str:
        self._serial += 1
        return f"{parent}.{self._serial}:{suffix}"

    def observe_timing(self, evidence: TimingEventEvidence) -> None:
        self._time(evidence.observed_at_s)
        position = self._event_positions.get(evidence.local_id)
        if position is None:
            self._event_positions[evidence.local_id] = len(self._events)
            self._events.append(evidence)
        else:
            self._events[position] = evidence

    def observe_keydown(self, evidence: MotorKeyDownEvidence) -> None:
        self._time(evidence.observed_at_s)
        if evidence.intended_local_id not in self._event_positions:
            raise ValueError("KeyDown references unseen Timing local ID")
        if any(row.input_id == evidence.input_id for row in self._keydowns):
            raise ValueError("input_id must be unique")
        self._keydowns.append(evidence)
        self._keydowns.sort(key=lambda row: (row.occurred_at_s, row.input_id))

    def observe_reservation(self, evidence: StockMotorReservationEvidence) -> None:
        self._time(evidence.observed_at_s)
        if evidence.local_id not in self._event_positions:
            raise ValueError("reservation references unseen Timing local ID")
        prior = self._reservations.get(evidence.reservation_id)
        if prior is not None and prior.local_id != evidence.local_id:
            raise ValueError("reservation_id cannot be rebound to another local ID")
        self._reservations[evidence.reservation_id] = evidence

    def advance_time(self, observed_at_s: float) -> StockFloorStateEstimate:
        self._time(observed_at_s)
        return self._estimate(StockEstimateStatus.NO_UPDATE, "time alone does not reveal a Stock floor transition")

    @staticmethod
    def _advances(label: str) -> bool:
        return label not in {"too_early", "too_late", "fail_overload", "over_press"}

    @staticmethod
    def _labels(delta: JudgementDelta) -> tuple[str, ...]:
        labels: list[str] = []
        for label, count in delta.deltas:
            if count < 0:
                raise ValueError("visible deltas cannot be negative")
            labels.extend([label] * count)
        return tuple(labels)

    def _current_id(self, hypothesis: StockFloorHypothesis) -> str | None:
        cursor = hypothesis.cursor
        while cursor < len(self._events) and self._events[cursor].status is TimingEventStatus.CANCELED:
            cursor += 1
        return self._events[cursor].local_id if cursor < len(self._events) else None

    def _apply_label(
        self,
        hypothesis: StockFloorHypothesis,
        label: str,
        observed_at_s: float,
        interval_upper_s: float,
    ) -> StockFloorHypothesis | None:
        cursor = hypothesis.cursor
        while cursor < len(self._events) and self._events[cursor].status is TimingEventStatus.CANCELED:
            cursor += 1
        judged = self._events[cursor].local_id if cursor < len(self._events) else None
        if label == "fail_miss":
            if self.automatic_failmiss_visible is not True:
                # A fixture may present the label, but a contract that has not
                # established visibility cannot use it as certain evidence.
                return replace(
                    hypothesis,
                    hypothesis_id=self._id(hypothesis.hypothesis_id, "untrusted_failmiss"),
                    unknown_search_remainder=True,
                    observation_gaps=hypothesis.observation_gaps + ("FailMiss visibility is not established",),
                )
            return replace(
                hypothesis,
                hypothesis_id=self._id(hypothesis.hypothesis_id, "visible_auto"),
                cursor=cursor + (1 if judged is not None else 0),
                attributions=hypothesis.attributions + (StockFloorAttribution(None, None, judged, label, observed_at_s, "visible_automatic", judged is not None),),
            )

        pending = [
            row for row in self._keydowns
            if row.input_id not in hypothesis.used_input_ids
            and row.occurred_at_s <= interval_upper_s + 1e-12
        ]
        if not pending:
            return None
        keydown = pending[0]
        advances = self._advances(label) and judged is not None
        return replace(
            hypothesis,
            hypothesis_id=self._id(hypothesis.hypothesis_id, "keydown"),
            cursor=cursor + (1 if advances else 0),
            used_input_ids=hypothesis.used_input_ids + (keydown.input_id,),
            attributions=hypothesis.attributions + (StockFloorAttribution(keydown.input_id, keydown.intended_local_id, judged, label, observed_at_s, "keydown", advances),),
        )

    def _hidden_branches(self, hypothesis: StockFloorHypothesis) -> tuple[StockFloorHypothesis, ...]:
        if self.automatic_failmiss_visible is True:
            return (hypothesis,)
        branches = [hypothesis]
        if hypothesis.cursor < len(self._events):
            branches.append(replace(
                hypothesis,
                hypothesis_id=self._id(hypothesis.hypothesis_id, "hidden_auto_possible"),
                cursor=hypothesis.cursor + 1,
                hidden_automatic_steps=hypothesis.hidden_automatic_steps + 1,
            ))
            branches.append(replace(
                hypothesis,
                hypothesis_id=self._id(hypothesis.hypothesis_id, "UNKNOWN_hidden_auto_remainder"),
                unknown_search_remainder=True,
                observation_gaps=hypothesis.observation_gaps + ("zero, one, or more invisible automatic floor transitions remain observationally possible",),
            ))
        return tuple(branches)

    def _set(self, candidates: list[StockFloorHypothesis]) -> bool:
        unique: dict[tuple[object, ...], StockFloorHypothesis] = {}
        for row in candidates:
            key = (row.cursor, row.attributions, row.used_input_ids, row.hidden_automatic_steps, row.unknown_search_remainder, row.observation_gaps)
            unique.setdefault(key, row)
        rows = list(unique.values())
        truncated = len(rows) > self.max_hypotheses
        if truncated:
            rows = rows[: self.max_hypotheses - 1]
            parent = rows[0]
            rows.append(replace(parent, hypothesis_id=self._id(parent.hypothesis_id, "UNKNOWN_cap"), unknown_search_remainder=True, observation_gaps=parent.observation_gaps + ("hypothesis cap reached",)))
        self._hypotheses = tuple(rows)
        return truncated

    def observe_judgement(self, delta: JudgementDelta) -> StockFloorStateEstimate:
        self._time(delta.observed_at_s)
        if delta.status in {JudgementDeltaStatus.BASELINE, JudgementDeltaStatus.UNCHANGED}:
            return self._estimate(StockEstimateStatus.NO_UPDATE, "no visible judgement increment")
        if delta.status in {JudgementDeltaStatus.INVALID, JudgementDeltaStatus.RESET} or delta.event_interval_s is None:
            self._hypotheses = tuple(replace(row, unknown_search_remainder=True, observation_gaps=row.observation_gaps + (f"counter {delta.status.value}",)) for row in self._hypotheses)
            return self._estimate(StockEstimateStatus.OBSERVATION_GAP, "counter discontinuity prevents unique state recovery")

        # Validate the counter before expanding a potentially enormous delta.
        if any(count < 0 for _, count in delta.deltas):
            raise ValueError("visible deltas cannot be negative")
        if delta.total_increment > MAX_VISIBLE_EVENTS:
            self._hypotheses = tuple(replace(
                row,
                unknown_search_remainder=True,
                observation_gaps=row.observation_gaps + ("visible delta too large",),
            ) for row in self._hypotheses)
            return self._estimate(StockEstimateStatus.TRUNCATED, "visible delta too large")
        labels = self._labels(delta)
        if not labels:
            return self._estimate(StockEstimateStatus.NO_UPDATE, "empty visible delta")

        # Bound all label orders *before* constructing any permutation. This
        # conservative budget includes possible hidden-automatic branches.
        branch_factor = 1 if self.automatic_failmiss_visible is True else 3
        budget = max(1, MAX_ORDER_EXPANSIONS // max(
            1, len(self._hypotheses) * branch_factor
        ))
        orders = bounded_unique_orders(labels, limit=budget)
        if orders is None:
            self._hypotheses = tuple(replace(
                row,
                unknown_search_remainder=True,
                observation_gaps=row.observation_gaps + ("visible order budget exceeded",),
            ) for row in self._hypotheses)
            return self._estimate(StockEstimateStatus.TRUNCATED, "visible order budget exceeded")

        candidates: list[StockFloorHypothesis] = []
        for hypothesis in self._hypotheses:
            for branch in self._hidden_branches(hypothesis):
                for order in orders:
                    current: StockFloorHypothesis | None = branch
                    for label in order:
                        if current is None:
                            break
                        current = self._apply_label(current, label, delta.observed_at_s, delta.event_interval_s[1])
                    if current is not None:
                        candidates.append(current)
        if not candidates:
            self._hypotheses = tuple(replace(row, unknown_search_remainder=True, observation_gaps=row.observation_gaps + ("visible increments cannot be paired with causal inputs",)) for row in self._hypotheses)
            return self._estimate(StockEstimateStatus.OBSERVATION_GAP, "no complete causal attribution")
        truncated = self._set(candidates)
        return self._estimate(StockEstimateStatus.TRUNCATED if truncated else StockEstimateStatus.AMBIGUOUS, "visible feedback updated sequential Stock-floor hypotheses")

    def _estimate(self, status: StockEstimateStatus, reason: str) -> StockFloorStateEstimate:
        possible = tuple(dict.fromkeys(filter(None, (self._current_id(row) for row in self._hypotheses))))
        identifiable = len(possible) == 1 and bool(self._hypotheses) and all(not row.unknown_search_remainder for row in self._hypotheses)
        if identifiable and status is StockEstimateStatus.AMBIGUOUS:
            status = StockEstimateStatus.IDENTIFIED_UNDER_MODEL
        return StockFloorStateEstimate(status, self._last_time, self._hypotheses, possible, identifiable, False, reason)

