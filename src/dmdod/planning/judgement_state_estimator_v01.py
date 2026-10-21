from __future__ import annotations

"""Set-valued, HUD-only judgement attribution research model.

This module is intentionally disconnected from production control.  It joins
causal Timing local IDs, the agent's own KeyDown history, and Phase 17 visible
counter deltas.  It never reads scorer cursors or target truth.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import permutations, product
from math import isfinite

from dmdod.features.judgement_observer_v01 import (
    JudgementDelta,
    JudgementDeltaStatus,
)


class TimingEventStatus(str, Enum):
    CONFIRMED = "confirmed"
    ABSTAINED = "abstained"
    CANCELED = "canceled"
    ELAPSED = "elapsed"


class EstimateStatus(str, Enum):
    NO_UPDATE = "no_update"
    MODEL_IDENTIFIED = "model_identified"
    AMBIGUOUS = "ambiguous"
    OBSERVATION_GAP = "observation_gap"


@dataclass(frozen=True, slots=True)
class TimingEventEvidence:
    local_id: str
    observed_at_s: float
    due_lower_s: float
    due_upper_s: float
    status: TimingEventStatus
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.local_id:
            raise ValueError("local_id must be non-empty")
        values = (self.observed_at_s, self.due_lower_s, self.due_upper_s)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("timing evidence timestamps must be finite")
        if self.due_lower_s > self.due_upper_s:
            raise ValueError("due interval is inverted")


@dataclass(frozen=True, slots=True)
class MotorKeyDownEvidence:
    input_id: str
    intended_local_id: str
    occurred_at_s: float
    observed_at_s: float

    def __post_init__(self) -> None:
        if not self.input_id or not self.intended_local_id:
            raise ValueError("input and intended local IDs must be non-empty")
        if not isfinite(self.occurred_at_s) or not isfinite(self.observed_at_s):
            raise ValueError("KeyDown timestamps must be finite")
        if self.observed_at_s + 1e-12 < self.occurred_at_s:
            raise ValueError("KeyDown cannot be observed before it occurs")


@dataclass(frozen=True, slots=True)
class AttributionHypothesis:
    """One model-compatible assignment, with no probability attached."""

    # input_id, intended local ID, possibly judged local ID, observed label
    attributions: tuple[tuple[str, str, str, str], ...]
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    within_interval_order_known: bool = False


@dataclass(frozen=True, slots=True)
class JudgementStateEstimate:
    status: EstimateStatus
    observed_at_s: float
    hypotheses: tuple[AttributionHypothesis, ...]
    possible_unresolved_local_ids: tuple[str, ...]
    aligned_state_possible: bool
    identifiable: bool
    stock_identity_verified: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RepositoryWindowModel:
    """Explicit simulator-specific symmetric judgement-window hypothesis.

    The caller must justify these values from causal/visible inputs.  Merely
    constructing this model does not establish parity with installed Stock.
    """

    perfect_s: float
    early_late_perfect_s: float
    pass_s: float

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.perfect_s
            <= self.early_late_perfect_s
            <= self.pass_s
        ):
            raise ValueError("judgement windows must be non-negative and nested")

    def possible_labels(
        self, event: TimingEventEvidence, keydown_s: float
    ) -> frozenset[str]:
        error_low = keydown_s - event.due_upper_s
        error_high = keydown_s - event.due_lower_s
        bands = (
            (float("-inf"), -self.pass_s, "too_early"),
            (-self.pass_s, -self.early_late_perfect_s, "very_early"),
            (-self.early_late_perfect_s, -self.perfect_s, "early_perfect"),
            (-self.perfect_s, self.perfect_s, "perfect"),
            (self.perfect_s, self.early_late_perfect_s, "late_perfect"),
            (self.early_late_perfect_s, self.pass_s, "very_late"),
            (self.pass_s, float("inf"), "too_late"),
        )
        return frozenset(
            label for low, high, label in bands
            if error_high >= low - 1e-12 and error_low <= high + 1e-12
        )


class JudgementStateEstimator:
    """Maintain all attributions compatible with an explicit window model."""

    def __init__(
        self,
        window_model: RepositoryWindowModel,
        *,
        max_predecessors: int = 3,
        max_hypotheses: int = 128,
    ) -> None:
        if max_predecessors < 0 or max_hypotheses <= 0:
            raise ValueError("hypothesis bounds must be positive")
        self.window_model = window_model
        self.max_predecessors = int(max_predecessors)
        self.max_hypotheses = int(max_hypotheses)
        self._events: list[TimingEventEvidence] = []
        self._event_positions: dict[str, int] = {}
        self._pending_keydowns: list[MotorKeyDownEvidence] = []

    def reset(self) -> None:
        self._events.clear()
        self._event_positions.clear()
        self._pending_keydowns.clear()

    def observe_timing(self, evidence: TimingEventEvidence) -> None:
        position = self._event_positions.get(evidence.local_id)
        if position is None:
            self._event_positions[evidence.local_id] = len(self._events)
            self._events.append(evidence)
        else:
            self._events[position] = evidence

    def observe_keydown(self, evidence: MotorKeyDownEvidence) -> None:
        if evidence.intended_local_id not in self._event_positions:
            raise ValueError("KeyDown references an unseen Timing local ID")
        if any(row.input_id == evidence.input_id for row in self._pending_keydowns):
            raise ValueError("input_id must be unique")
        self._pending_keydowns.append(evidence)

    def _candidate_events(
        self, keydown: MotorKeyDownEvidence, label: str
    ) -> tuple[TimingEventEvidence, ...]:
        intended_position = self._event_positions[keydown.intended_local_id]
        start = max(0, intended_position - self.max_predecessors)
        candidates = []
        for event in self._events[start:intended_position + 1]:
            if event.status is TimingEventStatus.CANCELED:
                continue
            if label in self.window_model.possible_labels(event, keydown.occurred_at_s):
                candidates.append(event)
        return tuple(candidates)

    @staticmethod
    def _expanded_labels(feedback: JudgementDelta) -> tuple[str, ...]:
        labels: list[str] = []
        for label, count in feedback.deltas:
            if count < 0:
                raise ValueError("visible judgement delta cannot be negative")
            labels.extend([label] * count)
        return tuple(labels)

    def _no_update(self, feedback: JudgementDelta, reason: str) -> JudgementStateEstimate:
        return JudgementStateEstimate(
            status=EstimateStatus.NO_UPDATE,
            observed_at_s=feedback.observed_at_s,
            hypotheses=(),
            possible_unresolved_local_ids=(),
            aligned_state_possible=True,
            identifiable=False,
            stock_identity_verified=False,
            reason=reason,
        )

    def observe_judgement(self, feedback: JudgementDelta) -> JudgementStateEstimate:
        if feedback.status in {JudgementDeltaStatus.INVALID, JudgementDeltaStatus.RESET}:
            self._pending_keydowns.clear()
            return JudgementStateEstimate(
                status=EstimateStatus.OBSERVATION_GAP,
                observed_at_s=feedback.observed_at_s,
                hypotheses=(),
                possible_unresolved_local_ids=(),
                aligned_state_possible=True,
                identifiable=False,
                stock_identity_verified=False,
                reason="visible counter discontinuity prevents causal attribution",
            )
        if feedback.status in {JudgementDeltaStatus.BASELINE, JudgementDeltaStatus.UNCHANGED}:
            return self._no_update(feedback, "no visible judgement increment")
        if feedback.status is not JudgementDeltaStatus.DELTA or feedback.event_interval_s is None:
            return self._no_update(feedback, "feedback has no attributable display interval")

        labels = self._expanded_labels(feedback)
        lower, upper = feedback.event_interval_s
        keydowns = [
            row for row in self._pending_keydowns
            if lower - 1e-12 <= row.occurred_at_s <= upper + 1e-12
        ]
        if len(keydowns) != len(labels) or not labels:
            self._pending_keydowns = [
                row for row in self._pending_keydowns if row.occurred_at_s > upper + 1e-12
            ]
            return JudgementStateEstimate(
                status=EstimateStatus.AMBIGUOUS,
                observed_at_s=feedback.observed_at_s,
                hypotheses=(),
                possible_unresolved_local_ids=tuple(
                    sorted({row.intended_local_id for row in keydowns})
                ),
                aligned_state_possible=True,
                identifiable=False,
                stock_identity_verified=False,
                reason="visible increment count does not uniquely pair with observed KeyDowns",
            )

        hypothesis_keys: set[tuple[tuple[str, str, str, str], ...]] = set()
        hypotheses: list[AttributionHypothesis] = []
        for label_order in set(permutations(labels)):
            choices = [
                self._candidate_events(keydown, label)
                for keydown, label in zip(keydowns, label_order)
            ]
            if any(not candidates for candidates in choices):
                continue
            for selected in product(*choices):
                selected_ids = tuple(event.local_id for event in selected)
                if len(set(selected_ids)) != len(selected_ids):
                    continue
                attributions = tuple(
                    (keydown.input_id, keydown.intended_local_id, event.local_id, label)
                    for keydown, event, label in zip(keydowns, selected, label_order)
                )
                if attributions in hypothesis_keys:
                    continue
                hypothesis_keys.add(attributions)
                hypotheses.append(AttributionHypothesis(
                    attributions=attributions,
                    supporting_evidence=tuple(
                        f"{label} is compatible with KeyDown {keydown.occurred_at_s:.9f}s "
                        f"and Timing {event.local_id} due interval"
                        for keydown, event, label in zip(keydowns, selected, label_order)
                    ),
                    contradicting_evidence=tuple(
                        f"intended Timing {keydown.intended_local_id} category is incompatible"
                        for keydown, event in zip(keydowns, selected)
                        if event.local_id != keydown.intended_local_id
                    ),
                    within_interval_order_known=feedback.within_interval_order_known,
                ))
                if len(hypotheses) >= self.max_hypotheses:
                    break
            if len(hypotheses) >= self.max_hypotheses:
                break

        used_ids = {row.input_id for row in keydowns}
        self._pending_keydowns = [
            row for row in self._pending_keydowns if row.input_id not in used_ids
        ]
        if not hypotheses:
            return JudgementStateEstimate(
                status=EstimateStatus.AMBIGUOUS,
                observed_at_s=feedback.observed_at_s,
                hypotheses=(),
                possible_unresolved_local_ids=tuple(
                    sorted({row.intended_local_id for row in keydowns})
                ),
                aligned_state_possible=True,
                identifiable=False,
                stock_identity_verified=False,
                reason="no attribution fits the explicit simulator window model",
            )

        aligned_possible = any(
            all(intended == judged for _, intended, judged, _ in hypothesis.attributions)
            for hypothesis in hypotheses
        )
        unresolved = sorted({
            intended
            for hypothesis in hypotheses
            for _, intended, judged, _ in hypothesis.attributions
            if intended != judged
        })
        identified = len(hypotheses) == 1
        return JudgementStateEstimate(
            status=(EstimateStatus.MODEL_IDENTIFIED if identified else EstimateStatus.AMBIGUOUS),
            observed_at_s=feedback.observed_at_s,
            hypotheses=tuple(hypotheses),
            possible_unresolved_local_ids=tuple(unresolved),
            aligned_state_possible=aligned_possible,
            identifiable=identified,
            stock_identity_verified=False,
            reason=(
                "one attribution remains under the explicit simulator-specific window model; "
                "Stock identity remains unverified"
                if identified else
                "multiple attributions remain under the simulator-specific window model"
            ),
        )
