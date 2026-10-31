from __future__ import annotations

"""Persistent, set-valued judgement-state research estimator.

The estimator is deliberately disconnected from production control.  It only
accepts Timing-local evidence, the motor's own reservations/KeyDowns, elapsed
time, and visible counter deltas.  Search limits are represented explicitly by
an UNKNOWN hypothesis; they can therefore never manufacture identification.
"""

from dataclasses import dataclass, replace
from enum import Enum
from itertools import permutations, product
from math import isfinite
from typing import Iterable

from dmdod.features.judgement_observer_v01 import (
    JudgementDelta,
    JudgementDeltaStatus,
)
from dmdod.planning.judgement_label_orders import (
    MAX_ORDER_EXPANSIONS,
    MAX_VISIBLE_EVENTS,
    bounded_unique_orders,
    tied_input_orders_exceed_budget,
)
from dmdod.planning.judgement_state_estimator_v01 import (
    MotorKeyDownEvidence,
    RepositoryWindowModel,
    TimingEventEvidence,
    TimingEventStatus,
)


class AllocationRule(str, Enum):
    """Explicit model assumption; neither value claims universal Stock parity."""

    REPOSITORY_FIFO = "repository_fifo"
    COMPATIBLE_ANY = "compatible_any"


class PersistentEstimateStatus(str, Enum):
    NO_UPDATE = "no_update"
    MODEL_IDENTIFIED = "model_identified"
    AMBIGUOUS = "ambiguous"
    TRUNCATED = "truncated"
    OBSERVATION_GAP = "observation_gap"
    CONTRADICTED = "contradicted"


class ReservationStatus(str, Enum):
    RESERVED = "reserved"
    CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class MotorReservationEvidence:
    reservation_id: str
    local_id: str
    observed_at_s: float
    status: ReservationStatus
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.local_id:
            raise ValueError("reservation and local IDs must be non-empty")
        if not isfinite(self.observed_at_s):
            raise ValueError("reservation timestamp must be finite")


@dataclass(frozen=True, slots=True)
class PersistentAttribution:
    input_id: str
    intended_local_id: str
    judged_local_id: str | None
    label: str
    visible_interval_s: tuple[float, float]
    resolves_target: bool
    ambiguous_order: bool


@dataclass(frozen=True, slots=True)
class TargetResolution:
    local_id: str
    kind: str  # keydown, automatic_miss, cancellation
    evidence_at_s: float
    input_id: str | None = None


@dataclass(frozen=True, slots=True)
class PersistentStateHypothesis:
    hypothesis_id: str
    attributions: tuple[PersistentAttribution, ...] = ()
    resolutions: tuple[TargetResolution, ...] = ()
    pending_input_ids: tuple[str, ...] = ()
    possible_unresolved_local_ids: tuple[str, ...] = ()
    observation_consistent: bool = True
    automatic_miss_possible: tuple[str, ...] = ()
    observation_gaps: tuple[str, ...] = ()
    model_assumptions: tuple[str, ...] = ()
    unknown_search_remainder: bool = False

    @property
    def resolved_local_ids(self) -> frozenset[str]:
        return frozenset(row.local_id for row in self.resolutions)


@dataclass(frozen=True, slots=True)
class PersistentJudgementStateEstimate:
    status: PersistentEstimateStatus
    observed_at_s: float
    hypotheses: tuple[PersistentStateHypothesis, ...]
    search_complete: bool
    identifiable: bool
    stock_identity_verified: bool
    reason: str


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class PersistentJudgementStateEstimator:
    """Transition prior hypotheses instead of reconstructing each interval.

    ``max_predecessors`` and ``max_hypotheses`` are computational limits only.
    Any potentially relevant omitted state is summarized by a persistent
    ``unknown_search_remainder`` hypothesis.  Such a set is never identifiable.
    """

    def __init__(
        self,
        window_model: RepositoryWindowModel,
        *,
        allocation_rule: AllocationRule = AllocationRule.REPOSITORY_FIFO,
        max_predecessors: int | None = 3,
        max_hypotheses: int = 128,
        automatic_miss_visible: bool | None = None,
    ) -> None:
        if max_predecessors is not None and max_predecessors < 0:
            raise ValueError("max_predecessors must be non-negative or None")
        if max_hypotheses <= 0:
            raise ValueError("max_hypotheses must be positive")
        self.window_model = window_model
        self.allocation_rule = AllocationRule(allocation_rule)
        self.max_predecessors = max_predecessors
        self.max_hypotheses = int(max_hypotheses)
        self.automatic_miss_visible = automatic_miss_visible
        self._events: list[TimingEventEvidence] = []
        self._event_positions: dict[str, int] = {}
        self._keydowns: dict[str, MotorKeyDownEvidence] = {}
        self._reservations: dict[str, MotorReservationEvidence] = {}
        self._hypotheses: tuple[PersistentStateHypothesis, ...] = (
            PersistentStateHypothesis(
                hypothesis_id="h0",
                model_assumptions=(
                    f"allocation_rule={self.allocation_rule.value}",
                    "repository symmetric due-time window",
                    f"automatic_miss_visible={self.automatic_miss_visible}",
                ),
            ),
        )
        self._last_observed_at_s = float("-inf")
        self._sequence = 0

    @property
    def hypotheses(self) -> tuple[PersistentStateHypothesis, ...]:
        return self._hypotheses

    def _next_id(self, parent: str, suffix: str) -> str:
        self._sequence += 1
        return f"{parent}.{self._sequence}:{suffix}"

    def _validate_time(self, observed_at_s: float) -> None:
        if not isfinite(observed_at_s):
            raise ValueError("observation timestamp must be finite")
        if observed_at_s + 1e-12 < self._last_observed_at_s:
            raise ValueError("observations must not move backward")
        self._last_observed_at_s = max(self._last_observed_at_s, observed_at_s)

    def _event_ids(self) -> tuple[str, ...]:
        return tuple(
            row.local_id for row in self._events
            if row.status is not TimingEventStatus.CANCELED
        )

    def _refresh(self, hypothesis: PersistentStateHypothesis) -> PersistentStateHypothesis:
        resolved = hypothesis.resolved_local_ids
        return replace(
            hypothesis,
            pending_input_ids=tuple(
                input_id for input_id in self._keydowns
                if all(a.input_id != input_id for a in hypothesis.attributions)
            ),
            possible_unresolved_local_ids=tuple(
                event_id for event_id in self._event_ids() if event_id not in resolved
            ),
        )

    def _set_hypotheses(
        self,
        candidates: Iterable[PersistentStateHypothesis],
        *,
        truncation_reason: str | None = None,
    ) -> bool:
        unique: dict[tuple, PersistentStateHypothesis] = {}
        for row in candidates:
            row = self._refresh(row)
            key = (
                row.attributions,
                row.resolutions,
                row.observation_consistent,
                row.automatic_miss_possible,
                row.observation_gaps,
                row.model_assumptions,
                row.unknown_search_remainder,
            )
            unique.setdefault(key, row)
        rows = list(unique.values())
        truncated = truncation_reason is not None or len(rows) > self.max_hypotheses
        if len(rows) > self.max_hypotheses:
            truncation_reason = (
                f"hypothesis cap {self.max_hypotheses} omitted compatible states"
            )
        if truncated:
            keep = max(0, self.max_hypotheses - 1)
            rows = rows[:keep]
            parent = rows[0] if rows else self._hypotheses[0]
            gaps = parent.observation_gaps + (truncation_reason or "search truncated",)
            rows.append(self._refresh(replace(
                parent,
                hypothesis_id=self._next_id(parent.hypothesis_id, "UNKNOWN"),
                observation_consistent=True,
                observation_gaps=_dedupe(gaps),
                unknown_search_remainder=True,
            )))
        self._hypotheses = tuple(rows)
        return truncated

    def observe_timing(self, evidence: TimingEventEvidence) -> None:
        self._validate_time(evidence.observed_at_s)
        position = self._event_positions.get(evidence.local_id)
        if position is None:
            self._event_positions[evidence.local_id] = len(self._events)
            self._events.append(evidence)
        else:
            self._events[position] = evidence
        if evidence.status is TimingEventStatus.CANCELED:
            updated = []
            for hypothesis in self._hypotheses:
                if evidence.local_id in hypothesis.resolved_local_ids:
                    updated.append(replace(
                        hypothesis,
                        observation_consistent=False,
                        observation_gaps=hypothesis.observation_gaps + (
                            f"Timing {evidence.local_id} canceled after a possible resolution",
                        ),
                    ))
                else:
                    updated.append(replace(
                        hypothesis,
                        resolutions=hypothesis.resolutions + (TargetResolution(
                            local_id=evidence.local_id,
                            kind="cancellation",
                            evidence_at_s=evidence.observed_at_s,
                        ),),
                    ))
            self._set_hypotheses(updated)
        else:
            self._set_hypotheses(self._hypotheses)

    def observe_reservation(self, evidence: MotorReservationEvidence) -> None:
        self._validate_time(evidence.observed_at_s)
        if evidence.local_id not in self._event_positions:
            raise ValueError("reservation references unseen Timing local ID")
        prior = self._reservations.get(evidence.reservation_id)
        if prior is not None and prior.local_id != evidence.local_id:
            raise ValueError("reservation_id cannot change local ID")
        self._reservations[evidence.reservation_id] = evidence

    def observe_keydown(self, evidence: MotorKeyDownEvidence) -> None:
        self._validate_time(evidence.observed_at_s)
        if evidence.intended_local_id not in self._event_positions:
            raise ValueError("KeyDown references unseen Timing local ID")
        if evidence.input_id in self._keydowns:
            raise ValueError("input_id must be unique")
        self._keydowns[evidence.input_id] = evidence
        self._set_hypotheses(self._hypotheses)

    def advance_time(self, observed_at_s: float) -> PersistentJudgementStateEstimate:
        """Advance model time while retaining uncertainty near due intervals.

        Automatic expiry is deterministic only after the latest possible due
        plus ``pass_s``.  Between lower and upper deadlines both branches are
        retained.  UNKNOWN states remain UNKNOWN rather than being repaired.
        """

        self._validate_time(observed_at_s)
        candidates: list[PersistentStateHypothesis] = []
        for hypothesis in self._hypotheses:
            branches = [hypothesis]
            for event in self._events:
                if event.status is TimingEventStatus.CANCELED:
                    continue
                next_branches: list[PersistentStateHypothesis] = []
                for branch in branches:
                    if event.local_id in branch.resolved_local_ids:
                        next_branches.append(branch)
                        continue
                    early_deadline = event.due_lower_s + self.window_model.pass_s
                    late_deadline = event.due_upper_s + self.window_model.pass_s
                    if observed_at_s <= early_deadline + 1e-12:
                        next_branches.append(branch)
                    elif observed_at_s <= late_deadline + 1e-12:
                        next_branches.extend((
                            replace(
                                branch,
                                hypothesis_id=self._next_id(branch.hypothesis_id, "auto_pending"),
                                automatic_miss_possible=_dedupe(
                                    branch.automatic_miss_possible + (event.local_id,)
                                ),
                            ),
                            replace(
                                branch,
                                hypothesis_id=self._next_id(branch.hypothesis_id, "auto_resolved"),
                                resolutions=branch.resolutions + (TargetResolution(
                                    local_id=event.local_id,
                                    kind="automatic_miss",
                                    evidence_at_s=observed_at_s,
                                ),),
                                automatic_miss_possible=_dedupe(
                                    branch.automatic_miss_possible + (event.local_id,)
                                ),
                            ),
                        ))
                    else:
                        next_branches.append(replace(
                            branch,
                            hypothesis_id=self._next_id(branch.hypothesis_id, "auto_resolved"),
                            resolutions=branch.resolutions + (TargetResolution(
                                local_id=event.local_id,
                                kind="automatic_miss",
                                evidence_at_s=observed_at_s,
                            ),),
                            automatic_miss_possible=_dedupe(
                                branch.automatic_miss_possible + (event.local_id,)
                            ),
                        ))
                branches = next_branches
            candidates.extend(branches)
        truncated = self._set_hypotheses(candidates)
        return self._estimate(
            observed_at_s,
            PersistentEstimateStatus.TRUNCATED if truncated else PersistentEstimateStatus.NO_UPDATE,
            "time advanced under explicit automatic-miss assumptions",
        )

    def _candidate_events(
        self,
        hypothesis: PersistentStateHypothesis,
        keydown: MotorKeyDownEvidence,
        label: str,
    ) -> tuple[tuple[TimingEventEvidence, ...], bool]:
        unresolved = [
            event for event in self._events
            if event.status is not TimingEventStatus.CANCELED
            and event.local_id not in hypothesis.resolved_local_ids
        ]
        if not unresolved:
            return (), hypothesis.unknown_search_remainder
        if self.allocation_rule is AllocationRule.REPOSITORY_FIFO:
            event = unresolved[0]
            return (
                (event,) if label in self.window_model.possible_labels(
                    event, keydown.occurred_at_s
                ) else (),
                hypothesis.unknown_search_remainder,
            )

        intended_position = self._event_positions[keydown.intended_local_id]
        eligible = [
            event for event in unresolved
            if self._event_positions[event.local_id] <= intended_position
        ]
        omitted = hypothesis.unknown_search_remainder
        if self.max_predecessors is not None:
            start = max(0, intended_position - self.max_predecessors)
            if any(self._event_positions[event.local_id] < start for event in eligible):
                omitted = True
            eligible = [
                event for event in eligible
                if self._event_positions[event.local_id] >= start
            ]
        return tuple(
            event for event in eligible
            if label in self.window_model.possible_labels(event, keydown.occurred_at_s)
        ), omitted

    @staticmethod
    def _labels(feedback: JudgementDelta) -> tuple[str, ...]:
        result: list[str] = []
        for label, count in feedback.deltas:
            if count < 0:
                raise ValueError("visible judgement delta cannot be negative")
            result.extend([label] * count)
        return tuple(result)

    @staticmethod
    def _resolves(label: str) -> bool:
        return label not in {"too_early", "fail_overload", "over_press"}

    def observe_judgement(
        self, feedback: JudgementDelta
    ) -> PersistentJudgementStateEstimate:
        self._validate_time(feedback.observed_at_s)
        if feedback.status in {JudgementDeltaStatus.INVALID, JudgementDeltaStatus.RESET}:
            rows = [replace(
                hypothesis,
                hypothesis_id=self._next_id(hypothesis.hypothesis_id, "gap"),
                observation_gaps=_dedupe(hypothesis.observation_gaps + (
                    f"{feedback.status.value} at {feedback.observed_at_s:.9f}s: {feedback.reason}",
                )),
                unknown_search_remainder=True,
            ) for hypothesis in self._hypotheses]
            self._set_hypotheses(rows)
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.OBSERVATION_GAP,
                "counter discontinuity preserved prior states plus an unknown remainder",
            )
        if feedback.status in {JudgementDeltaStatus.BASELINE, JudgementDeltaStatus.UNCHANGED}:
            # UNCHANGED is only absence of a visible increment in this frame.
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.NO_UPDATE,
                "no visible increment; hidden/delayed events are not ruled out",
            )
        if feedback.status is not JudgementDeltaStatus.DELTA or feedback.event_interval_s is None:
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.OBSERVATION_GAP,
                "delta has no attributable display interval",
            )

        # A huge counter increment must not be expanded into a huge tuple.
        if any(count < 0 for _, count in feedback.deltas):
            raise ValueError("visible judgement delta cannot be negative")
        if feedback.total_increment > MAX_VISIBLE_EVENTS:
            self._set_hypotheses(
                self._hypotheses, truncation_reason="visible delta too large"
            )
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.TRUNCATED,
                "visible delta too large",
            )
        labels = self._labels(feedback)
        budget = max(1, MAX_ORDER_EXPANSIONS // max(1, len(self._hypotheses)))
        label_orders = bounded_unique_orders(labels, limit=budget)
        if label_orders is None:
            self._set_hypotheses(
                self._hypotheses, truncation_reason="visible order budget exceeded"
            )
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.TRUNCATED,
                "visible order budget exceeded",
            )
        lower, upper = feedback.event_interval_s
        compatible: list[PersistentStateHypothesis] = []
        remaining_search_budget = MAX_ORDER_EXPANSIONS
        combination_budget_exceeded = False
        omitted_possible = False
        generation_capped = False
        for hypothesis in self._hypotheses:
            if generation_capped:
                break
            pending = [
                self._keydowns[input_id]
                for input_id in hypothesis.pending_input_ids
                if self._keydowns[input_id].occurred_at_s <= upper + 1e-12
            ]
            # Prefer interval-local inputs, but retain older pending inputs to
            # model display delay. Inputs after the interval cannot explain it.
            if len(pending) < len(labels) or not labels:
                compatible.append(replace(
                    hypothesis,
                    hypothesis_id=self._next_id(hypothesis.hypothesis_id, "unpaired"),
                    observation_gaps=_dedupe(hypothesis.observation_gaps + (
                        "visible increment count cannot be paired with observed KeyDowns; "
                        "automatic/missing/delayed judgement remains possible",
                    )),
                    unknown_search_remainder=True,
                ))
                omitted_possible = True
                continue
            # Motor edge timestamps retain input order even when the visible
            # counter loses judgement-label order within a 10 ms interval.
            # Exact timestamp ties are conservatively permuted.
            ordered_pending = sorted(
                pending, key=lambda row: (row.occurred_at_s, row.input_id)
            )
            selected_pending = ordered_pending[:len(labels)]
            if (
                len(selected_pending) > 1
                and len({row.occurred_at_s for row in selected_pending}) == 1
            ):
                if tied_input_orders_exceed_budget(
                    len(selected_pending), len(label_orders),
                    limit=max(1, MAX_ORDER_EXPANSIONS // max(1, len(self._hypotheses))),
                ):
                    compatible.append(replace(
                        hypothesis,
                        unknown_search_remainder=True,
                        observation_gaps=_dedupe(hypothesis.observation_gaps + (
                            "tied KeyDown order budget exceeded",
                        )),
                    ))
                    omitted_possible = True
                    continue
                input_orders = permutations(selected_pending)
            else:
                input_orders = (tuple(selected_pending),)
            for keydown_order in input_orders:
                if generation_capped:
                    break
                if len({row.input_id for row in keydown_order}) != len(keydown_order):
                    continue
                for label_order in label_orders:
                    if generation_capped:
                        break
                    if self.allocation_rule is AllocationRule.REPOSITORY_FIFO:
                        branch = hypothesis
                        sequential_attributions: list[PersistentAttribution] = []
                        sequential_resolutions: list[TargetResolution] = []
                        sequential_valid = True
                        for keydown, label in zip(keydown_order, label_order):
                            choices, omitted = self._candidate_events(
                                branch, keydown, label
                            )
                            omitted_possible |= omitted
                            if not choices:
                                sequential_valid = False
                                break
                            selected_event = choices[0]
                            sequential_attributions.append(PersistentAttribution(
                                input_id=keydown.input_id,
                                intended_local_id=keydown.intended_local_id,
                                judged_local_id=selected_event.local_id,
                                label=label,
                                visible_interval_s=(lower, upper),
                                resolves_target=self._resolves(label),
                                ambiguous_order=(
                                    not feedback.within_interval_order_known
                                    and len(labels) > 1
                                ),
                            ))
                            if self._resolves(label):
                                resolution = TargetResolution(
                                    local_id=selected_event.local_id,
                                    kind="keydown",
                                    evidence_at_s=feedback.observed_at_s,
                                    input_id=keydown.input_id,
                                )
                                sequential_resolutions.append(resolution)
                                branch = replace(
                                    branch,
                                    resolutions=branch.resolutions + (resolution,),
                                )
                        if sequential_valid:
                            compatible.append(replace(
                                hypothesis,
                                hypothesis_id=self._next_id(
                                    hypothesis.hypothesis_id, "delta"
                                ),
                                attributions=(
                                    hypothesis.attributions
                                    + tuple(sequential_attributions)
                                ),
                                resolutions=(
                                    hypothesis.resolutions
                                    + tuple(sequential_resolutions)
                                ),
                            ))
                            if len(compatible) >= self.max_hypotheses:
                                omitted_possible = True
                                generation_capped = True
                        continue
                    event_choices = []
                    branch_omitted = False
                    for keydown, label in zip(keydown_order, label_order):
                        choices, omitted = self._candidate_events(hypothesis, keydown, label)
                        event_choices.append(choices)
                        branch_omitted |= omitted
                    omitted_possible |= branch_omitted
                    if any(not choices for choices in event_choices):
                        continue
                    # The number of distinct visible orders is bounded, but
                    # each order can still expand into an enormous Cartesian
                    # product of compatible predecessor events.
                    combinations = 1
                    for choices in event_choices:
                        combinations *= len(choices)
                    if combinations > remaining_search_budget:
                        omitted_possible = True
                        combination_budget_exceeded = True
                        generation_capped = True
                        break
                    remaining_search_budget -= combinations
                    for selected in product(*event_choices):
                        resolving_ids = [
                            event.local_id for event, label in zip(selected, label_order)
                            if self._resolves(label)
                        ]
                        if len(set(resolving_ids)) != len(resolving_ids):
                            continue
                        attributions = tuple(
                            PersistentAttribution(
                                input_id=keydown.input_id,
                                intended_local_id=keydown.intended_local_id,
                                judged_local_id=event.local_id,
                                label=label,
                                visible_interval_s=(lower, upper),
                                resolves_target=self._resolves(label),
                                ambiguous_order=(
                                    not feedback.within_interval_order_known
                                    and len(labels) > 1
                                ),
                            )
                            for keydown, event, label in zip(
                                keydown_order, selected, label_order
                            )
                        )
                        resolutions = tuple(
                            TargetResolution(
                                local_id=event.local_id,
                                kind="keydown",
                                evidence_at_s=feedback.observed_at_s,
                                input_id=keydown.input_id,
                            )
                            for keydown, event, label in zip(
                                keydown_order, selected, label_order
                            )
                            if self._resolves(label)
                        )
                        compatible.append(replace(
                            hypothesis,
                            hypothesis_id=self._next_id(hypothesis.hypothesis_id, "delta"),
                            attributions=hypothesis.attributions + attributions,
                            resolutions=hypothesis.resolutions + resolutions,
                        ))
                        if len(compatible) >= self.max_hypotheses:
                            omitted_possible = True
                            generation_capped = True
                            break

        if not compatible and combination_budget_exceeded:
            self._set_hypotheses(
                self._hypotheses,
                truncation_reason="attribution combination budget exceeded",
            )
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.TRUNCATED,
                "attribution combination budget exceeded; causal attribution unknown",
            )
        if not compatible:
            # Do not silently resurrect discarded concrete histories.  A single
            # UNKNOWN continuation records model mismatch/observation loss.
            parent = self._hypotheses[0]
            compatible = [replace(
                parent,
                hypothesis_id=self._next_id(parent.hypothesis_id, "contradicted"),
                observation_consistent=False,
                observation_gaps=_dedupe(parent.observation_gaps + (
                    "visible delta contradicts every retained concrete model state",
                )),
                unknown_search_remainder=True,
            )]
            self._set_hypotheses(compatible)
            return self._estimate(
                feedback.observed_at_s,
                PersistentEstimateStatus.CONTRADICTED,
                "no concrete transition fits; only model-mismatch UNKNOWN remains",
            )

        truncated = self._set_hypotheses(
            compatible,
            truncation_reason=(
                "predecessor/search scope omitted potentially compatible states"
                if omitted_possible else None
            ),
        )
        return self._estimate(
            feedback.observed_at_s,
            PersistentEstimateStatus.TRUNCATED if truncated else PersistentEstimateStatus.AMBIGUOUS,
            "visible delta transitioned every surviving prior hypothesis",
        )

    def _estimate(
        self,
        observed_at_s: float,
        default_status: PersistentEstimateStatus,
        reason: str,
    ) -> PersistentJudgementStateEstimate:
        unknown = any(row.unknown_search_remainder for row in self._hypotheses)
        concrete = [
            row for row in self._hypotheses
            if row.observation_consistent and not row.unknown_search_remainder
        ]
        search_complete = not unknown
        identifiable = search_complete and len(concrete) == 1 and len(self._hypotheses) == 1
        status = default_status
        if identifiable and default_status not in {
            PersistentEstimateStatus.NO_UPDATE,
            PersistentEstimateStatus.OBSERVATION_GAP,
        }:
            status = PersistentEstimateStatus.MODEL_IDENTIFIED
        elif unknown and default_status not in {
            PersistentEstimateStatus.OBSERVATION_GAP,
            PersistentEstimateStatus.CONTRADICTED,
        }:
            status = PersistentEstimateStatus.TRUNCATED
        return PersistentJudgementStateEstimate(
            status=status,
            observed_at_s=observed_at_s,
            hypotheses=self._hypotheses,
            search_complete=search_complete,
            identifiable=identifiable,
            stock_identity_verified=False,
            reason=reason,
        )
