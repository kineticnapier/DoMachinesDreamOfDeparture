from __future__ import annotations

"""Set-valued research scheduler; no command or production integration API."""

from dataclasses import dataclass
from enum import Enum
import math


class FeasibilityStatus(str, Enum):
    SAFE = "safe_reservation_candidate"
    ABSTAIN = "abstain"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TimeInterval:
    lower_s: float
    upper_s: float

    def __post_init__(self):
        if not math.isfinite(self.lower_s) or not math.isfinite(self.upper_s) or self.lower_s > self.upper_s:
            raise ValueError("invalid time interval")

    def contains_interval(self, other: "TimeInterval") -> bool:
        return self.lower_s <= other.lower_s + 1e-12 and other.upper_s <= self.upper_s + 1e-12

    def overlaps(self, other: "TimeInterval") -> bool:
        return max(self.lower_s, other.lower_s) <= min(self.upper_s, other.upper_s) + 1e-12


@dataclass(frozen=True, slots=True)
class DueHypothesis:
    hypothesis_id: str
    due: TimeInterval
    acceptable_strike: TimeInterval
    causal_visible: bool


@dataclass(frozen=True, slots=True)
class FloorStateHypothesis:
    hypothesis_id: str
    intended_event_id: str | None
    target_for_next_keydown: str | None
    run_mode: str
    supported: bool


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_id: str
    axis: str
    occupied: TimeInterval
    baseline_hit_dependency: bool = True


@dataclass(frozen=True, slots=True)
class RobustScheduleDecision:
    status: FeasibilityStatus
    arrival: TimeInterval | None
    axis: str | None
    reasons: tuple[str, ...]
    evaluated_due_hypotheses: int
    evaluated_floor_hypotheses: int


def assess_reservation(
    *,
    intended_event_id: str,
    axis: str,
    motor_arrival: TimeInterval | None,
    motor_envelope_valid: bool,
    due_hypotheses: tuple[DueHypothesis, ...],
    floor_hypotheses: tuple[FloorStateHypothesis, ...],
    reservations: tuple[Reservation, ...] = (),
    declared_run_mode: str,
) -> RobustScheduleDecision:
    """Require safety in every retained causal hypothesis."""

    if not intended_event_id or not axis:
        raise ValueError("event and axis must be non-empty")
    reasons: list[str] = []
    unknown = False
    if not due_hypotheses:
        reasons.append("no_due_hypotheses")
        unknown = True
    if not floor_hypotheses:
        reasons.append("no_floor_state_hypotheses")
        unknown = True
    if motor_arrival is None or not motor_envelope_valid:
        reasons.append("motor_arrival_envelope_unknown")
        unknown = True
    if any(not row.causal_visible for row in due_hypotheses):
        reasons.append("noncausal_due_hypothesis_rejected")
        unknown = True
    if any(not row.supported for row in floor_hypotheses):
        reasons.append("unsupported_floor_branch")
        unknown = True

    if motor_arrival is not None:
        for row in due_hypotheses:
            if not row.acceptable_strike.contains_interval(motor_arrival):
                reasons.append(f"arrival_not_safe_for_due:{row.hypothesis_id}")
        for existing in reservations:
            if existing.axis == axis and existing.occupied.overlaps(motor_arrival):
                reasons.append(f"reservation_conflict:{existing.reservation_id}")
                if existing.baseline_hit_dependency:
                    reasons.append(f"baseline_hit_at_risk:{existing.reservation_id}")
    for row in floor_hypotheses:
        if row.run_mode != declared_run_mode:
            reasons.append(f"run_mode_mismatch:{row.hypothesis_id}")
        if row.target_for_next_keydown != intended_event_id:
            reasons.append(f"floor_attribution_not_robust:{row.hypothesis_id}")

    if unknown:
        status = FeasibilityStatus.UNKNOWN
    elif reasons:
        status = FeasibilityStatus.ABSTAIN
    else:
        status = FeasibilityStatus.SAFE
    return RobustScheduleDecision(
        status, motor_arrival if status is FeasibilityStatus.SAFE else None,
        axis if status is FeasibilityStatus.SAFE else None,
        tuple(dict.fromkeys(reasons)), len(due_hypotheses), len(floor_hypotheses),
    )
