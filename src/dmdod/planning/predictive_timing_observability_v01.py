from __future__ import annotations

"""Causal due-prefix and Motor-lead feasibility primitives.

This research module accepts only already-observed prefixes and declared due
intervals.  It has no chart, scorer, target ordinal, or future-frame API.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence


class EarlyDueStatus(str, Enum):
    OBSERVABLE = "early_due_observable"
    NONIDENTIFIABLE = "early_due_nonidentifiable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class LeadStatus(str, Enum):
    FEASIBLE = "lead_44ms_feasible"
    INFEASIBLE = "lead_44ms_infeasible"


@dataclass(frozen=True, slots=True)
class VisiblePrefixFrame:
    observed_at_s: float
    features: tuple[float, ...]
    cue: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s):
            raise ValueError("observation time must be finite")
        if not self.features or not all(isfinite(value) for value in self.features):
            raise ValueError("visible feature vector must be finite and non-empty")


@dataclass(frozen=True, slots=True)
class PrefixComparison:
    status: EarlyDueStatus
    compared_frames: int
    last_compared_at_s: float | None
    maximum_feature_difference: float
    cues_equal: bool
    first_difference_at_s: float | None


@dataclass(frozen=True, slots=True)
class DueInterval:
    lower_s: float
    upper_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.lower_s) or not isfinite(self.upper_s) or self.lower_s > self.upper_s:
            raise ValueError("due interval must be finite and ordered")


@dataclass(frozen=True, slots=True)
class AcceptableStrikeWindow:
    earliest_s: float
    latest_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.earliest_s) or not isfinite(self.latest_s) or self.earliest_s > self.latest_s:
            raise ValueError("strike window must be finite and ordered")


@dataclass(frozen=True, slots=True)
class LeadAssessment:
    status: LeadStatus
    observed_at_s: float
    required_lead_s: float
    earliest_motor_strike_s: float
    guaranteed_reservation_deadline_s: float
    shared_safe_strike_window: AcceptableStrikeWindow | None
    reason: str


def compare_visible_prefixes(
    left: Sequence[VisiblePrefixFrame],
    right: Sequence[VisiblePrefixFrame],
    *,
    deadline_s: float,
    tolerance: float = 0.0,
) -> PrefixComparison:
    """Compare only frames observed no later than ``deadline_s``."""

    if not isfinite(deadline_s) or not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("deadline and tolerance must be finite")
    lprefix = tuple(row for row in left if row.observed_at_s <= deadline_s + 1e-12)
    rprefix = tuple(row for row in right if row.observed_at_s <= deadline_s + 1e-12)
    if not lprefix or len(lprefix) != len(rprefix):
        return PrefixComparison(EarlyDueStatus.INSUFFICIENT_EVIDENCE, 0, None, 0.0, False, None)
    maximum = 0.0
    cues_equal = True
    first: float | None = None
    for lrow, rrow in zip(lprefix, rprefix, strict=True):
        if abs(lrow.observed_at_s - rrow.observed_at_s) > 1e-12 or len(lrow.features) != len(rrow.features):
            return PrefixComparison(EarlyDueStatus.INSUFFICIENT_EVIDENCE, 0, None, 0.0, False, None)
        delta = max(abs(a - b) for a, b in zip(lrow.features, rrow.features, strict=True))
        maximum = max(maximum, delta)
        cue_same = lrow.cue == rrow.cue
        cues_equal &= cue_same
        if first is None and (delta > tolerance or not cue_same):
            first = lrow.observed_at_s
    return PrefixComparison(
        EarlyDueStatus.NONIDENTIFIABLE if first is None else EarlyDueStatus.OBSERVABLE,
        len(lprefix), lprefix[-1].observed_at_s, maximum, cues_equal, first,
    )


def assess_shared_safe_strike(
    *,
    observed_at_s: float,
    due_interval: DueInterval,
    acceptable_windows: Sequence[AcceptableStrikeWindow],
    required_lead_s: float = 0.044,
) -> LeadAssessment:
    """Require one strike time safe for every still-compatible hypothesis."""

    if not isfinite(observed_at_s) or not isfinite(required_lead_s) or required_lead_s < 0.0:
        raise ValueError("observation and lead must be finite")
    deadline = due_interval.lower_s - required_lead_s
    motor_earliest = observed_at_s + required_lead_s
    if not acceptable_windows:
        return LeadAssessment(LeadStatus.INFEASIBLE, observed_at_s, required_lead_s, motor_earliest, deadline, None, "no sound acceptable strike windows")
    shared_lower = max(row.earliest_s for row in acceptable_windows)
    shared_upper = min(row.latest_s for row in acceptable_windows)
    executable_lower = max(shared_lower, motor_earliest)
    if observed_at_s > deadline + 1e-12:
        return LeadAssessment(LeadStatus.INFEASIBLE, observed_at_s, required_lead_s, motor_earliest, deadline, None, "observation arrived after the earliest compatible due's reservation deadline")
    if executable_lower > shared_upper + 1e-12:
        return LeadAssessment(LeadStatus.INFEASIBLE, observed_at_s, required_lead_s, motor_earliest, deadline, None, "compatible worlds have no common Motor-reachable hit time")
    shared = AcceptableStrikeWindow(executable_lower, shared_upper)
    return LeadAssessment(LeadStatus.FEASIBLE, observed_at_s, required_lead_s, motor_earliest, deadline, shared, "one strike time is safe in every compatible world")

