from __future__ import annotations

"""Research-only decomposition of Motor lead feasibility.

This module deliberately does not issue commands.  It distinguishes arrival of
a causal due observation, controller quantisation, the measured cold press,
and the Stock counted window.  Target truth may be supplied by an evaluator,
but there is no API that can feed it into production Timing or Motor code.
"""

from dataclasses import dataclass
from enum import Enum
import math


class EvidenceState(str, Enum):
    YES = "yes"
    NO = "no"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class LeadFeasibility:
    lead_s: float
    observed_at_s: float
    due_s_posthoc: float
    reservation_deadline_s_posthoc: float
    nominal_lead_temporally_available: bool
    due_observability: EvidenceState
    earliest_command_tick_s: float
    cold_keydown_s: float
    cold_keydown_error_ms: float
    cold_keydown_by_due: bool
    cold_keydown_in_stock_window: bool
    reservation_conflict_clear: EvidenceState
    all_due_aligned_conditions: bool
    all_counted_hit_conditions: bool


def next_control_tick(time_s: float, *, control_dt_s: float) -> float:
    """First control tick at or after ``time_s`` on a zero-phase grid."""

    if not math.isfinite(time_s) or not math.isfinite(control_dt_s):
        raise ValueError("times must be finite")
    if control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive")
    # Remove floating-point noise at an exact grid boundary without moving a
    # genuinely later request into the previous tick.
    ticks = math.ceil(time_s / control_dt_s - 1e-12)
    return ticks * control_dt_s


def calibrated_nominal_lead_s(
    *, cold_press_latency_s: float, control_dt_s: float
) -> float:
    """Reproduce the repository's measured-latency + half-tick convention."""

    if not math.isfinite(cold_press_latency_s) or cold_press_latency_s < 0.0:
        raise ValueError("cold press latency must be finite and non-negative")
    if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    return cold_press_latency_s + 0.5 * control_dt_s


def assess_lead(
    *,
    observed_at_s: float,
    due_s_posthoc: float,
    lead_s: float,
    due_observability: EvidenceState,
    stock_window_s: tuple[float, float],
    cold_press_latency_s: float = 0.039,
    control_dt_s: float = 0.010,
    reservation_conflict_clear: EvidenceState = EvidenceState.UNVERIFIED,
) -> LeadFeasibility:
    """Assess one declared lead without treating target truth as observation.

    ``due_s_posthoc`` and ``stock_window_s`` are evaluator-only labels.  The
    command is not allowed before both the observation and the nominal launch
    time.  The physical calculation is a conditional cold/free-axis projection,
    not a replay of the actual 12-axis state.
    """

    values = (
        observed_at_s,
        due_s_posthoc,
        lead_s,
        cold_press_latency_s,
        control_dt_s,
        *stock_window_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("lead assessment values must be finite")
    if lead_s < 0.0 or cold_press_latency_s < 0.0 or control_dt_s <= 0.0:
        raise ValueError("lead and latency must be non-negative; tick positive")
    earliest, latest = stock_window_s
    if earliest > latest:
        raise ValueError("Stock window must be ordered")

    deadline = due_s_posthoc - lead_s
    temporal = observed_at_s <= deadline + 1e-12
    requested_command_s = max(observed_at_s, deadline)
    command_tick = next_control_tick(requested_command_s, control_dt_s=control_dt_s)
    keydown_s = command_tick + cold_press_latency_s
    by_due = keydown_s <= due_s_posthoc + 1e-12
    in_window = earliest - 1e-12 <= keydown_s <= latest + 1e-12
    observation_proven = due_observability is EvidenceState.YES
    conflict_proven_clear = reservation_conflict_clear is EvidenceState.YES
    return LeadFeasibility(
        lead_s=float(lead_s),
        observed_at_s=float(observed_at_s),
        due_s_posthoc=float(due_s_posthoc),
        reservation_deadline_s_posthoc=deadline,
        nominal_lead_temporally_available=temporal,
        due_observability=due_observability,
        earliest_command_tick_s=command_tick,
        cold_keydown_s=keydown_s,
        cold_keydown_error_ms=(keydown_s - due_s_posthoc) * 1000.0,
        cold_keydown_by_due=by_due,
        cold_keydown_in_stock_window=in_window,
        reservation_conflict_clear=reservation_conflict_clear,
        all_due_aligned_conditions=(
            temporal and observation_proven and by_due and conflict_proven_clear
        ),
        all_counted_hit_conditions=(
            temporal and observation_proven and in_window and conflict_proven_clear
        ),
    )
