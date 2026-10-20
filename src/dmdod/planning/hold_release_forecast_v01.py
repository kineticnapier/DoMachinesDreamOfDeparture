from __future__ import annotations

"""Causal Hold release interval forecast from explicitly visible evidence.

No chart Hold action, holdLength, target time, or future speed is accepted.
Diagnostic proxy evidence can exercise the math but can never become
Motor-ready.  Real BPM is assumed to include current pitch.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite, pi

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.features.hold_mesh_observability_v01 import HoldDurationObservation


MOTOR_LEAD_S = 0.044
CONTROL_DT_S = 0.010


class HoldVisualEvidence(str, Enum):
    STOCK_PIXEL_VERIFIED = "stock_pixel_verified"
    DIAGNOSTIC_PROXY = "diagnostic_proxy"
    UNVERIFIED = "unverified"


class HoldReleaseForecastStatus(str, Enum):
    MOTOR_READY = "motor_ready"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    ABSTAIN_DURATION = "abstain_duration"
    ABSTAIN_KINEMATICS = "abstain_kinematics"
    ABSTAIN_FUTURE_TRANSITION = "abstain_future_transition"
    ABSTAIN_RELEASE_LATENCY = "abstain_release_latency"
    ABSTAIN_UNCERTAINTY = "abstain_uncertainty"
    ABSTAIN_INSUFFICIENT_LEAD = "abstain_insufficient_lead"


@dataclass(frozen=True, slots=True)
class VisibleHoldKinematics:
    observed_at_s: float
    base_angle_rad: TimeWindow
    real_bpm: TimeWindow
    valid: bool
    future_transition_free: bool
    source: str

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s):
            raise ValueError("kinematics observation time must be finite")
        if not isinstance(self.base_angle_rad, TimeWindow):
            raise TypeError("base angle must be a TimeWindow")
        if not isinstance(self.real_bpm, TimeWindow):
            raise TypeError("real BPM must be a TimeWindow")
        if self.base_angle_rad.lower_s < 0.0 or self.real_bpm.lower_s <= 0.0:
            raise ValueError("angle must be nonnegative and BPM positive")
        if type(self.valid) is not bool or type(self.future_transition_free) is not bool:
            raise TypeError("kinematics flags must be bool")
        if not self.source.strip():
            raise ValueError("kinematics source is required")


@dataclass(frozen=True, slots=True)
class MotorReleaseLatency:
    observed_at_s: float
    latency_s: TimeWindow
    valid: bool
    source: str = "frozen_12axis_motor_state_model"

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s):
            raise ValueError("latency observation time must be finite")
        if not isinstance(self.latency_s, TimeWindow):
            raise TypeError("latency must be a TimeWindow")
        if self.latency_s.lower_s < 0.0:
            raise ValueError("release latency cannot be negative")
        if type(self.valid) is not bool or not self.source.strip():
            raise ValueError("valid latency source is required")


@dataclass(frozen=True, slots=True)
class HoldReleaseForecastRequest:
    hold_start_local_id: int
    start_due_s: TimeWindow
    duration: HoldDurationObservation
    evidence: HoldVisualEvidence
    kinematics: VisibleHoldKinematics
    release_latency: MotorReleaseLatency

    def __post_init__(self) -> None:
        if type(self.hold_start_local_id) is not int or self.hold_start_local_id < 0:
            raise ValueError("Hold start local ID must be nonnegative integer")
        if not isinstance(self.start_due_s, TimeWindow):
            raise TypeError("start due must be a TimeWindow")
        if not isinstance(self.duration, HoldDurationObservation):
            raise TypeError("duration must be a causal HoldDurationObservation")
        if not isinstance(self.evidence, HoldVisualEvidence):
            raise TypeError("evidence must be HoldVisualEvidence")
        if not isinstance(self.kinematics, VisibleHoldKinematics):
            raise TypeError("kinematics must be VisibleHoldKinematics")
        if not isinstance(self.release_latency, MotorReleaseLatency):
            raise TypeError("release latency must be MotorReleaseLatency")


@dataclass(frozen=True, slots=True)
class HoldReleaseForecast:
    hold_start_local_id: int
    status: HoldReleaseForecastStatus
    available_at_s: float
    duration_candidates: tuple[int, ...]
    release_due_s: TimeWindow | None
    release_command_s: TimeWindow | None
    forecast_lead_lower_s: float | None
    motor_ready: bool
    reason: str


def forecast_hold_release(
    request: HoldReleaseForecastRequest,
) -> HoldReleaseForecast:
    if not isinstance(request, HoldReleaseForecastRequest):
        raise TypeError("expected HoldReleaseForecastRequest")
    duration = request.duration
    kinematics = request.kinematics
    latency = request.release_latency
    available_at = max(
        duration.observed_at_s, kinematics.observed_at_s, latency.observed_at_s
    )

    def abstain(status: HoldReleaseForecastStatus, reason: str):
        return HoldReleaseForecast(
            request.hold_start_local_id, status, available_at,
            duration.candidates, None, None, None, False, reason,
        )

    if not duration.valid or len(duration.candidates) != 1:
        return abstain(
            HoldReleaseForecastStatus.ABSTAIN_DURATION,
            "duration_not_uniquely_visible",
        )
    if not kinematics.valid:
        return abstain(
            HoldReleaseForecastStatus.ABSTAIN_KINEMATICS,
            "visible_angle_or_real_bpm_invalid",
        )
    if not kinematics.future_transition_free:
        return abstain(
            HoldReleaseForecastStatus.ABSTAIN_FUTURE_TRANSITION,
            "DO_NOT_RELAX_SETSPEED_GATE",
        )
    if not latency.valid:
        return abstain(
            HoldReleaseForecastStatus.ABSTAIN_RELEASE_LATENCY,
            "physical_release_latency_unavailable",
        )

    loops = duration.candidates[0]
    lower_angle = kinematics.base_angle_rad.lower_s + loops * 2.0 * pi
    upper_angle = kinematics.base_angle_rad.upper_s + loops * 2.0 * pi
    delta_lower = lower_angle / pi * 60.0 / kinematics.real_bpm.upper_s
    delta_upper = upper_angle / pi * 60.0 / kinematics.real_bpm.lower_s
    release_due = TimeWindow(
        request.start_due_s.lower_s + delta_lower,
        request.start_due_s.upper_s + delta_upper,
    )
    release_command = TimeWindow(
        release_due.lower_s - latency.latency_s.upper_s,
        release_due.upper_s - latency.latency_s.lower_s,
    )
    if release_command.lower_s < 0.0:
        return abstain(
            HoldReleaseForecastStatus.ABSTAIN_RELEASE_LATENCY,
            "release_command_precedes_episode_start",
        )
    lead_lower = release_command.lower_s - available_at

    if request.evidence is not HoldVisualEvidence.STOCK_PIXEL_VERIFIED:
        return HoldReleaseForecast(
            request.hold_start_local_id,
            HoldReleaseForecastStatus.DIAGNOSTIC_ONLY,
            available_at,
            duration.candidates,
            release_due,
            release_command,
            lead_lower,
            False,
            "visual_evidence_not_live_stock_verified",
        )
    if release_command.upper_s - release_command.lower_s > CONTROL_DT_S + 1e-12:
        return HoldReleaseForecast(
            request.hold_start_local_id,
            HoldReleaseForecastStatus.ABSTAIN_UNCERTAINTY,
            available_at,
            duration.candidates,
            release_due,
            release_command,
            lead_lower,
            False,
            "release_command_interval_exceeds_one_control_step",
        )
    if lead_lower < MOTOR_LEAD_S - 1e-12:
        return HoldReleaseForecast(
            request.hold_start_local_id,
            HoldReleaseForecastStatus.ABSTAIN_INSUFFICIENT_LEAD,
            available_at,
            duration.candidates,
            release_due,
            release_command,
            lead_lower,
            False,
            "release_command_lower_bound_has_less_than_44ms_lead",
        )
    return HoldReleaseForecast(
        request.hold_start_local_id,
        HoldReleaseForecastStatus.MOTOR_READY,
        available_at,
        duration.candidates,
        release_due,
        release_command,
        lead_lower,
        True,
        "unique_stock_pixel_duration_and_bounded_visible_kinematics",
    )
