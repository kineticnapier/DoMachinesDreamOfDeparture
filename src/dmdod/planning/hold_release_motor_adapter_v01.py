from __future__ import annotations

"""Fail-closed H8 release-forecast to H4 Motor-plan adapter.

This module does not infer a Hold duration, pick a finger, compensate Motor
latency, or read evaluator truth.  It only connects an already Motor-ready H8
forecast to the frozen Hold Motor overlay.  A non-point release command is
rejected because one deterministic KeyUp command cannot be proven safe for
every value in that interval.
"""

from dataclasses import dataclass
from math import isclose, isfinite

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.planning.hold_motor_overlay_v01 import (
    HoldMotorPlan,
    HoldTimingProvenance,
)
from dmdod.planning.hold_release_forecast_v01 import (
    HoldReleaseForecast,
    HoldReleaseForecastStatus,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import PhysicalFinger


@dataclass(frozen=True, slots=True)
class HoldReleaseMotorReservation:
    """Information owned by Timing/Fingering before Motor registration."""

    local_id: int
    finger: PhysicalFinger
    start_due_s: TimeWindow
    start_command_s: float
    release: HoldReleaseForecast

    def __post_init__(self) -> None:
        if type(self.local_id) is not int or self.local_id < 0:
            raise ValueError("local_id must be a nonnegative integer")
        if not isinstance(self.finger, PhysicalFinger):
            raise TypeError("finger must be PhysicalFinger")
        if not isinstance(self.start_due_s, TimeWindow):
            raise TypeError("start due must be a TimeWindow")
        if not isfinite(self.start_command_s) or self.start_command_s < 0.0:
            raise ValueError("start command must be finite and nonnegative")
        if not isinstance(self.release, HoldReleaseForecast):
            raise TypeError("release must be HoldReleaseForecast")


def build_causal_hold_motor_plan(
    reservation: HoldReleaseMotorReservation,
) -> HoldMotorPlan:
    """Convert only a causally verified, deterministic release forecast."""
    if not isinstance(reservation, HoldReleaseMotorReservation):
        raise TypeError("expected HoldReleaseMotorReservation")
    forecast = reservation.release
    if forecast.hold_start_local_id != reservation.local_id:
        raise ValueError("Hold release forecast local ID does not match reservation")
    if (
        not forecast.motor_ready
        or forecast.status is not HoldReleaseForecastStatus.MOTOR_READY
    ):
        raise ValueError("Hold release forecast is not Motor-ready")
    if forecast.release_due_s is None or forecast.release_command_s is None:
        raise ValueError("Motor-ready Hold forecast is missing release intervals")
    command = forecast.release_command_s
    if not isclose(command.lower_s, command.upper_s, abs_tol=1e-12):
        raise ValueError("non-point Hold release command remains unsafe for Motor")
    return HoldMotorPlan(
        local_id=reservation.local_id,
        finger=reservation.finger,
        start_due=reservation.start_due_s,
        release_due=forecast.release_due_s,
        start_command_s=reservation.start_command_s,
        release_command_s=command.lower_s,
        provenance=HoldTimingProvenance.CAUSAL_VISIBLE_INTERVAL,
    )
