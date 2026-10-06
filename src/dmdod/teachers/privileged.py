from __future__ import annotations

from dataclasses import dataclass

from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.env import MotorAction, MotorEnv, MotorObservation


@dataclass(frozen=True)
class LeadTeacherCalibration:
    """Privileged single-press timing calibration for imitation experiments.

    ``press_latency_s`` is measured from the frozen body itself by applying a
    full downward motor command from rest until the first key-down event.
    ``lead_s`` additionally compensates for the average half-control-step delay
    between an ideal command time and the next policy decision.
    """

    press_latency_s: float
    lead_s: float
    control_dt_s: float


def calibrate_single_press_lead(
    *,
    control_dt_s: float = 0.010,
    same_hand: bool = True,
    max_wait_s: float = 1.0,
) -> LeadTeacherCalibration:
    """Measure the body's open-loop press latency without chart information."""

    env = MotorEnv(same_hand=same_hand, control_dt_s=control_dt_s)
    env.reset()
    max_steps = max(1, int(max_wait_s / control_dt_s) + 1)

    for _ in range(max_steps):
        transition = env.step(MotorAction(1.0, 0.0))
        for event in transition.evaluator_events:
            if event.event is KeyEvent.DOWN and event.key == "left":
                latency = float(event.time_s)
                return LeadTeacherCalibration(
                    press_latency_s=latency,
                    lead_s=latency + 0.5 * control_dt_s,
                    control_dt_s=control_dt_s,
                )

    raise RuntimeError("single-press calibration produced no left key-down event")


class PrivilegedLeadTeacher:
    """P3 teacher that converts exact target time into a motor lead command.

    This teacher is intentionally privileged and must never be used as the
    student's observation.  It exists only to generate imitation targets.  The
    student still receives motor state plus visible geometry/motion and must
    infer the speed-dependent lead rule from those observations.
    """

    def __init__(self, calibration: LeadTeacherCalibration) -> None:
        self.calibration = calibration

    def action(
        self,
        *,
        now_s: float,
        target_time_s: float,
        motor: MotorObservation,
    ) -> MotorAction:
        # P3 uses one left-key target from rest. Keep the finger neutral until
        # the calibrated launch point so the measured open-loop latency remains
        # applicable, then drive fully downward. A resolved press ends the
        # one-note episode, but release is included for safe standalone use.
        if motor.left_pressed:
            return MotorAction(-1.0, 0.0)
        if now_s + self.calibration.lead_s >= target_time_s:
            return MotorAction(1.0, 0.0)
        return MotorAction(0.0, 0.0)
