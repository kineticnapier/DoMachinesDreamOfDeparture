from __future__ import annotations

from dataclasses import dataclass

from .motor_env import MotorAction, MotorObservation


@dataclass(frozen=True, slots=True)
class FingerChoice:
    side: str
    reason: str


def _projected_readiness(position_m: float, velocity_m_s: float, lead_s: float) -> float:
    """Cheap state-only estimate of which free finger can actuate sooner."""

    horizon = max(0.0, min(float(lead_s), 0.050))
    return float(position_m) + horizon * float(velocity_m_s)


def choose_press_finger(
    motor: MotorObservation,
    *,
    lead_s: float,
    preferred_action: MotorAction | None = None,
    preference_threshold: float = 0.15,
) -> FingerChoice | None:
    """Choose a physically available finger without assigning notes to finger IDs.

    If a student is already committing positive force to an available finger,
    follow that choice instead of correcting it to an arbitrary canonical
    fingering.  Otherwise choose the currently more-ready free finger from the
    visible motor state.  Target ordinal is deliberately not an input.
    """

    available_left = not motor.left_pressed
    available_right = not motor.right_pressed
    if not available_left and not available_right:
        return None

    if preferred_action is not None:
        left_preference = float(preferred_action.left) if available_left else float("-inf")
        right_preference = float(preferred_action.right) if available_right else float("-inf")
        best_preference = max(left_preference, right_preference)
        if best_preference >= preference_threshold:
            if right_preference > left_preference:
                return FingerChoice("right", "student-preference")
            return FingerChoice("left", "student-preference")

    if available_left and not available_right:
        return FingerChoice("left", "only-free")
    if available_right and not available_left:
        return FingerChoice("right", "only-free")

    left_readiness = _projected_readiness(
        motor.left_position_m,
        motor.left_velocity_m_s,
        lead_s,
    )
    right_readiness = _projected_readiness(
        motor.right_position_m,
        motor.right_velocity_m_s,
        lead_s,
    )
    if right_readiness > left_readiness:
        return FingerChoice("right", "more-ready")
    return FingerChoice("left", "more-ready")


def finger_agnostic_lead_action(
    *,
    now_s: float,
    target_time_s: float,
    lead_s: float,
    motor: MotorObservation,
    preferred_action: MotorAction | None = None,
) -> MotorAction:
    """Privileged timing teacher with no note-to-finger assignment.

    Before the launch point, pressed keys are released and free keys are left
    neutral.  At the launch point, either free finger may be used.  During
    DAgger, ``preferred_action`` lets the teacher preserve a student's already
    valid fingering choice instead of imposing left/right alternation.
    """

    left = -1.0 if motor.left_pressed else 0.0
    right = -1.0 if motor.right_pressed else 0.0
    if now_s + lead_s < target_time_s:
        return MotorAction(left, right)

    choice = choose_press_finger(
        motor,
        lead_s=lead_s,
        preferred_action=preferred_action,
    )
    if choice is None:
        return MotorAction(left, right)

    if choice.side == "left":
        left = 1.0
    else:
        right = 1.0
    return MotorAction(left, right)
