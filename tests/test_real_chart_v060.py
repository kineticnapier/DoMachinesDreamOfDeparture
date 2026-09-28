from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v060 as trainer  # noqa: E402
from dmdod.finger_agnostic_teacher import (  # noqa: E402
    choose_press_finger,
    finger_agnostic_lead_action,
)
from dmdod.motor_env import MotorAction, MotorObservation  # noqa: E402


def _motor(
    *,
    left_position: float = 0.0,
    right_position: float = 0.0,
    left_velocity: float = 0.0,
    right_velocity: float = 0.0,
    left_pressed: bool = False,
    right_pressed: bool = False,
) -> MotorObservation:
    return MotorObservation(
        left_position_m=left_position,
        right_position_m=right_position,
        left_velocity_m_s=left_velocity,
        right_velocity_m_s=right_velocity,
        left_pressed=left_pressed,
        right_pressed=right_pressed,
    )


def test_teacher_preserves_students_available_finger_choice():
    motor = _motor()
    right = finger_agnostic_lead_action(
        now_s=1.0,
        target_time_s=1.02,
        lead_s=0.043,
        motor=motor,
        preferred_action=MotorAction(-0.2, 0.8),
    )
    left = finger_agnostic_lead_action(
        now_s=1.0,
        target_time_s=1.02,
        lead_s=0.043,
        motor=motor,
        preferred_action=MotorAction(0.9, -0.1),
    )
    assert right.right == 1.0 and right.left == 0.0
    assert left.left == 1.0 and left.right == 0.0


def test_teacher_uses_other_finger_when_preferred_one_is_still_pressed():
    motor = _motor(left_pressed=True, left_position=0.0022)
    action = finger_agnostic_lead_action(
        now_s=1.0,
        target_time_s=1.01,
        lead_s=0.043,
        motor=motor,
        preferred_action=MotorAction(1.0, 0.0),
    )
    assert action.left == -1.0
    assert action.right == 1.0


def test_teacher_without_student_preference_uses_motor_readiness_not_note_parity():
    motor = _motor(
        left_position=0.0002,
        right_position=0.0012,
        left_velocity=0.0,
        right_velocity=0.002,
    )
    choice = choose_press_finger(motor, lead_s=0.043)
    assert choice is not None
    assert choice.side == "right"
    assert choice.reason == "more-ready"


def test_teacher_releases_before_launch_point_without_assigning_a_press():
    motor = _motor(left_pressed=True, left_position=0.0022)
    action = finger_agnostic_lead_action(
        now_s=0.5,
        target_time_s=1.0,
        lead_s=0.043,
        motor=motor,
        preferred_action=MotorAction(0.9, 0.0),
    )
    assert action == MotorAction(-1.0, 0.0)


def test_v060_starts_fresh_instead_of_reusing_fixed_fingering_checkpoint():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 8
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v060_finger_agnostic.pt")
