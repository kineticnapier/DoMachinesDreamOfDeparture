from pathlib import Path
import pytest

from dmdod.planning.fingering_reservation_counterfactual import (
    forecast_reservation_swap,
)
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES


def test_candidate_cannot_be_currently_reserved():
    motor = Coupled16ContactMotor()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    with pytest.raises(ValueError, match="unreserved"):
        forecast_reservation_swap(motor, action, {"left_1": (1,), "right_1": (2,)},
                                  {1: .05, 2: .06}, "left_1", "right_1")


def test_non_candidate_motor_kept_unmodified():
    motor = Coupled16ContactMotor()
    before = motor.observe()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    a = forecast_reservation_swap(motor, action, {"left_1": (1,)},
                                  {1: .05}, "left_1", "right_1")
    assert motor.observe() == before
    assert a == forecast_reservation_swap(motor, action, {"left_1": (1,)},
                                          {1: .05}, "left_1", "right_1")
