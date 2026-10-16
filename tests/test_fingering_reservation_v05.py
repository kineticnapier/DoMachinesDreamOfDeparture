from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import forecast_reservation_swap
from pathlib import Path
import sys

# scripts/ contains standalone entry points with sibling imports, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_nerfed_fingering_v05 import score_future_readiness


def test_no_future_note_no_penalty():
    motor = Coupled16ContactMotor()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    reservations = {"left_1": (0,)}
    note_times = {0: .05}
    f = forecast_reservation_swap(motor, action, reservations, note_times,
                                  "left_1", "right_1")
    assert score_future_readiness(motor, action, f, note_times, reservations) == 0


def test_near_upcoming_note_penalty_is_nonnegative():
    motor = Coupled16ContactMotor()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    reservations = {"left_1": (0,)}
    note_times = {0: .05, 1: .055}
    f = forecast_reservation_swap(motor, action, reservations, note_times,
                                  "left_1", "right_1")
    assert score_future_readiness(motor, action, f, note_times, reservations) >= 0
