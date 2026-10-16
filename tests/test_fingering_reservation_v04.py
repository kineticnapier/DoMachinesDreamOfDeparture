from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import (
    forecast_reservation_swap,
)


def test_all_candidates_share_unchanged_physical_state():
    motor = Coupled16ContactMotor()
    original = motor.observe()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    reservations = {"left_1": (0,)}
    due = {0: .050}
    for destination in PHYSICAL_AXES:
        if destination == "left_1":
            continue
        try:
            forecast_reservation_swap(motor, action, reservations, due,
                                      "left_1", destination)
        except ValueError:
            pass
        assert motor.observe() == original
