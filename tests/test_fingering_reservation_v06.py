from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_nerfed_fingering_v06 import next_note_rollout, choose_swap
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import forecast_reservation_swap


def test_two_note_rollout_is_deterministic_and_nonmutating():
    motor=Coupled16ContactMotor()
    before=motor.observe()
    action=ContactAction((0.,)*len(PHYSICAL_AXES))
    reservations={"left_1": (0,)}
    notes={0: .050, 1: .095}
    forecast=forecast_reservation_swap(motor,action,reservations,notes,"left_1","right_1")
    a=next_note_rollout(motor,action,forecast,notes,reservations)
    b=next_note_rollout(motor,action,forecast,notes,reservations)
    assert a==b
    assert a["next_ordinal"]==1
    assert a["evaluated"]>=1
    assert motor.observe()==before


def test_no_future_note_is_explicit():
    motor=Coupled16ContactMotor()
    action=ContactAction((0.,)*len(PHYSICAL_AXES))
    reservations={"left_1": (0,)}
    notes={0:.050}
    forecast=forecast_reservation_swap(motor,action,reservations,notes,"left_1","right_1")
    assert next_note_rollout(motor,action,forecast,notes,reservations)["next_ordinal"] is None
