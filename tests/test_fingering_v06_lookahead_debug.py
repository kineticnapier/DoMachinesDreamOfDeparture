from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_nerfed_fingering_v06_lookahead_debug import next_note_rollout
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import forecast_reservation_swap

def test_second_event_failure_diagnostics():
    motor=Coupled16ContactMotor()
    action=ContactAction((0.,)*len(PHYSICAL_AXES))
    reserve={"left_1": (0,)}
    times={0:.05, 1:.10}
    f=forecast_reservation_swap(motor,action,reserve,times,"left_1","right_1")
    outcome=next_note_rollout(motor,action,f,times,reserve)
    assert outcome["next_ordinal"]==1
    assert outcome["evaluated"]>=1
    if outcome["next_axis"] is None:
        assert outcome["failure_reason"]=="no_second_keydown"
        assert len(outcome["attempts"])==outcome["evaluated"]
