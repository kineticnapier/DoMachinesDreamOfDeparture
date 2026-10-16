from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_nerfed_fingering_v061 import next_note_rollout
from dmdod.motor.contact_16k import Coupled16ContactMotor, ContactAction, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import forecast_reservation_swap

def test_diagnostic_is_deterministic_and_does_not_advance_real_motor():
    m=Coupled16ContactMotor()
    a=ContactAction((0.,)*len(PHYSICAL_AXES))
    rs={"left_1":(0,)}
    times={0:.05,1:.11}
    f=forecast_reservation_swap(m,a,rs,times,"left_1","right_1")
    before=m.motor.time_s
    first=next_note_rollout(m,a,f,times,rs)
    assert first==next_note_rollout(m,a,f,times,rs)
    assert m.motor.time_s==before
    assert first["next_ordinal"]==1
    for row in first["attempts"]:
        assert row["command_start_ms"] >= row["first_tick_end_ms"]-1e-5
        assert row["stop_ms"] >= row["command_start_ms"]
        assert row["reason"] in ("keydown_observed","still_pressed","pressed_at_command","no_keydown_before_deadline")
