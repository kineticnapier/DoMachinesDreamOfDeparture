"""Portable Motor replay regression for Stage C v0.6.2."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trace_swapped_teacher_causality_v062 import replay, first_down
from dmdod.motor.contact_16k import Coupled16ContactMotor, ContactAction, PHYSICAL_AXES


def test_cloned_motor_replay_is_exact_and_does_not_mutate_snapshot():
    motor=Coupled16ContactMotor()
    values=[-1.]*len(PHYSICAL_AXES)
    values[PHYSICAL_AXES.index("left_1")]=1.
    action=ContactAction(tuple(values))
    controls=[action]*15
    assert replay(motor,controls)==replay(motor,controls)
    assert first_down(replay(motor,controls)) is None
    assert motor.motor.time_s == 0.
