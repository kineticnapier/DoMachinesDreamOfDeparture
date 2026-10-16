from __future__ import annotations
from dataclasses import FrozenInstanceError
import pytest
from dmdod.motor.contact_16k import Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.planning.fingering_counterfactual import evaluate_axis_alternatives, rank_feasible_axes


def test_same_state_repeatable_and_nonmutating():
    motor = Coupled16ContactMotor()
    before = motor.motor.observe()
    a = evaluate_axis_alternatives(motor, .050, candidates=("left_1","right_1"))
    b = evaluate_axis_alternatives(motor, .050, candidates=("left_1","right_1"))
    assert a == b
    assert motor.motor.observe() == before
    assert {x.axis for x in a} == {"left_1","right_1"}
    assert rank_feasible_axes(a) == rank_feasible_axes(b)


def test_invalid_candidates_and_inputs():
    motor = Coupled16ContactMotor()
    with pytest.raises(ValueError):
        evaluate_axis_alternatives(motor, 0, candidates=("left_1","left_1"))
    with pytest.raises(ValueError):
        evaluate_axis_alternatives(motor, 0, candidates=("left_7",))
    with pytest.raises(ValueError):
        evaluate_axis_alternatives(motor, float("nan"))
    with pytest.raises(ValueError):
        evaluate_axis_alternatives(motor, 0, horizon_frames=0)
    with pytest.raises(ValueError):
        evaluate_axis_alternatives(motor, 0, base_commands=(1.,))
