from __future__ import annotations
import pytest
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode
from dmdod.planning.fingering_reservation_counterfactual import forecast_reservation_swap


def test_single_reservation_swap_is_deterministic_nonmutating():
    motor=Coupled16ContactMotor()
    action=ContactAction(tuple(0. for _ in PHYSICAL_AXES))
    old=motor.observe()
    kw=dict(motor=motor, action=action, reservations={"left_1":(0,)},
            note_times={0:.050}, source_axis="left_1",candidate_axis="right_1")
    a=forecast_reservation_swap(**kw)
    b=forecast_reservation_swap(**kw)
    assert a==b
    assert motor.observe()==old
    assert a.reserved_ordinals==(0,)
    assert len(a.projected)>0


def test_rejects_occupied_and_incompatible_candidate():
    motor=Coupled16ContactMotor()
    action=ContactAction(tuple(0. for _ in PHYSICAL_AXES))
    with pytest.raises(ValueError):
        forecast_reservation_swap(motor,action,{"left_1":(0,),"right_1":(1,)},
                                  {0:.05,1:.06},"left_1","right_1")
    with pytest.raises(ValueError):
        forecast_reservation_swap(motor,action,{"left_1":(0,)},{0:.05},
                                  "left_1","left_5")


def test_paired_reservation_multiplicity():
    motor=Coupled16ContactMotor()
    action=ContactAction(tuple(0. for _ in PHYSICAL_AXES),
                         (ThumbMode.FLAT,ThumbMode.FLAT),(False,False))
    result=forecast_reservation_swap(
        motor,action,{"left_5":(0,1)},{0:.050,1:.051},"left_5","right_5")
    assert result.reserved_ordinals==(0,1)
    assert [x[0] for x in result.projected if x[0] is not None]==[0,1]
