from __future__ import annotations

from math import atan2, cos, pi, sin

import pytest
import torch

from dmdod.features.hud_features import encode_bpm
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_projective_orbit_audit as rp1
from dmdod.training import prestige1_autoplay_projective_unwrap_audit as u


def frame(angle:float,*,bpm:float=4000.,floor_x:float=0.,
          marker:bool=False,exit_angle:float=pi)->tuple[float,...]:
    data=[0.]*p.INPUT_DIM
    data[cd.MOTOR_FEATURES]=sin(angle)
    data[cd.MOTOR_FEATURES+1]=cos(angle)
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        j=cd.SLOT_START+slot*cd.FLOOR_DIM
        data[j:j+2]=[floor_x if slot==cd.CURRENT_SLOT else 0.,0.]
        data[j+2:j+6]=[0.,1.,sin(exit_angle),cos(exit_angle)]
        data[j+14]=1.
        if slot==cd.CURRENT_SLOT and marker:
            data[j+8]=1.
    data[cd.HUD_OFFSET]=encode_bpm(bpm)
    data[cd.HUD_OFFSET+1]=encode_bpm(bpm)
    return tuple(data)


def sequence(bpm:float=4000.,direction:int=1,*,floor_changed:bool=True,
             continuous_swaps:bool=True):
    step=direction*pi*bpm/60.*p.CONTROL_DT
    phase=.16
    return (
        frame(phase,bpm=bpm),
        frame(phase+step+(pi if continuous_swaps else 0),bpm=bpm),
        frame(phase+2*step+(2*pi if continuous_swaps else 0),
              bpm=bpm,floor_x=.2 if floor_changed else 0.),
        frame(phase+3*step+(3*pi if continuous_swaps else 0),
              bpm=bpm,floor_x=.2 if floor_changed else 0.),
    )


@pytest.mark.parametrize("bpm,direction,projected_deg",[
    (4000.,1,-60.),
    (4000.,-1,60.),
    (8000.,1,60.),
    (8000.,-1,-60.),
    (10000.,1,-60.),
    (10000.,-1,60.),
])
def test_bpm_guided_unwrapping_recovers_multiturn_motion(
    bpm:float,direction:int,projected_deg:float,
):
    data=sequence(bpm,direction)
    expected=direction*pi*bpm/60.*p.CONTROL_DT
    projective=rp1.projective_step(data[1],data[2])
    assert projective*180./pi==pytest.approx(projected_deg,abs=1e-8)
    step=u.unwrap_step(data[1],data[2])
    assert step.status=="eligible"
    assert step.radians==pytest.approx(expected,abs=1e-8)
    assert step.expected_abs_deg==pytest.approx(abs(expected)*180./pi)
    proposals,orig=u.causal_candidates(data)
    assert orig[2]=="floor_changed"
    for name in u.CANDIDATES:
        assert proposals[name][2].status=="eligible"
        assert proposals[name][2].step.radians==pytest.approx(expected,abs=1e-8)
        assert proposals[name][2].seconds is not None
        assert proposals[name][2].seconds>=0.


@pytest.mark.parametrize("bpm",[3000.,6000.,12000.])
def test_mathematically_ambiguous_direction_abstains(bpm:float):
    data=sequence(bpm)
    result=u.unwrap_step(data[1],data[2])
    assert result.radians is None
    assert result.status=="sign_or_alias_ambiguous"
    proposals,_=u.causal_candidates(data)
    for name in u.CANDIDATES:
        assert proposals[name][2].seconds is None
        assert proposals[name][2].status=="sign_or_alias_ambiguous"


def test_hud_speed_change_within_10ms_is_not_assumed_instantaneous():
    samples=sequence(4000.)
    now=list(samples[2])
    now[cd.HUD_OFFSET]=encode_bpm(8000.)
    changed=u.unwrap_step(samples[1],tuple(now))
    assert changed.radians is None
    assert changed.status=="bpm_changed_across_sample"


def test_original_eligible_high_bpm_geometry_may_wrap_direction_wrong():
    # At 7500BPM, the real +225deg step wraps to -135deg; frozen Geometry's
    # speed ratio is .6 and can be accepted although its sign is incorrect.
    frames=sequence(7500.,floor_changed=False,continuous_swaps=False)
    original,reason=cd.geometry_one(frames[1],frames[2])
    assert reason=="eligible_simple_floor"
    assert original is not None
    raw=cd._wrap(atan2(frames[2][cd.MOTOR_FEATURES],
                       frames[2][cd.MOTOR_FEATURES+1])-
                 atan2(frames[1][cd.MOTOR_FEATURES],
                       frames[1][cd.MOTOR_FEATURES+1]))
    assert raw*180/pi==pytest.approx(-135.)
    proposed,status=u.causal_candidates(frames)
    assert status[2]=="eligible_simple_floor"
    assert proposed["unwrap_speed"][2].status=="eligible"
    assert proposed["unwrap_speed"][2].step.radians*180/pi==pytest.approx(225.)


def test_low_bpm_frozen_eligible_is_not_overridden():
    frames=sequence(600.,floor_changed=False,continuous_swaps=False)
    predicted,status=u.causal_candidates(frames)
    assert status[2]=="eligible_simple_floor"
    for name in u.CANDIDATES:
        assert predicted[name][2].seconds is None
        assert predicted[name][2].status=="not_original_guard"


def test_original_special_event_still_abstains_after_floor_change():
    frames=sequence(4000.)
    changed=list(frames[2])
    start=cd.SLOT_START+cd.CURRENT_SLOT*cd.FLOOR_DIM
    changed[start+8]=1.
    data=(frames[0],frames[1],tuple(changed))
    choices,reason=u.causal_candidates(data)
    assert reason[2]=="floor_changed"
    for name in u.CANDIDATES:
        assert choices[name][2].seconds is None
        assert choices[name][2].status=="special_event"


def test_nonfinite_and_invalid_orbit_fail_closed():
    valid=sequence()
    with pytest.raises(ValueError,match="nonempty"):
        u.causal_candidates(())
    with pytest.raises(ValueError,match="263"):
        u.causal_candidates((valid[0][:-1],))
    bad=list(valid[2]);bad[3]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        u.causal_candidates((valid[0],valid[1],tuple(bad)))
    no_orbit=list(valid[2])
    no_orbit[cd.MOTOR_FEATURES]=0.
    no_orbit[cd.MOTOR_FEATURES+1]=0.
    assert u.unwrap_step(valid[1],tuple(no_orbit)).status=="invalid_orbit"


def test_causal_unwrap_predictions_unchanged_by_appended_future():
    frames=sequence(8000.)
    short=u.causal_candidates(frames[:3])
    long=u.causal_candidates(frames)
    assert short[0]=={k:v[:3] for k,v in long[0].items()}
    assert short[1]==long[1][:3]


def test_invalid_old_legacy_bpm_signal_is_not_sufficient_for_4000():
    frames=sequence(4000.)
    old=list(frames[2]);old[cd.HUD_OFFSET]=4.
    result=u.unwrap_step(frames[1],tuple(old))
    assert result.status=="bpm_changed_across_sample"


def test_metadata_guard_contract():
    assert u.REPORT!=rp1.REPORT
    assert "v2" in u.FORMAT
    assert p.CONTROL_DT==.010
    assert u.MAX_SPEED_REL_ERROR<.5
    assert u.MIN_UNIQUENESS_REL_GAP>0
