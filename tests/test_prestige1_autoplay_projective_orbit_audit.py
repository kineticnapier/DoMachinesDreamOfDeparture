from __future__ import annotations

from dataclasses import replace
from math import atan2, cos, pi, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_halfturn_audit as old_halfturn
from dmdod.training import prestige1_autoplay_projective_orbit_audit as projective
from dmdod.training import prestige1_autoplay_hybrid_audit as hybrid
from dmdod.training import prestige1_autoplay_readout as stage_t

BPM=600.
STEP=pi*BPM/60.*p.CONTROL_DT


def hud(angle:float, *, floor_x:float=0., bpm:float=BPM,
        special:bool=False) -> tuple[float,...]:
    arr=torch.zeros(p.INPUT_DIM,dtype=torch.float32)
    arr[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2]=torch.tensor(
        (sin(angle),cos(angle)))
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        i=cd.SLOT_START+slot*cd.FLOOR_DIM
        arr[i+2:i+6]=torch.tensor((0.,1.,0.,-1.))
        arr[i+14]=1.
        if slot==cd.CURRENT_SLOT:
            arr[i]=floor_x
            if special:
                arr[i+8]=1.  # Twirl retains the original special_event guard.
    arr[cd.HUD_OFFSET] = torch.log2(torch.tensor(bpm/cd.HUD_BPM_REFERENCE))
    return tuple(float(v) for v in arr.tolist())


def swapped(*, floor_changed:bool=False, phase_extra:float=0.,
            bpm:float=BPM):
    d=pi*bpm/60.*p.CONTROL_DT
    return (
        hud(.4,bpm=bpm),
        hud(.4+d+pi,bpm=bpm),
        hud(.4+2*d+2*pi+phase_extra,
            floor_x=.1 if floor_changed else 0.,bpm=bpm),
        hud(.4+3*d+2*pi+phase_extra,
            floor_x=.1 if floor_changed else 0.,bpm=bpm),
    )


def true_countdown(bpm:float=BPM)->float:
    angle=.4+2*pi*bpm/60.*p.CONTROL_DT
    return (pi-angle)/(pi*bpm/60.)


def sequence(anchor:int,*,switch:bool, floor_changed:bool=False,
             true_s:float|None=None)->stage_t.TimingSequence:
    data=swapped(floor_changed=floor_changed)
    oracle=true_countdown() if true_s is None else true_s
    return stage_t.TimingSequence(
        anchor=anchor,chart_sha256=(str(anchor)*64)[:64],
        visible=torch.tensor(data,dtype=torch.float32),
        remaining_s=torch.tensor(
            (.1,.09,oracle,oracle-.01) if switch else (.1,.09,.08,.07)),
        near=torch.tensor((False,False,True,True)),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(0,0,1,1) if switch else (0,0,0,0),
    )


def old_trace(seq):
    huds=tuple(tuple(float(x) for x in row) for row in seq.visible.tolist())
    values=[None]
    reasons=["no_previous_hud_frame"]
    for k in range(1,len(huds)):
        value,reason=cd.geometry_one(huds[k-1],huds[k])
        values.append(value)
        reasons.append(reason)
    fallback=(.1,.09,.15,.14)
    chosen=tuple(v if v is not None else fallback[i]
                 for i,v in enumerate(values))
    return hybrid.HybridTrace(
        chosen_s=chosen,fallback_s=fallback,geometry_s=tuple(values),
        mode=tuple("geometry" if v is not None else "gru" for v in values),
        abstain_reasons=tuple(reasons),
    )


def test_two_angle_removes_arbitrary_pi_parity_without_event_id():
    for turns in (-3,-2,-1,0,1,2,3):
        prev=hud(.6)
        now=hud(.6+STEP+turns*pi)
        assert projective.projective_step(prev,now)==pytest.approx(STEP,abs=1e-6)
    a=hud(2.8)
    b=hud(2.8-STEP+3*pi)
    assert projective.projective_step(a,b)==pytest.approx(-STEP,abs=1e-6)


def test_consecutive_swaps_fix_prior_180_degree_residual():
    data=swapped()
    assert cd.geometry_one(data[1],data[2])[1]=="inconsistent_orbit_motion"
    old=old_halfturn.hud_halfturn_snapshot(data,2)
    assert old["halfturn_minus_prior_absolute_deg"]==pytest.approx(180.,abs=.001)
    candidate,original,observed=projective.causal_projective_candidates(data)
    assert original[2]=="inconsistent_orbit_motion"
    assert observed[2]["projective_step_difference_abs_deg"]<.001
    assert observed[2]["current_speed_ratio"]==pytest.approx(1.,rel=1e-5)
    assert observed[2]["prior_speed_ratio"]==pytest.approx(1.,rel=1e-5)
    for name in projective.CANDIDATES:
        assert candidate[name][2].status=="eligible"
        assert candidate[name][2].seconds==pytest.approx(true_countdown(),abs=2e-7)
    assert candidate["projective_speed"][1].seconds is None


def test_floor_changed_old_unexplained_counterfactual_now_eligible():
    data=swapped(floor_changed=True)
    assert cd.geometry_one(data[1],data[2])[1]=="floor_changed"
    before=old_halfturn.hud_halfturn_snapshot(data,2)
    assert before["halfturn_minus_prior_absolute_deg"]>179.99
    values,reasons,_=projective.causal_projective_candidates(data)
    assert reasons[2]=="floor_changed"
    for name in projective.CANDIDATES:
        assert values[name][2].status=="eligible"
        assert values[name][2].seconds==pytest.approx(true_countdown(),abs=2e-7)


def test_prior_consistency_rejects_unexplained_extra_angle():
    frames=swapped(phase_extra=.08)
    options,reasons,observed=projective.causal_projective_candidates(frames)
    assert reasons[2]=="inconsistent_orbit_motion"
    assert observed[2]["projective_step_difference_abs_deg"]>1.
    assert options["projective_speed"][2].status=="eligible"
    assert options["projective_prior_agreement"][2].status=="prior_projective_step_mismatch"
    assert options["projective_prior_agreement"][2].seconds is None


def test_aliasing_guard_abstains_before_cosine_double_angle_wrap():
    frames=swapped(bpm=2100.)
    options,reasons,observed=projective.causal_projective_candidates(frames)
    assert reasons[2]=="inconsistent_orbit_motion"
    assert observed[2]["alias_risk"] is True
    for name in projective.CANDIDATES:
        assert options[name][2].status=="alias_risk"
        assert options[name][2].seconds is None


def test_same_event_eligibility_is_counted_as_false_positive():
    positive=sequence(2,switch=True,floor_changed=True)
    negative=sequence(3,switch=False)
    result=projective._score_split([positive,negative],{
        2:old_trace(positive),3:old_trace(negative),
    })
    assert result["pair_totals"]=={"all_switch":1,"same_event":5}
    assert result["classified_floor_halfturn_unexplained"]==1
    assert result["classified_floor_halfturn_explained"]==0
    assert result["prior_carried_previous_step"]["floor_halfturn_unexplained"]["eligible"]==0
    for name in projective.CANDIDATES:
        summary=result["double_angle_candidates"][name]
        assert summary["original_floor"]["eligible"]==1
        assert summary["floor_halfturn_unexplained"]["eligible"]==1
        assert summary["floor_halfturn_unexplained"]["candidate_timing"]["mae_ms"]<.001
        assert summary["same_event"]["eligible"]==1
        assert summary["false_positive_rate"]==pytest.approx(.2)
        assert summary["descriptive_precision"]==pytest.approx(.5)


def test_correct_orbit_step_does_not_guarantee_correct_target_time():
    wrong=sequence(2,switch=True,floor_changed=True,true_s=.5)
    normal=sequence(3,switch=False)
    result=projective._score_split([wrong,normal],{
        2:old_trace(wrong),3:old_trace(normal)
    })
    for name in projective.CANDIDATES:
        row=result["double_angle_candidates"][name]["floor_halfturn_unexplained"]
        assert row["candidate_timing"]["mae_ms"]>100.


def test_causal_prefix_is_independent_of_oracle_labels_and_future_hud():
    data=swapped()
    short=projective.causal_projective_candidates(data[:3])
    long=projective.causal_projective_candidates(data)
    assert short[0]=={key:value[:3] for key,value in long[0].items()}
    assert short[1]==long[1][:3]
    assert short[2]==long[2][:3]
    sample=sequence(2,switch=True)
    different=replace(
        sample,event_indices=(10,11,12,13),
        remaining_s=torch.ones(4),near=torch.zeros(4,dtype=torch.bool),
        mask=torch.zeros(4,dtype=torch.bool),
    )
    one=tuple(tuple(float(v) for v in row) for row in sample.visible.tolist())
    two=tuple(tuple(float(v) for v in row) for row in different.visible.tolist())
    assert projective.causal_projective_candidates(one)==projective.causal_projective_candidates(two)


def test_original_special_guard_and_input_validation():
    data=swapped()
    with pytest.raises(ValueError,match="nonempty"):
        projective.causal_projective_candidates(())
    with pytest.raises(ValueError,match="263"):
        projective.causal_projective_candidates((data[0][:-1],))
    bad=tuple(float("nan") if i==3 else value for i,value in enumerate(data[2]))
    with pytest.raises(ValueError,match="Nonfinite"):
        projective.causal_projective_candidates((data[0],data[1],bad))
    # Twirl introduced at k=2 also changes the current-floor 15D HUD.
    # Frozen geometry checks floor_changed BEFORE special_event.
    newly_marked=(data[0],data[1],hud(.4+2*STEP+2*pi,special=True))
    values,reasons,_=projective.causal_projective_candidates(newly_marked)
    assert reasons[2]=="floor_changed"
    for name in projective.CANDIDATES:
        assert values[name][2].status=="special_event"
        assert values[name][2].seconds is None

    # When the Twirl marker was already present before the transition,
    # the original Geometry itself must abstain as special_event.
    already_marked=(
        hud(.4,special=True),
        hud(.4+STEP+pi,special=True),
        hud(.4+2*STEP+2*pi,special=True),
    )
    values,reasons,_=projective.causal_projective_candidates(already_marked)
    assert reasons[2]=="special_event"
    for name in projective.CANDIDATES:
        assert values[name][2].status=="not_target_guard"
        assert values[name][2].seconds is None


def test_zero_orbit_and_status_guards():
    valid=hud(.8)
    empty=tuple(0. if i in (cd.MOTOR_FEATURES,cd.MOTOR_FEATURES+1) else x
                for i,x in enumerate(valid))
    assert projective.projective_step(valid,empty) is None
    assert projective.REPORT!=old_halfturn.REPORT
    assert projective.TARGET_REASONS==("inconsistent_orbit_motion","floor_changed")
    assert projective.PRIOR_DEG==1.
    assert p.STAGE_A_REPORT=="stage_a_v2.json"
