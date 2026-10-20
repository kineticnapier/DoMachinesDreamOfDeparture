from __future__ import annotations

from dataclasses import replace
from math import atan2, cos, pi, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_halfturn_countdown_audit as audit
from dmdod.training import prestige1_autoplay_hybrid_audit as hybrid
from dmdod.training import prestige1_autoplay_readout as stage_t


BPM = 600.0
STEP = pi*BPM/60.0*p.CONTROL_DT


def frame(theta: float, *, special:bool=False) -> tuple[float,...]:
    values=torch.zeros(p.INPUT_DIM)
    values[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2]=torch.tensor(
        (sin(theta),cos(theta)))
    for slot in (cd.CURRENT_SLOT-1, cd.CURRENT_SLOT, cd.NEXT_SLOT):
        base=cd.SLOT_START+slot*cd.FLOOR_DIM
        values[base+2:base+6]=torch.tensor((0.0,1.0,sin(pi),cos(pi)))
        values[base+14]=1.
    values[cd.HUD_OFFSET]=torch.log2(torch.tensor(BPM/120.))
    if special:
        values[cd.SLOT_START+cd.CURRENT_SLOT*cd.FLOOR_DIM+8]=1.  # Twirl
    return tuple(float(v) for v in values.tolist())


def input_frames(*, mismatch:float=0.) -> tuple[tuple[float,...],...]:
    angle=2.1
    return (
        frame(angle),
        frame(angle+STEP),
        frame(angle+2*STEP+pi+mismatch),
        frame(angle+3*STEP+pi+mismatch),
    )


def truth_after_switch() -> float:
    # Physically visible current orbit angle must be used by the existing
    # Geometry rule. Do not call the candidate or its helper here.
    actual_angle=atan2(sin(2.1+2*STEP+pi),cos(2.1+2*STEP+pi))
    return ((pi-actual_angle)%(2*pi))/(pi*BPM/60.)


def seq(anchor:int,*,event_switch:bool, true_remaining:float|None=None):
    visible=torch.tensor(input_frames(),dtype=torch.float32)
    remaining=truth_after_switch() if true_remaining is None else true_remaining
    samples = ((.30,.29,remaining,remaining-.01) if event_switch
               else (.30,.29,.28,.27))
    return stage_t.TimingSequence(
        anchor=anchor,
        chart_sha256=(str(anchor)*64)[:64],
        visible=visible,
        remaining_s=torch.tensor(samples),
        near=torch.tensor((False,False,True,True)),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(0,0,1,1) if event_switch else (0,0,0,0),
    )


def trace(s:stage_t.TimingSequence):
    frames=tuple(tuple(float(x) for x in r) for r in s.visible.tolist())
    geometry=[None]
    status=["no_previous_hud_frame"]
    for k in range(1,len(frames)):
        value,reason=cd.geometry_one(frames[k-1],frames[k])
        geometry.append(value)
        status.append(reason)
    fallback=(.30,.29,.20,.19)
    chosen=tuple(x if x is not None else fallback[i]
                 for i,x in enumerate(geometry))
    return hybrid.HybridTrace(
        chosen_s=chosen,fallback_s=fallback,geometry_s=tuple(geometry),
        mode=tuple("geometry" if x is not None else "gru" for x in geometry),
        abstain_reasons=tuple(status),
    )


def test_halfturn_candidates_repair_original_motion_rejection():
    visible=input_frames()
    _,reason=cd.geometry_one(visible[1],visible[2])
    assert reason=="inconsistent_orbit_motion"
    candidates,status= audit.causal_halfturn_candidates(visible)
    assert status[2]=="inconsistent_orbit_motion"
    for name in audit.CANDIDATES:
        prediction=candidates[name][2]
        assert prediction.reason=="eligible"
        assert prediction.seconds==pytest.approx(truth_after_switch(),abs=2e-7)
        assert candidates[name][1].reason=="not_original_motion_abstention"
        assert candidates[name][1].seconds is None


def test_prior_agreement_rejects_anomalies_while_speed_only_is_separate():
    frames=input_frames(mismatch=.1)
    candidate,original=audit.causal_halfturn_candidates(frames)
    assert original[2]=="inconsistent_orbit_motion"
    assert candidate["halfturn_speed_only"][2].reason=="eligible"
    assert candidate["halfturn_prior_agreement"][2].reason=="prior_angle_mismatch"
    assert candidate["halfturn_prior_agreement"][2].seconds is None


def test_non_halfturn_motion_anomaly_abstains():
    raw=(frame(2.1),frame(2.1+STEP),frame(2.1+STEP+.9))
    options,reasons=audit.causal_halfturn_candidates(raw)
    assert reasons[2]=="inconsistent_orbit_motion"
    for name in audit.CANDIDATES:
        assert options[name][2].reason=="corrected_speed_not_credible"
        assert options[name][2].seconds is None


def test_candidates_are_prefix_causal_and_independent_of_oracle_metadata():
    visible=input_frames()
    prior,ps=audit.causal_halfturn_candidates(visible[:3])
    future,fs=audit.causal_halfturn_candidates(visible)
    assert ps==fs[:3]
    for name in audit.CANDIDATES:
        assert prior[name]==future[name][:3]
    s=seq(2,event_switch=True)
    modified=replace(
        s,event_indices=(9,8,7,6),
        remaining_s=torch.full((4,),.01),
        near=torch.zeros(4,dtype=torch.bool),
        mask=torch.zeros(4,dtype=torch.bool))
    data1=tuple(tuple(float(x) for x in row) for row in s.visible.tolist())
    data2=tuple(tuple(float(x) for x in row) for row in modified.visible.tolist())
    assert audit.causal_halfturn_candidates(data1)==audit.causal_halfturn_candidates(data2)


def test_false_positive_accounting_is_not_hidden_by_oracle_switch_selection():
    positive=seq(2,event_switch=True)
    negative=seq(3,event_switch=False)
    result=audit._score([positive,negative],{
        positive.anchor:trace(positive),
        negative.anchor:trace(negative),
    })
    assert result["label_only_pair_counts"]=={"switch":1,"same":5}
    assert result["original_frozen_geometry_abstention_reasons"]["switch"]=={
        "inconsistent_orbit_motion":1
    }
    for name in audit.CANDIDATES:
        scored=result["candidate_results"][name]
        assert scored["switch"]["eligible"]==1
        assert scored["switch"]["errors"]["candidate"]["mae_ms"] < .001
        assert scored["switch"]["errors"]["gru"]["count"]==1
        assert scored["same_event"]["eligible"]==1
        assert scored["same_event"]["false_positive_rate"]==pytest.approx(.2)
        assert scored["descriptive_precision_on_scored_pairs"]==pytest.approx(.5)
        assert scored["same_event"]["errors"]["candidate"]["count"]==1


def test_nonzero_candidate_error_is_preserved_not_automatically_adopted():
    positive=seq(2,event_switch=True,true_remaining=.75)
    negative=seq(3,event_switch=False)
    scored=audit._score([positive,negative],{
        2:trace(positive),3:trace(negative)
    })
    prediction=scored["candidate_results"]["halfturn_prior_agreement"]["switch"]
    assert prediction["eligible"]==1
    assert prediction["errors"]["candidate"]["mae_ms"] > 100.
    assert audit.REPORT != hybrid.REPORT


def test_input_guards_and_original_special_event_guard_unchanged():
    visible=input_frames()
    with pytest.raises(ValueError,match="nonempty"):
        audit.causal_halfturn_candidates(())
    with pytest.raises(ValueError,match="263"):
        audit.causal_halfturn_candidates((visible[0][:-1],))
    nan=tuple(float("nan") if i==0 else x for i,x in enumerate(visible[2]))
    with pytest.raises(ValueError,match="Nonfinite"):
        audit.causal_halfturn_candidates((visible[0],visible[1],nan))
    special_frames=(
        visible[0],
        frame(2.1 + STEP, special=True),
        frame(2.1 + 2 * STEP + pi, special=True),
    )
    opts,reasons=audit.causal_halfturn_candidates(special_frames)
    assert reasons[2]=="special_event"
    for name in audit.CANDIDATES:
        assert opts[name][2].reason=="not_original_motion_abstention"


def test_no_validation_or_training_scope():
    assert audit.PRIOR_AGREEMENT_DEG == 1.0
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
    assert audit.REPORT != stage_t.REPORT
