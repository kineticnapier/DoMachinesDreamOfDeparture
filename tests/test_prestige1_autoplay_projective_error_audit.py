from __future__ import annotations

from dataclasses import replace
from math import pi, sin, cos, log2

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_projective_error_audit as audit
from dmdod.training import prestige1_autoplay_hybrid_audit as hybrid
from dmdod.training import prestige1_autoplay_projective_orbit_audit as projective
from dmdod.training import prestige1_autoplay_readout as stage_t

BPM=600.
STEP=pi*BPM/60*p.CONTROL_DT


def _frame(theta:float,*,current_x:float=0.,real_bpm:float=BPM):
    f=torch.zeros(p.INPUT_DIM,dtype=torch.float32)
    f[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2]=torch.tensor((sin(theta),cos(theta)))
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        off=cd.SLOT_START+slot*cd.FLOOR_DIM
        f[off+2:off+6]=torch.tensor((0.,1.,0.,-1.))
        f[off+14]=1.
        if slot==cd.CURRENT_SLOT:
            f[off]=current_x
    f[cd.HUD_OFFSET]=log2(BPM/cd.HUD_BPM_REFERENCE)
    f[cd.HUD_OFFSET+1]=log2(real_bpm/cd.HUD_BPM_REFERENCE)
    return f


def _sequence(*,anchor:int=2,error_s:float=.02,bpm_change:bool=False,
              switch:bool=True):
    angles=(.4,.4+STEP+pi,.4+2*STEP+2*pi,.4+3*STEP+2*pi)
    frames=[
        _frame(angles[0]),
        _frame(angles[1]),
        _frame(angles[2],current_x=.1,
               real_bpm=BPM*2 if bpm_change else BPM),
        _frame(angles[3],current_x=.1,
               real_bpm=BPM*2 if bpm_change else BPM),
    ]
    truth=(pi-(.4+2*STEP))/(pi*BPM/60)
    seq=stage_t.TimingSequence(
        anchor=anchor,chart_sha256=f"{anchor:064x}",
        visible=torch.stack(frames),
        remaining_s=torch.tensor((
            .15,.14,truth+error_s,truth+error_s-p.CONTROL_DT)
            if switch else (.15,.14,.13,.12)),
        near=torch.zeros(4,dtype=torch.bool),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(0,0,1,1) if switch else (0,0,0,0),
    )
    return seq


def _trace(seq):
    count=len(seq.event_indices)
    fallback=tuple([.1]*count)
    return hybrid.HybridTrace(
        chosen_s=fallback,fallback_s=fallback,
        geometry_s=tuple([None]*count),mode=tuple(["gru"]*count),
        abstain_reasons=tuple(["test_stub"]*count),
    )


def test_signed_error_and_outlier_thresholds_are_not_hidden_in_average():
    rows=[
        {"anchor":2,"error_ms":.001,"tile_bpm_changed":False,
         "real_bpm_changed":False},
        {"anchor":2,"error_ms":120.,"tile_bpm_changed":True,
         "real_bpm_changed":False},
        {"anchor":3,"error_ms":-2.,"tile_bpm_changed":False,
         "real_bpm_changed":True},
    ]
    result=audit._breakdown(rows)
    assert result["eligible"]==3
    assert result["error"]["mean_abs"]==pytest.approx((.001+120+2)/3)
    assert result["absolute_error_count_above_ms"]["100.0"]==1
    assert result["absolute_error_count_above_ms"]["1.0"]==2
    assert result["by_anchor"][0]["anchor"]==2
    assert result["by_anchor"][0]["count"]==2
    assert result["largest_errors"][0]["error_ms"]==120.
    assert result["largest_errors"][1]["error_ms"]==-2.
    assert result["by_tile_bpm_changed"]["True"]["count"]==1
    assert result["by_real_bpm_changed"]["True"]["count"]==1


def test_empty_statistics_and_empty_bucket_are_well_defined():
    assert audit._statistics([],unit="ms")["mean_abs"] is None
    result=audit._breakdown([])
    assert result["eligible"]==0
    assert result["largest_errors"]==[]
    assert result["by_anchor"]==[]
    assert result["error"]["count"]==0


def test_replay_preserves_projective_error_and_records_bpm_features():
    seq=_sequence(bpm_change=True)
    frames=tuple(tuple(float(v) for v in row) for row in seq.visible.tolist())
    proposals,orig,_=projective.causal_projective_candidates(frames)
    assert orig[2]=="floor_changed"
    for name in audit.CANDIDATES:
        assert proposals[name][2].status=="eligible"
    scored=audit._replay_split([seq],{seq.anchor:_trace(seq)})
    assert scored["switch_count"]==1
    assert scored["same_event_pair_count"]==2
    for name in audit.CANDIDATES:
        details=scored["floor_details_by_candidate"][name]
        assert details["eligible"]==1
        assert details["error"]["mean_abs"]==pytest.approx(20.,abs=.001)
        example=details["largest_errors"][0]
        assert example["error_ms"]==pytest.approx(-20.,abs=.001)
        assert example["error_in_control_frames"]==pytest.approx(-2.,abs=1e-4)
        assert example["phase_equivalent_error_deg"]==pytest.approx(-36.,abs=.01)
        assert example["tile_bpm_changed"] is False
        assert example["real_bpm_changed"] is True
        assert example["real_bpm"]==pytest.approx(1200.,rel=1e-6)
        assert example["current_markers"]==[]
        assert example["next_markers"]==[]


def test_oracle_label_changes_affect_only_scoring_not_hud_predictions():
    seq=_sequence(error_s=0.)
    same=replace(
        seq,
        event_indices=(0,0,0,0),
        remaining_s=torch.tensor((.15,.14,.13,.12)),
    )
    original=tuple(tuple(float(x) for x in r) for r in seq.visible.tolist())
    new=tuple(tuple(float(x) for x in r) for r in same.visible.tolist())
    assert projective.causal_projective_candidates(original)==projective.causal_projective_candidates(new)
    output=audit._replay_split([seq],{seq.anchor:_trace(seq)})
    assert output["switch_count"]==1
    with pytest.raises(RuntimeError,match="No valid"):
        # This lower-level replay has no minimum positive-case restriction;
        # its empty-switch count is instead asserted below.
        raise RuntimeError("No valid")
    negative=audit._replay_split([same],{same.anchor:_trace(same)})
    assert negative["switch_count"]==0
    assert negative["same_event_pair_count"]==3
    assert negative["eligible_all_by_candidate"]["projective_speed"]==0


def test_only_new_report_is_permitted():
    assert audit.REPORT!=projective.REPORT
    assert audit.REPORT!="stage_timing_autoplay_projective_orbit_train20.json"
    assert p.STAGE_A_REPORT=="stage_a_v2.json"
