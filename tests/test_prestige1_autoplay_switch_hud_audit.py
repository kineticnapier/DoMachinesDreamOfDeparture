from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_hybrid_audit as hybrid
from dmdod.training import prestige1_autoplay_switch_hud_audit as audited
from dmdod.training import prestige1_autoplay_readout as stage_t


def hud(theta: float, *, current_angle=0., next_angle=0.,
        feedback=False, tile_bpm=0.) -> torch.Tensor:
    result = torch.zeros(p.INPUT_DIM)
    result[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2] = torch.tensor(
        [sin(theta),cos(theta)])
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        off = cd.SLOT_START+slot*cd.FLOOR_DIM
        angle = current_angle if slot == cd.CURRENT_SLOT else next_angle
        result[off+2:off+6] = torch.tensor([sin(angle),cos(angle),0.,-1.])
        result[off+14] = 1.
    result[cd.HUD_OFFSET] = tile_bpm
    result[cd.HUD_OFFSET+2] = float(feedback)
    return result


def sequence() -> stage_t.TimingSequence:
    # Switches at frames 2 and 6; changes in the HUD at both transitions.
    visible = torch.stack([
        hud(.2 + .05*k,
            current_angle=0. if k<2 else (1. if k<6 else 2.),
            next_angle=0. if k<2 else (1. if k<6 else 2.),
            feedback=(k==2))
        for k in range(8)
    ])
    return stage_t.TimingSequence(
        anchor=2, chart_sha256="a"*64, visible=visible,
        remaining_s=torch.tensor([.35,.34,.45,.44,.43,.42,.50,.49]),
        near=torch.zeros(8,dtype=torch.bool),
        mask=torch.ones(8,dtype=torch.bool),
        event_indices=(0,0,1,1,1,1,2,2),
    )


def trace() -> hybrid.HybridTrace:
    modes = ("geometry","geometry","gru","gru",
             "geometry","geometry","gru","geometry")
    chosen = (.35,.34,.09,.09,.43,.42,.09,.49)
    fallback = (.1,)*8
    geom = tuple(
        chosen[i] if modes[i] == "geometry" else None
        for i in range(8))
    return hybrid.HybridTrace(
        chosen_s=chosen, fallback_s=fallback, geometry_s=geom,
        mode=modes, abstain_reasons=tuple(
            "eligible_simple_floor" if m=="geometry" else "floor_changed"
            for m in modes),
    )


def test_hud_delta_reports_semantically_grouped_floor_changes():
    before = tuple(hud(.2).tolist())
    after = tuple(hud(.25,current_angle=1.,next_angle=2.,feedback=True).tolist())
    observed = audited.hud_delta(before,after)
    assert observed["orbit_xy_changed"] is True
    assert observed["current_floor_changed"] is True
    assert observed["next_floor_changed"] is True
    assert observed["behind_floor_changed"] is True  # fixture changes all surrounding slots
    assert observed["current_angles_changed"] is True
    assert observed["current_position_changed"] is False
    assert observed["feedback_changed"] is True
    assert observed["tile_bpm_changed"] is False
    assert observed["real_bpm_changed"] is False
    assert observed["current_visibility_changed"] is False
    assert observed["any_floor_changed"] is True
    assert cd.CURRENT_SLOT in observed["changed_floor_slots"]


def test_hud_delta_has_no_oracle_inputs_and_is_zero_for_identical_hud():
    sample = tuple(hud(.2).tolist())
    actual = audited.hud_delta(sample, sample)
    assert not any(actual[k] for k in audited.FLAGS)
    assert actual["orbit_delta_degrees"] == pytest.approx(0.)
    assert actual["current_floor_linf"] == 0.
    with pytest.raises(ValueError, match="263D"):
        audited.hud_delta(sample[:-1], sample)
    with pytest.raises(ValueError, match="Nonfinite"):
        audited.hud_delta(sample, tuple(float("nan") if i==0 else x
                                       for i,x in enumerate(sample)))


def test_pair_summary_stratifies_after_hud_delta_not_based_on_future_target():
    seq = sequence()
    scored = audited._pair_summary([seq])
    assert scored["pair_counts"] == {"true_switch":2,"same_event":5}
    cues = scored["cues"]
    assert cues["current_floor_changed"]["switch_cue_true"] == 2
    assert cues["current_floor_changed"]["same_event_cue_true"] == 0
    assert cues["current_floor_changed"]["descriptive_precision"] == 1.
    assert cues["current_floor_changed"]["descriptive_recall"] == 1.
    assert cues["orbit_xy_changed"]["switch_change_rate"] == 1.
    assert cues["orbit_xy_changed"]["same_event_change_rate"] == 1.
    assert cues["orbit_xy_changed"]["descriptive_precision"] == pytest.approx(2/7)
    assert cues["tile_bpm_changed"]["descriptive_precision"] is None


def test_pair_summary_uses_only_adjacent_observable_pairs():
    seq = sequence()
    masked = replace(seq,mask=torch.tensor([True, True,False,True,True,True,True,True]))
    result = audited._pair_summary([masked])
    assert result["pair_counts"] == {"true_switch":1,"same_event":4}
    with pytest.raises(RuntimeError, match="switch and non-switch"):
        audited._pair_summary([replace(seq,mask=torch.zeros(8,dtype=torch.bool))])


def test_post_switch_recovery_does_not_cross_oracle_target_boundary():
    seq = sequence()
    result = audited._post_switch([seq],{2:trace()})
    assert result["scored_true_event_switches"] == 2
    recovered = result["first_eligible_geometry_same_event"]
    assert recovered["reached"] == 2
    assert recovered["not_reached"] == 0
    assert recovered["delay_frames_distribution"] == {1:1,2:1}
    assert recovered["delay_ms"]["mae_ms"] == pytest.approx(15.)
    assert result["offset_from_switch"]["0"]["frames_with_same_oracle_event_and_observable"] == 2
    assert result["offset_from_switch"]["0"]["modes"] == {"gru":2}
    assert result["offset_from_switch"]["1"]["frames_with_same_oracle_event_and_observable"] == 2
    assert result["offset_from_switch"]["2"]["frames_with_same_oracle_event_and_observable"] == 1
    assert result["offset_from_switch"]["4"]["frames_with_same_oracle_event_and_observable"] == 0
    assert result["largest_first_frame_errors"][0]["anchor"] == 2


def test_no_geometry_before_next_event_is_explicitly_censored():
    seq = sequence()
    original = trace()
    modes = ("geometry","geometry","gru","gru","gru","gru","gru","gru")
    censored = replace(
        original,
        mode=modes,
        geometry_s=tuple(v if modes[i]=="geometry" else None
                         for i,v in enumerate(original.geometry_s)),
    )
    result = audited._post_switch([seq],{2:censored})
    recorded = result["first_eligible_geometry_same_event"]
    assert recorded["reached"] == 0
    assert recorded["not_reached"] == 2
    assert recorded["reason_if_not_reached"] == {
        "next_event_before_geometry":1,
        "sequence_end_before_geometry":1,
    }
    assert recorded["delay_ms"]["count"] == 0


def test_hidden_event_ids_cannot_change_observed_hud_pair_features():
    a = sequence()
    b = replace(
        a,event_indices=(1,2,3,4,5,6,7,8),
        remaining_s=torch.full((8,),.2), near=torch.ones(8,dtype=torch.bool))
    for k in range(1,8):
        left=audited.hud_delta(tuple(a.visible[k-1].tolist()),
                               tuple(a.visible[k].tolist()))
        right=audited.hud_delta(tuple(b.visible[k-1].tolist()),
                                tuple(b.visible[k].tolist()))
        assert left == right


def test_untrained_audit_namespace_does_not_replace_hybrid_or_stage_a():
    assert audited.REPORT != hybrid.REPORT
    assert audited.REPORT != stage_t.REPORT
    assert audited.OFFSETS == (0,1,2,3,4)
    assert audited.CHANGE_EPS == 1e-6
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
