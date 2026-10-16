from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin
from types import SimpleNamespace

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_hybrid_audit as hy
from dmdod.training import prestige1_autoplay_first_frame_geometry_audit as a
from dmdod.training import prestige1_autoplay_readout as stage_t


def frame(theta: float, *, entry: float = 0.,
          special: bool = False, hidden_next: bool = False) -> tuple[float, ...]:
    hud = torch.zeros(p.INPUT_DIM)
    hud[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2] = torch.tensor(
        (sin(theta), cos(theta)))
    for slot in (cd.CURRENT_SLOT-1, cd.CURRENT_SLOT, cd.NEXT_SLOT):
        base = cd.SLOT_START+slot*cd.FLOOR_DIM
        angle = entry if slot == cd.CURRENT_SLOT else 0.
        hud[base+2:base+6] = torch.tensor(
            (sin(angle), cos(angle), sin(pi), cos(pi)))
        hud[base+14] = 1.
    if special:
        off = cd.SLOT_START+cd.CURRENT_SLOT*cd.FLOOR_DIM
        hud[off+8] = 1.  # Twirl
    if hidden_next:
        off = cd.SLOT_START+cd.NEXT_SLOT*cd.FLOOR_DIM
        hud[off+14] = 0.
    hud[cd.HUD_OFFSET] = 0.  # Tile BPM = 120
    return tuple(float(value) for value in hud.tolist())


def hud_sequence() -> tuple[tuple[float, ...], ...]:
    dtheta = 2*pi*.01
    return tuple(
        frame(2.5+i*dtheta, entry=0. if i<2 else .2)
        for i in range(4)
    )


def oracle_sequence() -> stage_t.TimingSequence:
    v = hud_sequence()
    exact = (pi - (2.5+2*2*pi*.01))/(2*pi)
    return stage_t.TimingSequence(
        anchor=2, chart_sha256="a"*64,
        visible=torch.tensor(v,dtype=torch.float32),
        remaining_s=torch.tensor((.30,.29,exact,exact-.01)),
        near=torch.tensor((False,False,True,True)),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(0,0,1,1),
    )


def trace(seq: stage_t.TimingSequence) -> hy.HybridTrace:
    truth_last = float(seq.remaining_s[3])
    return hy.HybridTrace(
        chosen_s=(.12,.12,.12,truth_last),
        fallback_s=(.12,)*4,
        geometry_s=(None,None,None,truth_last),
        mode=("gru","gru","gru","geometry"),
        abstain_reasons=("no_previous_hud_frame", "other",
                         "floor_changed", "eligible_simple_floor"),
    )


def test_existing_geometry_abstains_on_changed_floor_but_both_candidates_work():
    visible = hud_sequence()
    result, statuses = a.first_frame_candidates(visible)
    assert statuses[0] == "no_previous_hud_frame"
    assert statuses[2] == "floor_changed"
    expected = (pi - (2.5+2*2*pi*.01))/(2*pi)
    for name in a.CANDIDATES:
        candidate = result[name][2]
        assert candidate.status == "eligible"
        assert candidate.estimate_s == pytest.approx(expected,abs=5e-7)
        assert result[name][0].estimate_s is None
        assert result[name][1].estimate_s is None
        assert result[name][3].estimate_s is None


def test_carried_step_is_prior_visible_angular_change_not_oracle():
    dtheta = 2*pi*.01
    visible = (
        frame(2.5), frame(2.5+dtheta),
        frame(2.5+dtheta-dtheta),  # backward crossing orbit step
    )
    # Changed floor required to activate the counterfactual.
    changed = visible[:2]+(frame(2.5,entry=.2),)
    result, statuses = a.first_frame_candidates(changed)
    assert statuses[-1] == "floor_changed"
    raw = result["current_crossing_step"][-1]
    carry = result["carried_previous_step"][-1]
    assert carry.status == "eligible"
    assert carry.estimate_s == pytest.approx((pi-2.5)/(2*pi),abs=5e-7)
    # Crossing step is negative while carried prior step is positive.
    assert raw.estimate_s is None or abs(raw.estimate_s-carry.estimate_s) > .1


def test_hud_only_special_event_and_missing_floor_guards_still_apply():
    original = hud_sequence()
    altered = original[:2]+(
        frame(2.5+2*2*pi*.01,entry=.2,special=True),
    )
    proposals, original_status = a.first_frame_candidates(altered)
    assert original_status[2] == "floor_changed"
    for name in a.CANDIDATES:
        assert proposals[name][2].estimate_s is None
        assert proposals[name][2].status == "special_event"

    altered = original[:2]+(
        frame(2.5+2*2*pi*.01,entry=.2,hidden_next=True),
    )
    proposals, original_status = a.first_frame_candidates(altered)
    assert original_status[2] == "unavailable_ordinary_floor"
    for name in a.CANDIDATES:
        assert proposals[name][2].status == "not_floor_changed"


def test_model_proposals_are_prefix_causal_and_do_not_accept_oracle_labels():
    v = hud_sequence()
    partial, partial_status = a.first_frame_candidates(v[:3])
    full, full_status = a.first_frame_candidates(v)
    assert partial_status == full_status[:3]
    for name in a.CANDIDATES:
        assert partial[name] == full[name][:3]
    with pytest.raises(ValueError,match="263"):
        a.first_frame_candidates((v[0][:-1],))
    with pytest.raises(ValueError,match="Nonfinite"):
        a.first_frame_candidates((v[0],tuple(float("nan") if i==0 else x
                                              for i,x in enumerate(v[1]))))
    with pytest.raises(ValueError,match="nonempty"):
        a.first_frame_candidates(())


def test_scoring_is_posthoc_and_no_same_event_frames_added_to_switch():
    seq = oracle_sequence()
    report = a._score_split([seq],{2:trace(seq)})
    assert report["oracle_true_switches"] == 1
    assert report["original_floor_changed_at_switch"] == 1
    assert report["crossing_orbit_motion"]["with_valid_preceding_step"] == 1
    for name in a.CANDIDATES:
        row = report["candidates_on_same_oracle_switch_frames"][name]
        assert row["eligible_switches"] == 1
        assert row["absolute_time_error"]["mae_ms"] < .001
        assert row["same_frame_frozen_gru"]["count"] == 1
        assert row["near_time_error"]["count"] == 1
        assert row["next_frame_pair_count"] == 1
        assert row["next_frame_actual_geometry_countdown_residual"]["mae_ms"] < .001


def test_hidden_oracle_labels_change_scores_not_geometry_candidates():
    first = oracle_sequence()
    changed = replace(
        first,
        event_indices=(9,9,9,9),
        remaining_s=torch.full((4,),.100),
        near=torch.zeros(4,dtype=torch.bool),
        mask=torch.zeros(4,dtype=torch.bool),
    )
    v1 = tuple(tuple(float(x) for x in row) for row in first.visible.tolist())
    v2 = tuple(tuple(float(x) for x in row) for row in changed.visible.tolist())
    assert a.first_frame_candidates(v1) == a.first_frame_candidates(v2)
    with pytest.raises(RuntimeError,match="No true note switches"):
        a._score_split([changed],{2:trace(first)})


def test_missing_old_hud_report_is_fatal_and_no_write(tmp_path):
    source = SimpleNamespace(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="previous HUD switch"):
        a._verify_prior_report(source,SimpleNamespace(),{}, {}, {})
    assert list(tmp_path.iterdir()) == []


def test_audit_does_not_replace_model_or_prior_reports():
    assert a.REPORT != hy.REPORT
    assert a.REPORT != stage_t.REPORT
    assert a.FORMAT.endswith("train20-v1")
    assert a.CANDIDATES == ("current_crossing_step","carried_previous_step")
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
