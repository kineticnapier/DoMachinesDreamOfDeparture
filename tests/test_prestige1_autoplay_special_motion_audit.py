from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_special_motion_audit as audit
from dmdod.training import prestige1_autoplay_first_frame_geometry_audit as first
from dmdod.training import prestige1_autoplay_readout as stage_t


def hud(theta: float, *, entry: float = 0.,
        current_markers=(), next_markers=(), tile_log=0.) -> tuple[float,...]:
    values = torch.zeros(p.INPUT_DIM)
    values[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2] = torch.tensor(
        [sin(theta),cos(theta)])
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        start = cd.SLOT_START+slot*cd.FLOOR_DIM
        angle = entry if slot==cd.CURRENT_SLOT else 0.
        values[start+2:start+6] = torch.tensor(
            [sin(angle),cos(angle),sin(pi),cos(pi)])
        values[start+14] = 1.
        selected = (
            current_markers if slot == cd.CURRENT_SLOT else
            next_markers if slot == cd.NEXT_SLOT else ()
        )
        for name in selected:
            index = audit.MARKERS.index(name)
            values[start+cd.SKIP_MARKER_FROM+index] = 1.
    values[cd.HUD_OFFSET] = tile_log
    return tuple(float(x) for x in values.tolist())


def event_sequence(*, anchor:int, special:bool) -> stage_t.TimingSequence:
    step = 2*pi*.01
    if special:
        frames = [
            hud(2.1),hud(2.1+step),
            hud(2.1+2*step, entry=.2,
                current_markers=("Pause",),
                next_markers=("Checkpoint",)),
            hud(2.1+3*step, entry=.2,
                current_markers=("Pause",),
                next_markers=("Checkpoint",)),
        ]
    else:
        # Stable floor, but >1.5x expected crossing motion: original guard.
        frames = [
            hud(2.1),hud(2.1+step),
            hud(2.1+step+.5),hud(2.1+step+.5+step),
        ]
    return stage_t.TimingSequence(
        anchor=anchor, chart_sha256=str(anchor)*64,
        visible=torch.tensor(frames,dtype=torch.float32),
        remaining_s=torch.tensor([.2,.19,.35,.34]),
        near=torch.tensor([True,True,False,False]),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(1,1,2,2),
    )


def test_marker_position_and_overlap_are_explicit_and_named():
    values = hud(
        2.0, current_markers=("Pause","Checkpoint"),
        next_markers=("Checkpoint","SetSpeed"))
    snap = audit._marker_snapshot(values)
    assert snap["current"]["Pause"] is True
    assert snap["current"]["Checkpoint"] is True
    assert snap["next"]["Checkpoint"] is True
    assert snap["next"]["SetSpeed"] is True
    for name in audit.MARKERS:
        if name not in ("Pause","Checkpoint"):
            assert snap["current"][name] is False
    sample = {
        "snapshot":{"marker_flags":snap},"anchor":2,"frame":3,
        "original_status":"floor_changed","carried_status":"special_event",
    }
    report = audit._marker_report([sample])
    assert report["marker_occurrences_by_slot"]["current"]["Pause"] == 1
    assert report["marker_occurrences_by_slot"]["next"]["SetSpeed"] == 1
    assert report["same_marker_on_both_slots"]["Checkpoint"] == 1
    assert report["marker_patterns_exclusive"] == {
        "next:SetSpeed,current:Pause,current:Checkpoint,next:Checkpoint":1
    }
    assert report["count"] == 1


def test_hud_motion_reports_crossing_and_prior_ratios_against_tile_bpm():
    step = 2*pi*.01
    sequence = (hud(1.8),hud(1.8+step),hud(1.8+step+.5))
    m = audit.hud_motion_snapshot(sequence,2)
    assert m["crossing_step_deg"] == pytest.approx(.5*180/pi,abs=.001)
    assert m["prior_step_deg"] == pytest.approx(step*180/pi,abs=.001)
    assert m["crossing_speed_ratio"] == pytest.approx(.5/step,rel=1e-5)
    assert m["prior_to_current_speed_ratio"] == pytest.approx(1.,rel=1e-5)
    assert m["crossing_speed_class"] == "above_one_and_half_expected"
    assert m["prior_current_speed_class"] == "within_original_guard"
    assert m["direction_flip"] is False
    assert m["tile_bpm_changed"] is False


def test_bpm_shift_changes_current_ratio_not_prior_reference_ratio():
    step=2*pi*.01
    f0=hud(1.8)
    f1=hud(1.8+step)
    f2=hud(1.8+2*step,tile_log=1.)  # Tile BPM doubles
    m=audit.hud_motion_snapshot((f0,f1,f2),2)
    assert m["tile_bpm_changed"] is True
    assert m["prior_to_current_speed_ratio"] == pytest.approx(.5,rel=1e-5)
    assert m["prior_to_previous_speed_ratio"] == pytest.approx(1.,rel=1e-5)


def test_immutable_hud_precomputation_labels_special_and_motion_separately():
    special=event_sequence(anchor=2,special=True)
    inconsistent=event_sequence(anchor=3,special=False)
    report=audit._describe_split([special,inconsistent])
    assert report["true_note_switches"] == 2
    assert report["original_geometry_first_rejection"]["floor_changed"] == 1
    assert report["original_geometry_first_rejection"]["inconsistent_orbit_motion"] == 1
    assert report["counterfactual_carried_first_rejection"]["special_event"] == 1
    assert report["counterfactual_carried_first_rejection"]["not_floor_changed"] == 1
    assert report["carried_special_event"]["marker_breakdown"]["count"] == 1
    assert report["carried_special_event"]["marker_breakdown"]["marker_occurrences_by_slot"]["current"]["Pause"] == 1
    assert report["carried_special_event"]["marker_breakdown"]["marker_occurrences_by_slot"]["next"]["Checkpoint"] == 1
    assert report["original_inconsistent_orbit_motion"]["count"] == 1
    assert report["original_inconsistent_orbit_motion"]["speed_classes"]["crossing_speed_class"] == {
        "above_one_and_half_expected":1
    }
    assert report["original_inconsistent_orbit_motion"]["speed_classes"]["prior_current_speed_class"] == {
        "within_original_guard":1
    }
    assert report["floor_changed_then_carried_motion_inconsistent"]["count"] == 0
    assert len(report["original_vs_carried_cross_tab"]) == 2


def test_change_oracle_event_ids_does_not_change_hud_or_frozen_candidates():
    s=event_sequence(anchor=2,special=True)
    changed=replace(s,event_indices=(1,1,1,1),
                    remaining_s=torch.full((4,),.100),
                    near=torch.zeros(4,dtype=torch.bool))
    original=tuple(tuple(float(x) for x in row) for row in s.visible.tolist())
    altered=tuple(tuple(float(x) for x in row) for row in changed.visible.tolist())
    assert first.first_frame_candidates(original) == first.first_frame_candidates(altered)
    for k in range(1,len(original)):
        assert audit.hud_motion_snapshot(original,k) == audit.hud_motion_snapshot(altered,k)
    with pytest.raises(RuntimeError,match="No oracle-labelled note switches"):
        audit._describe_split([changed])


def test_no_finite_motion_input_rejects_before_scoring():
    v0=hud(1.8)
    with pytest.raises(ValueError,match="index"):
        audit.hud_motion_snapshot((v0,),0)
    with pytest.raises(ValueError,match="263D"):
        audit.hud_motion_snapshot((v0,v0[:-1]),1)
    invalid=tuple(float("nan") if i==1 else x for i,x in enumerate(v0))
    with pytest.raises(ValueError,match="Nonfinite"):
        audit.hud_motion_snapshot((v0,invalid),1)


def test_empty_population_reports_are_well_defined_without_division_by_zero():
    assert audit._motion_report([])["count"] == 0
    assert audit._motion_report([])["quantities"]["crossing_speed_ratio"]["mean_abs"] is None
    markers = audit._marker_report([])
    assert markers["count"] == 0
    assert all(value == 0 for value in markers["marker_occurrences_by_slot"]["next"].values())


def test_frozen_namespaces_and_original_guard_constants_untouched():
    assert audit.REPORT != first.REPORT
    assert audit.REPORT != stage_t.REPORT
    assert audit.SPEED_BAND == (0.5,1.5)
    assert audit.MARKERS == (
        "Twirl","SetSpeed","MultiPlanet","Pause","Checkpoint","SetFloorIcon"
    )
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
