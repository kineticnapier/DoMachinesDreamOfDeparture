from __future__ import annotations

from dataclasses import replace
from math import cos, log2, pi, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_halfturn_audit as audit
from dmdod.training import prestige1_autoplay_readout as stage_t


def frame(theta: float, *, current_entry: float = 0.,
          tile_bpm: float = 600.) -> tuple[float,...]:
    hud = torch.zeros(p.INPUT_DIM)
    hud[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2] = torch.tensor(
        [sin(theta),cos(theta)])
    for slot in (cd.CURRENT_SLOT-1,cd.CURRENT_SLOT,cd.NEXT_SLOT):
        base = cd.SLOT_START+slot*cd.FLOOR_DIM
        entry = current_entry if slot==cd.CURRENT_SLOT else 0.
        hud[base+2:base+6] = torch.tensor([
            sin(entry),cos(entry),sin(pi),cos(pi)])
        hud[base+14] = 1.
    hud[cd.HUD_OFFSET] = log2(tile_bpm/120.)
    return tuple(float(x) for x in hud.tolist())


def halfturn_frames() -> tuple[tuple[float,...],...]:
    step=pi/10  # BPM600 x control10ms => 18 degrees per frame
    return (
        frame(.5),
        frame(.5+step),
        frame(.5+2*step+pi),
        frame(.5+3*step+pi),
    )


def seq(*, halfturn: bool) -> stage_t.TimingSequence:
    source=halfturn_frames()
    if not halfturn:
        d=pi/10
        source=(
            frame(.5),
            frame(.5+d),
            frame(.5+d+.6),  # non-pi angular anomaly
            frame(.5+2*d+.6),
        )
    return stage_t.TimingSequence(
        anchor=2,chart_sha256="a"*64,
        visible=torch.tensor(source,dtype=torch.float32),
        remaining_s=torch.tensor((.35,.34,.33,.32)),
        near=torch.tensor((False,False,False,False)),
        mask=torch.ones(4,dtype=torch.bool),
        event_indices=(0,0,1,1),
    )


def test_exact_halfturn_generates_apparent_ninefold_speed_not_real_acceleration():
    frames=halfturn_frames()
    _,status=cd.geometry_one(frames[1],frames[2])
    assert status=="inconsistent_orbit_motion"
    snapshot=audit.hud_halfturn_snapshot(frames,2)
    assert snapshot["current_floor_unchanged"] is True
    assert snapshot["crossing_speed_ratio"]==pytest.approx(9.,rel=2e-6)
    assert snapshot["prior_speed_ratio"]==pytest.approx(1.,rel=2e-6)
    assert snapshot["halfturn_corrected_speed_ratio"]==pytest.approx(1.,rel=2e-6)
    assert snapshot["halfturn_minus_prior_absolute_deg"] < 0.001
    assert abs(snapshot["entry_exit_halfturn_residual_deg"]) < 0.001
    assert snapshot["crossing_step_deg"]==pytest.approx(-162.,abs=.001)
    assert snapshot["prior_step_deg"]==pytest.approx(18.,abs=.001)
    assert snapshot["halfturn_corrected_step_deg"]==pytest.approx(18.,abs=.001)


def test_non_halfturn_anomaly_does_not_pass_180_degree_hypothesis():
    frames=tuple(tuple(float(x) for x in row)
                 for row in seq(halfturn=False).visible.tolist())
    observed=audit.hud_halfturn_snapshot(frames,2)
    assert observed["halfturn_minus_prior_absolute_deg"] > 30.
    assert observed["corrected_within_original_speed_band"] is False
    assert observed["current_floor_unchanged"] is True


def test_counterfactual_does_not_know_oracle_events_or_labels():
    sample=seq(halfturn=True)
    other=replace(
        sample,event_indices=(10,11,12,13),
        remaining_s=torch.full((4,),.01),
        mask=torch.zeros(4,dtype=torch.bool),
    )
    original=tuple(tuple(float(x) for x in row)
                   for row in sample.visible.tolist())
    changed=tuple(tuple(float(x) for x in row)
                  for row in other.visible.tolist())
    assert audit.hud_halfturn_snapshot(original,2)==audit.hud_halfturn_snapshot(changed,2)
    prior=audit._describe([sample])
    assert prior["total_labelled_switches_with_prior_step"]==1
    assert prior["cohorts"]["original_inconsistent_orbit_motion"]["events"]==1
    assert prior["cohorts"]["original_inconsistent_orbit_motion"]["halfturn_residual_threshold_counts"]["1.0"]==1
    with pytest.raises(RuntimeError,match="No label-only true switches"):
        audit._describe([other])


def test_input_guards_check_three_visible_frames():
    frames=halfturn_frames()
    with pytest.raises(ValueError,match="three successive"):
        audit.hud_halfturn_snapshot(frames,1)
    with pytest.raises(ValueError,match="263D"):
        audit.hud_halfturn_snapshot((frames[0],frames[1],frames[2][:-1]),2)
    corrupted=tuple(float("nan") if i==0 else v for i,v in enumerate(frames[2]))
    with pytest.raises(ValueError,match="Nonfinite"):
        audit.hud_halfturn_snapshot((frames[0],frames[1],corrupted),2)


def test_empty_cohort_has_none_for_stats():
    observed=audit._cohort([])
    assert observed["events"]==0
    assert observed["halfturn_corrected_speed_ratio"]["mean"] is None
    assert observed["halfturn_residual_threshold_counts"]["1.0"]==0


def test_source_names_and_original_geometry_are_unchanged():
    assert audit.REPORT != cd.REPORT
    assert audit.COHORTS == ("original_inconsistent_orbit_motion","original_floor_changed")
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
