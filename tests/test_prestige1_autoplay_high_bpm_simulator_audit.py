from __future__ import annotations

from math import isfinite

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as audit
from dmdod.training import prestige1_autoplay_projective_unwrap_audit as unwrap


def test_fixed_bpm_fixture_includes_20k_and_alias_boundaries():
    assert audit.BPM_POINTS[-1]==20000.
    assert audit.BPM_POINTS[0]==600.
    assert {1920.,2380.,2999.,3000.,3001.,4000.,6000.,8000.,10000.,12000.,16000.,20000.}.issubset(
        set(audit.BPM_POINTS))
    assert audit.PHYSICS_DT_S==.001
    assert audit.CONTROL_DT_S==.010


@pytest.mark.parametrize("bpm,reverse",[(4000.,False),(8000.,True),(20000.,False),(20000.,True)])
def test_real_physics_and_renderer_1ms_recovers_signed_angle(bpm,reverse):
    result=audit._one(bpm,reverse)
    assert result["target_count"]>=24
    assert result["episode_done"] is True
    assert result["independent_10ms_replay_verified"] is True
    assert result["hud_decoded_tile_bpm"]==pytest.approx(bpm,abs=.03)
    assert result["hud_encoded_tile_bpm"]>4.
    assert result["samples_1ms"]>10*result["samples_10ms"]-10
    one=result["phase_1ms"]
    assert one["posthoc_same_floor_pairs"]>0
    assert one["eligible"]>0
    assert one["wrong_above_0p05deg"]==0
    assert one["correct_within_0p05deg"]==one["eligible"]
    assert one["max_error_deg"]<.05


def test_real_20k_10ms_crosses_multiple_distinct_target_times():
    result=audit._one(20000.,False)
    co=result["next_target_timing_10ms"]
    assert co["pairs_crossing_two_or_more_targets"]>0
    assert co["max_targets_crossed_per_sample"]>=3
    assert result["phase_10ms"]["posthoc_same_floor_pairs"]==0


def test_simulator_output_is_deterministic_for_same_fixture():
    x=audit._one(20000.,False)
    y=audit._one(20000.,False)
    assert x==y
    assert len(x["chart_sha256"])==64


def test_frozen_student_control_stays_10ms_and_probe_is_not_retraining():
    assert p.CONTROL_DT==.01
    assert p.PHYSICS_DT==.001
    assert audit.FORMAT.endswith("readonly-v1")
    assert audit.REPORT!=unwrap.REPORT


def test_fixed_fixture_rejects_unspecified_speed():
    with pytest.raises(ValueError,match="fixed deterministic"):
        audit._fixture(25000.,reverse=False)


def test_20k_bpm_encoder_and_10ms_output_keep_numeric_finiteness():
    result=audit._one(20000.,True)
    assert isfinite(result["hud_encoded_tile_bpm"])
    assert result["hud_encoded_tile_bpm"]>7.
    assert result["samples_10ms"]>5
    for candidate in result["next_target_timing_10ms"]["candidates"].values():
        if candidate["mae_ms"] is not None:
            assert isfinite(candidate["mae_ms"])
