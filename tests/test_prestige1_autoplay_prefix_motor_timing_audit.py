from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_prefix_motor_timing_audit as timing
from dmdod.training import prestige1_autoplay_visible_prefix_audit as prefix


@pytest.fixture(scope="module")
def calibration():
    return calibrate_n_key_press_lead(8,control_dt_s=.010,physics_dt_s=.001)


@pytest.mark.parametrize("bpm,notes,hits,misses", [
    (2000.,72,72,0),(2000.,144,144,0),
    (4000.,18,17,1),(4000.,72,65,7),(4000.,144,127,17),
])
def test_replays_unmodified_visible_prefix_physics(bpm,notes,hits,misses,calibration):
    trace=timing.analyze_case(bpm,notes,calibration)
    assert trace['hits']==hits
    assert trace['misses']==misses
    assert trace['no_motor_reset_between_notes']
    assert trace['no_oracle_inputs_to_planner']
    assert len(trace['all_target_outcomes'])==notes
    assert sum(row['hit'] for row in trace['all_target_outcomes'])==hits
    assert sum(row['miss'] for row in trace['all_target_outcomes'])==misses
    assert trace['physical_keydowns']==len(trace['physical_keydown_trace'])
    assert trace['command_started_with_insufficient_cold_lead']>0
    assert all(row['cold_lead_shortfall_ms'] is None or row['cold_lead_shortfall_ms']>=0
               for row in trace['all_target_outcomes'])
    # Independently rerun the original unchanged planner baseline.
    old=prefix._run(bpm,notes,calibration,partial=True)
    assert (old['hits'],old['misses'],old['planned_unique']) == (
        trace['hits'],trace['misses'],trace['reservations'])


def test_4k_failure_types_and_intended_physical_delay(calibration):
    result=timing.analyze_case(4000.,144,calibration)
    assert result['miss_attribution']=={
        'keydown_claimed_different_target':16,'no_reservation':1}
    assert result['intended_keydown_mae_ms'] == pytest.approx(39.070865844,abs=.0001)
    assert result['matched_intents_with_down']>=120
    assert result['intent_claim_offset_histogram']['-1']>50
    assert result['first_misses'][0]['target_ordinal']==20
    assert result['first_misses'][0]['physical_keydown_claimed_target_ordinal']==19
    assert result['first_misses'][0]['keydown_minus_intended_ms']==pytest.approx(38.)
    assert result['first_misses'][0]['command_start_velocity_mm_s'] is not None


def test_4k_short_chart_is_not_a_clean_clear(calibration):
    result=timing.analyze_case(4000.,18,calibration)
    assert result['miss_attribution']=={'keydown_claimed_different_target':1}
    assert result['first_misses'][0]['target_ordinal']==18


def test_2k_hit_count_does_not_imply_zero_intended_latency(calibration):
    result=timing.analyze_case(2000.,144,calibration)
    assert result['hits']==144 and result['misses']==0
    assert result['intended_keydown_mae_ms']>15
    assert result['x_accuracy_percent']<100


def test_rejects_nonfixed_case_and_invalid_calibration(calibration):
    with pytest.raises(ValueError,match='fixed'):
        timing.analyze_case(4000.,200,calibration)
    assert timing.FIXTURES==((2000.,72),(2000.,144),(4000.,18),(4000.,72),(4000.,144))


def test_report_refuses_overwrite_and_keeps_stage_a_frozen(tmp_path):
    cfg=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                output_dir=str(tmp_path))
    report=timing.run_audit(cfg)
    assert len(report['cases'])==5
    assert report['no_training'] and report['no_checkpoint']
    assert report['no_validation'] and report['no_final']
    assert report['stage_a_fail_preserved']
    assert report['policy_control_dt_ms']==10
    assert report['physical_dt_ms']==1
    out=tmp_path/'diagnostics'/'prestige1'/timing.REPORT
    initial=out.read_bytes()
    with pytest.raises(FileExistsError):
        timing.run_audit(cfg)
    assert out.read_bytes()==initial
