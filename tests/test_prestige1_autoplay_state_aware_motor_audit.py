from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.n_key import n_key_names
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_state_aware_motor_audit as model
from dmdod.training import prestige1_autoplay_visible_prefix_audit as prefix


@pytest.fixture(scope='module')
def calibration():
    return calibrate_n_key_press_lead(8,control_dt_s=.010,physics_dt_s=.001)


def test_actual_physical_8key_first_keydown_model_prediction(calibration):
    planner=model.ShadowDynamicsPlanner(calibration)
    first=planner._down_if_started_now(0,tuple([0.]*8))
    second=planner._down_if_started_now(1,tuple([0.]*8))
    assert first==pytest.approx(.039,abs=1e-10)
    assert second==pytest.approx(.042,abs=1e-10)
    assert planner.shadow.time_s==0.


def test_model_uses_only_own_actions_and_visible_hud(calibration):
    for bpm,notes in [(1200.,18),(4000.,72)]:
        result=model.simulate_case(bpm,notes,calibration,shadow=True)
        assert result['shadow_position_disagreements']==0
        assert result['shadow_rollout_count']>0
        assert result['keydown_count']==result['physical_keydowns']


@pytest.mark.parametrize('bpm,notes,hits,shadow_hits,baseline_x,shadow_x',[
    (1200.,18,18,18,95.28,100.),
    (2000.,18,18,18,85.00,100.),
    (4000.,18,17,17,63.06,63.06),
    (1200.,72,72,72,98.47,100.),
    (2000.,72,72,72,88.89,100.),
    (4000.,72,65,63,42.43,41.32),
    (1200.,144,144,144,99.24,100.),
    (2000.,144,144,144,88.54,100.),
    (4000.,144,127,127,41.25,38.44),
])
def test_fixed_physical_simulator_ab_regressions(
    bpm,notes,hits,shadow_hits,baseline_x,shadow_x,calibration,
):
    report=model.compare_case(bpm,notes,calibration)
    original,shadow=report['baseline'],report['shadow']
    assert original['hits']==hits
    assert shadow['hits']==shadow_hits
    assert original['misses']==notes-hits
    assert shadow['misses']==notes-shadow_hits
    assert original['x_accuracy']==pytest.approx(baseline_x,abs=.02)
    assert shadow['x_accuracy']==pytest.approx(shadow_x,abs=.02)
    assert shadow['shadow_position_disagreements']==0
    assert shadow['matched_armed_keydowns']<=shadow['physical_keydowns']
    assert report['hit_delta']==shadow_hits-hits


def test_model_based_2k_improves_precision_without_extra_physical_interface(calibration):
    report=model.compare_case(2000.,144,calibration)
    assert report['baseline']['hits']==report['shadow']['hits']==144
    assert report['shadow']['x_accuracy']==pytest.approx(100.)
    assert report['shadow']['intended_vs_actual_keydown_mae_ms']<2.
    assert report['shadow']['predicted_vs_actual_keydown_mae_ms']<4.
    assert model.PREDICTION_HORIZON_S==.09
    assert model.COMMAND_QUANTIZATION_MARGIN_S==.005
    assert p.CONTROL_DT==.010 and p.PHYSICS_DT==.001


def test_4k_prediction_accuracy_is_distinct_from_target_timing(calibration):
    report=model.compare_case(4000.,144,calibration)
    item=report['shadow']
    assert item['hits']==127
    assert report['x_accuracy_delta']<0.  # do not hide regression
    assert item['predicted_vs_actual_keydown_mae_ms']<3.
    assert item['intended_vs_actual_keydown_mae_ms']>35.


def test_no_16key_policy_conflation(calibration):
    assert p.KEY_COUNT==8
    assert len(n_key_names(16))==16
    assert len(n_key_names(8))==8
    assert model.FIXTURES==prefix.FIXTURES
    assert model.REPORT!=prefix.REPORT


def test_abort_on_unregistered_chart_and_no_overwrites(tmp_path):
    c=calibrate_n_key_press_lead(8,control_dt_s=.01,physics_dt_s=.001)
    with pytest.raises(ValueError,match='Unregistered'):
        model.simulate_case(4000.,100,c,shadow=True)
    cfg=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                output_dir=str(tmp_path))
    report=model.run_audit(cfg)
    assert len(report['cases'])==9
    assert report['no_training'] and report['no_student_checkpoint']
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    assert report['fixture_key_count']==8
    target=tmp_path/'diagnostics'/'prestige1'/model.REPORT
    before=target.read_bytes()
    with pytest.raises(FileExistsError):
        model.run_audit(cfg)
    assert target.read_bytes()==before
