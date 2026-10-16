from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_persistent_motor_audit as persist
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as fixtures
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv, encode_n_key_hud_real_chart_observation


def test_true_frozen_control_and_physics_contract():
    assert persist.DT == p.CONTROL_DT == .010
    assert persist.PHYSICS == p.PHYSICS_DT == .001
    assert persist.PREVIEW > persist.DT
    assert persist.REPORT != fixtures.REPORT


def test_extended_hud_only_prediction_does_not_need_true_target_time():
    segment = persist._fixture(1200.)
    env = DiagnosticHudNKeyRealChartMotorEnv(segment, key_count=8)
    hud = encode_n_key_hud_real_chart_observation(env.reset())
    pred = persist.forecast_visible(hud)
    assert pred.eligible
    assert pred.floor_offsets == (1,)
    assert pred.times_s == pytest.approx((.050,), abs=2e-6)
    assert persist.forecast_visible(hud, horizon=.020).times_s == ()


def test_persistent_key_assignment_survives_next_hud_observation():
    segment = persist._fixture(1200.)
    env = DiagnosticHudNKeyRealChartMotorEnv(segment, key_count=8)
    cal = calibrate_n_key_press_lead(8, control_dt_s=.010, physics_dt_s=.001)
    planner = persist.PersistentMotorPlanner(cal)
    obs = env.reset()
    for _ in range(3):
        command = planner.action(encode_n_key_hud_real_chart_observation(obs))
        obs = env.step(command).observation
    assert planner.planned_unique == 1
    assert len(planner.commitments) == 1
    assert sum(abs(a) > 0 for a in command.values) == 1


@pytest.mark.parametrize('bpm', [600., 900., 1200., 2000.])
def test_real_evaluator_hit_all_on_basic_ordinary_chart(bpm):
    r = persist.simulate(bpm)
    assert r['evaluator_hits'] == r['targets'] == 18
    assert r['evaluator_misses'] == 0
    assert r['physical_keydowns'] == 18
    assert r['motor_resets_after_start'] == 0
    assert r['continuous_10ms_policy_1ms_physics'] is True
    assert r['unique_commitments'] == 18


def test_4000_bpm_reveals_limit_without_claiming_clear():
    r = persist.simulate(4000.)
    assert r['evaluator_hits'] == 15
    assert r['evaluator_misses'] == 3
    assert r['late_at_discovery'] > 0


def test_20000_bpm_refuses_unsupported_70ms_horizon():
    r = persist.simulate(20000.)
    assert r['hud_forecast_frames'] == 0
    assert r['evaluator_hits'] == 0
    assert r['physical_keydowns'] == 0
    assert r['hud_abstain_frames'] > 0


def test_abstains_on_invalid_hud():
    assert not persist.forecast_visible((0.,)).eligible
    with pytest.raises(ValueError):
        persist.forecast_visible((0.,), horizon=0.)


def test_report_is_deterministic_and_refuses_overwrite(tmp_path):
    cfg = replace(p.PrestigeConfig.load('configs/training/prestige1.toml'), output_dir=str(tmp_path))
    report = persist.run_audit(cfg)
    assert len(report['cases']) == len(persist.FIXTURE_BPM) == 6
    assert report['no_training'] is True
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    path = tmp_path / 'diagnostics' / 'prestige1' / persist.REPORT
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        persist.run_audit(cfg)
    assert path.read_bytes() == content
    assert persist.simulate(1200.) == next(x for x in report['cases'] if x['bpm'] == 1200.)
