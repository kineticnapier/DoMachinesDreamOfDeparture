from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.n_key import NKeyAction
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_persistent_motor_audit as baseline
from dmdod.training import prestige1_autoplay_visible_prefix_audit as prefix


@pytest.fixture(scope='module')
def calibration():
    return calibrate_n_key_press_lead(8, control_dt_s=.010, physics_dt_s=.001)


def _chart(bpm=4000., angles=None, actions=()):
    angles = angles or [0.] * 30
    data = {
        'angleData': angles,
        'settings': {'bpm': bpm, 'pitch': 100, 'countdownTicks': 0},
        'actions': list(actions),
    }
    return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps(data))))


def _hud_trace(segment):
    env = DiagnosticHudNKeyRealChartMotorEnv(segment, key_count=8,
        control_dt_s=.01, physics_dt_s=.001)
    obs = env.reset()
    for tick in range(int((segment.duration_s+.5)/.01)+3):
        yield tick*.01, tuple(encode_n_key_hud_real_chart_observation(obs))
        state = env.step(NKeyAction((0.,)*8))
        obs = state.observation
        if state.done:
            return
    raise RuntimeError('Unbounded physical test episode')


def test_terminal_prefix_claims_only_visible_events():
    segment = baseline._fixture(4000., floors=18)
    frames = list(_hud_trace(segment))
    # At ~220ms, a full 70ms horizon cannot be inspected, while the
    # next four floor arrivals remain visible and accurately timed.
    now, hud = frames[22]
    assert now == pytest.approx(.22)
    assert not baseline.forecast_visible(hud).eligible
    forecast = prefix.forecast_visible_prefix(hud)
    assert forecast.eligible and forecast.status == 'visible_prefix'
    assert forecast.stop_reason == 'insufficient_lookahead'
    assert forecast.floor_offsets == (1, 2, 3, 4)
    assert forecast.times_s == pytest.approx((.005,.020,.035,.050), abs=2e-6)
    assert all(ts <= baseline.PREVIEW for ts in forecast.times_s)


@pytest.mark.parametrize('bpm,floors',[(1200.,18),(2000.,18),(4000.,18),
                                      (4000.,72),(20000.,18)])
def test_every_claimed_normal_floor_prefix_matches_real_oracle(bpm,floors):
    # The candidate predicts from a single HUD. Privileged chart times are
    # consulted ONLY here, after the candidate returned its decision.
    segment = baseline._fixture(bpm, floors=floors)
    truth = tuple(t.episode_time_s for t in segment.targets)
    claimed = partial = 0
    for now, hud in _hud_trace(segment):
        guess = prefix.forecast_visible_prefix(hud)
        if not guess.eligible:
            assert not guess.times_s
            continue
        claimed += len(guess.times_s)
        partial += guess.status == 'visible_prefix'
        assert guess.times_s == tuple(sorted(guess.times_s))
        assert len(set(guess.floor_offsets)) == len(guess.floor_offsets)
        for t in guess.times_s:
            assert 0 < t <= baseline.PREVIEW + 1e-7
            assert any(abs((target-now)-t) <= 5e-5
                       for target in truth if target > now + 1e-7)
    assert claimed > 0
    if bpm in (4000.,20000.):
        assert partial > 0


def test_partial_prefix_excludes_unmodeled_future_pause():
    segment = _chart(actions=[{'floor':3, 'eventType':'Pause','duration':1}])
    hud = next(_hud_trace(segment))[1]
    prediction = prefix.forecast_visible_prefix(hud)
    assert prediction.eligible
    assert prediction.status == 'visible_prefix'
    assert prediction.floor_offsets == (1,2)
    assert prediction.times_s == pytest.approx((.015,.030),abs=2e-6)
    assert prediction.stop_reason == 'direction_agreed_prefix'


def test_immediate_future_twirl_is_not_guessed():
    segment = _chart(actions=[{'floor':1,'eventType':'Twirl'}])
    hud = next(_hud_trace(segment))[1]
    prediction = prefix.forecast_visible_prefix(hud)
    assert not prediction.eligible
    assert not prediction.times_s
    assert prediction.status == 'special_future_floor'


def test_ambiguous_rotation_direction_does_not_become_guess():
    segment = _chart(angles=[0.,90.,180.,270.] * 20)
    hud = next(_hud_trace(segment))[1]
    prediction = prefix.forecast_visible_prefix(hud)
    assert not prediction.eligible
    assert prediction.status == 'ambiguous_direction'


def test_invalid_hud_abstains_without_actions():
    assert prefix.forecast_visible_prefix((0.,)).status == 'invalid_hud'
    with pytest.raises(ValueError, match='horizon'):
        prefix.forecast_visible_prefix((0.,),0.)


@pytest.mark.parametrize('bpm,length,base_hits,new_hits',[
    (1200.,18,18,18),(2000.,18,18,18),(4000.,18,15,17),
    (1200.,72,72,72),(2000.,72,72,72),(4000.,72,61,65),
    (1200.,144,144,144),(2000.,144,144,144),(4000.,144,124,127),
])
def test_fixed_ab_comparison_with_unchanged_motor(bpm,length,base_hits,new_hits,calibration):
    item=prefix.compare_case(bpm,length,calibration)
    assert item['original']['hits']==base_hits
    assert item['visible_prefix']['hits']==new_hits
    assert item['hit_delta']==new_hits-base_hits
    assert item['original']['motor_resets_after_start']==0
    assert item['visible_prefix']['motor_resets_after_start']==0
    assert item['original']['planned_unique']<=item['visible_prefix']['planned_unique']
    assert item['original']['misses']+base_hits==length
    assert item['visible_prefix']['misses']+new_hits==length
    assert item['visible_prefix']['visible_prefix_new_commitments']>=0


def test_new_forecast_marks_incomplete_instead_of_relabeling_complete():
    segment = baseline._fixture(20000.,floors=18)
    first = next(_hud_trace(segment))[1]
    forecast=prefix.forecast_visible_prefix(first)
    assert forecast.status == 'visible_prefix'
    assert forecast.stop_reason == 'direction_agreed_prefix'
    assert len(forecast.times_s)==12
    assert forecast.times_s[-1]==pytest.approx(.036,abs=2e-6)
    assert baseline.PREVIEW==.070


def test_fixed_report_refuses_overwrite_and_never_trains(tmp_path):
    cfg=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                output_dir=str(tmp_path))
    report=prefix.run_audit(cfg)
    assert len(report['cases'])==9
    assert report['no_training'] and report['no_checkpoint']
    assert report['stage_a_fail_preserved']
    assert not report['validation_evaluated'] and not report['final_evaluated']
    assert next(r for r in report['cases'] if r['bpm']==4000 and r['targets']==144)['hit_delta']==3
    dest=tmp_path/'diagnostics'/'prestige1'/prefix.REPORT
    prior=dest.read_bytes()
    with pytest.raises(FileExistsError):
        prefix.run_audit(cfg)
    assert dest.read_bytes()==prior
