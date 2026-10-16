from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_coupled_chart_patterns_audit as audit


@pytest.mark.parametrize('name,count,gaps,actions', [
    ('bends_1200', 72, 9, 0),
    ('events_1800', 96, 17, 4),
    ('tight_bends_4000', 108, 62, 0),
])
def test_predeclared_varied_angle_chart_geometry_and_action_events(name,count,gaps,actions):
    chart, metadata = audit.fixture(name)
    assert len(chart.targets)==count
    assert len({x.floor_index for x in chart.targets})==count
    assert metadata['angle_count']==count
    assert metadata['unique_angle_count']>=8
    assert metadata['consecutive_gaps_at_most_20ms']==gaps
    assert len(metadata['action_events'])==actions
    assert metadata['gap_min_ms']>0
    assert metadata['gap_max_ms']>metadata['gap_min_ms']
    assert len(metadata['sha256_adofai_json'])==64
    if name=='events_1800':
        assert {x['eventType'] for x in metadata['action_events']}=={'Twirl','Pause','SetSpeed'}
        assert any('Twirl' in x.event_markers for x in chart.chart.floors)
        assert any(x.pause_s>0 for x in chart.chart.floors)
        assert max(x.bpm for x in chart.targets)>min(x.bpm for x in chart.targets)
    elif name=='tight_bends_4000':
        assert metadata['gap_min_ms']<10


def test_oracle_separated_from_contact_observation():
    chart,_=audit.fixture('events_1800')
    env=CoupledContactRealChartEnv(chart)
    obs=env.reset()
    assert not hasattr(obs,'target_time_s')
    assert not hasattr(obs,'floor_index')
    assert not hasattr(obs,'target_ordinal')
    assert not hasattr(obs.motor,'target_time_s')
    assert not hasattr(obs.motor,'target_ordinal')
    assert len(obs.motor.logical_keys)==16
    assert len(obs.motor.physical.key_names)==12
    assert p.KEY_COUNT==8 and p.INPUT_DIM==263


@pytest.mark.parametrize('name,hit,miss', [
    ('bends_1200',72,0),
    ('events_1800',96,0),
    ('tight_bends_4000',108,0),
])
def test_high_bpm_oracle_8k_mode_with_real_physical_contact_attribution(name,hit,miss):
    r=audit.simulate(name,8)
    assert (r['hit'],r['miss'])==(hit,miss)
    assert r['continuous_motor']
    assert r['privileged_targets_given_only_to_teacher']
    assert r['too_early']==0
    assert not r['overloaded']
    assert r['physical_logical_keydowns']==r['scorer_judged_downs']
    assert r['physical_logical_keydowns']==r['keydowns_with_plan']+r['keydowns_without_plan']
    assert r['unjudged_due_to_terminal']==0
    for event in r['first_keydown_attributions']:
        assert event['key']
        assert isinstance(event['physical_keydown_ms'],float)
        if event['planned_ordinal'] is not None:
            assert event['planned_target_ms'] is not None
            assert event['planned_error_ms']==pytest.approx(
                event['physical_keydown_ms']-event['planned_target_ms'],abs=.000002)


def test_4000_bpm_planned_timing_error_is_not_equivalent_to_hitting_all_notes():
    r=audit.simulate('tight_bends_4000',8)
    assert r['hit']==108
    assert r['mean_abs_planned_keydown_error_ms']==pytest.approx(29.443,abs=.002)
    assert r['x_accuracy_percent']<60
    assert r['paired_reserved_axis_strokes']==17
    assert r['paired_contact_keydowns_with_plan']>=30
    assert r['paired_plan_target_gap_min_ms']>0
    assert r['paired_plan_target_gap_mean_ms']>0
    assert r['planned_vs_scorer_target_mismatch']==2
    assert r['keydowns_without_plan']==2


def test_4000_bpm_four_keys_per_hand_miss_and_mismatch_are_retained():
    a=audit.simulate('tight_bends_4000',4)
    b=audit.simulate('tight_bends_4000',5)
    assert (a['hit'],a['miss'])==(95,13)
    assert a['planned_vs_scorer_target_mismatch']==49
    assert a['paired_contact_keydowns_with_plan']==0
    assert (b['hit'],b['miss'])==(108,0)
    assert b['paired_reserved_axis_strokes']==0
    assert b['mean_abs_planned_keydown_error_ms']>30


def test_actual_two_contact_uses_same_physical_time_for_different_target_times():
    for mode in (6,7,8):
        r=audit.simulate('tight_bends_4000',mode)
        assert r['paired_reserved_axis_strokes']>0
        assert r['paired_plan_target_gap_min_ms']>0
        assert r['paired_contact_keydowns_with_plan']>0
        assert r['planned_error_over_20ms_count']>0
    # The purpose of the audit is to avoid hiding this gap behind credited Hits.
    r=audit.simulate('tight_bends_4000',8)
    events=[e for e in r['first_keydown_attributions'] if e['paired_physical_press']]
    assert events
    assert any(e['planned_target_ms'] != e['physical_keydown_ms'] for e in events)


def test_pair_selection_is_different_across_thumb_and_edge_models():
    thumb=audit.simulate('tight_bends_4000',6)
    edge=audit.simulate('tight_bends_4000',7)
    assert thumb['paired_reserved_axis_strokes']==9
    assert edge['paired_reserved_axis_strokes']==8
    assert edge['mean_abs_planned_keydown_error_ms'] < thumb['mean_abs_planned_keydown_error_ms']


def test_invalid_fixture_or_mode_fails_closed():
    with pytest.raises(ValueError,match='Unregistered'):
        audit.fixture('not_a_test')
    with pytest.raises(ValueError,match='Unsupported'):
        audit.simulate('events_1800',16)
    with pytest.raises(ValueError,match='Unregistered'):
        audit.simulate('made_up',8)


def test_report_immutable_without_student_or_validation(tmp_path):
    config=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                   output_dir=str(tmp_path))
    report=audit.run_audit(config)
    assert len(report['cases'])==15
    assert report['fixed_generator_not_existing_song_chart']
    assert report['physical_contact_keydowns_measured']
    assert report['planned_vs_scorer_attribution_separate']
    assert report['oracle_teacher_privileged_true_target_schedule']
    assert not report['hud_student_trained']
    assert not report['training']
    assert report['student_8key_263d_unchanged']
    assert report['stage_a_fail_preserved']
    assert not report['validation_evaluated']
    assert not report['final_evaluated']
    path=tmp_path/'diagnostics'/'prestige1'/audit.REPORT
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run_audit(config)
    assert path.read_bytes()==before
