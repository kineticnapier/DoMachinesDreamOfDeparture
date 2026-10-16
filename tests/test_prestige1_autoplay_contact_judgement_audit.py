from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.adofai_rules import TimingJudgement
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_contact_judgement_audit as audit


@pytest.mark.parametrize('name,hit,te,overload,margins', [
    ('600_single_finger_control', 1, 0, False, ['perfect']),
    ('600_edge_pair', 1, 1, False, ['perfect','too_early']),
    ('600_flat_thumb_pair', 1, 1, False, ['perfect','too_early']),
    ('600_bilateral_edge_pairs', 1, 3, True,
     ['perfect','too_early','too_early','fail_overload']),
    ('1200_edge_pair', 2, 0, False, ['perfect','very_early']),
    ('4000_edge_pair', 2, 0, False, ['perfect','perfect']),
    ('4000_bilateral_edge_pairs', 4, 0, False,
     ['late_perfect','perfect','perfect','perfect']),
])
def test_original_adofai_scorer_on_real_coupled_contact_events(name,hit,te,overload,margins):
    row=audit.run_case(name)
    assert row['hits']==hit
    assert row['too_early']==te
    assert row['overloaded'] is overload
    assert row['judgement_sequence']==margins
    assert row['raw_physical_contact_downs']==len(margins)
    assert row['judged_keydowns']==len(margins)
    assert row['hits']+row['too_early']==len(margins)
    assert {x['time_ms'] for x in row['judgement_events']} and len({
        x['time_ms'] for x in row['judgement_events']})==1
    assert row['multipress_state_machine_implemented'] is False
    assert row['motor_12_axis_16_contact'] is True


def test_600_single_and_coupled_edge_identify_one_surplus_te():
    solo=audit.run_case('600_single_finger_control')
    paired=audit.run_case('600_edge_pair')
    assert solo['hits']==paired['hits']==1
    assert solo['too_early']==0 and paired['too_early']==1
    assert paired['judgement_events'][0]['key']=='left_7'
    assert paired['judgement_events'][1]['key']=='left_8'
    assert paired['judgement_events'][0]['evaluator_target_ordinal']==1
    assert paired['judgement_events'][1]['evaluator_target_ordinal']==2
    assert paired['overload_gauge']==pytest.approx(.5)
    assert paired['overloaded'] is False


def test_bilateral_4_contact_chord_triggers_failoverload_only_on_third_te():
    row=audit.run_case('600_bilateral_edge_pairs')
    assert [x['overload_after'] for x in row['judgement_events']]==[0.,.5,1.,1.5]
    assert row['fail_overloads']==1
    assert row['overloaded']
    assert row['judgement_events'][-1]['margin']==TimingJudgement.FAIL_OVERLOAD.value
    # The unclaimed next target remains the same during all 3 early presses.
    assert [x['evaluator_target_ordinal'] for x in row['judgement_events']]==[1,2,2,2]


def test_at_high_bpm_surplus_simultaneous_press_claims_next_note():
    for bpm in (1200,4000):
        row=audit.run_case(f'{bpm}_edge_pair')
        assert row['hits']==2 and row['too_early']==0
        assert [x['evaluator_target_ordinal'] for x in row['judgement_events']]==[1,2]
        assert row['judgement_events'][0]['time_ms']==row['judgement_events'][1]['time_ms']
    # The present real-chart scorer lacks Multipress/OverPress queue semantics.
    assert audit.run_case('4000_bilateral_edge_pairs')['hits']==4


def test_actual_contact_event_times_are_aligned_to_first_target_not_fabricated():
    for case in audit.CASES:
        row=audit.run_case(case.name)
        assert abs(row['judgement_events'][0]['time_ms']-row['first_target_ms'])<=30
        assert row['second_target_ms']>row['first_target_ms']
        assert row['oracle_motor_control_for_diagnostic'] is True


def test_invalid_scenario_and_frozen_contract():
    with pytest.raises(ValueError,match='predeclared'):
        audit.run_case('not-a-case')
    assert p.KEY_COUNT==8 and p.INPUT_DIM==263
    assert p.CONTROL_DT==.010 and p.PHYSICS_DT==.001
    assert audit.PRESS_FRAMES==10


def test_diagnostic_report_refuses_overwrite_and_does_not_train(tmp_path):
    config=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                   output_dir=str(tmp_path))
    report=audit.run_audit(config)
    assert len(report['cases'])==len(audit.CASES)==7
    assert report['no_multipress_or_overpress_queue_semantics'] is True
    assert report['no_full_game_env_step'] is True
    assert report['no_student_training'] is True
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    destination=tmp_path/'diagnostics'/'prestige1'/audit.REPORT
    previous=destination.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run_audit(config)
    assert destination.read_bytes()==previous
