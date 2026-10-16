from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.contact_16k import (
    PHYSICAL_AXES, ContactAction, Coupled16ContactMotor, ThumbMode,
)
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_coupled_contact_sequences_audit as seq


@pytest.mark.parametrize('name,extra,total', [
    ('edge_requested_single_7',1,2),
    ('edge_requested_single_8',1,2),
    ('edge_requested_pair',0,2),
    ('thumb_requested_tip',0,1),
    ('thumb_requested_flat_pair',0,2),
    ('thumb_requested_second_alone',1,2),
    ('edge_requested_alternating_7_then_8',2,4),
    ('thumb_tip_then_flat',0,3),
    ('thumb_flat_then_tip',0,3),
    ('independent_finger_and_edge',0,3),
    ('bilateral_edge_chords',0,4),
])
def test_fixed_sequence_contacts_and_unwanted_keys(name,extra,total):
    run=seq.run_case(name)
    assert run['unexpected_keydown_count']==extra
    assert run['missing_keydown_count']==0
    assert run['actual_keydown_count']==total
    assert run['motor_resets_after_start']==0
    assert run['continuous_physical_motor'] is True
    assert run['no_adofai_evaluator_or_student'] is True
    for pulse in run['pulses']:
        assert len(pulse['physical_down_events'])==len(pulse['actual_downs'])
        assert len(pulse['physical_up_events'])==len(pulse['actual_downs'])
        assert len(set(x['key'] for x in pulse['physical_down_events']))==len(pulse['actual_downs'])
        assert {e['event'] for e in pulse['physical_down_events']}=={'down'}
        assert {e['event'] for e in pulse['physical_up_events']}=={'up'}
        assert pulse['elapsed_from_release_to_safe_ms']>0
        assert pulse['physical_recovery_frames']<=seq.MAX_RELEASE_FRAMES
        assert pulse['safe_to_switch_ms']>pulse['press_command_start_ms']


@pytest.mark.parametrize('case,target_extra', [
    ('edge_requested_single_7','left_8'),
    ('edge_requested_single_8','left_7'),
    ('thumb_requested_second_alone','left_5'),
])
def test_extra_contact_is_real_physical_keydown_at_same_timestamp(case,target_extra):
    result=seq.run_case(case)['pulses'][0]
    assert result['unexpected_downs']==[target_extra]
    assert len(result['physical_down_events'])==2
    assert len({e['time_ms'] for e in result['physical_down_events']})==1
    assert result['actual_downs']==sorted(set(result['requested'])|{target_extra})


def test_alternating_side_single_keys_does_not_create_independent_input():
    result=seq.run_case('edge_requested_alternating_7_then_8')
    assert result['unexpected_keydown_count']==2
    assert [x['unexpected_downs'] for x in result['pulses']]==[['left_8'],['left_7']]
    for pulse in result['pulses']:
        assert {e['key'] for e in pulse['physical_down_events']}=={'left_7','left_8'}
        assert len({e['time_ms'] for e in pulse['physical_down_events']})==1


def test_thumb_mode_switch_requires_release_and_no_phantom_keydown():
    result=seq.run_case('thumb_tip_then_flat')
    first,second=result['pulses']
    assert first['requested']==['left_5']
    assert second['requested']==['left_5','left_6']
    assert first['safe_to_switch_ms']==second['press_command_start_ms']
    assert first['physical_up_events'][0]['time_ms']<=first['safe_to_switch_ms']
    assert second['physical_down_events'][0]['time_ms']>first['safe_to_switch_ms']
    assert result['unexpected_keydown_count']==0


def test_physical_guard_rejects_mid_stroke_switch_without_time_progression():
    motor=Coupled16ContactMotor()
    tip=seq.THUMB_TIP.action(positive=True)
    flat=seq.THUMB_FLAT.action(positive=True)
    for _ in range(6):
        motor.step(tip)
    time=motor.motor.time_s
    before=motor.observe()
    with pytest.raises(ValueError,match='during physical stroke'):
        motor.step(flat)
    assert motor.motor.time_s==time
    assert motor.observe()==before
    assert motor.observe().thumb_modes[0] is ThumbMode.TIP


def test_physical_guard_rejects_edge_disengage_until_motor_reset():
    motor=Coupled16ContactMotor()
    active=seq.EDGE_PAIR.action(positive=True)
    for _ in range(6):
        motor.step(active)
    off=ContactAction(tuple(-1. if x=='left_6' else 0. for x in PHYSICAL_AXES))
    time=motor.motor.time_s
    with pytest.raises(ValueError,match='during physical stroke'):
        motor.step(off)
    assert motor.motor.time_s==time
    for _ in range(seq.MAX_RELEASE_FRAMES):
        motor.step(seq.EDGE_PAIR.action(positive=False))
        if seq._can_switch_and_rearm(motor,seq.EDGE_PAIR):
            break
    else:
        pytest.fail('Failed physical recovery')
    motor.step(off)
    assert not motor.observe().edge_enabled[0]


def test_fixed_press_is_not_a_claim_about_human_pose_or_speed():
    assert seq.PRESS_FRAMES==6
    assert p.KEY_COUNT==8 and p.INPUT_DIM==263
    assert p.CONTROL_DT==.010 and p.PHYSICS_DT==.001
    assert len(PHYSICAL_AXES)==12
    assert seq.FORMAT!=seq.REPORT
    # The current model charges no extra time for pose switching at rest.
    case=seq.run_case('thumb_flat_then_tip')
    assert case['pulses'][0]['safe_to_switch_ms']==case['pulses'][1]['press_command_start_ms']


def test_nonregistered_case_and_invalid_requested_logical_contacts_are_rejected():
    with pytest.raises(ValueError,match='predeclared'):
        seq.run_case('invented')
    with pytest.raises(ValueError,match='unique nonempty'):
        seq.Pulse('invalid',('left_5','left_5'),('left_5',),left_thumb=ThumbMode.TIP)
    with pytest.raises(ValueError,match='Unknown requested'):
        seq.Pulse('invalid',('left_123',),('left_5',),left_thumb=ThumbMode.TIP)
    with pytest.raises(ValueError,match='Unknown physical'):
        seq.Pulse('invalid',('left_5',),('left_123',),left_thumb=ThumbMode.TIP)


def test_immutable_independent_report(tmp_path):
    config=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                   output_dir=str(tmp_path))
    report=seq.run_audit(config)
    assert len(report['cases'])==len(seq.FIXTURES)==11
    assert report['motor_physical_axes']==12
    assert report['keyboard_logical_contacts']==16
    assert report['no_chart_oracle']
    assert report['no_adofai_evaluator']
    assert report['no_training'] and report['no_student_checkpoint']
    assert report['stage_a_fail_preserved']
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    saved=tmp_path/'diagnostics'/'prestige1'/seq.REPORT
    raw=saved.read_bytes()
    with pytest.raises(FileExistsError):
        seq.run_audit(config)
    assert saved.read_bytes()==raw
