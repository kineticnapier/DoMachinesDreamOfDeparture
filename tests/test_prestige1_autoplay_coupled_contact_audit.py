from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.contact_16k import (
    HANDS, PHYSICAL_AXES, LOGICAL_KEYS, ContactAction, Coupled16ContactMotor,
    ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import n_key_names
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_coupled_contact_audit as contact


def _step_many(env, action, count=30):
    events = []
    for _ in range(count):
        events.extend(env.step(action).logical_events)
    return events


@pytest.mark.parametrize('mode,edge,expected', [
    (ThumbMode.OFF,False,4),
    (ThumbMode.TIP,False,5),
    (ThumbMode.FLAT,False,6),
    (ThumbMode.TIP,True,7),
    (ThumbMode.FLAT,True,8),
])
@pytest.mark.parametrize('hand', HANDS)
def test_each_per_hand_4k_to_8k_mode_is_physically_actuated_and_released(
    hand,mode,edge,expected,
):
    r = contact.run_case(hand, f'{expected}K', mode, edge, expected)
    assert r['expected_logical_contacts']==expected
    assert r['first_press']['logical_down_count']==expected
    assert r['release_up_count']==expected
    assert r['second_press']['logical_down_count']==expected
    assert r['no_motor_reset_between_strokes']
    keytimes = r['first_press']['down_timestamps_ms']
    assert len(keytimes)==expected
    assert set(keytimes)==set(Coupled16ContactMotor.visible_keys_for_hand(
        hand,mode,edge_enabled=edge))
    assert 0<r['first_press']['first_down_ms']<=r['first_press']['last_down_ms']<300
    assert r['first_press']['physical_used_axes']==(expected - int(mode is ThumbMode.FLAT)-int(edge))
    if edge:
        assert keytimes[f'{hand}_7']==keytimes[f'{hand}_8']
    if mode is ThumbMode.FLAT:
        assert keytimes[f'{hand}_5']==keytimes[f'{hand}_6']
    if mode is ThumbMode.TIP:
        assert f'{hand}_6' not in keytimes


def test_bilateral_16_contacts_from_only_12_physical_axes():
    r=contact.run_case('both','16K',ThumbMode.FLAT,True,16)
    assert r['expected_logical_contacts']==16
    assert r['first_press']['physical_used_axes']==12
    assert r['first_press']['logical_down_count']==16
    assert r['release_up_count']==16
    assert r['second_press']['logical_down_count']==16
    assert set(r['first_press']['down_timestamps_ms'])==set(LOGICAL_KEYS)
    assert len(PHYSICAL_AXES)==12
    assert len(n_key_names(16))==16
    assert p.KEY_COUNT==8 and p.INPUT_DIM==263


def test_single_side_actuator_cannot_press_just_one_of_two_edge_keys():
    env=Coupled16ContactMotor()
    action=ContactAction(
        tuple(1. if key=='left_6' else 0. for key in PHYSICAL_AXES),
        edge_enabled=(True,False),
    )
    events=_step_many(env,action)
    down=[evt for evt in events if evt.event is KeyEvent.DOWN]
    assert len(down)==2
    assert {e.key for e in down}=={'left_7','left_8'}
    assert down[0].time_s==down[1].time_s
    assert env.observe().pressed('left_7')
    assert env.observe().pressed('left_8')
    assert not env.observe().pressed('right_7')
    assert not env.observe().pressed('left_5')


def test_one_thumb_axis_generates_both_flat_contacts_not_two_fingers():
    env=Coupled16ContactMotor()
    action=ContactAction(
        tuple(1. if key=='left_5' else 0. for key in PHYSICAL_AXES),
        thumb_modes=(ThumbMode.FLAT,ThumbMode.OFF),
    )
    events=_step_many(env,action)
    downs=[e for e in events if e.event is KeyEvent.DOWN]
    assert {e.key for e in downs}=={'left_5','left_6'}
    assert len({e.time_s for e in downs})==1
    physical_idx=env.motor.key_names.index('left_5')
    assert env.motor.observe().pressed_flags[physical_idx]


def test_thumb_can_switch_tip_to_flat_only_after_full_physical_release():
    env=Coupled16ContactMotor()
    tip=ContactAction(tuple(1. if x=='left_5' else 0. for x in PHYSICAL_AXES),
                      thumb_modes=(ThumbMode.TIP,ThumbMode.OFF))
    flat=ContactAction(tip.commands,thumb_modes=(ThumbMode.FLAT,ThumbMode.OFF))
    downs=_step_many(env,tip)
    assert {e.key for e in downs if e.event is KeyEvent.DOWN}=={'left_5'}
    t=env.motor.time_s
    with pytest.raises(ValueError,match='during physical stroke'):
        env.step(flat)
    assert env.motor.time_s==t  # fail-closed: no physical advance
    assert env.observe().thumb_modes[0] is ThumbMode.TIP
    released=ContactAction(tuple(-1. if x=='left_5' else 0. for x in PHYSICAL_AXES),
                           thumb_modes=(ThumbMode.TIP,ThumbMode.OFF))
    ups=_step_many(env,released,40)
    assert {e.key for e in ups if e.event is KeyEvent.UP}=={'left_5'}
    assert not env.observe().pressed('left_5')
    flat_downs=_step_many(env,flat)
    assert {e.key for e in flat_downs if e.event is KeyEvent.DOWN}=={'left_5','left_6'}


def test_thumb_pose_change_during_prethreshold_forward_motion_is_rejected():
    env=Coupled16ContactMotor()
    tip=ContactAction(tuple(1. if x=='left_5' else 0. for x in PHYSICAL_AXES),
                      thumb_modes=(ThumbMode.TIP,ThumbMode.OFF))
    # The muscle starts moving before it reaches the 2mm KeyDown threshold.
    env.step(tip)
    index=env.motor.key_names.index('left_5')
    obs=env.motor.observe()
    assert 0 < obs.positions_m[index] < env.motor.keyboard.config.reset_m
    assert obs.velocities_m_s[index] > 0
    flat=ContactAction(tip.commands,thumb_modes=(ThumbMode.FLAT,ThumbMode.OFF))
    before=env.motor.time_s
    with pytest.raises(ValueError,match='physical stroke'):
        env.step(flat)
    assert env.motor.time_s==before


def test_side_cannot_disengage_while_pressed_or_spring_position_high():
    env=Coupled16ContactMotor()
    side=ContactAction(tuple(1. if x=='left_6' else 0. for x in PHYSICAL_AXES),
                       edge_enabled=(True,False))
    _step_many(env,side)
    off=ContactAction(tuple(-1. if x=='left_6' else 0. for x in PHYSICAL_AXES))
    with pytest.raises(ValueError,match='physical stroke'):
        env.step(off)
    release=ContactAction(off.commands,edge_enabled=(True,False))
    ups=_step_many(env,release,40)
    assert {e.key for e in ups if e.event is KeyEvent.UP}=={'left_7','left_8'}
    env.step(off)
    assert env.observe().edge_enabled==(False,False)


@pytest.mark.parametrize('action', [
    ContactAction((0.,)*12),
])
def test_rest_command_never_generates_phantom_keys(action):
    env=Coupled16ContactMotor()
    for event in _step_many(env,action):
        pytest.fail(f'unexpected event from rest: {event}')
    assert not any(env.observe().logical_pressed)


def test_invalid_commands_and_mode_selection_are_rejected():
    with pytest.raises(ValueError):ContactAction((0.,)*16)
    with pytest.raises(ValueError):ContactAction((float('nan'),)+(0.,)*11)
    with pytest.raises(ValueError):ContactAction((2.,)+(0.,)*11)
    with pytest.raises(ValueError):ContactAction((0.,)*12,thumb_modes=('flat',ThumbMode.OFF))
    with pytest.raises(ValueError,match='disabled thumb'):
        ContactAction(tuple(1. if x=='left_5' else 0. for x in PHYSICAL_AXES))
    with pytest.raises(ValueError,match='disabled hand-edge'):
        ContactAction(tuple(1. if x=='left_6' else 0. for x in PHYSICAL_AXES))
    with pytest.raises(ValueError):
        Coupled16ContactMotor.visible_keys_for_hand('invalid',ThumbMode.OFF,edge_enabled=False)
    with pytest.raises(ValueError):
        Coupled16ContactMotor.contact_keys('left',7,ThumbMode.OFF,edge_enabled=False)


def test_fixed_report_is_readonly_and_leaves_stage_a_student_frozen(tmp_path):
    cfg=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                output_dir=str(tmp_path))
    report=contact.run_audit(cfg)
    assert len(report['cases'])==11
    assert report['underlying_motor_physical_axes']==12
    assert report['two_hand_logical_keyboard_contacts']==16
    assert report['not_bpm_hit_or_accuracy_measurement']
    assert report['not_human_16k_speed_validation']
    assert report['not_trained_student_or_teacher']
    assert report['no_training'] and report['frozen_8key_263d_unchanged']
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    saved=tmp_path/'diagnostics'/'prestige1'/contact.REPORT
    original=saved.read_bytes()
    with pytest.raises(FileExistsError):contact.run_audit(cfg)
    assert saved.read_bytes()==original
