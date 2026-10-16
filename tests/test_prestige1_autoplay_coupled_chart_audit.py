from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction, PHYSICAL_AXES, ThumbMode, LOGICAL_KEYS
from dmdod.motor.keyboard import KeyEvent
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_coupled_chart_audit as audit


def _command(axis: str, *, edge=False, thumb=ThumbMode.OFF, sign=1.):
    return ContactAction(tuple(sign if channel == axis else 0. for channel in PHYSICAL_AXES),
                         thumb_modes=(thumb,ThumbMode.OFF),
                         edge_enabled=(edge,False))


def test_coupled_motor_step_full_real_chart_600_bpm_single_edge_press_is_te():
    chart=audit._fixture(600.,24)
    env=CoupledContactRealChartEnv(chart)
    obs=env.reset()
    assert obs.motor.logical_keys == LOGICAL_KEYS
    assert len(obs.motor.physical.key_names)==12
    assert len(obs.floors)>1
    for _ in range(6):
        result=env.step(ContactAction((0.,)*12,edge_enabled=(True,False)))
    for _ in range(10):
        result=env.step(_command('left_6',edge=True))
        if env.physical_keydowns:
            break
    assert env.physical_keydowns==2
    assert env.stats.hits==1
    assert env.stats.too_early_presses==1
    assert not env.stats.overloaded
    events=tuple(e for e in env.last_events if e.event is KeyEvent.DOWN)
    assert {e.key for e in events} == {'left_7','left_8'}
    assert events[0].time_s==events[1].time_s
    assert [x['target_index'] for x in env.judgement_trace]==[0,1]
    assert env.motor.motor.time_s==pytest.approx(0.1)
    assert result.observation.motor.pressed('left_7')


def test_full_step_high_bpm_simultaneous_contact_can_claim_two_notes():
    env=CoupledContactRealChartEnv(audit._fixture(4000.,72))
    env.reset()
    for _ in range(10):
        result=env.step(_command('left_6',edge=True))
        if env.physical_keydowns:
            break
    assert env.physical_keydowns==2
    assert env.stats.hits==2
    assert env.stats.too_early_presses==0
    assert env.judgement_trace[0]['time_s']==env.judgement_trace[1]['time_s']
    assert result.done is False


def test_step_guards_invalid_action_and_end_of_episode():
    env=CoupledContactRealChartEnv(audit._fixture(600.,24))
    env.reset()
    with pytest.raises(AttributeError):
        env.step('not a contact action')
    env._done=True
    with pytest.raises(RuntimeError,match='episode is done'):
        env.step(ContactAction((0.,)*12))


@pytest.mark.parametrize('bpm,notes,expected_hits',[
    (600.,24,[24,24,24,24,24]),
    (1200.,48,[48,48,48,48,48]),
    (4000.,72,[60,72,72,72,72]),
])
def test_long_continuous_chart_with_each_left_and_right_fingering_mode(bpm,notes,expected_hits):
    rows=[audit.simulate(bpm,notes,k) for k in audit.MODES]
    assert [r['hits'] for r in rows]==expected_hits
    assert all(r['hits']+r['misses']==notes for r in rows)
    assert all(r['real_physical_contact_keydowns']==r['recorded_keydowns'] for r in rows)
    assert all(r['single_continuous_motor'] for r in rows)
    assert all(r['oracle_target_schedule_supplied_to_teacher'] for r in rows)
    assert all(r['too_early']==0 and not r['overloaded'] for r in rows)
    if bpm==4000:
        assert rows[0]['misses']==12
        assert rows[1]['two_contact_reservations']==0
        assert rows[2]['two_contact_reservations']>0  # flat thumb shared contacts
        assert rows[3]['two_contact_reservations']>0  # hand-edge shared contacts
        assert rows[4]['two_contact_reservations']>0  # both shared contact types
        assert {'left_6','right_6'}<=set(rows[3]['used_physical_axes'])
        assert {'left_5','right_5','left_6','right_6'}<=set(rows[4]['used_physical_axes'])


def test_teacher_mode_mapping_and_logical_key_groups_are_not_independent_axes():
    assert len(audit._physical_channels(4))==8
    assert len(audit._physical_channels(5))==10
    assert len(audit._physical_channels(6))==10
    assert len(audit._physical_channels(7))==12
    assert len(audit._physical_channels(8))==12
    contact_pairs=dict(audit._physical_channels(8))
    assert contact_pairs['left_5']==('left_5','left_6')
    assert contact_pairs['left_6']==('left_7','left_8')
    assert contact_pairs['right_5']==('right_5','right_6')
    assert contact_pairs['right_6']==('right_7','right_8')
    assert len(LOGICAL_KEYS)==16 and len(PHYSICAL_AXES)==12


def test_no_evaluator_privilege_leak_into_contact_observation():
    env=CoupledContactRealChartEnv(audit._fixture(600.,24))
    obs=env.reset()
    assert not hasattr(obs,'target_time_s')
    assert not hasattr(obs,'target_index')
    assert not hasattr(obs.motor,'target_index')
    assert not hasattr(obs.motor,'target_time_s')
    assert p.KEY_COUNT==8 and p.INPUT_DIM==263
    assert p.CONTROL_DT==.010 and p.PHYSICS_DT==.001


def test_reject_unregistered_chart_and_mode():
    with pytest.raises(ValueError,match='fixed'):
        audit._fixture(4000.,100)
    with pytest.raises(ValueError,match='Unsupported'):
        audit.simulate(600.,24,9)
    with pytest.raises(ValueError):
        audit.ContactOracleTeacher(9)


def test_immutable_report_and_no_student_training(tmp_path):
    cfg=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                output_dir=str(tmp_path))
    report=audit.run_audit(cfg)
    assert len(report['cases'])==15
    assert report['oracle_motor_only'] and report['not_hud_or_student']
    assert report['not_true_multipress'] and report['no_training']
    assert report['stage_a_fail_preserved']
    assert not report['validation_evaluated'] and not report['final_evaluated']
    outfile=tmp_path/'diagnostics'/'prestige1'/audit.REPORT
    raw=outfile.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run_audit(cfg)
    assert outfile.read_bytes()==raw
