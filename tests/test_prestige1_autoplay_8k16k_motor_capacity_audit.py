from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.n_key import n_key_names
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_8k16k_motor_capacity_audit as audit


def test_three_fixed_motor_only_protocols_preserve_8k_student_contract():
    assert audit.PROTOCOLS == ((8, 8), (16, 8), (16, 16))
    assert p.KEY_COUNT == 8 and p.INPUT_DIM == 263
    assert p.CONTROL_DT == .010 and p.PHYSICS_DT == .001
    assert len(n_key_names(16)) == 16
    assert set(audit.ComparisonProtocol(16, 8).allowed) == {
        'left_4', 'left_3', 'left_2', 'left_1',
        'right_1', 'right_2', 'right_3', 'right_4',
    }
    assert len(audit.ComparisonProtocol(16, 16).allowed) == 16


def test_physical_cold_calibration_is_fixed_and_matches_lead():
    for size in (8, 16):
        cal = calibrate_n_key_press_lead(size, control_dt_s=.010, physics_dt_s=.001)
        assert cal.lead_s == pytest.approx(.044, abs=1e-12)
        assert len(cal.key_latencies_s) == size


def test_refuse_unregistered_chart_and_invalid_actuator_protocol():
    with pytest.raises(ValueError, match='Unregistered'):
        audit.make_fixture(4000., 100)
    with pytest.raises(ValueError, match='predeclared'):
        audit.run_case(4000., 100, audit.ComparisonProtocol(8, 8))
    with pytest.raises(ValueError):
        audit.ComparisonProtocol(8, 16)
    with pytest.raises(ValueError):
        audit.ComparisonProtocol(12, 8)
    with pytest.raises(ValueError, match='fixed'):
        audit.run_case(4000., 144, audit.ComparisonProtocol(8, 8), lead_s=.080)


@pytest.mark.parametrize('bpm,notes,hit8,hit16restricted,hit16full', [
    (1200., 144, 144, 144, 144),
    (2000., 144, 144, 144, 144),
    (4000., 144, 120, 123, 144),
    (4000., 576, 472, 479, 576),
])
def test_fixed_oracle_motor_only_capacity_with_three_controls(
    bpm, notes, hit8, hit16restricted, hit16full,
):
    results = [audit.run_case(bpm, notes, audit.ComparisonProtocol(k, a))
               for k, a in audit.PROTOCOLS]
    assert [r['hit'] for r in results] == [hit8, hit16restricted, hit16full]
    assert all(r['miss'] + r['hit'] == notes for r in results)
    assert all(r['physical_keydowns'] == r['hit'] for r in results)
    assert all(r['no_motor_reset_between_notes'] for r in results)
    assert all(r['oracle_target_schedule_used_by_teacher'] for r in results)
    assert all(r['student_or_hud_model_executed'] is False for r in results)
    assert len({r['fixture_sha256'] for r in results}) == 1
    for r in results:
        assert r['fingers_with_actual_downs'] <= r['allowed_key_count']
        assert set(r['physical_downs_per_key']) <= set(r['allowed_key_names'])
        assert set(r['positive_action_frames_per_key']) <= set(r['allowed_key_names'])
        assert r['max_simultaneous_positive_action_keys'] <= r['allowed_key_count']
    if bpm == 4000.:
        assert results[0]['fingers_with_actual_downs'] == 8
        assert results[1]['fingers_with_actual_downs'] == 8
        assert results[2]['fingers_with_actual_downs'] == 10
        assert results[0]['unique_due_targets_blocked_at_some_point'] > 0
        assert results[2]['unique_due_targets_blocked_at_some_point'] == 0


def test_motor_key_count_alone_can_change_dynamics_without_more_active_keys():
    a = audit.run_case(2000., 144, audit.ComparisonProtocol(8, 8))
    b = audit.run_case(2000., 144, audit.ComparisonProtocol(16, 8))
    assert a['hit'] == b['hit'] == 144
    # Coordination layout differs when the same number of usable fingers
    # is placed inside the 16-actuator body, even with common hand capacity.
    assert a['x_accuracy_percent'] != b['x_accuracy_percent']


def test_report_is_readonly_immutable_and_never_claims_human_16k(tmp_path):
    config = replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                     output_dir=str(tmp_path))
    report = audit.run_audit(config)
    assert report['format'] == audit.VERSION
    assert len(report['cases']) == 4
    assert report['motor_only_oracle_teacher'] is True
    assert report['not_a_human_16key_fingering_model'] is True
    assert report['hud_or_student_used'] is False
    assert report['fixed_8k_student_untouched'] is True
    assert report['no_training'] is True
    assert report['validation_evaluated'] is False
    assert report['final_evaluated'] is False
    for case in report['cases']:
        assert len(case['protocol_results']) == 3
    file = tmp_path / 'diagnostics' / 'prestige1' / audit.REPORT
    old = file.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run_audit(config)
    assert file.read_bytes() == old
