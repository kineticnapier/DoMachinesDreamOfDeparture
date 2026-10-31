from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_dmdogo_h21_human_transition_rules as h21


def make_rows():
    return [
        {'event_index': i, 'due_s': t, 'human_index': j,
         'human_hand': hand, 'human_finger': finger, 'human_key': key,
         'dmd_axis': axis, 'dmd_hand': axis[0].upper(),
         'dmd_time_s': t + .003}
        for i, (t, j, hand, finger, key, axis) in enumerate([
            (1.0, 0, 'R', 'RI', 'K09', 'left_1'),
            (1.010, 1, 'R', 'RM', 'K10', 'left_2'),
            (1.104, 2, 'L', 'LI', 'K04', 'right_1'),
            (1.292, 4, 'R', 'RI', 'K09', 'right_2'),
            (1.480, 5, 'R', 'RI', 'K09', 'left_1'),
        ])]


def make_motor(rows):
    return {'chart': h21.CHART, 'candidate_id': h21.CANDIDATE,
            'stage': 'executed', 'termination_reason': 'chart_complete',
            'chart_event_count': len(rows),
            'physical_local_replan_count': 0,
            'stats': {'hits': len(rows), 'misses': 0},
            'chart_sha256': 'test',
            'decisions': [{
                'event_index': r['event_index'], 'due_ms': r['due_s'] * 1000,
                'chosen': r['dmd_axis'],
                'keydowns': [{'axis': r['dmd_axis'], 'episode_s': r['dmd_time_s']}]
            } for r in rows]}


def test_exact_motor_event_verification():
    rows = make_rows()
    h21.verify_motor(rows, make_motor(rows))
    motor = make_motor(rows)
    motor['decisions'][1]['keydowns'][0]['axis'] = 'right_4'
    with pytest.raises(ValueError, match='axis mismatch'):
        h21.verify_motor(rows, motor)
    motor = make_motor(rows)
    motor['decisions'][2]['keydowns'][0]['episode_s'] += .001
    with pytest.raises(ValueError, match='time mismatch'):
        h21.verify_motor(rows, motor)


def test_only_consecutive_both_sources_are_compared():
    pairs = h21.transitions(make_rows())
    # Human index 2->4 skips an unmatched Human keydown: exclude that link.
    assert [p['event_start'] for p in pairs] == [0, 1, 3]
    assert [p['human_switch'] for p in pairs] == [False, True, False]
    assert [p['gap_bin'] for p in pairs] == ['<=20', '(50,100]', '(150,300]']


def test_gap_boundaries():
    assert [h21.gap_bin(ms) for ms in [10, 20, 47, 50, 94, 100, 140, 150, 188, 300, 375]] == [
        '<=20', '<=20', '(20,50]', '(20,50]', '(50,100]', '(50,100]',
        '(100,150]', '(100,150]', '(150,300]', '(150,300]', '>300']


def test_analyze_refuses_wrong_run_and_is_descriptive():
    rows = make_rows()
    motor = make_motor(rows)
    result = h21.analyze(rows, motor)
    assert result['input']['comparable_consecutive_pairs'] == 3
    assert result['human_preference_observed'] is False
    assert result['training_performed'] is False
    assert result['all_pairs']['human_switches'] == 1
    motor['candidate_id'] = 'wrong'
    with pytest.raises(ValueError, match='wrong Motor'):
        h21.analyze(rows, motor)


def test_csv_rejects_invalid_hand_alignment(tmp_path):
    fields = ('event_index', 'reference_due_s', 'human_index', 'human_hand',
              'human_finger', 'human_key', 'dmd_actual_axis', 'dmd_actual_hand',
              'dmd_actual_time_s')
    f = tmp_path / 'sample.csv'
    with f.open('w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerow({'event_index': 0, 'reference_due_s': 1.,
                    'human_index': 0, 'human_hand': 'R', 'human_finger': 'RI',
                    'human_key': 'K09', 'dmd_actual_axis': 'left_1',
                    'dmd_actual_hand': 'R', 'dmd_actual_time_s': 1.003})
    with pytest.raises(ValueError, match='DMD hand'):
        h21.read_events(f)


def test_within_chart_windows_never_fit_test_window():
    samples = [
        {'start_s': 1., 'gap_bin': '<=20', 'human_switch': False, 'dmd_switch': True},
        {'start_s': 2., 'gap_bin': '<=20', 'human_switch': False, 'dmd_switch': True},
        {'start_s': 21., 'gap_bin': '<=20', 'human_switch': True, 'dmd_switch': False},
    ]
    result = h21.heldout_windows(samples)
    assert result['evaluated_pairs'] == 3
    # Test windows are never included in their own fitted majority rule.
    assert result['interval_correct'] == 0
    assert result['global_correct'] == 0
