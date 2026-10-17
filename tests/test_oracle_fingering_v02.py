from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_oracle_fingering_v02 as teacher
from dmdod.motor.contact_16k import Coupled16ContactMotor


def fixture_chart(tmp_path: Path) -> Path:
    p = tmp_path / 'fixture.adofai'
    p.write_text(json.dumps({"angleData": [180] * 19,
                             "settings": {"bpm": 1200}, "actions": []}), encoding='utf-8')
    return p


def test_fingers_have_physical_contact_semantics():
    assert len(teacher.AXES) == 12
    assert len(teacher.finger_options(1)) == 10
    assert any(o[0].mode == 'edge' for o in teacher.finger_options(2))
    assert any(o[0].mode == 'flat' for o in teacher.finger_options(2))
    assert teacher.Finger('right_6', 'edge').logical_keys == ('right_7', 'right_8')
    assert teacher.Finger('left_5', 'flat').logical_keys == ('left_5', 'left_6')
    assert sum(f.contact_count for f in teacher.finger_options(2)[0]) == 2


def test_synthetic_lookahead_is_physical_deterministic_nonmutating():
    motor = Coupled16ContactMotor()
    initial = motor.observe()
    class Event:
        def __init__(self, i, t):
            self.event_index=i
            self.episode_time_s=t
            self.chord_size=1
    events = tuple(Event(i, t) for i,t in enumerate((.045, .060, .075)))
    a = teacher.rank_first_choices(motor, (), events)
    b = teacher.rank_first_choices(motor, (), events)
    assert motor.observe() == initial
    assert len(a) == len(b) == 10
    assert [(r['finger'], r['score'], r.get('best_continuation')) for r in a] == [
        (r['finger'], r['score'], r.get('best_continuation')) for r in b]
    selected = min((v for v in a if v['eligible']), key=lambda x: (x['score'],x['finger']))
    assert selected['lookahead_depth'] == 3
    assert len(selected['best_continuation']) == 2
    assert len(selected['physical_preview']['per_event']) == 3


def test_out_of_range_and_invalid_depth(tmp_path):
    path = fixture_chart(tmp_path)
    with pytest.raises(ValueError):
        teacher.report_chart(path, end_s=0)
    with pytest.raises(ValueError):
        teacher.run_teacher(teacher.build_playable_segment(
            teacher.compile_adofai(teacher.load_adofai(path)), end_s=.2), max_depth=5)


def test_report_and_viewer_write_once(tmp_path):
    report = teacher.report_chart(fixture_chart(tmp_path), end_s=.4)
    assert report['physical_axes'] == 12 and report['logical_contacts'] == 16
    assert report['student_trained'] is False and report['oracle_assisted'] is True
    assert len(report['teacher']['decisions']) > 0
    assert any(r['depth'] >= 2 for r in report['teacher']['decisions'])
    assert all(len(r['candidates']) == 10 for r in report['teacher']['decisions'])
    out = tmp_path/'result.json'
    html, csv = teacher.save_report(report, out)
    assert '候補比較' in html.read_text(encoding='utf-8')
    assert '12軸' in html.read_text(encoding='utf-8')
    assert 'chosen' in csv.read_text(encoding='utf-8-sig')
    with pytest.raises(FileExistsError):
        teacher.save_report(report, out)


def test_thumb_flat_and_edge_emit_two_real_logical_keydowns():
    motor = Coupled16ContactMotor()
    for finger, expected in [
        (teacher.Finger('left_5', 'flat'), {'left_5', 'left_6'}),
        (teacher.Finger('right_6', 'edge'), {'right_7', 'right_8'}),
    ]:
        plan = teacher.PlanItem(0, .044, (finger,))
        result = teacher.simulated_sequence(motor, (plan,), until_s=.08)
        assert {e['contact'] for e in result['physical_downs']} == expected
        assert result['missing'] == result['extra'] == 0
    assert motor.motor.time_s == 0.0