from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_dmdogo_h22_frozen_transfer as h22

def h21():
    return {'format': 'dmdogo-h21-human-transition-rules-v1',
            'chart': h22.SOURCE_CHART,
            'by_reference_interval_ms': {
                b: {'pairs': 10, 'human_switches': 10 if b in ('(50,100]', '(100,150]') else 0}
                for b in h22.BINS}}

def motor(chart='other-chart.adofai'):
    items = [(1500, 'left_1'), (1594, 'right_1'), (1782, 'right_2')]
    return {'stage':'executed','termination_reason':'chart_complete',
            'chart':chart,'chart_event_count':len(items),
            'physical_local_replan_count':0,
            'stats':{'hits':len(items),'misses':0},
            'decisions':[{'event_index':i,'due_ms':ms,'chosen':axis,
                          'keydowns':[{'axis':axis,'episode_s':ms/1000+.003}]}
                         for i,(ms,axis) in enumerate(items)]}

def human():
    return [{'time_s': t, 'hand': hand} for t,hand in
            [(4.23,'R'), (4.324,'L'), (4.512,'L')]]

def test_frozen_rule_not_retrained():
    rule=h22.frozen_rule(h21())
    assert rule['(50,100]'] is True
    assert rule['(150,300]'] is False

def test_training_chart_is_refused():
    with pytest.raises(ValueError,match='training chart'):
        h22.evaluate(h21(), human(), motor(h22.SOURCE_CHART))

def test_incomplete_motor_is_refused():
    run=motor()
    run['decisions'][1]['keydowns']=[]
    with pytest.raises(ValueError,match='missing or ambiguous'):
        h22.motor_events(run)

def test_transfer_synthetic():
    result=h22.evaluate(h21(),human(),motor())
    assert result['overall']['pairs']==2
    assert result['overall']['frozen_correct']==2
    assert result['overall']['motor_correct']==2
    assert result['training_performed'] is False

def test_boundary_bins():
    assert [h22.gap_bin(v) for v in (20,47,94,140,188,400)]==list(h22.BINS)
