from __future__ import annotations
import hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_dmdogo_h23_actual_motor import compile_report

CHART=b'abc'

def fixture():
    s={'planning_mode':'gesture_v08','chart':'level.adofai',
       'chart_sha256':hashlib.sha256(CHART).hexdigest(),
       'total_events':3,'termination_reason':'chart_complete',
       'stats':{'targets':3,'hits':3,'misses':0,'too_early':0,'overloaded':False,'physical_keydowns':3}}
    d=[{'event_index':i,'due_ms':1000+90*i,'chosen':axis,
        'axes':[axis],'desired_axes':['right_1'],'local_replan':i==1,
        'outcome':'Hit','keydowns':[{'axis':axis,'episode_s':1.004+0.09*i}]}
        for i,axis in enumerate(['right_1','left_1','right_2'])]
    t={'decisions':[d[0],d[2],d[1]],'physical':{'local_replans':1,'unassigned_events':0}}
    return s,t

def test_out_of_order_preserved_while_events_sorted():
    s,t=fixture();a=compile_report(s,t,CHART,'level.adofai',variant='interval_only')
    assert a['h23_all_hit_verified'] is True
    assert a['h23_exact_original_plan_verified'] is False
    assert a['original_selection_event_order']==[0,2,1]
    assert [d['event_index'] for d in a['decisions']]==[0,1,2]
    assert a['physical_local_replan_count']==1

def test_no_replan_valid():
    s,t=fixture();t['physical']['local_replans']=0
    t['decisions'][-1]['local_replan']=False
    a=compile_report(s,t,CHART,'level.adofai',variant='right_after_long_rest')
    assert a['h23_exact_original_plan_verified']

@pytest.mark.parametrize('mutator,pattern',[
    (lambda s,t:t['decisions'][2].update(event_index=0),'duplicate/missing'),
    (lambda s,t:t['physical'].update(local_replans=0),'replan provenance'),
    (lambda s,t:t['decisions'][0]['keydowns'].clear(),'incomplete keydown'),
    (lambda s,t:s.update(chart_sha256='bad'),'checksum'),
])
def test_invalid_evidence_rejected(mutator,pattern):
    s,t=fixture();mutator(s,t)
    with pytest.raises(ValueError,match=pattern):compile_report(s,t,CHART,'level.adofai',variant='interval_only')

def test_miss_is_not_all_hit():
    s,t=fixture();s['stats'].update(hits=2,misses=1)
    t['decisions'][0]['outcome']='Miss'
    a=compile_report(s,t,CHART,'level.adofai',variant='interval_only')
    assert not a['h23_all_hit_verified']
