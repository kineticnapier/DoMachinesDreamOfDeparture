from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training.train_counterfactual_motor_v12 import Snapshot
from dmdod.training.train_counterfactual_motor_v14 import (
    HORIZONS, Intervention, Trace, REPRESENTATIVES, _check_output_dir,
    assert_reproduced, case_from_row, compare_horizon,
    local_neighbors, select_cases, snapshot_at_common,
    _visible_neighbors,
)


def snap(*, t=.8, hits=(0,1,2),misses=(),early=3,down=6,
         overloaded=False, reason='Horizon cutoff',terminated=False):
    return Snapshot(
        time_s=t,hits=len(hits),misses=len(misses),too_early=early,
        overloaded=overloaded,overload_time_s=t if overloaded else None,
        terminated=terminated,termination_reason=reason,
        hit_ids=tuple(hits),miss_ids=tuple(misses),
        keydowns=tuple(('left_2',round(i*.01,3)) for i in range(down)),
    )


def trace(v8,*,v16=None,v32=None, frames=None, features=()):
    return Trace(at={.8:v8,1.6:v16 or v8,3.2:v32 or v16 or v8},
                 frames=tuple(frames) if frames is not None else (v8,),
                 observable=features)


def test_horizons_fixed_three_points():
    assert HORIZONS==(.8,1.6,3.2)


def test_800ms_exact_physical_reproduction_and_mismatch():
    old=snap()
    assert_reproduced(old,replace(old),label='baseline')
    with pytest.raises(RuntimeError,match='physical outcome'):
        assert_reproduced(replace(old,misses=1,miss_ids=(3,)),old,label='baseline')
    with pytest.raises(RuntimeError,match='timing mismatch'):
        assert_reproduced(replace(old,keydowns=tuple([('right_3',0.05)]+list(old.keydowns[1:]))),old,label='baseline')
    with pytest.raises(RuntimeError,match='timing mismatch'):
        assert_reproduced(replace(old,time_s=.7),old,label='baseline')


def test_early_overload_avoids_unobserved_future_imputation():
    baseline=snap(t=.08,hits=(0,1),early=3,down=3,overloaded=True,
                  reason='Overload',terminated=True)
    # Common 80ms has temporarily lagging target 1; it is recovered by .8s.
    candidate_at_80=snap(t=.08,hits=(0,),early=1,down=1)
    candidate_end=snap(hits=(0,1,2,3,4,5,6,7,8,9),misses=(10,11),early=3,down=12)
    # The first frame's 80ms TooEarly is lower, new area has 2 misses.
    ref=trace(baseline)
    cand=trace(candidate_end,frames=(candidate_at_80,candidate_end))
    result=compare_horizon(ref,cand,.8)
    assert result['reference_reached_horizon'] is False
    assert result['candidate_reached_horizon'] is True
    assert result['full_misses_not_increased'] is None
    assert result['recovered_lagging_hit_ids']==[1]
    assert result['extension_miss_ids_unpaired']==[10,11]
    assert result['overload_rescued']
    assert result['verdict']=='conditional_rescue_review'
    assert not result['safe_observed_improvement']


def test_same_800ms_outcome_early_reduction_with_fewer_keydowns_is_allowed():
    orig=snap(hits=(0,1,2,3,4),misses=(5,),early=3,down=9)
    better=snap(hits=(0,1,2,3,4),misses=(5,),early=2,down=7)
    compared=compare_horizon(trace(orig),trace(better),.8)
    assert compared['safe_observed_improvement']
    assert compared['verdict']=='verified_common_horizon_improvement'
    assert compared['candidate_keydowns']<compared['reference_keydowns']


def test_zero_hit_no_press_cannot_qualify_as_improvement():
    baseline=snap(hits=(),early=3,down=1)
    silence=snap(hits=(),early=0,down=0)
    result=compare_horizon(trace(baseline),trace(silence),.8)
    assert result['verdict']=='insufficient_physical_evidence'
    assert not result['safe_observed_improvement']


def test_lost_target_ids_are_detected_even_equal_total_hits():
    baseline=snap(hits=(0,1,2),early=3)
    candidate=snap(hits=(0,2,3),early=1)
    result=compare_horizon(trace(baseline),trace(candidate),.8)
    assert result['hit_ids_lost_by_horizon']==[1]
    assert result['verdict']=='hit_suppression'


def test_1600ms_late_overload_is_not_counted_as_stable_800ms_improvement():
    baseline=snap(t=.8,early=3)
    good=snap(t=.8,early=1)
    bad=snap(t=1.1,hits=(0,1,2,3),early=3,overloaded=True,terminated=True,reason='Overload')
    old16=snap(t=1.6,hits=(0,1,2,3),early=3)
    early_good=trace(good,v16=bad,frames=(good,bad))
    reference=trace(baseline,v16=old16,frames=(baseline,old16))
    assert compare_horizon(reference,early_good,.8)['safe_observed_improvement']
    assert compare_horizon(reference,early_good,1.6)['verdict']=='deferred_failure'


def test_local_perturbations_keep_original_duration_and_nonnegative_start():
    c=Intervention(6,(2,),-.20,0,8)
    neighbors=local_neighbors(c)
    assert len(neighbors)==5
    assert all(x.frames==8 and x.keys==(2,) and x.start in (0,1) for x in neighbors)
    assert all(x.identity()!=c.identity() and abs(x.delta-c.delta)<=.0200001
               for x in neighbors)
    assert {x.delta for x in neighbors}=={-.22,-.20,-.18}
    c2=Intervention(18,(5,),-.20,4,1)
    assert len(local_neighbors(c2))==8
    assert {x.start for x in local_neighbors(c2)}=={3,4,5}


def test_case_selection_six_original_representatives_conditional_and_controls():
    names=('left_4','left_3','left_2','left_1','right_1','right_2','right_3','right_4')
    rows=[]
    def make(c,*,source='v11_short_pass',safe=True,conditional=False,group='unique'):
        return {'anchor':str(c.anchor),'keys':','.join(str(k) for k in c.keys),
                'delta':str(c.delta),'start_frame':str(c.start),'duration_frames':str(c.frames),
                'kind':'all_8' if len(c.keys)==8 else 'single',
                'source':source,'safe_review_candidate':str(safe),
                'conditional_review_candidate':str(conditional),
                'outcome_group_signature':group, 'hits_800': '0', 'misses_800':'0', 'early_800':'0'}
    for i,raw in enumerate(REPRESENTATIVES):
        c=Intervention(raw[0],raw[1],raw[2],raw[3],raw[4])
        rows.append(make(c,source='v1_manual_unsampled' if c.anchor==20 else 'v11_short_pass',group=f'rep{i}'))
    for i,score in enumerate(((4,1,2),(4,1,1),(5,0,1),(11,0,3),(11,0,2),(13,0,2))):
        rows[i].update(hits_800=str(score[0]), misses_800=str(score[1]), early_800=str(score[2]))
    conditional=make(Intervention(8,tuple(range(8)),-.20,0,8),safe=False,conditional=True)
    conditional.update(hits_800='10',misses_800='2')
    rows.append(conditional)
    # Include 1 same-outcome alternative shape for #06 without relying on later success.
    rows.append(make(Intervention(6,(0,),+.02,0,4),group='rep0'))
    for k in range(152-len(rows)):
        rows.append(make(Intervention(1,(1,),.02+k*.0001,1,1),safe=False))
    selected=select_cases(rows,names)
    assert sum(role=='representative' for _,_,role in selected)==6
    assert sum(role=='conditional_08' for _,_,role in selected)==1
    assert any(role=='same_outcome_control' for _,_,role in selected)
    assert selected[-1][0].anchor==6
    rows[0]['safe_review_candidate']='False'
    with pytest.raises(RuntimeError,match='was not v1.3 safe'):
        select_cases(rows,names)


def test_numeric_key_indices_from_saved_v12_are_parsed_exactly():
    names=('left_4','left_3','left_2','left_1','right_1','right_2','right_3','right_4')
    row={'anchor':'6','keys':'2','delta':'-0.2','start_frame':'0',
         'duration_frames':'8','kind':'single'}
    assert case_from_row(row,names)==Intervention(6,(2,),-.2,0,8)
    row['keys']='0,1,2,3,4,5,6,7';row['kind']='all_8'
    assert case_from_row(row,names).keys==tuple(range(8))

def test_selected_candidate_missing_or_duplicate_fails_closed():
    names=('left_4','left_3','left_2','left_1','right_1','right_2','right_3','right_4')
    with pytest.raises(RuntimeError,match='152'):
        select_cases([],names)
    with pytest.raises(RuntimeError,match='intervention geometry invalid'):
        case_from_row({'anchor':'1','keys':'left_2,left_2','delta':'-.2',
                       'start_frame':'0','duration_frames':'8','kind':'paired'},names)


def test_no_existing_v14_output_can_be_overwritten(tmp_path):
    inputs=[tmp_path/'v12',tmp_path/'v13']
    best=tmp_path/'checkpoints'/'best.pt'
    directory=tmp_path/'out'
    directory.mkdir();(directory/'important.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        _check_output_dir(directory,inputs,best)
    assert (directory/'important.txt').read_text()=='preserve'
    with pytest.raises(ValueError,match='overlaps'):
        _check_output_dir(inputs[0],inputs,best)


def test_visible_neighbor_search_never_crosses_to_validation():
    a=snap()
    tracks={i:trace(a,features=tuple(
        tuple(float(i)+step/10 for _ in range(263))+tuple(float(step%2) for _ in range(8))
        for step in range(10))) for i in range(1,21)}
    c=Intervention(6,(2,),-.2,0,8)
    report=_visible_neighbors(tracks,[(c,{},'representative')])
    assert len(report)==1
    assert report[0]['status']=='Train_only_observable_similarity'
    assert len(report[0]['nearest_other_train_states'])==3
    assert all(x['anchor']!=6 for x in report[0]['nearest_other_train_states'])
    assert not report[0]['transferred_intervention_evaluated']


def test_late_snapshots_are_observed_not_extrapolated():
    baseline=snap(t=.08,hits=(0,),overloaded=True,terminated=True,reason='Overload')
    tr=trace(baseline)
    assert tr.at[.8].time_s==.08
    assert tr.at[1.6].time_s==.08
    assert tr.at[3.2].time_s==.08
    with pytest.raises(RuntimeError,match='no common'):
        snapshot_at_common((baseline,),.01)


def test_v13_provenance_and_original_152_rows_loaded_without_gpu(tmp_path,monkeypatch):
    """Exercise the exact existing v1.2->v1.3 artifact layout, not invented JSON."""
    import test_train_counterfactual_motor_v13 as oldtests
    import dmdod.training.train_counterfactual_motor_v14 as mod
    from dmdod.training.train_counterfactual_motor_v13 import run_v13
    previous=oldtests._fixture(tmp_path)
    result=run_v13(previous,tmp_path/'v13')
    best=tmp_path/'best.pt'
    best.write_bytes(b'fake CPU-only checkpoint; actual hash guarded separately')
    real_hash=mod.file_hash
    monkeypatch.setattr(mod,'file_hash',lambda p: mod.EXPECTED_BEST_SHA256
                        if Path(p)==best else real_hash(p))
    files,hashes,rows,baselines,checks=mod._load_inputs(previous,tmp_path/'v13',best)
    assert len(rows)==152 and len(baselines)==20 and len(checks)==152
    assert len(files)==9
    assert result['input_candidates']==152
    # Tampered source must fail closed even though v1.3 has already been emitted.
    with (previous/'candidate_comparison.csv').open('a',encoding='utf-8') as f:
        f.write('corruption\n')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        mod._load_inputs(previous,tmp_path/'v13',best)


def test_sequential_263d_student_replay_records_three_real_checkpoints(monkeypatch):
    """Exercise 320 control steps with deterministic mock physical environment."""
    import dmdod.training.train_counterfactual_motor_v14 as v14
    import sys,types
    class Model:
        def __init__(self):self.restarts=0;self.inferences=0
        def initial_state(self,device):
            self.restarts+=1
            return 0
        def eval(self):return self
        def forward_step(self,x,state):
            self.inferences+=1
            assert x.shape==(263,)
            assert state==self.inferences-1 if self.restarts==1 else True
            return torch.zeros(8),None,None,state+1
    class Env:
        def __init__(self,stop=None):
            self._done=False;self.stop=stop;self.n=0
            self._used=[False]*10;self._missed=[False]*10
            self._hits=0;self._misses=0;self._too_early=0
            self._overload=SimpleNamespace(overloaded=False)
            self._trace_overload_event_time_s=None
            self._trace_physical_keydowns=[];self._failed_on_miss=False
        def reset(self):
            self.n=0
            return SimpleNamespace(t=0)
        def privileged_episode_time_s(self):return self.n*.01
        def step(self,action):
            assert len(action.values)==8
            self.n+=1
            if self.n==50:self._used[0]=True;self._hits=1
            self._done=(self.n==self.stop if self.stop else False)
            if self._done:
                self._overload.overloaded=True
                self._trace_overload_event_time_s=self.n*.01
            return SimpleNamespace(observation=SimpleNamespace(t=self.n),done=self._done)
    class Action:
        def __init__(self,values):self.values=values
    fake_env_module=types.ModuleType('dmdod.envs.n_key')
    fake_env_module.encode_n_key_hud_real_chart_observation=lambda obs:[float(obs.t)]*263
    fake_motor_module=types.ModuleType('dmdod.motor.n_key')
    fake_motor_module.NKeyAction=Action
    fake_motor_sens=types.ModuleType('dmdod.training.motor_sensitivity_audit')
    env_instances=[]
    def make_env(prep,named):
        env=Env(stop=named.stop)
        env_instances.append(env)
        return env
    fake_motor_sens._environment=make_env
    monkeypatch.setitem(sys.modules,'dmdod.envs.n_key',fake_env_module)
    monkeypatch.setitem(sys.modules,'dmdod.motor.n_key',fake_motor_module)
    monkeypatch.setitem(sys.modules,'dmdod.training.motor_sensitivity_audit',fake_motor_sens)
    student=Model()
    prepared=SimpleNamespace(model=student,device=torch.device('cpu'),control_dt_s=.01)
    with torch.no_grad():
        live=v14.run_long_trace(prepared,SimpleNamespace(stop=None),case=None,capture_visible=True)
    assert len(live.frames)==320
    assert [live.at[h].time_s for h in HORIZONS]==pytest.approx([.8,1.6,3.2])
    assert len(live.observable)==20
    assert all(len(x)==271 for x in live.observable)
    assert student.restarts==1 and student.inferences==320
    short=v14.run_long_trace(prepared,SimpleNamespace(stop=8),case=Intervention(6,(2,),-.2,0,4))
    assert len(short.frames)==8
    assert short.at[3.2].time_s==pytest.approx(.08)
    assert short.at[.8].overloaded
    assert student.restarts==2 and student.inferences==328
