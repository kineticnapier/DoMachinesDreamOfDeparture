from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training.train_counterfactual_motor_v12 import Snapshot
from dmdod.training.train_counterfactual_motor_v14 import HORIZONS, Intervention, Trace
from dmdod.training.train_counterfactual_motor_v15 import (
    GRID_06, FrameContext, DetailedReplay, ReplayCache, _required_layout,
    _snapshot, _compare_saved_metric, _load_previous, first_divergence,
    first_hit_loss, first_miss_excess, milestone_gate, persistence,
    rescue_quality, boundary_grid, run_detailed_trace,
)


def snap(t=.8, hits=(0,1), misses=(), early=3, downs=3, overload=False,
         terminated=False):
    return Snapshot(
        time_s=t,hits=len(hits),misses=len(misses),too_early=early,
        overloaded=overload,overload_time_s=t if overload else None,
        terminated=terminated,
        termination_reason=('Overload' if overload else
                            'Miss failure' if terminated else 'Horizon cutoff'),
        hit_ids=tuple(hits),miss_ids=tuple(misses),
        keydowns=tuple(('left_2',.010*i) for i in range(downs)),
    )


def sample_replay(*,v8=None,v16=None,v32=None,steps=None):
    s8=v8 or snap();s16=v16 or replace(s8,time_s=1.6)
    s32=v32 or replace(s16,time_s=3.2)
    milestones={.8:s8,1.6:s16,3.2:s32}
    frames=tuple(steps) if steps is not None else (s8,s16,s32)
    info=tuple(FrameContext(s.time_s,(0.,)*263,(0.,)*8,(0.,)*8,()) for s in frames)
    return DetailedReplay(Trace(milestones,frames,()),info)


def test_fixed_9x2_boundary_and_nonmonotonic_region():
    assert len(GRID_06)==9
    r=sample_replay(v8=snap(hits=(0,1,2),early=3),
                    v16=snap(t=1.6,hits=(0,1,2,3),early=3),
                    v32=snap(t=3.2,hits=(0,1,2,3,4),early=3))
    rows=[]
    for i in range(18):
        good=i in (0,4,9,17)  # deliberately nonmonotonic
        s8=snap(hits=(0,1,2,3,4) if good else (0,1),early=1 if good else 3,misses=())
        s16=snap(t=1.6,hits=(0,1,2,3,4,5) if good else (0,1,2),early=1 if good else 3)
        s32=snap(t=3.2,hits=(0,1,2,3,4,5,6) if good else (0,1,2),early=1 if good else 3)
        case=Intervention(6,(2,),GRID_06[i%9],i//9,8)
        rows.append((case,sample_replay(v8=s8,v16=s16,v32=s32)))
    result=boundary_grid(rows,r)
    assert len(result)==18
    assert [i for i,x in enumerate(result) if x['success_grid_point']]==[0,4,9,17]
    assert result[0]['source_800ms_5H_0M_1E']
    assert not result[2]['success_grid_point']


def test_paired_post800_gains_distinguish_initial_gain():
    base=sample_replay(v8=snap(hits=(0,1)),v16=snap(t=1.6,hits=(0,1,2)),
                       v32=snap(t=3.2,hits=(0,1,2,3),misses=(9,)))
    alt=sample_replay(v8=snap(hits=(0,1,4)),v16=snap(t=1.6,hits=(0,1,2,4,5)),
                      v32=snap(t=3.2,hits=(0,1,2,3,4,5,6),misses=()))
    p=persistence(base,alt)
    assert p['initial_hit_delta_800']==1
    assert p['post800_hit_gain']==(7-3)-(4-2)==2
    assert p['post800_miss_gain']==(0-0)-(1-0)==-1


def test_early_overload_yields_unpaired_continuation_not_fake_gain():
    stopped=snap(t=.1,hits=(0,1),overload=True,terminated=True)
    base=sample_replay(v8=stopped,v16=stopped,v32=stopped,steps=(stopped,))
    cand=sample_replay(v8=snap(hits=(0,1,2,3)),
                       v16=snap(t=1.6,hits=(0,1,2,3,4)),
                       v32=snap(t=3.2,hits=(0,1,2,3,4,5),misses=(8,)))
    p=persistence(base,cand)
    assert p['paired_800_to_3200_observed'] is False
    assert p['post800_hit_gain'] is None
    assert p['post800_miss_gain'] is None
    assert p['candidate_continuation_hits_after800']==2
    assert p['baseline_unobserved_future']
    assert not milestone_gate(base.trace,cand.trace,3.2)['passed']


def test_first_target_loss_and_recovery_and_confirmation_clock():
    base=sample_replay(
        v8=snap(hits=(0,1,2)),v16=snap(t=1.6,hits=(0,1,2,3)),
        v32=snap(t=3.2,hits=(0,1,2,3,4)),
        steps=(snap(t=.4,hits=(0,1)),snap(t=.8,hits=(0,1,2)),
               snap(t=1.6,hits=(0,1,2,3)),snap(t=3.2,hits=(0,1,2,3,4))))
    cand=sample_replay(
        v8=snap(hits=(0,2)),v16=snap(t=1.6,hits=(0,1,2,3)),
        v32=snap(t=3.2,hits=(0,1,2,3,4)),
        steps=(snap(t=.4,hits=(0,)),snap(t=.8,hits=(0,2)),
               snap(t=1.2,hits=(0,1,2)),snap(t=1.6,hits=(0,1,2,3)),
               snap(t=3.2,hits=(0,1,2,3,4))))
    loss=first_hit_loss(base,cand)
    assert loss['first_milestone_hit_loss_s']==.8
    assert loss['lost_target_id']==1
    assert loss['baseline_hit_confirmed_s']==.4
    assert loss['candidate_hit_confirmed_s']==1.2
    assert loss['recovered_by_3200']


def test_late_miss_excess_discovered_at_observed_time():
    b=sample_replay(steps=(snap(t=.1),snap(t=.2),snap(t=.3)))
    c=sample_replay(steps=(snap(t=.1),snap(t=.2,misses=(4,)),
                           snap(t=.3,misses=(4,))))
    assert first_miss_excess(b,c)==.2


def test_first_divergence_is_observed_and_keeps_hud_raw_and_motor_separate():
    b=sample_replay(steps=(snap(t=.1,downs=0),snap(t=.2,downs=1)))
    c=sample_replay(steps=(snap(t=.1,downs=0),snap(t=.2,downs=0)))
    result=first_divergence(b,c)
    assert result['first_divergence_s']==.2
    assert result['observed_not_causal']
    assert len(result['baseline_before_step_visible_hud'])==263
    assert len(result['candidate_student_action'])==8
    assert len(result['candidate_executed_action'])==8


def test_rescue_survival_and_quality_independent_and_denominator_kept():
    base=snap(t=.08,hits=(0,),overload=True,terminated=True)
    reference=sample_replay(v8=base,v16=base,v32=base)
    good=sample_replay(v8=snap(hits=(0,1),misses=(3,),early=1),
                       v16=snap(t=1.6,hits=(0,1,2),misses=(3,)),
                       v32=snap(t=3.2,hits=(0,1,2,4),misses=(3,)))
    out=rescue_quality(reference,good,Intervention(8,tuple(range(8)),-.2,0,8))
    assert all(x['survival_success'] for x in out)
    assert all(not x['quality_success'] for x in out)
    assert out[-1]['miss_fraction_of_resolved']==pytest.approx(1/5)
    assert out[-1]['new_misses_after800']==0
    bad=sample_replay(v8=snap(t=.11,hits=(0,),misses=(),overload=True,terminated=True))
    assert not rescue_quality(reference,bad,Intervention(8,(2,),-.2,0,8))[0]['survival_success']


def test_missing_hit_evidence_never_passes_boundaries_or_quality():
    base=sample_replay(v8=snap(hits=(),downs=0))
    cand=sample_replay(v8=snap(hits=(),downs=0,early=0))
    assert not milestone_gate(base.trace,cand.trace,.8)['passed']
    assert not rescue_quality(base,cand,Intervention(8,(2,),-.2,0,8))[0]['quality_success']


def test_v14_metric_mismatch_fails_closed():
    old={'horizon_s':.8,'reference_hits':3,'candidate_hits':5}
    with pytest.raises(RuntimeError,match='Fail Closed'):
        _compare_saved_metric({'horizon_s':.8,'reference_hits':3,'candidate_hits':4},old,'test')
    with pytest.raises(RuntimeError,match='Fail Closed'):
        _compare_saved_metric({'horizon_s':.8,'reference_hits':3,'candidate_hits':5},old,'missing')


def test_v14_snapshots_support_3200ms_and_strict_corruption_checks():
    row={'time_s':'3.2','hits':'1','misses':'0','too_early':'2',
         'overloaded':'False','overload_time_s':'','terminated':'False',
         'termination_reason':'Horizon cutoff','hit_ids':'[1]',
         'miss_ids':'[]','keydowns':'[["left_2",0.5]]'}
    s=_snapshot(row,'v1.4/3.2')
    assert s.time_s==3.2 and s.hit_ids==(1,)
    broken=dict(row,hits='2')
    with pytest.raises(RuntimeError,match='Fail Closed'):
        _snapshot(broken,'broken')
    broken=dict(row,hit_ids='[1, 1]')
    with pytest.raises(RuntimeError,match='Fail Closed'):
        _snapshot(broken,'duplicate ids')


def test_replay_cache_never_replays_same_case(monkeypatch):
    from dmdod.training import train_counterfactual_motor_v15 as m
    calls=[]
    def mock(prepared,named,*,case):
        calls.append(case.identity() if case is not None else 'baseline')
        return sample_replay()
    monkeypatch.setattr(m,'run_detailed_trace',mock)
    runner=ReplayCache(SimpleNamespace(anchors=[object()]*20))
    a=Intervention(6,(2,),-.2,0,8)
    assert runner.get(a) is runner.get(a)
    assert runner.get(None,anchor=6) is runner.get(None,anchor=6)
    assert runner.executions==2 and runner.cache_hits==2
    assert len(calls)==2


def test_reject_existing_output_and_input_overlap_and_wrong_train_geometry(tmp_path):
    source=tmp_path/'best.pt';source.write_bytes(b'fixed')
    inputs=(tmp_path/'v12',tmp_path/'v13',tmp_path/'v14')
    prepared=SimpleNamespace(anchors=[object()]*20,key_count=8,input_dim=263,
        control_dt_s=.010,physics_dt_s=.001,source_checkpoint=source)
    out=tmp_path/'new'
    _required_layout(prepared,out,inputs)
    out.mkdir()
    with pytest.raises(FileExistsError):_required_layout(prepared,out,inputs)
    with pytest.raises(ValueError,match='overlaps'):_required_layout(prepared,inputs[0]/'nested',inputs)
    prepared.input_dim=262
    with pytest.raises(ValueError,match='263D'):_required_layout(prepared,tmp_path/'unused',inputs)


def test_runner_rejects_corrupt_v14_before_any_replay(tmp_path,monkeypatch):
    import dmdod.training.train_counterfactual_motor_v15 as m
    p=SimpleNamespace(anchors=[object()]*20,key_count=8,input_dim=263,
        control_dt_s=.010,physics_dt_s=.001,source_checkpoint=tmp_path/'best.pt')
    p.source_checkpoint.write_bytes(b'checkpoint')
    def fail(*args,**kwargs):raise RuntimeError('Fail Closed: v1.4 hash mismatch')
    monkeypatch.setattr(m,'_load_previous',fail)
    monkeypatch.setattr(m,'run_detailed_trace',lambda *a,**k:pytest.fail('GPU/physical replay invoked before validation'))
    with pytest.raises(RuntimeError,match='Fail Closed'):
        m.run_v15(p,v12_dir=tmp_path/'v12',v13_dir=tmp_path/'v13',
                  v14_dir=tmp_path/'v14',output=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_no_training_validation_or_final_imports_in_diagnostic_source():
    import inspect
    from dmdod.training import train_counterfactual_motor_v15 as mod
    s=inspect.getsource(mod)
    assert 'optimizer.step(' not in s
    assert 'evaluate_role_continuous(' not in s
    assert '_collect_dagger_trajectories(' not in s
    assert 'build_validation_segments(' not in s
    assert 'torch.save(' not in s


def test_saved_v14_reproduction_must_have_real_hash_and_9_cases(tmp_path,monkeypatch):
    """Guard is evaluated before Stage A; no mocked physics can bypass it."""
    import dmdod.training.train_counterfactual_motor_v15 as mod
    monkeypatch.setattr(mod,'_load_inputs',lambda *args: ({}, {}, [], {}, {}))
    d=tmp_path/'v14';d.mkdir()
    for f,body in (
        ('diagnostic.json',{'kind':'train_counterfactual_motor_v14','best_sha256':'bad'}),
        ('reproduction_checks.json',{'status':'PASS'}),
        ('baseline_continuation.csv','anchor,horizon_s\n1,0.8\n'),
        ('long_horizon_comparison.csv','anchor\n1\n'),
        ('local_neighbors.csv','anchor\n1\n'),
        ('local_robustness.csv','anchor\n1\n'),
    ):
        path=d/f
        path.write_text(json.dumps(body) if isinstance(body,dict) else body,encoding='utf-8')
    with pytest.raises(RuntimeError,match='Fail Closed'):
        mod._load_previous(tmp_path/'v12',tmp_path/'v13',d,tmp_path/'best.pt')


def test_hit_lag_between_milestones_can_recover_before_1600():
    base=sample_replay(
        v8=snap(t=.8,hits=(0,)),v16=snap(t=1.6,hits=(0,1)),
        v32=snap(t=3.2,hits=(0,1,2)),
        steps=(snap(t=.8,hits=(0,)),snap(t=1.1,hits=(0,1)),
               snap(t=1.6,hits=(0,1)),snap(t=3.2,hits=(0,1,2))))
    cand=sample_replay(
        v8=snap(t=.8,hits=(0,)),v16=snap(t=1.6,hits=(0,1)),
        v32=snap(t=3.2,hits=(0,1,2)),
        steps=(snap(t=.8,hits=(0,)),snap(t=1.1,hits=(0,)),
               snap(t=1.6,hits=(0,1)),snap(t=3.2,hits=(0,1,2))))
    result=first_hit_loss(base,cand)
    assert result['first_observed_hit_lag_s']==1.1
    assert result['first_milestone_hit_loss_s'] is None
    assert result['lost_target_id']==1
    assert result['recovered_by_3200']


def test_320_control_step_mock_replay_hud_is_263d_and_no_training(monkeypatch):
    import sys,types
    from dmdod.training import train_counterfactual_motor_v15 as mod
    class Student:
        def __init__(self):self.restarts=0;self.steps=0
        def initial_state(self,device):self.restarts+=1;return 0
        def eval(self):return self
        def forward_step(self,features,state):
            assert features.shape==(263,)
            self.steps+=1
            return torch.zeros(8),None,None,state+1
    class Environment:
        def __init__(self):
            self.n=0;self._used=[False]*30;self._missed=[False]*30
            self._hits=0;self._misses=0;self._too_early=0
            self._overload=SimpleNamespace(overloaded=False)
            self._trace_overload_event_time_s=None
            self._trace_physical_keydowns=[];self._failed_on_miss=False
        def reset(self):self.n=0;return SimpleNamespace(t=0.)
        def privileged_episode_time_s(self):return self.n*.01
        def step(self,action):
            self.n+=1
            assert len(action.values)==8
            if self.n==10:self._hits=1;self._used[0]=True
            return SimpleNamespace(observation=SimpleNamespace(t=self.n*.01),done=False)
    class NKeyAction:
        def __init__(self,values):self.values=values
    hud=types.ModuleType('dmdod.envs.n_key')
    hud.encode_n_key_hud_real_chart_observation=lambda obs:[obs.t]*263
    motor=types.ModuleType('dmdod.motor.n_key');motor.NKeyAction=NKeyAction
    sensitivity=types.ModuleType('dmdod.training.motor_sensitivity_audit')
    sensitivity._environment=lambda prep,named:Environment()
    monkeypatch.setitem(sys.modules,'dmdod.envs.n_key',hud)
    monkeypatch.setitem(sys.modules,'dmdod.motor.n_key',motor)
    monkeypatch.setitem(sys.modules,'dmdod.training.motor_sensitivity_audit',sensitivity)
    student=Student();prepared=SimpleNamespace(model=student,device=torch.device('cpu'),control_dt_s=.01)
    result=run_detailed_trace(prepared,object(),case=Intervention(6,(2,),-.2,0,8))
    assert student.steps==320 and student.restarts==1
    assert [result.trace.at[h].time_s for h in HORIZONS]==pytest.approx([.8,1.6,3.2])
    assert len(result.context)==320 and len(result.context[0].visible_hud)==263
    assert result.context[0].student_action==(0.,)*8
    assert result.context[0].executed_action[2]==pytest.approx(-.2)
    assert result.context[8].executed_action==(0.,)*8
    assert result.trace.at[3.2].hits==1


def test_runner_cpu_mock_integration_outputs_all_six_files(monkeypatch,tmp_path):
    import sys,types
    from dmdod.training import train_counterfactual_motor_v15 as mod
    from dmdod.training.train_counterfactual_motor_v14 import REPRESENTATIVES
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.]))
        def prepare_recurrent_runtime(self):pass
    model=Model()
    best=tmp_path/'best.pt';best.write_bytes(b'best is read only')
    prepared=SimpleNamespace(
        anchors=[SimpleNamespace(chart_name=f'Train {i}') for i in range(20)],
        model=model,device=torch.device('cpu'),key_count=8,input_dim=263,
        control_dt_s=.010,physics_dt_s=.001,source_checkpoint=best)
    def fake_trace(prepared,named,*,case):
        # A stable physical result independent of the intervention is valid
        # for the serialization / caching / no-side-effects integration path.
        a=snap(t=.8,hits=(0,),early=3)
        b=snap(t=1.6,hits=(0,1),early=3)
        c=snap(t=3.2,hits=(0,1,2),early=3)
        return sample_replay(v8=a,v16=b,v32=c,steps=(a,b,c))
    monkeypatch.setattr(mod,'run_detailed_trace',fake_trace)
    selected=[]
    for raw in REPRESENTATIVES:
        selected.append((Intervention(*raw),{},'representative'))
    selected.append((Intervention(8,tuple(range(8)),-.2,0,8),{},'conditional_08'))
    selected.append((Intervention(6,(0,),-.05,0,1),{},'same_outcome_control'))
    selected.append((Intervention(18,(3,),-.05,0,1),{},'same_outcome_control'))
    static=fake_trace(None,None,case=None)
    saved_base={(a,h):static.trace.at[h] for a in range(1,21) for h in HORIZONS}
    saved={case.identity():{'candidate_snapshots':{str(h):static.trace.at[h].as_json()
                    for h in HORIZONS}} for case,_,_ in selected}
    monkeypatch.setattr(mod,'_load_previous',lambda *a:( {},{},selected,saved_base,saved,{}))
    monkeypatch.setattr(mod,'file_hash',lambda path:mod.EXPECTED_BEST_SHA256)
    monkeypatch.setattr(mod,'_hash_inputs',lambda inputs:{})
    fixed=types.ModuleType('dmdod.training.fixed_state_action_audit')
    fixed.assert_state_equal=lambda a,b,*,context: assert_tensors_equal(a,b)
    real_chart=types.ModuleType('dmdod.training.real_chart')
    real_chart.clone_model_state=lambda m:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    monkeypatch.setitem(sys.modules,'dmdod.training.fixed_state_action_audit',fixed)
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',real_chart)
    output=tmp_path/'v15'
    report=mod.run_v15(prepared,v12_dir=tmp_path/'v12',v13_dir=tmp_path/'v13',
                       v14_dir=tmp_path/'v14',output=output)
    assert report['kind']=='train_counterfactual_motor_v15'
    assert report['v14_representatives']==9
    assert report['boundary_06']['tested_grid_points']==18
    assert report['late_drift_18']['tested_grid_points']==18
    assert report['replay_cache']['cache_hits']>0
    assert len(list(output.iterdir()))==7
    assert set(x.name for x in output.iterdir()) == {
        'diagnostic.json','gain_persistence.csv','boundary_grid_06.csv',
        'late_drift_18.csv','event_divergence.csv','rescue_survival_quality.csv',
        'reproduction_checks.json'}
    with (output/'rescue_survival_quality.csv').open(encoding='utf-8-sig') as stream:
        import csv
        rescue=list(csv.DictReader(stream))
    assert len(rescue)==36 # 2 anchors x6 cases x3 horizons
    assert all(row['anchor'] in ('8','20') for row in rescue)
    assert model.weight.item()==1.
    assert best.read_bytes()==b'best is read only'
    with pytest.raises(FileExistsError):
        mod.run_v15(prepared,v12_dir=tmp_path/'v12',v13_dir=tmp_path/'v13',
                    v14_dir=tmp_path/'v14',output=output)


def assert_tensors_equal(actual,expected):
    assert set(actual)==set(expected)
    for key in actual:
        assert torch.equal(actual[key],expected[key])


def test_first_divergence_includes_before_and_after_visible_actions():
    b=sample_replay(steps=(snap(t=.1,downs=0),snap(t=.2,downs=1),snap(t=.3,downs=1)))
    c=sample_replay(steps=(snap(t=.1,downs=0),snap(t=.2,downs=0),snap(t=.3,downs=1)))
    observed=first_divergence(b,c)
    assert observed['baseline_previous_step']['time_s']==pytest.approx(.1)
    assert observed['candidate_next_step']['time_s']==pytest.approx(.3)
    assert len(observed['baseline_previous_step']['visible_hud'])==263
    assert len(observed['candidate_next_step']['student_action'])==8


def test_too_early_not_called_continuous_if_benefit_disappears_at_1600():
    from dmdod.training.train_counterfactual_motor_v15 import too_early_reduction_duration
    base=sample_replay(v8=snap(early=3),v16=snap(t=1.6,early=4),v32=snap(t=3.2,early=5))
    cand=sample_replay(v8=snap(early=1),v16=snap(t=1.6,early=4),v32=snap(t=3.2,early=4))
    report=too_early_reduction_duration(base,cand)
    assert report['continuously_reduced_through_s']==.8
    assert report['last_observed_lower_s']==3.2


def test_paired_target_id_deltas_never_impute_unobserved_baseline():
    from dmdod.training.train_counterfactual_motor_v15 import horizon_target_deltas
    ref=sample_replay(v8=snap(hits=(0,1),misses=(8,)))
    alt=sample_replay(v8=snap(hits=(0,2),misses=(7,)))
    delta=horizon_target_deltas(ref.trace,alt.trace,.8)
    assert delta['paired_hit_gains']==[2]
    assert delta['paired_hit_losses']==[1]
    assert delta['paired_miss_gains']==[7]
    assert delta['paired_miss_losses']==[8]
    lost=snap(t=.1,hits=(0,),overload=True,terminated=True)
    premature=sample_replay(v8=lost,v16=lost,v32=lost)
    not_pair=horizon_target_deltas(premature.trace,alt.trace,.8)
    assert not_pair['both_observed'] is False
    assert not_pair['paired_miss_gains'] is None
    assert not_pair['paired_hit_losses'] is None


def test_first_miss_target_id_change_detected_even_with_same_miss_count():
    from dmdod.training.train_counterfactual_motor_v15 import first_miss_id_difference
    ref=sample_replay(steps=(snap(t=.1,misses=()),snap(t=.2,misses=(8,))))
    alt=sample_replay(steps=(snap(t=.1,misses=()),snap(t=.2,misses=(7,))))
    delta=first_miss_id_difference(ref,alt)
    assert delta['time_s']==.2
    assert delta['candidate_new_miss_ids']==[7]
    assert delta['baseline_only_miss_ids']==[8]
