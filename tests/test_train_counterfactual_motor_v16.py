"""CPU-only, deterministic contracts for Train state-conditioned v1.6 diagnosis."""
from __future__ import annotations

from dataclasses import replace
import csv
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmdod.training.train_counterfactual_motor_v12 import Snapshot
from dmdod.training.train_counterfactual_motor_v14 import Intervention, Trace
from dmdod.training.train_counterfactual_motor_v15 import DetailedReplay, FrameContext
from dmdod.training import train_counterfactual_motor_v16 as m


def snap(h=3.2, hits=(0,1), misses=(), early=3,overload=False,downs=None):
    kd=downs if downs is not None else (("left_1",.02),("left_2",.04))
    return Snapshot(time_s=h,hits=len(hits),misses=len(misses),too_early=early,
        overloaded=overload,overload_time_s=h-.01 if overload else None,
        terminated=overload, termination_reason='Overload' if overload else 'Horizon cutoff',
        hit_ids=tuple(hits),miss_ids=tuple(misses),keydowns=kd)


def playback(s8=None,s16=None,s32=None,n=10, action=0.11):
    s8=s8 or snap(.8);s16=s16 or snap(1.6);s32=s32 or snap(3.2)
    c=tuple(FrameContext(time_s=(i+1)*.01,
          visible_hud=tuple([float(i)]*263),student_action=tuple([action]*8),
          executed_action=tuple([action]*8),new_keydowns=()) for i in range(n))
    return DetailedReplay(Trace({.8:s8,1.6:s16,3.2:s32},
                                (s8,s16,s32),()),c)


def case(anchor=6,keys=(2,),delta=-.20,start=0,frames=8):
    return Intervention(anchor,keys,delta,start,frames)


def test_exact_discovery_matrix_excludes_focus_and_has_all_keys():
    cases=m.discovery_cases()
    assert len(cases)==288==len({c.identity() for c in cases})
    assert {c.anchor for c in cases}==set(range(1,21))-{6,8,18,20}
    assert {c.start for c in cases}=={0,5}
    assert {c.keys for c in cases}=={*((i,) for i in range(8)),tuple(range(8))}
    assert all((c.delta,c.frames)==(-.20,8) for c in cases)


def test_preaction_reads_origin_not_perturbed_future():
    ref=playback(n=10)
    p=m.checked_preaction(ref,case(start=5))
    assert p['visible_hud']==[5.]*263
    assert p['student_raw_action']==[.11]*8
    bad=list(ref.context)
    bad[5]=replace(bad[5],visible_hud=tuple([99.]*263),
                   executed_action=tuple([-.2]*8))
    altered=DetailedReplay(ref.trace,tuple(bad))
    with pytest.raises(RuntimeError,match='pre-intervention state differs'):
        m.assert_preaction_unchanged(ref,altered,case(start=5))


def test_preaction_disallows_original_termination_and_bad_shapes():
    ref=playback(n=5)
    assert m.checked_preaction(ref,case(start=8)) is None
    bad=list(ref.context)
    bad[0]=replace(bad[0],student_action=(.1,)*7)
    with pytest.raises(RuntimeError,match='dimensions'):
        m.checked_preaction(DetailedReplay(ref.trace,tuple(bad)),case())


def test_known_safe_improvement_and_target_loss_recovery():
    base=playback(s8=snap(.8,hits=(0,1),early=3),s16=snap(1.6,hits=(0,1,2)),
                  s32=snap(3.2,hits=(0,1,2,3),early=4))
    alt=playback(s8=snap(.8,hits=(0,1,4),early=2),s16=snap(1.6,hits=(0,1,2,4),early=2),
                 s32=snap(3.2,hits=(0,1,2,3,4),early=2))
    result=m.label_case(base,alt,case=case())
    assert result['label']=='verified_safe_improvement'
    assert result['initial_hit_gain']==1
    assert result['post800_hit_gain']==0  # gained one by 800ms; not double-counted
    lag_alt=playback(s8=snap(.8,hits=(0,),early=2),s16=snap(1.6,hits=(0,1,2),early=2),
                     s32=snap(3.2,hits=(0,1,2,3,4),early=2))
    result=m.label_case(base,lag_alt,case=case())
    assert result['observed_hit_lag_at_800']==[1]
    assert result['recovered_hit_ids_by_3200']==[1]
    assert result['hit_suppression'] is False


def test_loss_at_last_milestone_is_hit_suppression():
    base=playback()
    alt=playback(s32=snap(3.2,hits=(0,),early=0))
    r=m.label_case(base,alt,case=case())
    assert r['label']=='hit_suppression'
    assert r['lost_hit_ids_by_3200']==[1]


def test_miss_regression_is_separate_from_reduced_early():
    base=playback()
    alt=playback(s32=snap(3.2,hits=(0,1),misses=(5,),early=1))
    r=m.label_case(base,alt,case=case())
    assert r['label']=='miss_regression'
    assert r['reason_flags']['miss_regression']


def test_early_overload_not_falsely_scored_against_unknown_future():
    terminated=snap(.08,hits=(0,1),early=6,overload=True)
    base=playback(terminated,terminated,terminated)
    c8=snap(.8,hits=(0,1,2,3),misses=(4,),early=2)
    c16=snap(1.6,hits=(0,1,2,3,5),misses=(4,6),early=2)
    c32=snap(3.2,hits=tuple(range(10)),misses=(10,11),early=2)
    alt=playback(c8,c16,c32)
    r=m.label_case(base,alt,case=case(anchor=8))
    assert r['label']=='conditional_rescue'
    assert r['per_horizon']['3.2']['miss_delta'] is None
    assert r['post800_hit_gain'] is None
    assert r['unpaired_extension_misses']==2
    assert r['reason_flags']['baseline_unobserved_future']


def test_keydown_suppression_without_hits_never_called_safe():
    no_hit=snap(3.2,hits=(),early=5,downs=(("left_1",.01),))
    no_press=snap(3.2,hits=(),early=0,downs=())
    base=playback(no_hit,no_hit,no_hit)
    alt=playback(no_press,no_press,no_press)
    assert m.label_case(base,alt,case=case())['label']=='insufficient_evidence'


def test_zero_material_change_and_underdetermined_separated():
    same=playback()
    assert m.label_case(same,same,case=case())['label']=='no_material_change'
    trunc=snap(.08,hits=(),early=0,overload=False,downs=())
    early=playback(trunc,trunc,trunc)
    assert m.label_case(early,same,case=case())['label']=='insufficient_evidence'


def test_features_fixed_and_future_outcomes_not_part_of_distance():
    p=m.checked_preaction(playback(),case(start=0))
    x=m.feature_blocks(p,case())
    assert tuple(map(len,x))==(263,8,8,1,1)
    assert m.feature_distance(x,x)==0
    assert m.feature_distance(x,m.feature_blocks(p,case(delta=-.19)))>0
    for group in x:
        assert all(math.isfinite(y) for y in group)
    assert 'target' not in p and 'future' not in p


def test_similarity_leave_anchor_out_and_do_not_count_case_variants_as_independent():
    rows=[]
    for anchor in (1,2,3):
        p=m.checked_preaction(playback(),case(anchor=anchor))
        for delta in (-.2,-.19):
            c=case(anchor=anchor,delta=delta)
            rows.append({'anchor':anchor,'case':c,'pre_state':p,
                         'label':'verified_safe_improvement' if anchor<=2 else 'miss_regression'})
    sim,oppo,summary=m.similarity_analysis(rows)
    assert len(sim)==6 and len(oppo)>0
    assert all(r['neighbor_anchor']!=r['anchor'] for r in sim)
    assert summary['successful_anchor_count']==2
    assert summary['failed_anchor_count']==1
    assert summary['distinct_pre_intervention_states']==3
    assert summary['generalization_status']=='insufficient_independent_evidence'


def test_unknown_label_not_used_for_nn_reference():
    good=m.checked_preaction(playback(),case(anchor=1))
    bad=m.checked_preaction(playback(),case(anchor=2))
    rows=[{'anchor':1,'case':case(anchor=1),'pre_state':good,
           'label':'verified_safe_improvement'},
          {'anchor':2,'case':case(anchor=2),'pre_state':bad,
           'label':'insufficient_evidence'}]
    comparisons,_,summary=m.similarity_analysis(rows)
    assert comparisons==[] and summary['anchor_held_out_nn_agreement'] is None


def test_csv_reproduction_mismatch_fails_closed():
    with pytest.raises(RuntimeError,match='reproduction differs'):
        m._csv_equal({'hits':'3'},{'hits':5},('hits',),'mock v1.5')
    m._csv_equal({'hits':'5','ids':'[0,1]'},{'hits':5,'ids':[0,1]},('hits','ids'),'mock')
    with pytest.raises(RuntimeError,match='missing prior'):
        m._csv_equal({}, {'hits':5},('hits',),'mock')


def test_output_existing_path_refused_without_mutation(tmp_path):
    output=tmp_path/'v16';output.mkdir()
    marker=output/'marker';marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        m._check_output(output,tuple(),tmp_path/'ckpt')
    assert marker.read_text()=='unchanged'


def test_empty_opposite_neighbors_written_as_header_only(tmp_path):
    p=tmp_path/'v16'
    m._publish(p,{
      'onset_states.jsonl':[{'frame':0}],
      'known_case_labels.csv':[{'anchor':1}],
      'discovery_probes.csv':[{'anchor':2}],
      'anchor_summary.csv':[{'anchor':1}],
      'state_similarity.csv':[],
      'opposite_label_neighbors.csv':[]},
      {'kind':'test'},{'status':'PASS'})
    assert p.is_dir()
    with (p/'opposite_label_neighbors.csv').open(encoding='utf-8-sig') as f:
        assert list(csv.DictReader(f))==[]
    assert json.loads((p/'diagnostic.json').read_text())['kind']=='test'


def test_no_training_or_validation_or_final_calls_in_source():
    code=inspect.getsource(m)
    assert 'optimizer.step(' not in code
    assert 'backward(' not in code
    assert 'evaluate_role_continuous(' not in code
    assert '_collect_dagger_trajectories(' not in code
    assert 'teacher_actions=' not in code


def test_preaction_is_from_baseline_even_if_candidate_raw_action_differs():
    ref=playback(n=10)
    alt=list(ref.context)
    alt[4]=replace(alt[4],student_action=tuple([-.4]*8))
    with pytest.raises(RuntimeError,match='pre-intervention'):
        m.assert_preaction_unchanged(ref,DetailedReplay(ref.trace,tuple(alt)),case(start=4))


def test_cached_replay_returns_same_object(monkeypatch):
    from dmdod.training import train_counterfactual_motor_v15 as v15
    calls=[]
    def simulate(prepared,named,*,case):
        calls.append(case)
        return playback()
    monkeypatch.setattr(v15,'run_detailed_trace',simulate)
    prep=SimpleNamespace(anchors=list(range(20)))
    cache=v15.ReplayCache(prep)
    x=case()
    assert cache.get(x) is cache.get(x)
    assert cache.executions==1 and cache.cache_hits==1 and len(calls)==1


def test_runner_mock_executes_full_288_without_validation_or_student_updates(monkeypatch,tmp_path):
    """Full CPU orchestration with immutable model, strict Train-only fixtures."""
    import sys
    import types
    import torch
    from dmdod.training.train_counterfactual_motor_v14 import REPRESENTATIVES

    basecases=[Intervention(a,k,d,s,f) for a,k,d,s,f in REPRESENTATIVES]
    conditional=Intervention(8,tuple(range(8)),-.20,0,8)
    extras=[Intervention(6,(2,),-.19,0,8),Intervention(18,(5,),-.18,4,1)]
    selected=[(c,{},'conditional_08' if c.anchor==8 else 'representative') for c in (*basecases,conditional)]
    selected.extend((c,{},'same_outcome_control') for c in extras)
    assert len(selected)==9
    baseline=playback(n=11)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([1.]))
            self.restores=0
        def prepare_recurrent_runtime(self):pass
        def load_state_dict(self,state_dict,strict=True):
            self.restores+=1
            return super().load_state_dict(state_dict,strict=strict)
    policy=Policy()
    prepared=SimpleNamespace(anchors=list(range(20)),key_count=8,input_dim=263,
              control_dt_s=.01,physics_dt_s=.001,model=policy,device='cpu',
              source_checkpoint=tmp_path/'best.pt')
    prepared.source_checkpoint.write_bytes(b'best')
    baseline_saved={(a,h):baseline.trace.at[h] for a in range(1,21) for h in m.HORIZONS}
    saved_cases={c.identity():{'candidate_snapshots':{str(h):baseline.trace.at[h].as_json()
                    for h in m.HORIZONS}} for c,_,_ in selected}
    monkeypatch.setattr(m,'_load_previous',lambda *args:({}, {}, selected,baseline_saved,saved_cases,{}))
    monkeypatch.setattr(m,'_load_v15',lambda *args:({}, {},{
        'boundary_06':{'successful_grid_points':0},
        'late_drift_18':{'retained_all_horizons':18},
        'rescue_survival_quality':{'8':{'3.2':{'survived':6}},'20':{'3.2':{'survived':6}}}},
        {}, {'gain_persistence.csv':[{'anchor':str(c.anchor),'keys':json.dumps(c.keys),
             'delta':str(c.delta),'start':str(c.start),'frames':str(c.frames)} for c,_,_ in selected],
           'boundary_grid_06.csv':[], 'late_drift_18.csv':[],
           'rescue_survival_quality.csv':[], 'event_divergence.csv':[]}))
    monkeypatch.setattr(m,'EXPECTED_V15_AGGREGATES',{'boundary_06':0,'drift_18':18,'rescue_8':6,'rescue_20':6})
    monkeypatch.setattr(m,'file_hash',lambda _:m.EXPECTED_BEST_SHA256)
    monkeypatch.setattr(m,'_hash_inputs',lambda _: {})
    monkeypatch.setattr(m,'assert_reproduced',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'_csv_equal',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'_compare_saved_metric',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'verify_saved_event_traces',lambda *args:57)

    class Cache:
        def __init__(self,prep):self.prep=prep;self._cached={};self.executions=0;self.cache_hits=0
        def get(self,case=None,*,anchor=None):
            k=(case.identity() if case else ('base',anchor))
            if k in self._cached:self.cache_hits+=1
            else:self._cached[k]=baseline;self.executions+=1
            return self._cached[k]
    monkeypatch.setattr(m,'ReplayCache',Cache)
    freeze=types.ModuleType('dmdod.training.fixed_state_action_audit')
    freeze.assert_state_equal=lambda state,expected,**kwargs: assert_equal(state,expected)
    real=types.ModuleType('dmdod.training.real_chart')
    real.clone_model_state=lambda model: {k:v.detach().clone() for k,v in model.state_dict().items()}
    monkeypatch.setitem(sys.modules,'dmdod.training.fixed_state_action_audit',freeze)
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',real)
    class SavedRows(dict):
        def get(self,key,default=None):return {}
        def __getitem__(self,key):return {}
    monkeypatch.setattr(m,'_prior_case_index',lambda rows,**kwargs:SavedRows())
    def assert_equal(one,two):
        assert all(torch.equal(one[k],two[k]) for k in one)
    output=tmp_path/'result'
    result=m.run_v16(prepared,v12_dir=tmp_path/'v12',v13_dir=tmp_path/'v13',
              v14_dir=tmp_path/'v14',v15_dir=tmp_path/'v15',output=output)
    assert result['discovery_probe_count']==288
    assert result['validation_evaluations']==0 and result['optimizer_updates']==0
    assert result['replay_cache']['unique_replays']<20+9+18+18+12+288
    assert policy.restores==1
    assert torch.equal(policy.weight.detach(),torch.tensor([1.]))
    assert {p.name for p in output.iterdir()}=={
        'diagnostic.json','onset_states.jsonl','known_case_labels.csv',
        'discovery_probes.csv','anchor_summary.csv','state_similarity.csv',
        'opposite_label_neighbors.csv','reproduction_checks.json'}
    with (output/'discovery_probes.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==288
    with (output/'onset_states.jsonl').open(encoding='utf-8') as f:
        assert len(f.readlines())==80
