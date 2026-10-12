from __future__ import annotations

import csv
from dataclasses import replace
import json
import sys
import types
from pathlib import Path

import pytest

from dmdod.training.train_counterfactual_motor_v12 import (
    EXPECTED_BEST_SHA256, Snapshot, assert_output_safe, assert_prefix_consistent,
    assert_v11_reproduced, case_identity, classify_student_counterfactual,
    common_interval_score, load_v11_short_passers, outcome_signature,
    snapshot_at, source_case,
)


def snap(time=0.8, *, used=(), missed=(), early=0, keys=(),
         overloaded=False, overload_time=None, done=False):
    return Snapshot(
        time_s=time, hits=len(used), misses=len(missed), too_early=early,
        overloaded=overloaded, overload_time_s=overload_time,
        terminated=done,
        termination_reason="Overload" if overloaded else "Horizon cutoff",
        hit_ids=tuple(used), miss_ids=tuple(missed),
        keydowns=tuple(keys),
    )


def sample(base400, base800, baseframes, cand400, cand800, candframes, *, teacher_hits=100):
    return classify_student_counterfactual(
        base400, base800, baseframes, cand400, cand800, candframes,
        expert_800=snap(used=tuple(range(teacher_hits))),
    )


def test_400_and_800_snapshot_monotonic_and_no_extrapolation():
    earlier = snap(.4, used=(1,), early=1, keys=(("a", .1),))
    later = snap(.8, used=(1, 2), early=2, keys=(("a", .1), ("b", .5)))
    assert_prefix_consistent(earlier, later)
    assert snapshot_at((earlier, later), .45) is earlier
    with pytest.raises(RuntimeError, match="no common"):
        snapshot_at((earlier, later), .1)
    with pytest.raises(RuntimeError, match="disappeared"):
        assert_prefix_consistent(earlier, snap(.8, used=(2,)))


def test_student_early_overload_not_compared_to_unseen_800ms_misses():
    base = snap(.10, used=(0,), keys=(("a", .08),), early=3,
                overloaded=True, overload_time=.095, done=True)
    c_at_base = snap(.10, used=(0,), keys=(("a", .08),), early=1)
    c_400 = snap(.4, used=(0,1), keys=(("a", .08), ("b", .35)), early=1)
    c_800 = snap(.8, used=(0,1,2), missed=(4,5),
                 keys=(("a",.08), ("b",.35), ("c",.7)), early=1)
    report = sample(base,base,(base,),c_400,c_800,(c_at_base,c_400,c_800))
    assert report["comparison_common"]["observed_until_s"] == pytest.approx(.1)
    assert report["comparison_common"]["misses_not_increased"]
    assert report["extended_observation"]["new_miss_ids"] == [4,5]
    assert report["extended_observation"]["cannot_compare_to_unobserved_baseline_future"]
    assert report["category"] == "overload_rescue"
    assert report["eligible_for_teacher_review"]
    assert not report["teacher_adopted"]


def test_target_hit_ids_not_just_total_hit_counts():
    reference=snap(used=(0, 2))
    changed=snap(used=(0, 3))
    report=sample(reference,reference,(reference,),changed,changed,(changed,))
    assert report["category"]=="hit_suppression"
    assert report["comparison_common"]["lost_hit_ids"]==[2]
    assert report["baseline_hit_ids_lost_by_800"]==[2]
    assert not report["eligible_for_teacher_review"]


def test_silence_cannot_be_scored_as_overload_rescue():
    original=snap(.1,used=(),early=3, keys=(("a", .08),),overloaded=True,
                  overload_time=.09,done=True)
    candidate=snap(.8,used=(),early=0,keys=())
    report=sample(original,original,(original,),candidate,candidate,(snap(.1),candidate))
    assert report["category"] == "hit_suppression"
    assert report["checks"]["overload_rescue"]
    assert not report["checks"]["no_press_suppression"]
    assert not report["eligible_for_teacher_review"]


def test_delayed_overload_is_not_rescue():
    base=snap(.1,early=3,keys=(("a", .08),),overloaded=True,
              overload_time=.095,done=True)
    c400=snap(.4,early=1,keys=(("a", .08),))
    c800=snap(.6,early=3,keys=(("a",.08),),overloaded=True,
              overload_time=.59,done=True)
    report=sample(base,base,(base,),c400,c800,(snap(.1),c400,c800))
    assert report["category"]=="deferred_failure"
    assert report["checks"]["overload_delayed_not_rescued"]
    assert not report["checks"]["overload_rescue"]


def test_teacher_absolute_hits_do_not_affect_student_classification():
    base=snap(.8,used=(0,),early=2,keys=(("a",.2),))
    candidate=snap(.8,used=(0,),early=1,keys=(("a",.2),))
    reports=[sample(base,base,(base,),candidate,candidate,(candidate,),teacher_hits=t)
             for t in (1,100,10000)]
    assert {r["category"] for r in reports}=={"early_reduction_preserved_hits"}
    assert all(r["eligible_for_teacher_review"] for r in reports)
    assert all(not r["teacher_adopted"] for r in reports)


def test_miss_worsening_common_interval_is_regression():
    base=snap(.8,used=(0,),missed=(),keys=(("a",.2),))
    candidate=snap(.8,used=(0,),missed=(2,),keys=(("a",.2),))
    r=sample(base,base,(base,),candidate,candidate,(candidate,))
    assert r["category"]=="regression"
    assert not r["checks"]["misses_not_increased_common"]


def test_outcome_signature_deduplicates_identical_physical_results():
    a=snap(.8,used=(1,),keys=(("a",.201),))
    b=snap(.8,used=(1,),keys=(("a",.201),))
    c=snap(.8,used=(2,),keys=(("a",.201),))
    assert outcome_signature(6,a)==outcome_signature(6,b)
    assert outcome_signature(6,a)!=outcome_signature(6,c)
    assert outcome_signature(6,a)!=outcome_signature(8,a)


def _old_row(short,long):
    return {
        "short_hits":str(short.hits),"short_early":str(short.too_early),
        "short_keydowns":str(len(short.keydowns)),"short_overload":str(short.overloaded),
        "followup_hits":str(long.hits),"followup_early":str(long.too_early),
        "followup_misses":str(long.misses),"followup_keydowns":str(len(long.keydowns)),
        "followup_overload":str(long.overloaded),"followup_duration_s":str(long.time_s),
        "followup_new_hits":str(long.hits-short.hits),
        "followup_new_keydowns":str(len(long.keydowns)-len(short.keydowns)),
        "followup_new_too_early":str(long.too_early-short.too_early),
    }


def test_saved_400_800_replay_reproduction_fail_closed():
    short=snap(.4,used=(1,),early=2,keys=(("a",.1),))
    long=snap(.8,used=(1,2),early=3,keys=(("a",.1),("b",.6)))
    row=_old_row(short,long)
    assert_v11_reproduced(row,short,long)
    row["followup_hits"]="999"
    with pytest.raises(RuntimeError,match="Fail Closed"):
        assert_v11_reproduced(row,short,long)


def test_v11_parser_disallows_incomplete_shortlist(tmp_path,monkeypatch):
    mod=types.ModuleType("dmdod.training.train_counterfactual_motor_v11")
    class Probe:
        def __init__(self,keys,start,duration,delta,lead):
            self.keys,self.start,self.duration,self.delta,self.lead=keys,start,duration,delta,lead
        @property
        def kind(self):
            return "single" if len(self.keys)==1 else "all_8" if len(self.keys)==8 else "paired"
    mod.Probe=Probe
    monkeypatch.setitem(sys.modules,mod.__name__,mod)
    original={"kind":"train_counterfactual_motor_diagnostic_v11",
              "source_sha256":EXPECTED_BEST_SHA256,
              "validation_evaluations":0,"optimizer_updates":0,"final_used":False,
              "followup_horizon_s":.8,"train_candidate_evaluations":1}
    (tmp_path/"diagnostic.json").write_text(json.dumps(original))
    row={"anchor":"1","keys":"left_1","kind":"single","start_frame":"0",
         "duration_frames":"4","lead_frames":"0","delta":"-0.2",
         "short_status":"overload_rescue_short","followup_checked":"true",
         "followup_hits":"2","followup_early":"1","followup_misses":"0",
         "followup_keydowns":"3","followup_overload":"false",
         "followup_duration_s":"0.8"}
    with (tmp_path/"train_candidate_gates.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(RuntimeError,match="151"):
        load_v11_short_passers(tmp_path,("left_1",))
    assert source_case(row,key_names=("left_1",)).start==0


def test_no_output_overwrite_and_read_only_prior(tmp_path):
    previous=tmp_path/"v11"
    previous.mkdir()
    source=tmp_path/"best.pt"
    source.write_bytes(b"selected-best")
    result=tmp_path/"v12"
    assert_output_safe(result,previous,source)
    result.mkdir()
    with pytest.raises(FileExistsError):
        assert_output_safe(result,previous,source)
    with pytest.raises(FileExistsError):
        assert_output_safe(previous,previous,source)
    assert source.read_bytes()==b"selected-best"


def test_only_train_codes_importable_without_validation_runner(monkeypatch):
    # Import/pure classification does not instantiate runner, optimizer, or FINAL.
    base=snap(.8,used=(0,))
    candidate=snap(.8,used=(0,))
    report=sample(base,base,(base,),candidate,candidate,(candidate,))
    assert report["category"]=="hit_suppression" or report["category"]=="inconclusive"


def test_old_baseline_400ms_and_origin_must_reproduce():
    from dmdod.training.train_counterfactual_motor_v12 import verify_v11_baselines
    anchors=[{
        "anchor":i,"chart":f"train{i:02d}",
        "base_overload_400ms":False,"base_early_400ms":0,
    } for i in range(1,21)]
    originals=[{"chart":f"train{i:02d}",
                "student_400":snap(.4,used=(0,)).as_json()}
               for i in range(1,21)]
    selected=[{"anchor":6,"old":{"base_hits":"1","base_early":"0",
                                 "base_overload":"False"}}]
    verify_v11_baselines({"anchors":anchors},selected,originals)
    selected[0]["old"]["base_hits"]="3"
    with pytest.raises(RuntimeError,match="candidate origin mismatch"):
        verify_v11_baselines({"anchors":anchors},selected,originals)


def test_empty_qualified_csv_has_header_and_zero_candidates(tmp_path):
    from dmdod.training.train_counterfactual_motor_v12 import csv_write_new
    path=tmp_path/"qualified_corrections.csv"
    csv_write_new(path,[],columns=("anchor","category","eligible_for_teacher_review"))
    with path.open(encoding="utf-8-sig",newline="") as f:
        contents=list(csv.DictReader(f))
    assert contents==[]


def test_end_to_end_151_train_only_no_optimizer_checkpoint_or_validation(monkeypatch,tmp_path):
    import torch
    import dmdod.training.train_counterfactual_motor_v12 as v12
    repo = Path(tmp_path)
    source = repo / 'best.pt'
    source.write_bytes(b'original-best-checkpoint')
    prior = repo / 'train_counterfactual_motor_v11'
    prior.mkdir()
    (prior/'diagnostic.json').write_text('original-v11')
    (prior/'train_candidate_gates.csv').write_text('original-candidates')

    class Probe:
        def __init__(self,start,delta):
            self.keys=(0,)
            self.start=start
            self.duration=1
            self.delta=delta
            self.lead=0
        @property
        def kind(self):return 'single'
        @property
        def end(self):return self.start+self.duration

    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([.1]))
        def prepare_recurrent_runtime(self):pass

    model=FrozenModel()
    old_state={k:t.clone() for k,t in model.state_dict().items()}
    before_hash = 'faked-best-hash'
    old400=snap(.4,used=(0,),early=1,keys=(("left_1", .2),))
    old800=snap(.8,used=(0,),early=1,keys=(("left_1", .2),))
    frames=(old400,old800)
    original_row=_old_row(old400,old800)
    original_row.update({'base_hits':'1','base_early':'1','base_overload':'False',
                         'gate_expert_relative_hits':'False','followup_qualified':'False',
                         'short_status':'safe_early_reduction_review_only'})
    cases=[{'anchor':(i%20)+1,'case':Probe(i,float(i)/1000),'old':dict(original_row),
            'source':'v11_short_pass'} for i in range(151)]
    report={'anchors':[
        {'anchor':i,'chart':f'train{i:02d}',
         'base_overload_400ms':False,'base_early_400ms':1}
         for i in range(1,21)]}
    def stub_hash(path):
        if path==source:
            assert path.read_bytes()==b'original-best-checkpoint'
            return v12.EXPECTED_BEST_SHA256
        return before_hash
    monkeypatch.setattr(v12,'load_v11_short_passers',lambda directory, key_names:(report,cases))
    monkeypatch.setattr(v12,'run_trace',lambda prepared,named,*,case=None,expert=False:
                        (old400,old800,frames))
    import sys
    def install(name,**attrs):
        module=types.ModuleType(name)
        for k,v in attrs.items():setattr(module,k,v)
        monkeypatch.setitem(sys.modules,name,module)
    install('dmdod.motor',__path__=[])
    install('dmdod.motor.n_key',n_key_names=lambda count:('left_1','left_2','left_3','left_4','right_1','right_2','right_3','right_4'))
    install('dmdod.training.fixed_state_action_audit',
            assert_state_equal=lambda a,b,context:None if all(torch.equal(a[k],b[k]) for k in b) else (_ for _ in ()).throw(AssertionError(context)),
            file_sha256=stub_hash)
    install('dmdod.training.real_chart',
            clone_model_state=lambda model:{k:v.detach().clone() for k,v in model.state_dict().items()})
    install('dmdod.training.train_counterfactual_motor_v11',Probe=lambda **kw:
            Probe(kw['start'],kw['delta']))

    class NoValidationAccess:
        def __len__(self):raise AssertionError('Validation MUST NOT be consulted')
        def __iter__(self):raise AssertionError('Validation MUST NOT be consulted')
    prepared=types.SimpleNamespace(
        anchors=[types.SimpleNamespace(chart_name=f'train{i:02d}') for i in range(1,21)],
        validation=NoValidationAccess(),key_count=8,input_dim=263,
        control_dt_s=.01,physics_dt_s=.001,source_checkpoint=source,model=model)
    output=repo/'v12'
    result=v12.run_v12(prepared,previous=prior,output=output)
    assert result['v11_candidates_reproduced']==151
    assert result['validation_evaluations']==result['optimizer_updates']==0
    assert result['teacher_adoptions']==result['checkpoint_writes']==0
    assert len(list(csv.DictReader((output/'candidate_comparison.csv').open(encoding='utf-8-sig'))))==152
    assert (output/'diagnostic.json').exists()
    assert (output/'reproduction_checks.json').exists()
    assert (output/'qualified_corrections.csv').exists()
    assert not list(output.glob('*.pt'))
    assert source.read_bytes()==b'original-best-checkpoint'
    assert (prior/'diagnostic.json').read_text()=='original-v11'
    assert (prior/'train_candidate_gates.csv').read_text()=='original-candidates'
    assert all(torch.equal(model.state_dict()[k],v) for k,v in old_state.items())
    with pytest.raises(FileExistsError):
        v12.run_v12(prepared,previous=prior,output=output)


def test_teacher_only_rejection_requires_all_other_followup_gates():
    from dmdod.training.train_counterfactual_motor_v12 import rejected_solely_by_followup_teacher
    row={
        "short_status":"overload_rescue_short",
        "followup_expert_relative_hits":"False",
        "followup_reached_horizon":"True",
        "followup_no_overload":"True",
        "followup_short_hits_retained":"True",
        "followup_no_hit_regression":"True",
        "followup_late_early_within_expert_plus_one":"True",
        "followup_late_misses_within_expert_plus_one":"True",
    }
    assert rejected_solely_by_followup_teacher(row)
    row["followup_no_hit_regression"]="False"
    assert not rejected_solely_by_followup_teacher(row)
    row["followup_no_hit_regression"]="True"
    row["short_status"]="safe_early_reduction_review_only"
    assert not rejected_solely_by_followup_teacher(row)


def test_one_closed_loop_trace_yields_400_and_800_snapshots(monkeypatch):
    import torch
    import dmdod.training.train_counterfactual_motor_v12 as v12

    class EvalEnv:
        def __init__(self):
            self.steps=0
            self._hits=0
            self._misses=0
            self._too_early=0
            self._overload=types.SimpleNamespace(overloaded=False)
            self._trace_overload_event_time_s=None
            self._trace_physical_keydowns=[]
            self._used=[False]*10
            self._missed=[False]*10
            self._failed_on_miss=False
        def reset(self):return object()
        def privileged_episode_time_s(self):return self.steps/100
        def step(self,action):
            self.steps+=1
            if self.steps==10:
                self._used[0]=True
                self._hits=1
            if self.steps==65:
                self._used[1]=True
                self._hits=2
            return types.SimpleNamespace(observation=object(),done=False)

    class Model:
        def __init__(self):self.initial=0;self.forwards=0
        def eval(self):pass
        def initial_state(self,device):self.initial+=1;return torch.zeros(1)
        def forward_step(self,visible,state):
            assert visible.shape==(263,)
            self.forwards+=1
            return torch.zeros(8),None,None,state+1

    model=Model()
    simulated=EvalEnv()
    modules={
        'dmdod.training.motor_sensitivity_audit':{'_environment':lambda prepared,named:simulated},
        'dmdod.envs.n_key':{'encode_n_key_hud_real_chart_observation':lambda obs:[0.0]*263},
        'dmdod.motor':{'__path__':[]},
        'dmdod.motor.n_key':{'NKeyAction':lambda a:a},
        'dmdod.n_key_capacity':{'CenterFirstNKeyTeacher':lambda n:object()},
        'dmdod.training.n_key':{'_teacher_action':lambda *a,**kw:(0.0,)*8},
    }
    for name,attrs in modules.items():
        module=types.ModuleType(name)
        for k,v in attrs.items():setattr(module,k,v)
        monkeypatch.setitem(sys.modules,name,module)
    prepared=types.SimpleNamespace(model=model,key_count=8,
                                  control_dt_s=.01,device=torch.device('cpu'))
    early,late,frames=v12.run_trace(prepared,object())
    assert early.time_s==pytest.approx(.4)
    assert late.time_s==pytest.approx(.8)
    assert early.hit_ids==(0,)
    assert late.hit_ids==(0,1)
    assert len(frames)==80
    assert model.initial==1 and model.forwards==80


def test_short_terminated_trace_not_padded_to_800ms(monkeypatch):
    import torch
    import dmdod.training.train_counterfactual_motor_v12 as v12

    class Env:
        def __init__(self):
            self.steps=0
            self._hits=self._misses=0
            self._too_early=3
            self._used=[False]*8
            self._missed=[False]*8
            self._failed_on_miss=False
            self._overload=types.SimpleNamespace(overloaded=False)
            self._trace_overload_event_time_s=None
            self._trace_physical_keydowns=[]
        def reset(self):return None
        def privileged_episode_time_s(self):return self.steps/100
        def step(self,action):
            self.steps+=1
            done=self.steps==10
            if done:
                self._overload.overloaded=True
                self._trace_overload_event_time_s=.095
            return types.SimpleNamespace(observation=None,done=done)

    env=Env()
    for name,attrs in {
        'dmdod.training.motor_sensitivity_audit':{'_environment':lambda prepared,named:env},
        'dmdod.envs.n_key':{'encode_n_key_hud_real_chart_observation':lambda obs:[0.]*263},
        'dmdod.motor':{'__path__':[]},
        'dmdod.motor.n_key':{'NKeyAction':lambda action:action},
        'dmdod.n_key_capacity':{'CenterFirstNKeyTeacher':lambda key_count:object()},
        'dmdod.training.n_key':{'_teacher_action':lambda *args,**kwargs:(0.,)*8},
    }.items():
        m=types.ModuleType(name)
        for k,v in attrs.items():setattr(m,k,v)
        monkeypatch.setitem(sys.modules,name,m)

    class Model:
        def eval(self):pass
        def initial_state(self,device):return torch.zeros(1)
        def forward_step(self,x,state):return torch.zeros(8),None,None,state

    prepared=types.SimpleNamespace(model=Model(),key_count=8,
                                  control_dt_s=.01,device=torch.device('cpu'))
    short,long,frames=v12.run_trace(prepared,object())
    assert len(frames)==10
    assert short==long
    assert long.time_s==pytest.approx(.1)
    assert long.overload_time_s==pytest.approx(.095)
    assert long.termination_reason=='Overload'
