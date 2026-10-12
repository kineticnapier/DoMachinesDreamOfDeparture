from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from dmdod.training.train_counterfactual_motor_v12 import Snapshot, outcome_signature
from dmdod.training.train_counterfactual_motor_v13 import (
    EXPECTED_BEST_SHA256, classify, outcome_group_signature, run_v13,
    snapshot_from_dict, validate_inputs,
)


def snap(hits=(), misses=(), early=0, downs=None, *, t=.8, overload=False):
    if downs is None:
        downs = tuple(('left_1', .01 + .001*i) for i in range(len(hits)+early))
    return Snapshot(time_s=t, hits=len(hits), misses=len(misses),
                    too_early=early, overloaded=overload,
                    overload_time_s=t-.001 if overload else None,
                    terminated=overload, termination_reason=('Overload' if overload else
                                                          'Horizon cutoff'),
                    hit_ids=tuple(hits), miss_ids=tuple(misses), keydowns=tuple(downs))


def common_and_extension(base, cand, *, common_candidate_ids=None, common_candidate_miss=0,
                         common_candidate_early=0, common_candidate_downs=0):
    is_short=base.time_s<cand.time_s
    b_ids=list(base.hit_ids)
    c_ids=list(common_candidate_ids if is_short else cand.hit_ids)
    b_miss=base.misses
    c_miss=common_candidate_miss if is_short else cand.misses
    b_early=base.too_early
    c_early=common_candidate_early if is_short else cand.too_early
    b_down=len(base.keydowns)
    c_down=common_candidate_downs if is_short else len(cand.keydowns)
    c={
        'observed_until_s':min(base.time_s,cand.time_s),
        'reference_hit_ids':b_ids, 'candidate_hit_ids':c_ids,
        'lost_hit_ids':sorted(set(b_ids)-set(c_ids)),
        'gained_hit_ids':sorted(set(c_ids)-set(b_ids)),
        'hits_preserved':not set(b_ids)-set(c_ids),
        'common_hit_gain':len(set(c_ids)-set(b_ids)),
        'reference_misses':b_miss, 'candidate_misses':c_miss,
        'misses_not_increased':c_miss<=b_miss,
        'reference_too_early':b_early, 'candidate_too_early':c_early,
        'too_early_reduced':c_early<b_early,
        'reference_keydowns':b_down, 'candidate_keydowns':c_down,
    }
    e={
        'reference_stopped_s':base.time_s,
        'candidate_observed_until_s':cand.time_s,
        'nonoverlap_s':max(0,cand.time_s-base.time_s),
        'new_hit_ids':sorted(set(cand.hit_ids)-set(c_ids)),
        'new_miss_ids':list(cand.miss_ids)[c_miss:],
        'new_keydowns':len(cand.keydowns)-c_down,
        'cannot_compare_to_unobserved_baseline_future':is_short,
    }
    return c,e


def test_06_prioritizes_hit_gain_and_records_early_reduction():
    base=snap((0,1,2),(9,10),3,downs=[('left_1',.01)]*6)
    good=snap((0,1,2,3,4),(),1,downs=[('left_1',.01)]*6)
    common,extension=common_and_extension(base,good)
    result=classify(base,good,common,extension,teacher={'hits':1000})
    assert result['category']=='hit_gain'
    assert result['too_early_reduced']
    assert result['safe_review_candidate']
    assert result['new_hit_ids_by_800']==[3,4]
    assert classify(base,good,common,extension,teacher={'hits':0})==result


def test_18_accepts_useful_fewer_keydowns_but_rejects_lost_hits():
    base=snap((0,1),(),3,downs=[('left_1',.01)]*5)
    good=snap((0,1),(),1,downs=[('left_1',.01)]*3)
    c,e=common_and_extension(base,good)
    out=classify(base,good,c,e)
    assert out['category']=='early_reduction_preserved_hits'
    assert out['safe_review_candidate']
    bad=snap((0,),(),1,downs=[('left_1',.01)]*3)
    c,e=common_and_extension(base,bad)
    out=classify(base,bad,c,e)
    assert out['category']=='hit_suppression'
    assert not out['safe_review_candidate']


def test_zero_hits_or_zero_keydowns_not_improvement():
    baseline=snap((),(),3,downs=[('left_1',.01)]*3)
    candidate=snap((),(),1,downs=[('left_1',.01)])
    c,e=common_and_extension(baseline,candidate)
    assert classify(baseline,candidate,c,e)['category']=='insufficient_evidence'
    candidate2=snap((),(),0,downs=[])
    c,e=common_and_extension(baseline,candidate2)
    assert classify(baseline,candidate2,c,e)['category']=='insufficient_evidence'


def test_08_recovers_lagging_hit_but_extension_misses_conditional_only():
    base=snap((0,1),(),3,downs=[('left_1',.01)]*5,t=.08,overload=True)
    cand=snap(tuple(range(10)),(10,11),3,downs=[('left_1',.01)]*11)
    c,e=common_and_extension(base,cand,common_candidate_ids=(0,),
                             common_candidate_miss=0,common_candidate_early=1,
                             common_candidate_downs=2)
    out=classify(base,cand,c,e)
    assert out['category']=='overload_rescue_review'
    assert out['temporarily_lagging_hit_ids']==[1]
    assert out['recovered_hit_ids_by_800']==[1]
    assert out['unrecovered_hit_ids_by_800']==[]
    assert out['observed_extension_miss_ids']==[10,11]
    assert out['conditional_review_candidate'] and not out['safe_review_candidate']


def test_20_explicit_overload_rescue_remains_qualified():
    base=snap((0,1,2),(),3,downs=[('left_1',.01)]*6,t=.14,overload=True)
    cand=snap(tuple(range(13)),(),2,downs=[('left_1',.01)]*15)
    c,e=common_and_extension(base,cand,common_candidate_ids=(0,1),
                             common_candidate_miss=0,common_candidate_early=1,
                             common_candidate_downs=3)
    out=classify(base,cand,c,e)
    assert out['category']=='overload_rescue'
    assert out['safe_review_candidate']
    assert out['recovered_hit_ids_by_800']==[2]


def test_no_unobserved_future_miss_extrapolation():
    base=snap((0,),(),3,t=.08,overload=True)
    cand=snap((0,1,2),(),1)
    c,e=common_and_extension(base,cand,common_candidate_ids=(0,),
                             common_candidate_downs=1)
    result=classify(base,cand,c,e)
    assert result['common_misses_not_increased']
    assert result['both_800ms_misses_not_increased'] is None
    assert not result['both_800ms_observed']


def test_outcome_group_ignores_keydown_timing_but_old_signature_differs():
    a=snap((0,1),(),1,downs=[('left_1',.01)]*3)
    b=snap((0,1),(),1,downs=[('left_1',.015)]*3)
    assert outcome_group_signature(18,a)==outcome_group_signature(18,b)
    assert outcome_signature(18,a)!=outcome_signature(18,b)
    assert outcome_group_signature(18,a)!=outcome_group_signature(19,a)


def test_corrupted_snapshots_rejected():
    raw=snap((0,1),(),1).as_json()
    raw['hits']=3
    with pytest.raises(RuntimeError,match='Fail Closed'):
        snapshot_from_dict(raw,'corrupted')
    raw=snap((0,1),(),1).as_json()
    raw['keydowns'].append(['left_1',5.0])
    with pytest.raises(RuntimeError,match='Fail Closed'):
        snapshot_from_dict(raw,'out of clock')


def _fixture(tmp: Path, *, corrupt=None):
    """152 completely synthetic, internally consistent, old-style records."""
    previous=tmp/'v12'
    previous.mkdir()
    baseline=[]
    for i in range(1,21):
        b=(snap((0,1,2),(9,10),3,downs=[('left_1',.01)]*6) if i==6 else
           snap((0,1),(),3,downs=[('left_1',.01)]*5) if i==18 else
           snap((0,1),(),3,downs=[('left_1',.01)]*5,t=.08,overload=True) if i==8 else
           snap((0,1,2),(),3,downs=[('left_1',.01)]*6,t=.14,overload=True) if i==20 else
           snap((0,1),(),0,downs=[('left_1',.01)]*2))
        baseline.append({'anchor':i,'chart':f'chart{i}',
             'student_400':json.dumps(b.as_json() if b.terminated else replace(b,time_s=.4).as_json()),
             'student_800':json.dumps(b.as_json()),
             'teacher_400':json.dumps(snap((0,),(),0,t=.4).as_json()),
             'teacher_800':json.dumps(snap((0,),(),0).as_json())})
    original=[]
    checks=[]
    distribution={6:40,18:38,8:1}
    others=[i for i in range(1,21) if i not in (6,8,18,20)]
    for i in range(72):
        distribution[others[i%len(others)]]=distribution.get(others[i%len(others)],0)+1
    assert sum(distribution.values())==151
    for anchor,count in [*sorted(distribution.items()),(20,1)]:
        base=snapshot_from_dict(json.loads(baseline[anchor-1]['student_800']),'base')
        for j in range(count):
            source='v1_manual_unsampled' if anchor==20 else 'v11_short_pass'
            delta=-.2 if anchor==20 else [-.02,-.05,-.10,-.20][j%4]
            start=0 if anchor==20 else j
            duration=4 if anchor==20 else 8
            keys=(6,) if anchor==20 else (j%8,)
            if anchor==6:
                candidate=snap((0,1,2,3,4),(),1,downs=[('left_1',.01)]*6)
                common,e=common_and_extension(base,candidate)
            elif anchor==18:
                candidate=snap((0,1),(),1,downs=[('left_1',.01)]*3) if j<29 else snap((0,),(),1,downs=[('left_1',.01)]*3)
                common,e=common_and_extension(base,candidate)
            elif anchor==8:
                candidate=snap(tuple(range(10)),(10,11),3,downs=[('left_1',.01)]*11)
                common,e=common_and_extension(base,candidate,common_candidate_ids=(0,),common_candidate_early=1,common_candidate_downs=2)
            elif anchor==20:
                candidate=snap(tuple(range(13)),(),2,downs=[('left_1',.01)]*15)
                common,e=common_and_extension(base,candidate,common_candidate_ids=(0,1),common_candidate_early=1,common_candidate_downs=3)
            else:
                candidate=base
                common,e=common_and_extension(base,candidate)
            short=(candidate if candidate.terminated else replace(candidate,time_s=.4))
            identity=[anchor,list(keys),start,duration,delta]
            row={
                'anchor':anchor,'source':source,'kind':'single',
                'keys':','.join(map(str,keys)),'start_frame':start,'duration_frames':duration,
                'delta':delta,'lead_frames':0,'old_status':'safe_early_reduction_review_only',
                'old_qualified':False,'old_teacher_gate_passed':False,
                'old_teacher_only_rejection':False,
                'short_hits':short.hits,'short_misses':short.misses,
                'short_early':short.too_early,'short_overload':short.overloaded,
                'short_duration_s':short.time_s,
                'hits_800':candidate.hits,'misses_800':candidate.misses,
                'early_800':candidate.too_early,'overload_800':candidate.overloaded,
                'overload_time_s':candidate.overload_time_s,'duration_s':candidate.time_s,
                'hit_ids_800':json.dumps(candidate.hit_ids),
                'miss_ids_800':json.dumps(candidate.miss_ids),
                'keydowns_800':json.dumps(candidate.keydowns),
                'category':'hit_suppression' if anchor in (8,18) else 'inconclusive',
                'eligible_for_teacher_review':False,
                'teacher_adopted':False,
                'checks':'{}','comparison_common':json.dumps(common),
                'extended_observation':json.dumps(e),
                'teacher_reference':json.dumps({'hits':100}),
                'lost_baseline_hit_ids':'[]',
                'outcome_signature':outcome_signature(anchor,candidate),
            }
            original.append(row)
            checks.append({'anchor':anchor,'source':source,'case':identity,
                           '400ms':short.as_json(),'800ms':candidate.as_json(),
                           'reproduced':source=='v11_short_pass'})
    assert len(original)==152
    if corrupt=='target':
        original[0]['hit_ids_800']='[999]'
    if corrupt=='repro':
        checks[2]['reproduced']=False
    if corrupt=='signature':
        original[0]['outcome_signature']='0000000000000000'
    diagnostic={'kind':'train_counterfactual_motor_v12',
                'source_checkpoint_sha256':EXPECTED_BEST_SHA256,
                'validation_evaluations':0,'final_used':False,
                'optimizer_updates':0,'checkpoint_writes':0,'teacher_adoptions':0,
                'v11_candidates_reproduced':151,'v11_candidates_selected':151,
                'v11_reproduction_rate':1.0, 'v1_train20_explicit_replay':True,
                'focus_anchors':[],
                'source_diagnostic_hashes':{'v11':'aaaa'}}
    for anchor in (6,8,18,20):
        records=[(row,check) for row,check in zip(original,checks) if row['anchor']==anchor]
        diagnostic['focus_anchors'].append({
            'anchor':anchor,
            'baseline':{'student_800':json.loads(baseline[anchor-1]['student_800'])},
            'cases':[
                {'outcome_signature':row['outcome_signature'],
                 'hit_ids_800':json.loads(row['hit_ids_800']),
                 'miss_ids_800':json.loads(row['miss_ids_800']),
                 'keydowns_800':json.loads(row['keydowns_800']),
                 'hits_800':row['hits_800'],
                 'early_800':row['early_800']}
                for row,check in records
            ],
        })
    repro={'status':'PASS','source_sha256':EXPECTED_BEST_SHA256,
           'v11_input_sha256':{'v11':'aaaa'},'candidates':checks}
    (previous/'diagnostic.json').write_text(json.dumps(diagnostic))
    (previous/'reproduction_checks.json').write_text(json.dumps(repro))
    for name,rows in [('baseline_800ms.csv',baseline),('candidate_comparison.csv',original)]:
        with (previous/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return previous


def test_full_152_offline_reclassification_integrity_and_no_training(tmp_path):
    source=_fixture(tmp_path)
    hashes={p.name:p.read_bytes() for p in source.iterdir()}
    result=run_v13(source,tmp_path/'v13')
    assert result['safe_improvement_original_151']==40+29
    assert result['conditional_review_original_151']==1
    assert len(result['qualified_train_anchors_original_151'])==2
    assert result['input_candidates']==152
    assert result['hit_suppression_original_151']==9
    output=tmp_path/'v13'
    with (output/'reclassified_candidates.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==152
    assert len([r for r in rows if r['anchor']=='6'])==40
    assert len([r for r in rows if r['anchor']=='18' and r['new_category']=='early_reduction_preserved_hits'])==29
    assert len([r for r in rows if r['anchor']=='18' and r['new_category']=='hit_suppression'])==9
    assert rows[-1]['anchor']=='20' and rows[-1]['new_category']=='overload_rescue'
    assert all(r['outcome_signature'] for r in rows)
    assert len(set(r['outcome_group_signature'] for r in rows if r['anchor']=='6'))==1
    with (output/'qualified_corrections.csv').open(encoding='utf-8-sig') as f:
        qualified=list(csv.DictReader(f))
    assert len(qualified)==70
    with (output/'conditional_review.csv').open(encoding='utf-8-sig') as f:
        review=list(csv.DictReader(f))
    assert len(review)==1 and review[0]['anchor']=='8'
    assert {p.name:p.read_bytes() for p in source.iterdir()}==hashes
    with pytest.raises(FileExistsError):
        run_v13(source,output)


@pytest.mark.parametrize('problem', ['target','repro','signature'])
def test_corruption_fails_closed_without_creating_output(tmp_path,problem):
    source=_fixture(tmp_path,corrupt=problem)
    dest=tmp_path/'v13'
    with pytest.raises(RuntimeError,match='Fail Closed'):
        run_v13(source,dest)
    assert not dest.exists()


def test_checkpoint_is_not_loaded_and_validation_not_accessed(tmp_path,monkeypatch):
    source=_fixture(tmp_path)
    import torch
    monkeypatch.setattr(torch,'load',lambda *args,**kw: (_ for _ in ()).throw(AssertionError('checkpoint load forbidden')))
    result=run_v13(source,tmp_path/'v13')
    assert result['optimizer_updates']==0 and result['validation_evaluations']==0


def test_v12_measurement_fields_preserved_verbatim(tmp_path):
    source=_fixture(tmp_path)
    with (source/'candidate_comparison.csv').open(encoding='utf-8-sig') as f:
        old=list(csv.DictReader(f))
    run_v13(source,tmp_path/'v13')
    with (tmp_path/'v13'/'reclassified_candidates.csv').open(encoding='utf-8-sig') as f:
        new=list(csv.DictReader(f))
    from dmdod.training.train_counterfactual_motor_v13 import MEASUREMENT_FIELDS
    for before,after in zip(old,new):
        assert {key:before[key] for key in MEASUREMENT_FIELDS}=={
            key:after[key] for key in MEASUREMENT_FIELDS}
