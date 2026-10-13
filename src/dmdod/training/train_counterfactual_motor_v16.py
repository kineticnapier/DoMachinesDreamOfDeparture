from __future__ import annotations

"""Train-only v1.6 audit: state-conditioned motor correction *evidence*, not training.

Features are frozen BEST Student's pre-action 263D HUD and 8D raw action,
plus proposed key mask, delta and duration. No targets, post-action state,
expert or future outcome is ever used as a classifier input.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from dmdod.training.train_counterfactual_motor_v12 import EXPECTED_BEST_SHA256
from dmdod.training.train_counterfactual_motor_v13 import file_hash
from dmdod.training.train_counterfactual_motor_v14 import (
    HORIZONS, TOL, Intervention, _hash_inputs, assert_reproduced, compare_horizon,
)
from dmdod.training.train_counterfactual_motor_v15 import (
    ReplayCache, _load_previous, _saved_horizons, _compare_saved_metric,
    boundary_grid, first_divergence, first_hit_loss, first_miss_excess,
    first_miss_id_difference, horizon_target_deltas,
    persistence, rescue_quality, too_early_reduction_duration, GRID_06,
    GRID_18, _json_field,
)

ONSET_FRAMES = (0, 1, 5, 8)
DISCOVERY_FRAMES = (0, 5)
FOCUS = frozenset((6, 8, 18, 20))
LABELS = frozenset((
    'verified_safe_improvement', 'hit_suppression', 'miss_regression',
    'conditional_rescue', 'no_material_change', 'insufficient_evidence',
))
FEATURE_NORMALIZATION = 'signed_saturation_x_over_1_plus_abs_x'
# Fixed, non-fitted group weights. Each segment contributes comparable norm.
FEATURE_GROUP_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
EXPECTED_V15_AGGREGATES = {'boundary_06':11,'drift_18':2,'rescue_8':4,'rescue_20':1}


def _identity(case: Intervention) -> tuple:
    return case.identity()


def discovery_cases() -> tuple[Intervention, ...]:
    cases = tuple(Intervention(anchor, keys, -.20, start, 8)
                  for anchor in range(1, 21) if anchor not in FOCUS
                  for start in DISCOVERY_FRAMES
                  for keys in (*((k,) for k in range(8)), tuple(range(8))))
    if len(cases) != 288 or len({_identity(c) for c in cases}) != 288:
        raise RuntimeError('Fail Closed: expected 288 distinct Train-only probes')
    return cases


def checked_preaction(reference, case: Intervention) -> dict | None:
    """Only BASELINE's state immediately BEFORE executing `case.start`."""
    if case.start < 0 or case.start >= len(reference.context):
        return None
    frame = reference.context[case.start]
    hud, raw = tuple(frame.visible_hud), tuple(frame.student_action)
    if (len(hud) != 263 or len(raw) != 8 or
            not all(math.isfinite(float(x)) for x in (*hud, *raw)) or
            not all(-1.000001 <= x <= 1.000001 for x in raw)):
        raise RuntimeError('Fail Closed: invalid visible pre-action feature dimensions')
    pretime = case.start * .010
    if abs(frame.time_s - pretime - .010) > TOL:
        raise RuntimeError('Fail Closed: pre-action frame alignment drift')
    if tuple(frame.executed_action) != raw:
        raise RuntimeError('Fail Closed: baseline has intervened action')
    motor={'positions_m_normalized_by_0_006':list(hud[:8]),
           'velocities_m_s_normalized_by_1_0':list(hud[8:16]),
           'pressed_flags':list(hud[16:24])}
    digest=hashlib.sha256(json.dumps([hud,raw],separators=(',',':')).encode()).hexdigest()
    return {'anchor':case.anchor, 'frame':case.start, 'time_s':pretime,
            'visible_hud':list(hud), 'student_raw_action':list(raw),
            # First 24 dimensions are the existing visible HUD motor features.
            # No privileged motor/target fields are consulted.
            'observable_motor_from_hud_only':motor,
            'state_id':f'{case.anchor}:{digest}',
            'state_sha256':digest}


def assert_preaction_unchanged(reference, modified, case: Intervention):
    origin=checked_preaction(reference,case)
    if origin is None:
        raise RuntimeError('Fail Closed: intervened after baseline terminated')
    if len(modified.context)<=case.start:
        raise RuntimeError('Fail Closed: candidate ended before intervention')
    actual=modified.context[case.start]
    if (tuple(actual.visible_hud)!=tuple(origin['visible_hud']) or
            tuple(actual.student_action)!=tuple(origin['student_raw_action'])):
        raise RuntimeError('Fail Closed: pre-intervention state differs from baseline')
    return origin


def _is_observed(snapshot, horizon: float) -> bool:
    return snapshot.time_s>=horizon-TOL


def label_case(reference, candidate, *, case: Intervention) -> dict:
    """Separate paired physical comparisons from counterfactual continuation.

    No unobserved baseline future is treated as failure. Persist ALL adverse
    flags even if priority assigns a single label.
    """
    milestones={}
    for h in HORIZONS:
        b,c=reference.trace.at[h],candidate.trace.at[h]
        paired=_is_observed(b,h) and _is_observed(c,h)
        milestones[str(h)]={
            'baseline_observed':_is_observed(b,h),
            'candidate_observed':_is_observed(c,h),
            'paired':paired,
            'baseline':b.as_json(), 'candidate':c.as_json(),
            'hit_ids_lost':sorted(set(b.hit_ids)-set(c.hit_ids)) if paired else None,
            'hit_ids_gained':sorted(set(c.hit_ids)-set(b.hit_ids)) if paired else None,
            'miss_delta':c.misses-b.misses if paired else None,
            'early_delta':c.too_early-b.too_early if paired else None,
        }
    b8,c8=reference.trace.at[.8],candidate.trace.at[.8]
    b32,c32=reference.trace.at[3.2],candidate.trace.at[3.2]
    paired_all=all(m['paired'] for m in milestones.values())
    paired_32=milestones['3.2']['paired']
    lost=sorted(set(b32.hit_ids)-set(c32.hit_ids)) if paired_32 else []
    miss_regression=any(m['miss_delta'] is not None and m['miss_delta']>0
                        for m in milestones.values())
    hit_suppression=bool(lost)
    # A 800ms temporary lag that recovers by 3200ms is not permanent loss.
    observed_early_lags=sorted(set().union(*(set(m['hit_ids_lost'] or ())
                            for m in milestones.values() if m['paired'])))
    recovered=sorted(set(observed_early_lags)-set(lost)) if paired_32 else []
    safe_by_h={}
    for h,m in milestones.items():
        c=m['candidate']; b=m['baseline']
        safe_by_h[h]=bool(m['paired'] and not c['overloaded']
            and c['hits']>0 and len(c['keydowns'])>0
            and not m['hit_ids_lost'] and m['miss_delta']<=0
            and (m['early_delta']<0 or c['hits']>b['hits']))
    verified=paired_all and all(safe_by_h.values())
    base_terminal_early=not _is_observed(b32,3.2)
    extended=_is_observed(c32,3.2) and base_terminal_early
    kept_known=set(b8.hit_ids).issubset(c32.hit_ids)
    rescue=bool(extended and b8.overloaded and not c32.overloaded
                and c32.hits>0 and bool(c32.keydowns) and kept_known)
    extra_unpaired_misses=(c32.misses-b8.misses) if extended else None
    if verified:
        label='verified_safe_improvement'
    elif paired_32 and lost:
        label='hit_suppression'
    elif miss_regression:
        label='miss_regression'
    elif rescue:
        label='conditional_rescue'
    elif paired_32 and not lost and not miss_regression and (
            c32.hits==b32.hits and c32.misses==b32.misses and
            c32.too_early==b32.too_early and c32.overloaded==b32.overloaded and
            c32.hit_ids==b32.hit_ids and c32.miss_ids==b32.miss_ids):
        label='no_material_change'
    else:
        label='insufficient_evidence'
    if label not in LABELS:raise AssertionError('unreachable')
    return {'label':label,
            'safe_improvement':verified,'hit_suppression':hit_suppression,
            'miss_regression':miss_regression,'conditional_rescue':rescue,
            'known_hits_retained_by_3200':kept_known,
            'has_usable_hit_and_keydown':c32.hits>0 and bool(c32.keydowns),
            'temporarily_lagging_hit_ids':observed_early_lags,
            'recovered_hit_ids_by_3200':recovered,
        'observed_hit_lag_at_800':milestones['0.8']['hit_ids_lost'],
            'lost_hit_ids_by_3200':lost if paired_32 else None,
            'unpaired_extension_misses':extra_unpaired_misses,
            'baseline_future_unobserved':base_terminal_early,
            'per_horizon':milestones,
            'safe_by_horizon':safe_by_h,
            'initial_hit_gain':(c8.hits-b8.hits) if milestones['0.8']['paired'] else None,
            'post800_hit_gain':persistence(reference,candidate)['post800_hit_gain'],
            'reason_flags':{
                'baseline_early_overload':b8.overloaded,
                'candidate_overload':any(s.overloaded for s in (c8,c32)),
                'baseline_unobserved_future':base_terminal_early,
                'candidate_early_termination':not _is_observed(c32,3.2),
                'hit_id_loss':hit_suppression,
                'miss_regression':miss_regression,
                'too_early_reduced_3200':(c32.too_early<b32.too_early) if paired_32 else None,
                'too_early_increased_3200':(c32.too_early>b32.too_early) if paired_32 else None,
                'no_effective_physical_hit':c32.hits==0,
            },
    }


def _sat(x:float)->float:
    return float(x)/(1.+abs(float(x)))


def feature_blocks(pre:dict, case:Intervention):
    hud=pre['visible_hud'];act=pre['student_raw_action']
    if len(hud)!=263 or len(act)!=8:raise RuntimeError('Fail Closed: feature geometry')
    if not (0<=case.start<320 and 1<=case.frames<=320 and len(case.keys)>0
            and all(0<=k<8 for k in case.keys)):
        raise RuntimeError('Fail Closed: intervention geometry')
    if not all(math.isfinite(float(x)) for x in (*hud,*act,case.delta)):
        raise RuntimeError('Fail Closed: nonfinite feature')
    key_mask=tuple(float(k in case.keys) for k in range(8))
    return (
        tuple(_sat(x) for x in hud),
        tuple(float(x) for x in act),
        key_mask,
        (float(case.delta),),
        (min(1.,case.frames/8.),),
    )


def feature_distance(a:tuple,b:tuple)->float:
    """Predefined 5-block normalized root-mean-square Euclidean distance."""
    if len(a)!=5 or len(b)!=5:raise RuntimeError('Fail Closed: feature group mismatch')
    total=0.
    for weight,x,y in zip(FEATURE_GROUP_WEIGHTS,a,b):
        if len(x)!=len(y) or not x:raise RuntimeError('Fail Closed: feature dimension drift')
        total+=weight*sum((u-v)**2 for u,v in zip(x,y))/len(x)
    return math.sqrt(total/sum(FEATURE_GROUP_WEIGHTS))


def _resolved_labels(items):
    return [x for x in items if x['label'] in ('verified_safe_improvement',
            'hit_suppression','miss_regression','no_material_change')]


def similarity_analysis(rows: list[dict]) -> tuple[list[dict],list[dict],dict]:
    """Anchor-held-out 1NN, not an optimizer or a trained classifier."""
    reliable=_resolved_labels(rows)
    positive=lambda r:r['label']=='verified_safe_improvement'
    for r in rows:
        if r.get('pre_state') is not None:
            r['_blocks']=feature_blocks(r['pre_state'],r['case'])
    comparisons=[];opposites=[];correct=0;count=0
    # One result per (anchor, frame, mask, delta, duration) probe, not 1 per
    # bootstrap sample. Leave the whole anchor out of the reference pool.
    for r in reliable:
        if r.get('pre_state') is None:continue
        options=[o for o in reliable if o['anchor']!=r['anchor']
                 and o.get('pre_state') is not None]
        if not options:continue
        neighbor=min(options,key=lambda o:(feature_distance(r['_blocks'],o['_blocks']),
                  o['anchor'],o['case'].identity()))
        distance=feature_distance(r['_blocks'],neighbor['_blocks'])
        agreed=positive(r)==positive(neighbor)
        comparisons.append({'anchor':r['anchor'],'start':r['case'].start,
           'case':asdict(r['case']),'label':r['label'],
           'neighbor_anchor':neighbor['anchor'],'neighbor_case':asdict(neighbor['case']),
           'neighbor_label':neighbor['label'],'distance':distance,
           'success_vs_failure_agrees':agreed,'held_out_anchor':True})
        if not agreed:
            opposites.append(comparisons[-1])
        correct+=int(agreed);count+=1
    # Collapse before reporting independence: one anchor can contain many probes.
    anchor_to_labels=defaultdict(set);anchor_to_states=defaultdict(set)
    for r in rows:
        if r.get('pre_state') is None:continue
        anchor_to_states[r['anchor']].add(r['pre_state']['state_id'])
        if r['label'] in ('verified_safe_improvement','hit_suppression',
                          'miss_regression','no_material_change'):
            anchor_to_labels[r['anchor']].add('success' if positive(r) else 'failure')
    successes={a for a,labels in anchor_to_labels.items() if 'success' in labels}
    failures={a for a,labels in anchor_to_labels.items() if 'failure' in labels}
    report={
      'successful_anchor_count':len(successes),'failed_anchor_count':len(failures),
      'unresolved_anchor_count':20-len(successes|failures),
      'successful_anchors':sorted(successes),'failed_anchors':sorted(failures),
      'mixed_success_failure_anchors':sorted(successes&failures),
      'distinct_pre_intervention_states':sum(map(len,anchor_to_states.values())),
      'distinct_anchors_with_states':len(anchor_to_states),
      'anchor_held_out_nn_comparisons':count,
      'probe_weighted_held_out_nn_agreement_diagnostic_only':correct/count if count else None,
      'anchor_held_out_nn_agreement':correct/count if count else None,
      'anchors_evaluated_with_held_out_nn':len({r['anchor'] for r in comparisons}),
      'anchor_macro_nn_agreement':(
          sum(sum(int(c['success_vs_failure_agrees']) for c in comparisons if c['anchor']==a)/
              sum(1 for c in comparisons if c['anchor']==a)
              for a in {c['anchor'] for c in comparisons}) /
          len({c['anchor'] for c in comparisons}) if comparisons else None),
      'generalization_status':('insufficient_independent_evidence' if len(successes)<5
          else 'diagnostic_only_at_least_five_success_anchors'),
      'can_conclude_separation':bool(len(successes)>=5 and count>=5 and
                                 correct==count and successes and failures),
      'notes':'Multiple interventions at one anchor and onset frame are not independent evidence. No fitted parameters or Validation.',
    }
    return comparisons,opposites,report


def _csv_equal(saved: dict, computed:dict, keys, label:str):
    for key in keys:
        if key not in saved:
            raise RuntimeError(f'Fail Closed: missing prior {label}/{key}')
        current=computed.get(key)
        value='' if current is None else str(_json_field(current))
        if saved[key]!=value:
            raise RuntimeError(f'Fail Closed: physical reproduction differs {label}/{key}: saved={saved[key]!r} actual={value!r}')


def _load_csv(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise RuntimeError('Fail Closed: malformed prior CSV '+str(path))
        return list(reader)


def _load_v15(v15:Path, v14_hashes:dict):
    names=('diagnostic.json','reproduction_checks.json','boundary_grid_06.csv',
           'late_drift_18.csv','rescue_survival_quality.csv','gain_persistence.csv',
           'event_divergence.csv')
    files={name:v15/name for name in names}
    if any(not f.is_file() for f in files.values()):
        raise FileNotFoundError('Fail Closed: v1.5 diagnostic files missing')
    hashes=_hash_inputs(files)
    report=json.loads(files['diagnostic.json'].read_text(encoding='utf-8'))
    checks=json.loads(files['reproduction_checks.json'].read_text(encoding='utf-8'))
    if (report.get('kind')!='train_counterfactual_motor_v15' or
            report.get('best_sha256')!=EXPECTED_BEST_SHA256 or
            report.get('input_sha256')!=v14_hashes or
            report.get('train_anchors')!=20 or
            report.get('validation_evaluations')!=0 or
            report.get('final_used') is not False or
            report.get('optimizer_updates')!=0 or
            report.get('teacher_adoptions')!=0 or
            report.get('checkpoint_writes')!=0 or
            report.get('reference_v14_reproduced') is not True or
            report.get('v14_representatives')!=9 or
            report.get('horizons_s')!=list(HORIZONS) or
            checks.get('status')!='PASS' or
            checks.get('best_sha256')!=EXPECTED_BEST_SHA256 or
            checks.get('prior_input_sha256')!=v14_hashes or
            checks.get('baseline_reproduced')!=20 or
            checks.get('representative_reproduced')!=9 or
            checks.get('frozen_student') is not True or
            checks.get('validation_evaluations')!=0 or
            checks.get('final_used') is not False or
            checks.get('optimizer_updates')!=0):
        raise RuntimeError('Fail Closed: v1.5 provenance or reproduction status mismatch')
    # Preserve all physically recorded trials; do not trust a verbal expected count.
    sources={name:_load_csv(files[name]) for name in names if name.endswith('.csv')}
    if (len(sources['boundary_grid_06.csv'])!=18 or
            len(sources['late_drift_18.csv'])!=18 or
            len(sources['rescue_survival_quality.csv'])!=36 or
            len(sources['gain_persistence.csv'])!=9 or
            int(report['boundary_06']['tested_grid_points'])!=18 or
            int(report['late_drift_18']['tested_grid_points'])!=18):
        raise RuntimeError('Fail Closed: v1.5 saved trial counts invalid')
    return files,hashes,report,checks,sources


def _prior_case_index(rows, *, horizons=False):
    result={}
    for row in rows:
        try:
            case=Intervention(int(row['anchor']),tuple(json.loads(row['keys'])),
                              float(row['delta']),int(row['start']),int(row['frames']))
            h=float(row['horizon_s']) if horizons else None
            key=(case.identity(),h) if horizons else case.identity()
            if key in result:raise ValueError('duplicate')
            result[key]=row
        except (KeyError,TypeError,ValueError, json.JSONDecodeError) as e:
            raise RuntimeError('Fail Closed: invalid saved intervention identity') from e
    return result


def _check_output(output:Path,inputs:tuple[Path,...],checkpoint:Path):
    if output.exists():raise FileExistsError('v1.6 output directory already exists: '+str(output))
    dest=output.resolve()
    for path in (*inputs,checkpoint):
        src=path.resolve()
        if dest==src or dest in src.parents or src in dest.parents:
            raise RuntimeError('Fail Closed: output overlaps read-only source or Best')


def _check_prepared(prepared):
    if (len(prepared.anchors)!=20 or prepared.key_count!=8 or prepared.input_dim!=263 or
            abs(prepared.control_dt_s-.010)>1e-12 or
            abs(prepared.physics_dt_s-.001)>1e-12):
        raise RuntimeError('Fail Closed: Train20 8key 263D 10ms 1ms contract changed')


def _publish(output:Path, tables:dict[str,list[dict]], report:dict, checks:dict):
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.v16-staging-',dir=output.parent) as tmp:
        stage=Path(tmp)
        empty_csv_headers={
            'state_similarity.csv':('anchor','start','case','label','neighbor_anchor','neighbor_case',
                                    'neighbor_label','distance','success_vs_failure_agrees','held_out_anchor'),
            'opposite_label_neighbors.csv':('anchor','start','case','label','neighbor_anchor',
                                             'neighbor_case','neighbor_label','distance',
                                             'success_vs_failure_agrees','held_out_anchor'),
        }
        for name,rows in tables.items():
            if name.endswith('.jsonl'):
                with (stage/name).open('x',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            else:
                if not rows and name not in empty_csv_headers:
                    raise RuntimeError('Fail Closed: empty required v1.6 CSV '+name)
                keys=tuple(rows[0]) if rows else empty_csv_headers[name]
                with (stage/name).open('x',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=keys,extrasaction='raise')
                    writer.writeheader()
                    for r in rows:
                        writer.writerow({k:_json_field(r.get(k)) for k in keys})
        (stage/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (stage/'reproduction_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        if output.exists():raise FileExistsError('v1.6 output appeared during audit')
        os.rename(stage,output)



def verify_saved_event_traces(rows:list[dict],cache:ReplayCache) -> int:
    """Audit v1.5's recorded keydown/timing divergence for ALL saved trials."""
    fields=(
        'first_divergence_s','time_resolution_s','observed_not_causal',
        'no_observed_divergence','different_termination_time_without_shared_result_divergence',
        'baseline_previous_step','candidate_previous_step',
        'baseline_next_step','candidate_next_step',
        'baseline_before_step_visible_hud','candidate_before_step_visible_hud',
        'baseline_student_action','candidate_student_action',
        'baseline_executed_action','candidate_executed_action',
        'baseline_new_keydowns','candidate_new_keydowns',
        'baseline_hit_ids','candidate_hit_ids','baseline_miss_ids','candidate_miss_ids',
    )
    if len(rows)!=57:
        raise RuntimeError('Fail Closed: v1.5 event trace count differs (expected 57)')
    seen=set()
    for row in rows:
        try:
            c=Intervention(int(row['anchor']),tuple(json.loads(row['keys'])),
                           float(row['delta']),int(row['start']),int(row['frames']))
            role=row['role']
            identity=(c.identity(),role)
            if identity in seen or not role:
                raise RuntimeError('Fail Closed: repeated v1.5 event identity')
            seen.add(identity)
        except (KeyError,ValueError,TypeError,json.JSONDecodeError) as exc:
            raise RuntimeError('Fail Closed: malformed v1.5 event identity') from exc
        base=cache.get(None,anchor=c.anchor)
        candidate=cache.get(c)
        divergence=first_divergence(base,candidate)
        _csv_equal(row,divergence,fields,f'v1.5 saved keydown event {identity}')
    return len(seen)


def run_v16(prepared, *, v12_dir:Path,v13_dir:Path,v14_dir:Path,v15_dir:Path,
            output:Path,seed:int=0)->dict:
    """GPU diagnostic only on user's Windows; no parameter updates or promotion."""
    _check_prepared(prepared)
    _check_output(output,(v12_dir,v13_dir,v14_dir,v15_dir),prepared.source_checkpoint)
    from dmdod.training.fixed_state_action_audit import assert_state_equal
    from dmdod.training.real_chart import clone_model_state
    model_state=clone_model_state(prepared.model)
    before_best=file_hash(prepared.source_checkpoint)
    if before_best!=EXPECTED_BEST_SHA256:
        raise RuntimeError('Fail Closed: Best checkpoint SHA256 differs')
    files,hashes,selected,saved_base,saved_cases,saved_neighbors=_load_previous(
        v12_dir,v13_dir,v14_dir,prepared.source_checkpoint)
    if len(selected)!=9 or len(saved_base)!=60:
        raise RuntimeError('Fail Closed: v1.4 representatives or baseline coverage changed')
    v15_files,v15_hashes,v15_report,v15_checks,prior=_load_v15(v15_dir,hashes)
    all_files={**files,**{f'v15_{key}':p for key,p in v15_files.items()}}
    all_hashes=_hash_inputs(all_files)
    cache=ReplayCache(prepared)
    labeled=[];known=[];discovered=[];states=[]
    reproduced={'baselines':0,'v14_representatives':0,'v15_boundary':0,
                'v15_drift':0,'v15_rescue_horizons':0,'v15_persistence':0,
                'v14_neighbors':0,'v15_event_traces':0}
    try:
        for anchor in range(1,21):
            base=cache.get(None,anchor=anchor)
            for h in HORIZONS:
                assert_reproduced(base.trace.at[h],saved_base[anchor,h],label=f'v1.4 base #{anchor}/{h}')
            reproduced['baselines']+=1
            for frame in ONSET_FRAMES:
                probe=Intervention(anchor,(0,),-.20,frame,8)
                state=checked_preaction(base,probe)
                if state is not None:
                    states.append(state)
            print(f'v1.6 Stage A Train baseline {anchor:02d}/20',flush=True)
        # Reproduce the v1.4 nine representatives and 1.5 saved persistence.
        saved_persist=_prior_case_index(prior['gain_persistence.csv'])
        for case,_,role in selected:
            base=cache.get(None,anchor=case.anchor)
            run=cache.get(case)
            for h,reference in _saved_horizons(saved_cases[case.identity()],
                         'candidate_snapshots',f'v1.4 #{case.anchor}').items():
                assert_reproduced(run.trace.at[h],reference,label=f'v1.4 #{case.anchor}/{h}')
            computed={'initial_hit_delta_800':persistence(base,run)['initial_hit_delta_800'],
                      'post800_hit_gain':persistence(base,run)['post800_hit_gain'],
                      'post800_miss_gain':persistence(base,run)['post800_miss_gain'],
                      **{f'cmp_{h}':compare_horizon(base.trace,run.trace,h) for h in HORIZONS}}
            saved=saved_persist.get(case.identity())
            if saved is None:raise RuntimeError('Fail Closed: missing v1.5 persistence case')
            _csv_equal(saved,computed,tuple(computed),f'v1.5 persistence #{case.anchor}')
            reproduced['v15_persistence']+=1
            reproduced['v14_representatives']+=1
        # Recheck saved v1.4 neighborhoods and all v1.5 grids before adoption.
        for (anchor,_),record in saved_neighbors.items():
            from dmdod.training.train_counterfactual_motor_v15 import case_from_dict
            case=case_from_dict(record['case'])
            base=cache.get(None,anchor=anchor)
            changed=cache.get(case)
            for h,key in ((.8,'at_800ms'),(1.6,'at_1600ms')):
                _compare_saved_metric(compare_horizon(base.trace,changed.trace,h),record[key],
                                      f'v1.4 neighbor #{anchor}/{h}')
            reproduced['v14_neighbors']+=1
        def add(case,stage,*,role='known'):
            base=cache.get(None,anchor=case.anchor)
            pre=checked_preaction(base,case)
            if pre is None:
                return {'anchor':case.anchor,'stage':stage,'role':role,
                        'case':case,'pre_state':None,'label':'insufficient_evidence',
                        'executable':False,'reason':'baseline_terminated_before_start',
                        'result':None}
            run=cache.get(case)
            assert_preaction_unchanged(base,run,case)
            result=label_case(base,run,case=case)
            return {'anchor':case.anchor,'stage':stage,'role':role,
                    'case':case,'pre_state':pre,'label':result['label'],
                    'executable':True,'reason':None,'result':result}
        # v1.5 Stage #06, #18 and rescue cases, each checked against recorded CSV.
        boundary=_prior_case_index(prior['boundary_grid_06.csv'])
        for start in (0,1):
            for delta in GRID_06:
                case=Intervention(6,(2,),delta,start,8)
                base=cache.get(None,anchor=6);run=cache.get(case)
                current=boundary_grid([(case,run)],base)[0]
                _csv_equal(boundary[case.identity()],current,
                           ('source_800ms_5H_0M_1E','success_grid_point','milestones',
                            'post800_hit_gain','post800_miss_gain'),f'v1.5 boundary {case.identity()}')
                known.append(add(case,'known_boundary_06'))
                reproduced['v15_boundary']+=1
        drift=_prior_case_index(prior['late_drift_18.csv'])
        for keys,frames,starts in (((5,),1,(3,4,5)),(tuple(range(8)),8,(7,8,9))):
            for start in starts:
                for delta in GRID_18:
                    case=Intervention(18,keys,delta,start,frames)
                    base=cache.get(None,anchor=18);run=cache.get(case)
                    first=first_hit_loss(base,run)
                    computed={
                      **{f'hit_ids_retained_{int(h*1000)}':not bool(set(base.trace.at[h].hit_ids)-set(run.trace.at[h].hit_ids)) if (
                           _is_observed(base.trace.at[h],h) and _is_observed(run.trace.at[h],h)) else None for h in HORIZONS},
                      **first, 'first_extra_miss_s':first_miss_excess(base,run),
                      'first_miss_id_difference':first_miss_id_difference(base,run),
                      'first_divergence':first_divergence(base,run),
                      'horizon_target_id_deltas':{str(h):horizon_target_deltas(base.trace,run.trace,h) for h in HORIZONS},
                      'too_early_reduction':too_early_reduction_duration(base,run),
                      'post800_hit_gain':persistence(base,run)['post800_hit_gain'],
                      'post800_miss_gain':persistence(base,run)['post800_miss_gain']}
                    _csv_equal(drift[case.identity()],computed,tuple(computed),
                               f'v1.5 drift {case.identity()}')
                    known.append(add(case,'known_drift_18'))
                    reproduced['v15_drift']+=1
        rescue=_prior_case_index(prior['rescue_survival_quality.csv'],horizons=True)
        for anchor in (8,20):
            origin=[c for c,_,role in selected if c.anchor==anchor and
                    role in ('representative','conditional_08')]
            if len(origin)!=1:raise RuntimeError('Fail Closed: rescue origin identity')
            from dmdod.training.train_counterfactual_motor_v14 import local_neighbors
            cases=(origin[0],*local_neighbors(origin[0]))
            if len(cases)!=6:raise RuntimeError('Fail Closed: expected six rescue neighbors')
            for case in cases:
                base=cache.get(None,anchor=anchor);run=cache.get(case)
                for measured in rescue_quality(base,run,case):
                    h=measured['horizon_s']
                    saved=rescue.get((case.identity(),h))
                    if saved is None:raise RuntimeError('Fail Closed: missing prior rescue metric')
                    _csv_equal(saved,measured,('hits','misses','too_early','keydowns',
                        'hit_ids','miss_ids','overload','overload_time_s','reached_time_s',
                        'survival_success','quality_success','known_baseline_hit_ids_retained'),
                        f'v1.5 rescue #{anchor}/{h}')
                    reproduced['v15_rescue_horizons']+=1
                known.append(add(case,'known_rescue'))
        # Add any distinct v1.4 controls not already in known grids.
        seen={r['case'].identity() for r in known}
        for case,_,role in selected:
            if case.identity() not in seen:
                known.append(add(case,'known_representative',role=role))
                seen.add(case.identity())
        if reproduced['v15_boundary']!=18 or reproduced['v15_drift']!=18 or reproduced['v15_rescue_horizons']!=36:
            raise RuntimeError('Fail Closed: incomplete known reproduction')
        # Four published v1.5 aggregates must agree with the independently
        # replayed physical evidence; never use reported successes as labels.
        six_grid=[Intervention(6,(2,),d,start,8) for start in (0,1) for d in GRID_06]
        grid_actual=boundary_grid([(c,cache.get(c)) for c in six_grid],cache.get(None,anchor=6))
        counts={
          'boundary_06':sum(x['success_grid_point'] for x in grid_actual),
          'drift_18':sum(all(
              not (set(cache.get(None,anchor=18).trace.at[h].hit_ids)-set(cache.get(c).trace.at[h].hit_ids))
              and _is_observed(cache.get(None,anchor=18).trace.at[h],h)
              and _is_observed(cache.get(c).trace.at[h],h) for h in HORIZONS)
              for c in (Intervention(18,keys,d,start,frames)
                  for keys,frames,starts in (((5,),1,(3,4,5)),(tuple(range(8)),8,(7,8,9)))
                  for start in starts for d in GRID_18)),
        }
        for anchor in (8,20):
            origin=next(c for c,_,role in selected if c.anchor==anchor and role in ('representative','conditional_08'))
            from dmdod.training.train_counterfactual_motor_v14 import local_neighbors
            counts[f'rescue_{anchor}']=sum(
                _is_observed(cache.get(c).trace.at[3.2],3.2) and
                not cache.get(c).trace.at[3.2].overloaded
                for c in (origin,*local_neighbors(origin)))
        saved_counts={
          'boundary_06':int(v15_report['boundary_06']['successful_grid_points']),
          'drift_18':int(v15_report['late_drift_18']['retained_all_horizons']),
          'rescue_8':int(v15_report['rescue_survival_quality']['8']['3.2']['survived']),
          'rescue_20':int(v15_report['rescue_survival_quality']['20']['3.2']['survived']),
        }
        if counts!=saved_counts:
            raise RuntimeError(f'Fail Closed: v1.5 aggregate reproduction mismatch {counts} != {saved_counts}')
        if saved_counts!=EXPECTED_V15_AGGREGATES:
            raise RuntimeError('Fail Closed: v1.5 stated known outcome counts changed')
        reproduced['v15_event_traces']=verify_saved_event_traces(
            prior['event_divergence.csv'],cache)
        print('v1.6 Stage B saved physical reproduction: PASS',flush=True)
        probes=discovery_cases()
        for n,case in enumerate(probes,1):
            discovered.append(add(case,'discovery',role='exhaustive'))
            if n%18==0:
                print(f'v1.6 Stage C discovery {n:03d}/288 (Train #{case.anchor:02d})',flush=True)
        if len(discovered)!=288:raise RuntimeError('Fail Closed: discovery coverage changed')
        labeled=known+discovered
        similar,opposite,generalization=similarity_analysis(labeled)
        # Restore in finally, but verify the *unmodified* checkpoint/model before output.
        assert_state_equal(clone_model_state(prepared.model),model_state,
                           context='v1.6 frozen Student prior to output')
        if file_hash(prepared.source_checkpoint)!=before_best or _hash_inputs(all_files)!=all_hashes:
            raise RuntimeError('Fail Closed: checkpoint or previous diagnostics changed')
        by_anchor=defaultdict(list)
        for r in labeled:by_anchor[r['anchor']].append(r)
        summaries=[]
        for anchor in range(1,21):
            rows=by_anchor[anchor]
            summaries.append({'anchor':anchor,'tested':len(rows),
                'executable':sum(r['executable'] for r in rows),
                'known':sum(r['stage']!='discovery' for r in rows),
                'discovery':sum(r['stage']=='discovery' for r in rows),
                'distinct_preaction_states':len({r['pre_state']['state_id'] for r in rows if r['pre_state']}),
                'labels':dict(Counter(r['label'] for r in rows)),
                'has_safe_improvement':any(r['label']=='verified_safe_improvement' for r in rows),
                'has_verified_failure':any(r['label'] in ('hit_suppression','miss_regression') for r in rows)})
        def output_record(r):
            current={'anchor':r['anchor'],'stage':r['stage'],'role':r['role'],
                     **asdict(r['case']), 'executable':r['executable'],
                     'state_id':r['pre_state']['state_id'] if r['pre_state'] else None,
                     'pre_state_sha256':r['pre_state']['state_sha256'] if r['pre_state'] else None,
                     'pre_intervention_visible_hud':r['pre_state']['visible_hud'] if r['pre_state'] else None,
                     'pre_intervention_student_action':r['pre_state']['student_raw_action'] if r['pre_state'] else None,
                     'label':r['label'],'reason':r['reason']}
            result=r['result']
            if result:current.update(result)
            return current
        known_rows=[output_record(r) for r in known]
        discovery_rows=[output_record(r) for r in discovered]
        labels=Counter(r['label'] for r in labeled)
        report={'kind':'train_counterfactual_motor_v16',
                'best_sha256':before_best,'input_sha256':all_hashes,
                'train_anchors':20,'feature_dimensions':{'visible_hud':263,'student_action':8},
                'control_dt_s':.010,'physics_dt_s':.001,'horizons_s':list(HORIZONS),
                'fixed_onset_frames':list(ONSET_FRAMES),
                'discovery_probe_count':len(discovered),
                'discovery_executable':sum(r['executable'] for r in discovered),
                'known_case_count':len(known),'known_case_labels':dict(Counter(r['label'] for r in known)),
                'discovery_labels':dict(Counter(r['label'] for r in discovered)),
                'all_labels':dict(labels),
                'state_similarity':{'normalization':FEATURE_NORMALIZATION,
                    'group_weights':list(FEATURE_GROUP_WEIGHTS),'distance':'five_block_rms_l2'},
                'independent_evidence':generalization,
                'replay_cache':{'unique_replays':cache.executions,
                                'cache_hits':cache.cache_hits},
                'validation_evaluations':0,'final_used':False,
                'optimizer_updates':0,'teacher_adoptions':0,'checkpoint_writes':0,
                'reproduction_status':'PASS',
                'limitations':['Train-only, 3.2s horizon and selected counterfactual starts.',
                   'No teacher labels were adopted; known cases and exploratory probes are diagnostic only.',
                   'No imputing future scores after either rollout terminates.',
                   'Anchor-held-out similarity is not an estimate of deployment accuracy.',
                   'The fixed 5-anchor threshold is evidence sufficiency, NOT training approval.']}
        checks={'status':'PASS','best_sha256':before_best,'input_sha256':all_hashes,
                'reproduced':reproduced,'train_probes_expected':288,
                'train_probes_recorded':len(discovered),'model_unchanged':True,
                'no_validation_or_final':True,'optimizer_updates':0,
                'fixed_seed':seed,'feature_provenance':'unintervened_pre_action_only'}
        tables={'onset_states.jsonl':states,'known_case_labels.csv':known_rows,
                'discovery_probes.csv':discovery_rows,'anchor_summary.csv':summaries,
                'state_similarity.csv':similar,
                'opposite_label_neighbors.csv':opposite}
        if _hash_inputs(all_files)!=all_hashes or file_hash(prepared.source_checkpoint)!=before_best:
            raise RuntimeError('Fail Closed: source changed before publication')
        _publish(output,tables,report,checks)
        return report
    finally:
        prepared.model.load_state_dict(model_state,strict=True)
        prepared.model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(prepared.model),model_state,
                           context='v1.6 restored frozen Best Student')
