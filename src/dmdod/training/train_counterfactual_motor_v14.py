from __future__ import annotations

"""Train-only 800/1600/3200ms counterfactual motor continuation audit.

No training, teacher adoption, Validation/FINAL evaluation, checkpoint writes,
Motor parameter changes, or privileged inputs to the Student.
"""

from dataclasses import asdict, dataclass, replace
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from dmdod.training.train_counterfactual_motor_v12 import (
    EXPECTED_BEST_SHA256, Snapshot, _environment_snapshot,
    _assert_snapshot_consistent, csv_write_new,
)
from dmdod.training.train_counterfactual_motor_v13 import (
    MEASUREMENT_FIELDS, bool_value, file_hash, integer, parsed_json,
    snapshot_from_dict, validate_inputs,
)

HORIZONS = (0.8, 1.6, 3.2)
TOL = 1e-6
NEIGHBOR_DELTA = 0.02
# The numeric key indices are the original eight-key motor indices.
REPRESENTATIVES = (
    (6, (2,), +.02, 0, 4),
    (6, tuple(range(8)), -.20, 0, 8),
    (6, (2,), -.20, 0, 8),
    (18, (5,), -.20, 4, 1),
    (18, tuple(range(8)), -.20, 8, 8),
    (20, (6,), -.20, 0, 4),
)
EXPECTED_800MS = ((4,1,2),(4,1,1),(5,0,1),(11,0,3),(11,0,2),(13,0,2))


@dataclass(frozen=True)
class Intervention:
    anchor: int
    keys: tuple[int, ...]
    delta: float
    start: int
    frames: int

    @property
    def end(self) -> int:
        return self.start + self.frames

    def identity(self) -> tuple:
        return (self.anchor, self.keys, round(self.delta, 8), self.start, self.frames)


@dataclass
class Trace:
    at: dict[float, Snapshot]
    frames: tuple[Snapshot, ...]
    observable: tuple[tuple[float, ...], ...]

    def at_or_terminal(self, horizon: float) -> Snapshot:
        return self.at[horizon]


def _hash_inputs(files: dict[str, Path]) -> dict[str, str]:
    return {k: file_hash(p) for k,p in files.items()}


def _read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise RuntimeError(f'Fail Closed: invalid CSV header: {path}')
        rows = list(reader)
    if not rows:
        raise RuntimeError(f'Fail Closed: empty CSV: {path}')
    return rows


def case_from_row(row: dict, names: tuple[str, ...]) -> Intervention:
    try:
        raw_keys=row['keys'].split(',')
        keys=tuple(int(k) if k.isdecimal() else names.index(k) for k in raw_keys)
        result = Intervention(integer(row['anchor'],'anchor'), keys,
                              float(row['delta']), integer(row['start_frame'],'start'),
                              integer(row['duration_frames'],'duration'))
    except (ValueError, KeyError) as exc:
        raise RuntimeError('Fail Closed: malformed saved intervention') from exc
    if (result.anchor not in range(1,21) or not keys or len(set(keys))!=len(keys)
            or any(k<0 or k>=len(names) for k in keys)
            or not math.isfinite(result.delta) or result.delta==0
            or result.start<0 or result.frames<=0
            or row['kind'] != ('all_8' if len(keys)==8 else
                               'single' if len(keys)==1 else 'paired')):
        raise RuntimeError('Fail Closed: intervention geometry invalid')
    return result


def select_cases(rows: list[dict], names: tuple[str, ...]):
    if len(rows)!=152:
        raise RuntimeError(f'Fail Closed: expected 152 v1.3 candidates, got {len(rows)}')
    identity={}
    for row in rows:
        case=case_from_row(row,names)
        if case.identity() in identity:
            raise RuntimeError('Fail Closed: duplicate intervention identity')
        identity[case.identity()]=row
    selected=[]
    for slot,(a,keys,d,start,n) in enumerate(REPRESENTATIVES):
        c=Intervention(a,keys,d,start,n)
        if c.identity() not in identity:
            raise RuntimeError(f'Fail Closed: representative missing: {c.identity()}')
        row=identity[c.identity()]
        if not bool_value(row['safe_review_candidate'],'safe_review_candidate'):
            raise RuntimeError(f'Fail Closed: representative was not v1.3 safe: {c.identity()}')
        measured=(integer(row['hits_800'],'hits_800'),
                  integer(row['misses_800'],'misses_800'),
                  integer(row['early_800'],'early_800'))
        if measured!=EXPECTED_800MS[slot]:
            raise RuntimeError(f'Fail Closed: representative saved 800ms score differs: {c.identity()} {measured}')
        selected.append((c,row,'representative'))
    if selected[-1][1]['source']!='v1_manual_unsampled':
        raise RuntimeError('Fail Closed: Train #20 must be separately sampled v1 extra')
    conditional=[(case_from_row(r,names),r,'conditional_08') for r in rows
                 if int(r['anchor'])==8 and
                 bool_value(r['conditional_review_candidate'],'conditional_review_candidate')]
    if len(conditional)!=1:
        raise RuntimeError(f'Fail Closed: expected 1 Train #08 conditional candidate, got {len(conditional)}')
    if (integer(conditional[0][1]['hits_800'],'#08 hits')!=10 or
        integer(conditional[0][1]['misses_800'],'#08 misses')!=2):
        raise RuntimeError('Fail Closed: Train #08 conditional reference unexpectedly changed')
    selected.extend(conditional)
    # Different control shapes with the SAME saved 800ms outcome, when present.
    controls=[]
    used={c.identity() for c,_,_ in selected}
    for c,row,_ in selected[:5]:
        found=[]
        for raw in rows:
            other=case_from_row(raw,names)
            if (other.anchor==c.anchor and other.identity() not in used
                    and raw['outcome_group_signature']==row['outcome_group_signature']):
                found.append((other,raw,'same_outcome_control'))
        # Selection is deterministic and independent of future 1600/3200ms outcomes.
        found.sort(key=lambda e:(e[0].keys==c.keys, len(e[0].keys),
                                 e[0].start,e[0].frames,abs(e[0].delta),e[0].keys))
        if found:
            candidate=found[0]
            controls.append(candidate)
            used.add(candidate[0].identity())
    selected.extend(controls)
    return selected


def local_neighbors(case: Intervention) -> tuple[Intervention,...]:
    """Independent +/- 1 frame and +/- 0.02 perturbations (no negative start)."""
    variants=set()
    for shift in (-1,0,1):
        for change in (-NEIGHBOR_DELTA,0.,NEIGHBOR_DELTA):
            start=case.start+shift
            delta=round(case.delta+change,8)
            if (start<0 or (shift==0 and change==0) or
                    abs(delta)>1.0 or abs(delta)<1e-9):
                continue
            variants.add(replace(case,start=start,delta=delta))
    return tuple(sorted(variants,key=lambda c:(c.start,c.delta)))


def _termination(s: Snapshot, horizon: float) -> dict:
    return {**s.as_json(), 'requested_horizon_s':horizon,
            'horizon_observed': s.time_s >= horizon - TOL,
            'unobserved_future': s.time_s < horizon-TOL}


def run_long_trace(prepared, named, *, case: Intervention|None,
                   capture_visible: bool=False) -> Trace:
    """One causal closed-loop replay; snapshots at 0.8, 1.6, 3.2s.

    If the episode terminates early, the final *observed* state is reused for
    table layout, explicitly marked as horizon_observed=False. No future is
    simulated or scored. Inputs to Student are only the same original 263D.
    """
    from dmdod.training.motor_sensitivity_audit import _environment
    from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
    from dmdod.motor.n_key import NKeyAction
    env=_environment(prepared,named)
    observation=env.reset()
    model=prepared.model
    state=model.initial_state(prepared.device)
    model.eval()
    steps=[]
    visible=[]
    horizon_snapshots={}
    with torch.no_grad():
        for frame in range(math.ceil(HORIZONS[-1]/prepared.control_dt_s)):
            vector=encode_n_key_hud_real_chart_observation(observation)
            if len(vector)!=263:
                raise RuntimeError('Student 263D visible contract violated')
            x=torch.tensor(vector,dtype=torch.float32,device=prepared.device)
            means,_,_,state=model.forward_step(x,state)
            action=tuple(float(value.item()) for value in torch.tanh(means))
            if capture_visible and frame<20:
                visible.append(tuple(float(v) for v in vector)+action)
            if case and case.start<=frame<case.end:
                a=list(action)
                for key in case.keys:
                    a[key]=max(-1.,min(1.,a[key]+case.delta))
                action=tuple(a)
            result=env.step(NKeyAction(action))
            observation=result.observation
            snap=_environment_snapshot(env,finished=bool(result.done))
            _assert_snapshot_consistent(snap)
            steps.append(snap)
            for horizon in HORIZONS:
                if horizon not in horizon_snapshots and snap.time_s>=horizon-TOL:
                    horizon_snapshots[horizon]=snap
            if result.done:
                break
    if not steps:
        raise RuntimeError('empty Train rollout')
    for horizon in HORIZONS:
        horizon_snapshots.setdefault(horizon,steps[-1])
    return Trace(at=horizon_snapshots,frames=tuple(steps),observable=tuple(visible))


def assert_reproduced(actual: Snapshot, expected: Snapshot, *, label: str) -> None:
    """Compare 800ms physical outcomes (including event-key/time tuples)."""
    a,b=actual,expected
    if (a.hits,a.misses,a.too_early,a.overloaded,a.terminated,
        a.termination_reason,a.hit_ids,a.miss_ids)!=(
        b.hits,b.misses,b.too_early,b.overloaded,b.terminated,
        b.termination_reason,b.hit_ids,b.miss_ids):
        raise RuntimeError('Fail Closed: '+label+' 800ms physical outcome mismatch')
    if (abs(a.time_s-b.time_s)>TOL or
        (a.overload_time_s is None)!=(b.overload_time_s is None) or
        (a.overload_time_s is not None and
         abs(a.overload_time_s-b.overload_time_s)>TOL) or
        len(a.keydowns)!=len(b.keydowns) or any(
            x[0]!=y[0] or abs(x[1]-y[1])>1e-5
            for x,y in zip(a.keydowns,b.keydowns))):
        raise RuntimeError('Fail Closed: '+label+' 800ms physical timing mismatch')


def snapshot_at_common(frames: tuple[Snapshot,...], until: float) -> Snapshot:
    if not frames:
        raise RuntimeError('no physical frames')
    eligible=[f for f in frames if f.time_s<=until+TOL]
    if not eligible:
        raise RuntimeError('no common observed interval')
    return eligible[-1]


def compare_horizon(baseline: Trace, changed: Trace, horizon:float) -> dict:
    ref=baseline.at[horizon]
    alt=changed.at[horizon]
    common_t=min(ref.time_s,alt.time_s,horizon)
    r=snapshot_at_common(baseline.frames,common_t)
    c=snapshot_at_common(changed.frames,common_t)
    lost_common=sorted(set(r.hit_ids)-set(c.hit_ids))
    lost_final=sorted(set(ref.hit_ids)-set(alt.hit_ids))
    recovered_lagging=sorted(set(lost_common) & set(alt.hit_ids))
    unrecovered_lagging=sorted(set(lost_common) - set(alt.hit_ids))
    gained_final=sorted(set(alt.hit_ids)-set(ref.hit_ids))
    ref_full=ref.time_s>=horizon-TOL
    alt_full=alt.time_s>=horizon-TOL
    extension=[f for f in changed.frames if f.time_s>common_t+TOL
               and f.time_s<=alt.time_s+TOL]
    later=extension[-1] if extension else c
    new_miss_ids=sorted(set(later.miss_ids)-set(c.miss_ids))
    new_hit_ids=sorted(set(later.hit_ids)-set(c.hit_ids))
    extra_keydowns=len(later.keydowns)-len(c.keydowns)
    retains=not lost_final
    common_miss_ok=c.misses<=r.misses
    full_miss_ok=(alt.misses<=ref.misses) if ref_full and alt_full else None
    physical=alt.hits>0 and len(alt.keydowns)>0
    rescue=ref.overloaded and not alt.overloaded and alt.time_s>ref.time_s+TOL
    no_deferred=not alt.overloaded and (alt_full or not ref_full)
    no_suppression=retains and physical and common_miss_ok and extra_keydowns>=0
    early_lower=(alt.too_early<ref.too_early if ref_full and alt_full
                 else c.too_early<r.too_early)
    gain=len(gained_final)>0
    # A temporary lag at the common cutoff can be recovered by the
    # candidate's observed 800/1600/3200ms continuation.
    shared_valid=common_miss_ok and not unrecovered_lagging
    safe=(shared_valid and no_suppression and no_deferred
          and (full_miss_ok is not False)
          and (early_lower or gain or rescue)
          and not (rescue and new_miss_ids))
    conditional=(rescue and shared_valid and physical and
                 (bool(new_miss_ids) or not alt_full))
    if not physical:
        verdict='insufficient_physical_evidence'
    elif not retains:
        verdict='hit_suppression'
    elif not shared_valid or full_miss_ok is False:
        verdict='regression'
    elif alt.overloaded:
        verdict='deferred_failure'
    elif conditional:
        verdict='conditional_rescue_review'
    elif safe and ref_full and alt_full:
        verdict='verified_common_horizon_improvement'
    elif safe:
        verdict='observed_continuation_unpaired_future'
    else:
        verdict='inconclusive'
    return {
        'horizon_s':horizon,'common_observed_until_s':common_t,
        'reference_reached_horizon':ref_full,'candidate_reached_horizon':alt_full,
        'reference_terminated':ref.terminated,'candidate_terminated':alt.terminated,
        'reference_hits':ref.hits,'candidate_hits':alt.hits,
        'reference_misses':ref.misses,'candidate_misses':alt.misses,
        'reference_too_early':ref.too_early,'candidate_too_early':alt.too_early,
        'reference_overloaded':ref.overloaded,'candidate_overloaded':alt.overloaded,
        'reference_keydowns':len(ref.keydowns),'candidate_keydowns':len(alt.keydowns),
        'hit_ids_lost_common':lost_common,'recovered_lagging_hit_ids':recovered_lagging,
        'unrecovered_lagging_hit_ids':unrecovered_lagging,
        'hit_ids_lost_by_horizon':lost_final,
        'hit_ids_gained_by_horizon':gained_final,
        'extension_hit_ids_unpaired':new_hit_ids,
        'extension_miss_ids_unpaired':new_miss_ids,
        'extension_keydowns_unpaired':extra_keydowns,
        'full_misses_not_increased':full_miss_ok,
        'too_early_reduced':early_lower,'physical_hit_and_keydown':physical,
        'overload_rescued':rescue,'safe_observed_improvement':safe,
        'conditional_review':conditional,'verdict':verdict,
    }


def _load_inputs(v12_dir:Path,v13_dir:Path,source_checkpoint:Path):
    v12_dir=v12_dir.resolve();v13_dir=v13_dir.resolve()
    files={
        'v13_diagnostic':v13_dir/'diagnostic.json',
        'v13_checks':v13_dir/'reclassification_checks.json',
        'v13_candidates':v13_dir/'reclassified_candidates.csv',
        'v13_qualified':v13_dir/'qualified_corrections.csv',
        'v13_conditional':v13_dir/'conditional_review.csv',
        'v12_candidates':v12_dir/'candidate_comparison.csv',
        'v12_baselines':v12_dir/'baseline_800ms.csv',
        'v12_diagnostic':v12_dir/'diagnostic.json',
        'v12_reproduction':v12_dir/'reproduction_checks.json',
    }
    for label,file in files.items():
        if not file.is_file():
            raise FileNotFoundError(f'Fail Closed: missing {label}: {file}')
    hashes=_hash_inputs(files)
    v13_report=json.loads(files['v13_diagnostic'].read_text(encoding='utf-8'))
    v13_checks=json.loads(files['v13_checks'].read_text(encoding='utf-8'))
    v12_report=json.loads(files['v12_diagnostic'].read_text(encoding='utf-8'))
    reproduction=json.loads(files['v12_reproduction'].read_text(encoding='utf-8'))
    if (v13_report.get('kind')!='train_counterfactual_motor_v13_offline_reclassification'
            or v13_report.get('input_candidates')!=152
            or v13_report.get('source_checkpoint_sha256')!=EXPECTED_BEST_SHA256
            or v13_checks.get('status')!='PASS'
            or v13_report.get('v12_reproduction_status')!='PASS'
            or v13_checks.get('checked_candidates')!=152
            or v13_checks.get('reproduced_v11_count')!=151
            or v13_report.get('original_v11_candidates')!=151
            or v13_report.get('explicit_v1_train20_candidates')!=1
            or v13_report.get('validation_evaluations')!=0
            or v13_report.get('optimizer_updates')!=0
            or v13_report.get('final_used') is not False
            or v13_report.get('teacher_adoptions')!=0
            or v12_report.get('source_checkpoint_sha256')!=EXPECTED_BEST_SHA256
            or v12_report.get('v11_candidates_reproduced')!=151
            or reproduction.get('status')!='PASS'
            or len(reproduction.get('candidates',[]))!=152):
        raise RuntimeError('Fail Closed: prior provenance/reproduction inconsistent')
    if (v13_report.get('v12_input_sha256')!=v13_checks.get('v12_input_sha256') or
            set(v13_checks.get('v12_input_sha256',{}))!=
            {'diagnostic.json','candidate_comparison.csv',
             'baseline_800ms.csv','reproduction_checks.json'}):
        raise RuntimeError('Fail Closed: v1.3 recorded v1.2 hashes incomplete')
    for k,v in v13_checks['v12_input_sha256'].items():
        # Those keys are original v1.2 filenames, not the v1.3 output names.
        p=v12_dir/k
        if not p.is_file() or file_hash(p)!=v:
            raise RuntimeError(f'Fail Closed: v1.2 input hash mismatch: {k}')
    if v13_checks.get('source_checkpoint_sha256')!=EXPECTED_BEST_SHA256:
        raise RuntimeError('Fail Closed: v1.3 source SHA256 mismatch')
    if file_hash(source_checkpoint)!=EXPECTED_BEST_SHA256:
        raise RuntimeError('Fail Closed: Best checkpoint SHA256 mismatch')
    r13=_read_csv(files['v13_candidates']);r12=_read_csv(files['v12_candidates'])
    q13=_read_csv(files['v13_qualified']);c13=_read_csv(files['v13_conditional'])
    if len(r13)!=152 or len(r12)!=152:
        raise RuntimeError('Fail Closed: expected 152 original rows')
    # Re-run v1.3's strict v1.2 integrity checks on the original stored facts.
    original_baselines=_read_csv(files['v12_baselines'])
    checked_baselines,checked_candidates=validate_inputs(
        v12_report,reproduction,r12,original_baselines)
    if len(checked_baselines)!=20 or len(checked_candidates)!=152:
        raise RuntimeError('Fail Closed: v1.2 Train replay consistency incomplete')
    # Exact unmodified physical measurements must still match original v1.2.
    keys=('anchor','keys','start_frame','duration_frames','delta')
    by_original={tuple(row[k] for k in keys):row for row in r12}
    if len(by_original)!=152:
        raise RuntimeError('Fail Closed: duplicate original intervention')
    for row in r13:
        old=by_original.get(tuple(row[k] for k in keys))
        if old is None or any(row[k]!=old[k] for k in MEASUREMENT_FIELDS):
            raise RuntimeError('Fail Closed: v1.3 measurement differs from v1.2')
    # Qualified and conditional files must be subsets of identical physical rows.
    for subset,flag in ((q13,'safe_review_candidate'),(c13,'conditional_review_candidate')):
        for row in subset:
            original=next((r for r in r13 if all(r[k]==row[k] for k in keys)),None)
            if original is None or not bool_value(original[flag],flag):
                raise RuntimeError(f'Fail Closed: invalid {flag} subset')
    baseline_rows=original_baselines
    if len(baseline_rows)!=20:
        raise RuntimeError('Fail Closed: Train20 baseline missing')
    baselines={}
    for row in baseline_rows:
        a=integer(row['anchor'],'baseline anchor')
        if a in baselines or not 1<=a<=20:
            raise RuntimeError('Fail Closed: duplicate/invalid Train baseline anchor')
        baselines[a]=snapshot_from_dict(parsed_json(row['student_800'],'student_800',dict),f'Train #{a}')
    saved_check={}
    for row in reproduction['candidates']:
        key=(int(row['anchor']),tuple(row['case'][1]),int(row['case'][2]),int(row['case'][3]),round(float(row['case'][4]),8))
        if key in saved_check:
            raise RuntimeError('Fail Closed: duplicate reproduction checks')
        saved_check[key]=snapshot_from_dict(row['800ms'],f'reproduced #{key[0]}')
    return files,hashes,r13,baselines,saved_check


def _check_output_dir(output:Path,inputs:list[Path],best:Path):
    output=output.resolve()
    if output.exists():
        raise FileExistsError(f'v1.4 output directory already exists: {output}')
    for parent in inputs:
        p=parent.resolve()
        if output==p or p in output.parents or output in p.parents:
            raise ValueError('v1.4 output overlaps read-only inputs')
    if output==best.resolve() or output in best.resolve().parents:
        raise ValueError('v1.4 output overlaps Best')


def _visible_neighbors(base:dict[int,Trace],selected:list[tuple],*,top:int=3)->list[dict]:
    """Observed visible 263D plus 8D action; Train-only normalization."""
    items=[(a,i,torch.tensor(vector,dtype=torch.float64))
           for a,trace in sorted(base.items()) for i,vector in enumerate(trace.observable)]
    if len(items)<21:
        return []
    x=torch.stack([vector for _,_,vector in items])
    mean=x.mean(0);std=x.std(0,unbiased=False)
    mask=std>1e-5
    if not bool(mask.any()):
        return []
    z=(x[:,mask]-mean[mask])/std[mask]
    mapping={(anchor,step):idx for idx,(anchor,step,_) in enumerate(items)}
    result=[]
    for case,row,role in selected:
        query=mapping.get((case.anchor,case.start))
        if query is None:
            result.append({'case':case.identity(),'source':role,'status':'onset_state_not_observed'})
            continue
        distances=((z-z[query]).square().mean(dim=1)).sqrt()
        matches=sorted((float(distances[j]),a,step) for j,(a,step,_) in enumerate(items)
                       if a!=case.anchor)
        result.append({'case':case.identity(),'source':role,'status':'Train_only_observable_similarity',
                       'metric':'RMS standardized visible_263d + Student_action_8d',
                       'dimensions_used':int(mask.sum()),
                       'nearest_other_train_states':[
                           {'anchor':a,'frame':i,'distance':d,
                            'onset_overloaded_by_800ms':base[a].at[.8].overloaded}
                           for d,a,i in matches[:top]],
                       'transferred_intervention_evaluated':False,
                       'state_dependent_rule_learned':False})
    return result


def run_v14(prepared,*,v12_dir:Path,v13_dir:Path,output:Path)->dict:
    """Full Train-only diagnostic; no optimizer, DAgger, Validation or FINAL."""
    if (len(prepared.anchors)!=20 or prepared.key_count!=8 or prepared.input_dim!=263
            or abs(prepared.control_dt_s-.010)>1e-12
            or abs(prepared.physics_dt_s-.001)>1e-12):
        raise ValueError('requires 20 Train, 263D, 8-key, 10ms/1ms motor')
    _check_output_dir(output,[v12_dir,v13_dir],prepared.source_checkpoint)
    files,hashes,rows,expected_base,expected_checks=_load_inputs(
        v12_dir,v13_dir,prepared.source_checkpoint)
    from dmdod.motor.n_key import n_key_names
    from dmdod.training.real_chart import clone_model_state
    from dmdod.training.fixed_state_action_audit import assert_state_equal
    names=n_key_names(prepared.key_count)
    selected=select_cases(rows,names)
    original=clone_model_state(prepared.model)
    before_hash=file_hash(prepared.source_checkpoint)
    all_results=[]
    robustness=[]
    try:
        baselines={}
        for index,named in enumerate(prepared.anchors,1):
            t=run_long_trace(prepared,named,case=None,capture_visible=True)
            assert_reproduced(t.at[.8],expected_base[index],label=f'Train #{index} baseline')
            baselines[index]=t
            print(f'v1.4 baseline Train #{index:02d}/20: '
                  f'{t.at[.8].hits}H @ {t.at[.8].time_s:.3f}s',flush=True)
        # Every representative must exactly match its saved 800ms score and timing.
        for idx,(case,row,role) in enumerate(selected,1):
            named=prepared.anchors[case.anchor-1]
            tr=run_long_trace(prepared,named,case=case)
            identity=(case.anchor,case.keys,case.start,case.frames,round(case.delta,8))
            old=expected_checks.get(identity)
            if old is None:
                raise RuntimeError('Fail Closed: saved 800ms candidate reproduction missing')
            assert_reproduced(tr.at[.8],old,label=f'{role} #{case.anchor}')
            horizons={str(h):compare_horizon(baselines[case.anchor],tr,h) for h in HORIZONS}
            late_trend={}
            for h in (1.6,3.2):
                earlier=tr.at[.8];current=tr.at[h]
                late_trend[str(h)]={
                    'retained_800ms_hit_ids':set(earlier.hit_ids).issubset(current.hit_ids),
                    'new_hits_since_800':sorted(set(current.hit_ids)-set(earlier.hit_ids)),
                    'new_misses_since_800':sorted(set(current.miss_ids)-set(earlier.miss_ids)),
                    'new_too_early_since_800':current.too_early-earlier.too_early,
                    'new_keydowns_since_800':len(current.keydowns)-len(earlier.keydowns),
                    'episode_observed_until_s':current.time_s,
                    'overloaded':current.overloaded,
                }
            all_results.append({'role':role,'case':asdict(case),
                                'late_continuation':late_trend,
                                'v13_category':row['new_category'],
                                'v13_outcome_group_signature':row['outcome_group_signature'],
                                'v13_outcome_signature':row['outcome_signature'],
                                'reproduced_800ms':True,
                                'base_snapshots':{str(h):_termination(baselines[case.anchor].at[h],h)
                                                  for h in HORIZONS},
                                'candidate_snapshots':{str(h):_termination(tr.at[h],h)
                                                       for h in HORIZONS},
                                'horizon_comparisons':horizons})
            print(f'v1.4 representative {idx}/{len(selected)}: Train #{case.anchor:02d} '
                  f'{case.keys} {case.delta:+.2f} => 1600={horizons["1.6"]["verdict"]} '
                  f'3200={horizons["3.2"]["verdict"]}',flush=True)
            if role not in ('representative','conditional_08'):
                continue
            neighbors=local_neighbors(case)
            trials=[]
            for alt in neighbors:
                trial=run_long_trace(prepared,named,case=alt)
                h8=compare_horizon(baselines[case.anchor],trial,.8)
                h16=compare_horizon(baselines[case.anchor],trial,1.6)
                trials.append({'case':asdict(alt),'at_800ms':h8,'at_1600ms':h16,
                               'observed_to_s':trial.at[1.6].time_s,
                               'relative_to_representative':{
                                   'new_800ms_hit_ids':sorted(set(tr.at[.8].hit_ids)^set(trial.at[.8].hit_ids)),
                                   'new_800ms_miss_ids':sorted(set(tr.at[.8].miss_ids)^set(trial.at[.8].miss_ids)),
                               }})
            n_ok=sum(x['at_800ms']['safe_observed_improvement'] for x in trials)
            n_maintained=sum(x['at_800ms']['safe_observed_improvement']
                             and x['at_1600ms']['safe_observed_improvement'] for x in trials)
            verdict=('locally_robust' if trials and n_ok>=math.ceil(len(trials)/2)
                     and n_maintained>=math.ceil(len(trials)/2)
                     else 'fragile' if trials and n_ok==0 else 'partially_robust_or_inconclusive')
            robustness.append({'role':role,'anchor':case.anchor,'case':asdict(case),
                               'neighbor_delta_step':NEIGHBOR_DELTA,
                               'negative_start_excluded':case.start==0,
                               'neighbors_tested':len(trials),
                               'neighbors_success_800ms':n_ok,
                               'success_rate_800ms':n_ok/len(trials) if trials else None,
                               'neighbors_maintained_1600ms':n_maintained,
                               'retention_rate_1600ms':n_maintained/len(trials) if trials else None,
                               'robustness':verdict,'trials':trials})
        # Stage C does not intervene on other charts: similarity only.
        generalization=_visible_neighbors(baselines,selected)
        original_best_unchanged=True
        assert_state_equal(clone_model_state(prepared.model),original,
                           context='Train v1.4 Student unchanged')
        if file_hash(prepared.source_checkpoint)!=before_hash:
            raise RuntimeError('Fail Closed: Best checkpoint modified')
        if _hash_inputs(files)!=hashes:
            raise RuntimeError('Fail Closed: prior diagnostics modified')
        overview={
            'long_term_by_horizon':{str(h):{
                'candidate_reached_horizon':sum(r['horizon_comparisons'][str(h)]['candidate_reached_horizon'] for r in all_results),
                'baseline_reached_horizon':sum(r['horizon_comparisons'][str(h)]['reference_reached_horizon'] for r in all_results),
                'observed_safe_improvement':sum(r['horizon_comparisons'][str(h)]['safe_observed_improvement'] for r in all_results),
                'conditional_review':sum(r['horizon_comparisons'][str(h)]['conditional_review'] for r in all_results),
                'candidate_overloaded':sum(r['horizon_comparisons'][str(h)]['candidate_overloaded'] for r in all_results),
                'verdict_counts':dict(Counter(r['horizon_comparisons'][str(h)]['verdict'] for r in all_results)),
            } for h in HORIZONS},
            'robustness_counts':dict(Counter(r['robustness'] for r in robustness)),
            'other_train_state_similarity_only':True,
            'transfer_improvement_not_established':True,
        }
        summary={
            'kind':'train_counterfactual_motor_v14',
            'best_sha256':before_hash,'input_hashes':hashes,
            'train_anchors':20,'validation_evaluations':0,'final_used':False,
            'optimizer_updates':0,'teacher_adoptions':0,'checkpoint_writes':0,
            'horizons_s':list(HORIZONS),'physical_reproduction_status':'PASS',
            'representative_count':sum(r['role']=='representative' for r in all_results),
            'conditional_08_count':sum(r['role']=='conditional_08' for r in all_results),
            'same_outcome_control_count':sum(r['role']=='same_outcome_control' for r in all_results),
            'overview':overview,
            'stage_a':all_results,'stage_b':robustness,
            'stage_c':generalization,'original_student_state_unchanged':original_best_unchanged,
            'notes':[
                'Train only, no optimizer/model/teacher updates.',
                'All baselines and selected candidates fail closed if saved 800ms trace disagrees.',
                'Early-ending baseline future is never imputed as a failure.',
                '1600/3200ms conclusions require observable comparison; unpaired continuation is explicitly labelled.',
                'Fragile/locally robust are descriptive local probes, not training decisions.',
                'Stage C uses only visible 263D and Student actions from Train, no cross-chart intervention.',
            ],
        }
        # Only after all Train replay checks PASS and input hashes remain stable.
        output.mkdir(parents=True,exist_ok=False)
        try:
            (output/'diagnostic.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            csv_write_new(output/'baseline_continuation.csv',[
                {'anchor':a,'chart':prepared.anchors[a-1].chart_name,
                 'horizon_s':h,**_termination(tr.at[h],h)}
                for a,tr in sorted(baselines.items()) for h in HORIZONS])
            csv_write_new(output/'state_similarity.csv',[
                {'source_case':json.dumps(r['case']),
                 'status':r['status'],'neighbor':json.dumps(v,ensure_ascii=False),
                 'metric':r.get('metric'),'dimensions_used':r.get('dimensions_used')}
                for r in generalization for v in r.get('nearest_other_train_states',[{}])
            ], columns=('source_case','status','neighbor','metric','dimensions_used'))
            csv_write_new(output/'long_horizon_comparison.csv',[
                {'role':x['role'],**x['case'],'horizon_s':float(h),
                 **{k:v for k,v in result.items() if k not in
                    ('hit_ids_lost_common','recovered_lagging_hit_ids','unrecovered_lagging_hit_ids',
                     'hit_ids_lost_by_horizon','hit_ids_gained_by_horizon',
                     'extension_hit_ids_unpaired','extension_miss_ids_unpaired')},
                 'hit_ids_lost_common':json.dumps(result['hit_ids_lost_common']),
                 'recovered_lagging_hit_ids':json.dumps(result['recovered_lagging_hit_ids']),
                 'unrecovered_lagging_hit_ids':json.dumps(result['unrecovered_lagging_hit_ids']),
                 'hit_ids_lost_by_horizon':json.dumps(result['hit_ids_lost_by_horizon']),
                 'hit_ids_gained_by_horizon':json.dumps(result['hit_ids_gained_by_horizon']),
                 'extension_hit_ids_unpaired':json.dumps(result['extension_hit_ids_unpaired']),
                 'extension_miss_ids_unpaired':json.dumps(result['extension_miss_ids_unpaired'])}
                for x in all_results for h,result in x['horizon_comparisons'].items()])
            csv_write_new(output/'local_robustness.csv',[
                {k:v for k,v in entry.items() if k!='trials'}
                for entry in robustness])
            csv_write_new(output/'local_neighbors.csv',[
                {'role':entry['role'],'anchor':entry['anchor'],
                 'case':entry['case'],'neighbor_case':trial['case'],
                 'at_800ms':trial['at_800ms']['verdict'],
                 'at_1600ms':trial['at_1600ms']['verdict'],
                 'safe_800ms':trial['at_800ms']['safe_observed_improvement'],
                 'safe_1600ms':trial['at_1600ms']['safe_observed_improvement'],
                 'lost_hit_ids_1600':trial['at_1600ms']['hit_ids_lost_by_horizon'],
                 'extension_miss_ids_1600':trial['at_1600ms']['extension_miss_ids_unpaired']}
                for entry in robustness for trial in entry['trials']])
            (output/'reproduction_checks.json').write_text(json.dumps({
                'status':'PASS','best_sha256':before_hash,
                'prior_input_sha256':hashes,'baseline_reproduced':20,
                'candidate_reproduced':len(all_results),
                'no_validation_or_final':True,'optimizer_updates':0,
            },indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            if file_hash(prepared.source_checkpoint)!=before_hash or _hash_inputs(files)!=hashes:
                raise RuntimeError('Fail Closed: input changed during result publishing')
        except Exception:
            (output/'diagnostic.json').unlink(missing_ok=True)
            (output/'FAILED_NO_TEACHER_ADOPTION.txt').write_text(
                'Fail Closed: no training or teacher adoption performed.\n',encoding='utf-8')
            raise
        return summary
    finally:
        prepared.model.load_state_dict(original,strict=True)
        prepared.model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(prepared.model),original,
                           context='v1.4 restores untouched Student')
