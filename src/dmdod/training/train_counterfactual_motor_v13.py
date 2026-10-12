from __future__ import annotations

"""Deterministic, offline reclassification of *recorded* Train v1.2 motor outcomes.

No policy, simulator, optimizer or manifest is imported or executed. Input files
are read-only and output is created exclusively in a previously absent folder.
Teacher-relative gates are retained as context, never applied to suitability.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dmdod.training.train_counterfactual_motor_v12 import (
    EXPECTED_BEST_SHA256, Snapshot, assert_prefix_consistent, csv_write_new,
    outcome_signature as v12_timing_signature,
)

HORIZON = 0.8
TOL = 1e-6
EXPECTED_ORIGINAL = 151
EXPECTED_TOTAL = 152
VALID_OLD_CATEGORIES = {
    'overload_rescue', 'early_reduction_preserved_hits', 'hit_gain',
    'deferred_failure', 'hit_suppression', 'regression', 'inconclusive',
}
SAFE_CATEGORIES = {'hit_gain', 'early_reduction_preserved_hits', 'overload_rescue'}
REVIEW_CATEGORIES = {'overload_rescue_review'}
MEASUREMENT_FIELDS = (
    'anchor', 'source', 'kind', 'keys', 'start_frame', 'duration_frames',
    'delta', 'lead_frames', 'short_hits', 'short_misses', 'short_early',
    'short_overload', 'short_duration_s', 'hits_800', 'misses_800',
    'early_800', 'overload_800', 'overload_time_s', 'duration_s',
    'hit_ids_800', 'miss_ids_800', 'keydowns_800', 'comparison_common',
    'extended_observation', 'outcome_signature',
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, reason: str):
    if not condition:
        raise RuntimeError('Fail Closed: ' + reason)


def parsed_json(value: str, label: str, expected_type: type):
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Fail Closed: invalid JSON in ' + label) from exc
    require(isinstance(result, expected_type), label + ' has wrong JSON type')
    return result


def integer(raw: str, field: str) -> int:
    try:
        value = int(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Fail Closed: invalid integer ' + field) from exc
    return value


def scalar(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Fail Closed: invalid scalar ' + field) from exc
    require(math.isfinite(value), field + ' is not finite')
    return value


def bool_value(raw: str, field: str) -> bool:
    require(raw in ('True', 'False', 'true', 'false', '1', '0'),
            field + ' invalid boolean')
    return raw.lower() in ('true', '1')


def snapshot_from_dict(raw: dict[str, Any], label: str) -> Snapshot:
    require(isinstance(raw, dict), label + ' must be object')
    try:
        hit_ids = tuple(raw['hit_ids'])
        miss_ids = tuple(raw['miss_ids'])
        keys = tuple((k, float(t)) for k, t in raw['keydowns'])
        terminal = raw['terminated']
        overloaded = raw['overloaded']
        if (type(terminal) is not bool or type(overloaded) is not bool
                or any(type(i) is not int or i < 0 for i in (*hit_ids, *miss_ids))):
            raise ValueError('invalid fields')
        result = Snapshot(
            time_s=float(raw['time_s']), hits=int(raw['hits']),
            misses=int(raw['misses']), too_early=int(raw['too_early']),
            overloaded=overloaded,
            overload_time_s=(None if raw['overload_time_s'] is None else
                             float(raw['overload_time_s'])),
            terminated=terminal, termination_reason=str(raw['termination_reason']),
            hit_ids=hit_ids, miss_ids=miss_ids, keydowns=keys,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError('Fail Closed: invalid snapshot ' + label) from exc
    require(0 < result.time_s <= HORIZON + TOL, label + ' invalid clock')
    require(result.hits == len(set(hit_ids)) == len(hit_ids), label + ' Hits/IDs mismatch')
    require(result.misses == len(set(miss_ids)) == len(miss_ids), label + ' Misses/IDs mismatch')
    require(not set(hit_ids).intersection(miss_ids), label + ' Hit/Miss IDs overlap')
    require(result.hits >= 0 and result.misses >= 0 and result.too_early >= 0,
            label + ' negative counts')
    require(all(math.isfinite(t) and 0 <= t <= result.time_s+TOL and isinstance(k, str)
                and bool(k) for k, t in keys), label + ' invalid physical KeyDown')
    require(not result.overloaded or result.overload_time_s is not None,
            label + ' missing overload time')
    require(result.overload_time_s is None or
            0 <= result.overload_time_s <= result.time_s+TOL,
            label + ' invalid overload time')
    require(not result.terminated or result.termination_reason != 'Horizon cutoff',
            label + ' contradictory termination')
    require(result.terminated or result.time_s >= HORIZON-TOL or
            result.termination_reason == 'Horizon cutoff',
            label + ' unexplained short observation')
    return result


def _same(a: float, b: float) -> bool:
    return abs(a-b) <= TOL


def validate_common(common: dict, extension: dict, base: Snapshot, cand: Snapshot,
                    *, label: str) -> None:
    require(isinstance(common, dict) and isinstance(extension, dict),
            label + ' missing common/extension data')
    common_t = float(common['observed_until_s'])
    require(_same(common_t, min(base.time_s, cand.time_s)), label + ' common clock wrong')
    for role, end in (('reference', base), ('candidate', cand)):
        ids = common[f'{role}_hit_ids']
        require(isinstance(ids, list) and len(ids) == len(set(ids)),
                label + ' common Hit IDs duplicated')
        require(set(ids).issubset(end.hit_ids), label + ' common Hits not in final state')
        for metric, final in [('misses', end.misses), ('too_early', end.too_early),
                              ('keydowns', len(end.keydowns))]:
            n = common[f'{role}_{metric}']
            require(type(n) is int and 0 <= n <= final,
                    label + ' common '+metric+' out of bounds')
        if _same(common_t, end.time_s):
            require(set(ids) == set(end.hit_ids), label + ' common full Hits disagreement')
            require(common[f'{role}_misses'] == end.misses and
                    common[f'{role}_too_early'] == end.too_early and
                    common[f'{role}_keydowns'] == len(end.keydowns),
                    label + ' common full score disagreement')
    require(common['hits_preserved'] == (not set(common['reference_hit_ids']) -
                                           set(common['candidate_hit_ids'])),
            label + ' common hits_preserved contradictory')
    require(common['misses_not_increased'] ==
            (common['candidate_misses'] <= common['reference_misses']),
            label + ' common misses condition contradictory')
    require(common['too_early_reduced'] ==
            (common['candidate_too_early'] < common['reference_too_early']),
            label + ' common TooEarly condition contradictory')
    require(sorted(common['lost_hit_ids']) == sorted(set(common['reference_hit_ids']) -
                                                     set(common['candidate_hit_ids'])),
            label + ' common lost IDs contradictory')
    require(sorted(common['gained_hit_ids']) == sorted(set(common['candidate_hit_ids']) -
                                                       set(common['reference_hit_ids'])),
            label + ' common gained IDs contradictory')
    require(common['common_hit_gain'] == len(common['gained_hit_ids']),
            label + ' common gain contradictory')
    require(common['reference_hit_ids'] == sorted(common['reference_hit_ids']) and
            common['candidate_hit_ids'] == sorted(common['candidate_hit_ids']),
            label + ' common Hit IDs not canonical')
    expected_extended = sorted(set(cand.hit_ids) - set(common['candidate_hit_ids']))
    require(sorted(extension['new_hit_ids']) == expected_extended,
            label + ' extension Hit IDs contradictory')
    require(len(extension['new_miss_ids']) == cand.misses - common['candidate_misses'],
            label + ' extension misses contradictory')
    require(extension['new_keydowns'] == len(cand.keydowns) - common['candidate_keydowns'],
            label + ' extension KeyDowns contradictory')
    require(_same(extension['reference_stopped_s'], base.time_s) and
            _same(extension['candidate_observed_until_s'], cand.time_s),
            label + ' extension clocks contradictory')


def classify(base: Snapshot, candidate: Snapshot, common: dict, extension: dict,
             *, teacher: dict | None = None) -> dict:
    """Pure classification from recorded physical facts. Teacher is deliberately unused."""
    del teacher
    missing = set(base.hit_ids) - set(candidate.hit_ids)
    lagging = set(common['reference_hit_ids']) - set(common['candidate_hit_ids'])
    recovered = lagging & set(candidate.hit_ids)
    unrecovered = lagging - set(candidate.hit_ids)
    both_full = (base.time_s >= HORIZON-TOL and candidate.time_s >= HORIZON-TOL)
    no_candidate_overload = not candidate.overloaded
    baseline_preserved = not missing
    common_misses_ok = common['candidate_misses'] <= common['reference_misses']
    full_misses_ok = candidate.misses <= base.misses if both_full else None
    premature = (candidate.time_s + TOL < base.time_s or
                 (candidate.terminated and candidate.time_s < HORIZON-TOL))
    physical = candidate.hits > 0 and len(candidate.keydowns) > 0
    early_less = candidate.too_early < base.too_early if both_full else (
        common['candidate_too_early'] < common['reference_too_early'])
    new_hits = len(set(candidate.hit_ids) - set(base.hit_ids))
    new_downs = extension['new_keydowns']
    extension_misses = len(extension['new_miss_ids'])
    rescued = base.overloaded and base.time_s < HORIZON-TOL and (
        candidate.time_s >= HORIZON-TOL and not candidate.overloaded)
    continuation = extension['new_hit_ids'] and new_downs > 0 and physical
    delayed = (candidate.overloaded and base.overloaded and
               candidate.overload_time_s is not None and
               base.overload_time_s is not None and
               candidate.overload_time_s > base.overload_time_s + .01)
    suppression = bool(missing or (unrecovered and both_full))
    extra_bad = bool((both_full and not full_misses_ok) or
                     (not common_misses_ok) or
                     (candidate.overloaded and not base.overloaded))
    no_press = not physical
    deferred = bool((candidate.overloaded and not base.overloaded) or
                    (candidate.overloaded and candidate.time_s > base.time_s+TOL) or
                    (candidate.too_early >= base.too_early and
                     common['too_early_reduced'] and both_full) or
                    (candidate.time_s > base.time_s+TOL and not continuation and
                     candidate.hits <= base.hits))
    # Explicit priority: new retained Hit targets > early reduction.
    if suppression:
        category = 'hit_suppression'
    elif premature or (both_full and extra_bad):
        category = 'regression'
    elif rescued and baseline_preserved and continuation and common_misses_ok:
        if extension_misses or candidate.misses > common['reference_misses']:
            category = 'overload_rescue_review'
        else:
            category = 'overload_rescue'
    elif rescued:
        category = ('regression' if not common_misses_ok else
                    'deferred_failure' if not continuation else 'inconclusive')
    elif delayed or deferred:
        category = 'deferred_failure'
    elif no_press:
        category = 'insufficient_evidence'
    elif both_full and baseline_preserved and full_misses_ok and no_candidate_overload:
        if new_hits > 0:
            category = 'hit_gain'
        elif early_less:
            category = 'early_reduction_preserved_hits'
        else:
            category = 'inconclusive'
    elif extra_bad:
        category = 'regression'
    else:
        category = 'inconclusive'
    # Do not promote increased Hit counts obtained only by a shorter observation.
    safe = category in SAFE_CATEGORIES
    review = category in REVIEW_CATEGORIES
    reasons = []
    if not baseline_preserved:
        reasons.append('unrecovered_baseline_hit_ids')
    if not common_misses_ok:
        reasons.append('common_interval_misses_increased')
    if both_full and not full_misses_ok:
        reasons.append('final_misses_increased')
    if not both_full and extension_misses:
        reasons.append('extension_misses_not_baseline_comparable')
    if premature:
        reasons.append('shortened_candidate_observation')
    if candidate.overloaded:
        reasons.append('candidate_overload')
    if no_press:
        reasons.append('insufficient_hit_and_keydown_evidence')
    if not safe and not review and not reasons:
        reasons.append('no_verified_student_relative_benefit')
    return {
        'category': category, 'safe_review_candidate': safe,
        'conditional_review_candidate': review,
        'teacher_adopted': False,
        'too_early_reduced': early_less,
        'baseline_hit_ids_preserved_by_800': baseline_preserved,
        'temporarily_lagging_hit_ids': sorted(lagging),
        'recovered_hit_ids_by_800': sorted(recovered),
        'unrecovered_hit_ids_by_800': sorted(unrecovered),
        'new_hit_ids_by_800': sorted(set(candidate.hit_ids)-set(base.hit_ids)),
        'candidate_has_evidence_of_useful_keydown': physical,
        'both_800ms_observed': both_full,
        'common_misses_not_increased': common_misses_ok,
        'both_800ms_misses_not_increased': full_misses_ok,
        'observed_extension_miss_ids': sorted(extension['new_miss_ids']),
        'observed_extension_keydowns': new_downs,
        'overload_rescued': rescued,
        'overload_only_delayed': delayed,
        'no_press_suppression': not no_press,
        'rejection_reasons': reasons,
    }


def outcome_group_signature(anchor: int, cand: Snapshot) -> str:
    """Outcome-only identity: deliberately excludes physical KeyDown timestamps."""
    data = {'anchor': anchor, 'hit_ids': sorted(cand.hit_ids),
            'miss_ids': sorted(cand.miss_ids), 'too_early': cand.too_early,
            'overloaded': cand.overloaded, 'keydown_count': len(cand.keydowns),
            'duration_ms': round(cand.time_s * 1000, 3),
            'termination_reason': cand.termination_reason}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and
                all(x in reader.fieldnames for x in MEASUREMENT_FIELDS),
                'candidate CSV missing required fields')
        rows = list(reader)
    require(all(None not in row for row in rows), 'candidate CSV has excess columns')
    return rows


def _read_inputs(previous: Path):
    paths = {n: previous/n for n in ('diagnostic.json', 'candidate_comparison.csv',
                                     'baseline_800ms.csv', 'reproduction_checks.json')}
    require(all(p.is_file() for p in paths.values()), 'v1.2 input missing')
    hashes = {k: file_hash(p) for k, p in paths.items()}
    diagnostic = json.loads(paths['diagnostic.json'].read_text(encoding='utf-8'))
    reproduced = json.loads(paths['reproduction_checks.json'].read_text(encoding='utf-8'))
    candidates = _load_csv(paths['candidate_comparison.csv'])
    with paths['baseline_800ms.csv'].open(encoding='utf-8-sig', newline='') as stream:
        baseline = list(csv.DictReader(stream))
    return paths, hashes, diagnostic, reproduced, candidates, baseline


def _identity(row: dict[str, str]):
    return (integer(row['anchor'],'anchor'),
            tuple(integer(k, 'key') for k in row['keys'].split(',')),
            integer(row['start_frame'],'start_frame'),
            integer(row['duration_frames'],'duration_frames'),
            scalar(row['delta'],'delta'))


def validate_inputs(diagnostic: dict, reproduced: dict, candidates: list[dict],
                    baselines: list[dict]) -> tuple[dict[int, Snapshot], list[dict]]:
    require(diagnostic.get('kind') == 'train_counterfactual_motor_v12', 'wrong v1.2 kind')
    require(diagnostic.get('source_checkpoint_sha256') == EXPECTED_BEST_SHA256,
            'Best checkpoint SHA256 mismatch')
    require(reproduced.get('source_sha256') == EXPECTED_BEST_SHA256,
            'reproduction checkpoint SHA256 mismatch')
    require(reproduced.get('status') == 'PASS', 'v1.2 reproduction not PASS')
    require(diagnostic.get('validation_evaluations') == 0 and
            diagnostic.get('final_used') is False and
            diagnostic.get('optimizer_updates') == 0 and
            diagnostic.get('checkpoint_writes') == 0 and
            diagnostic.get('teacher_adoptions') == 0,
            'v1.2 violated diagnostic-only contract')
    require(diagnostic.get('v11_candidates_reproduced') == EXPECTED_ORIGINAL and
            diagnostic.get('v11_candidates_selected') == EXPECTED_ORIGINAL and
            diagnostic.get('v11_reproduction_rate') == 1.0 and
            diagnostic.get('v1_train20_explicit_replay') is True,
            '151-candidate reproduction or explicit #20 replay incomplete')
    require(diagnostic.get('source_diagnostic_hashes') ==
            reproduced.get('v11_input_sha256'), 'v1.1 provenance hashes differ')
    require(len(candidates) == EXPECTED_TOTAL and len(reproduced.get('candidates', [])) == EXPECTED_TOTAL,
            'expected exactly 152 candidates and checks')
    require(len(baselines) == 20, 'expected 20 Train baselines')
    require(sum(row.get('source') == 'v11_short_pass' for row in candidates) == EXPECTED_ORIGINAL,
            '151 original v1.1 candidates missing')
    require(sum(row.get('source') == 'v1_manual_unsampled' for row in candidates) == 1,
            'missing one independent Train #20 extra')
    require(candidates[-1]['source'] == 'v1_manual_unsampled',
            'v1 manual candidate must be distinct final row')
    require(_identity(candidates[-1]) == (20, (6,), 0, 4, -.2),
            'Train #20 explicit correction differs from historical intervention')
    baseline_map = {}
    for position, row in enumerate(baselines, 1):
        require(integer(row['anchor'], 'baseline.anchor') == position,
                'Train baseline order/anchor mismatch')
        short = snapshot_from_dict(parsed_json(row['student_400'], 'student_400', dict),
                                   f'baseline #{position} at 400ms')
        long = snapshot_from_dict(parsed_json(row['student_800'], 'student_800', dict),
                                  f'baseline #{position} at 800ms')
        assert_prefix_consistent(short, long)
        require(long.keydowns[:len(short.keydowns)] == short.keydowns,
                'baseline KeyDown prefix mismatch')
        require(long.time_s >= short.time_s - TOL, 'baseline time reversed')
        baseline_map[position] = long
        # Teacher is diagnostic only but source must be internally consistent.
        teacher_short = snapshot_from_dict(parsed_json(row['teacher_400'],'teacher_400',dict),
                                           'teacher short')
        teacher_long = snapshot_from_dict(parsed_json(row['teacher_800'],'teacher_800',dict),
                                          'teacher long')
        assert_prefix_consistent(teacher_short, teacher_long)
    identities=set()
    checked=[]
    for i, (row, check) in enumerate(zip(candidates, reproduced['candidates']), 1):
        label=f'candidate {i}'
        identity=_identity(row)
        require(identity not in identities, label + ' duplicate intervention')
        identities.add(identity)
        require(1<=identity[0]<=20 and len(identity[1]) in (1,2,8) and
                all(0<=key<8 for key in identity[1]),label+' invalid key/anchor')
        require(identity == tuple([check['case'][0],tuple(check['case'][1]),
                                   *check['case'][2:]]),label+' case/check identity differs')
        require(integer(row['anchor'],'anchor') == check['anchor'] and
                row['source'] == check['source'],label+' source/check mismatch')
        require(check['reproduced'] is (row['source']=='v11_short_pass'),
                label+' reproduction flag mismatch')
        short = snapshot_from_dict(check['400ms'],label+' 400ms')
        long = snapshot_from_dict(check['800ms'],label+' 800ms')
        assert_prefix_consistent(short,long)
        require(long.keydowns[:len(short.keydowns)] == short.keydowns,
                label + ' 400ms/800ms physical KeyDown prefix mismatch')
        for field,measured in [
            ('short_hits',short.hits),('short_misses',short.misses),
            ('short_early',short.too_early),('hits_800',long.hits),
            ('misses_800',long.misses),('early_800',long.too_early),
        ]:
            require(integer(row[field],field)==measured,label+' '+field+' mismatch')
        for field,measured in [('short_overload',short.overloaded),
                               ('overload_800',long.overloaded)]:
            require(bool_value(row[field],field)==measured,label+' '+field+' mismatch')
        require(_same(scalar(row['short_duration_s'],'short_duration_s'),short.time_s)
                and _same(scalar(row['duration_s'],'duration_s'),long.time_s),
                label+' duration mismatch')
        overloaded_time = row['overload_time_s']
        require((overloaded_time in ('',None) and long.overload_time_s is None) or
                (overloaded_time not in ('',None) and long.overload_time_s is not None and
                 _same(scalar(overloaded_time,'overload_time_s'),long.overload_time_s)),
                label+' overload time mismatch')
        require(parsed_json(row['hit_ids_800'],'hit_ids_800',list)==list(long.hit_ids)
                and parsed_json(row['miss_ids_800'],'miss_ids_800',list)==list(long.miss_ids),
                label+' final target IDs mismatch')
        saved_downs=parsed_json(row['keydowns_800'],'keydowns_800',list)
        require(saved_downs==[[k,t] for k,t in long.keydowns],
                label+' final physical KeyDown timing mismatch')
        common=parsed_json(row['comparison_common'],'comparison_common',dict)
        extended=parsed_json(row['extended_observation'],'extended_observation',dict)
        validate_common(common,extended,baseline_map[identity[0]],long,label=label)
        require(row['outcome_signature'] == v12_timing_signature(identity[0], long),
                label+' original timing signature mismatches physical trace')
        require(row['category'] in VALID_OLD_CATEGORIES,label+' invalid original category')
        checked.append({'raw':row,'check':check,'short':short,'long':long,
                        'common':common,'extension':extended})
    require(sum(x['check']['reproduced'] for x in checked)==151,
            '151 stored reproduction checks not PASS')
    focus = diagnostic.get('focus_anchors')
    require(isinstance(focus,list) and [x.get('anchor') for x in focus]==[6,8,18,20],
            'v1.2 focus-anchor JSON missing or reordered')
    for group in focus:
        anchor=group['anchor']
        reference=group['baseline']
        require(reference['student_800']==baseline_map[anchor].as_json(),
                f'v1.2 focus #{anchor} baseline differs from CSV')
        candidates_on_anchor=[x for x in checked if int(x['raw']['anchor'])==anchor]
        cases=group['cases']
        require(len(cases)==len(candidates_on_anchor),
                f'v1.2 focus #{anchor} case count differs')
        for saved,record in zip(cases,candidates_on_anchor):
            raw=record['raw']
            require(saved['outcome_signature']==raw['outcome_signature'] and
                    saved['hit_ids_800']==list(record['long'].hit_ids) and
                    saved['miss_ids_800']==list(record['long'].miss_ids) and
                    saved['keydowns_800']==[list(d) for d in record['long'].keydowns] and
                    saved['hits_800']==record['long'].hits and
                    saved['early_800']==record['long'].too_early,
                    f'v1.2 focus #{anchor} candidate measurement differs from CSV')
    return baseline_map, checked


def run_v13(previous: Path, output: Path) -> dict:
    """Strict offline pass. No checkpoint loaded or updated, no simulation run."""
    previous = previous.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    if output == previous or previous in output.parents or output in previous.parents:
        raise ValueError('v1.3 output cannot overlap v1.2 input')
    paths, hashes, prior, reproduced, candidates, baselines = _read_inputs(previous)
    baseline_map, validated = validate_inputs(prior,reproduced,candidates,baselines)
    old_counter=Counter()
    current_counter=Counter()
    candidates_out=[]
    qualified=[]
    review=[]
    patterns=defaultdict(list)
    for entry in validated:
        row=entry['raw']
        result=classify(baseline_map[int(row['anchor'])],entry['long'],
                        entry['common'],entry['extension'],
                        teacher=parsed_json(row['teacher_reference'],'teacher_reference',dict))
        signature=outcome_group_signature(int(row['anchor']),entry['long'])
        new={**row, 'old_category':row['category'], 'new_category':result['category'],
             'safe_review_candidate':result['safe_review_candidate'],
             'conditional_review_candidate':result['conditional_review_candidate'],
             'too_early_reduced':result['too_early_reduced'],
             'reclassification':result,
             'temporarily_lagging_hit_ids':result['temporarily_lagging_hit_ids'],
             'recovered_hit_ids_by_800':result['recovered_hit_ids_by_800'],
             'unrecovered_hit_ids_by_800':result['unrecovered_hit_ids_by_800'],
             'outcome_group_signature':signature,
             'physical_measurements_preserved':True}
        for field in MEASUREMENT_FIELDS:
            require(new[field]==row[field],f'physical metric mutated: {field}')
        candidates_out.append(new)
        old_counter[row['category']]+=1
        current_counter[result['category']]+=1
        if result['safe_review_candidate']:
            qualified.append(new)
        if result['conditional_review_candidate']:
            review.append(new)
        patterns[(row['anchor'],signature)].append(new)
    require(len(candidates_out)==EXPECTED_TOTAL,'candidate count changed during classification')
    # Original #20 explicitly excluded from the 151-case improvement count.
    v11=[r for r in candidates_out if r['source']=='v11_short_pass']
    extra=[r for r in candidates_out if r['source']=='v1_manual_unsampled']
    unique_success=set((r['anchor'],r['outcome_group_signature']) for r in qualified)
    unique_v11_success={(r['anchor'],r['outcome_group_signature']) for r in v11
                        if r['safe_review_candidate']}
    comparison_detail={str(anchor):[
        {k:r[k] for k in ('keys','delta','start_frame','duration_frames',
                          'old_category','new_category','too_early_reduced',
                          'temporarily_lagging_hit_ids','recovered_hit_ids_by_800',
                          'unrecovered_hit_ids_by_800','safe_review_candidate',
                          'conditional_review_candidate','hits_800','misses_800',
                          'early_800','duration_s','outcome_signature',
                          'outcome_group_signature')}
        for r in candidates_out if int(r['anchor'])==anchor]
        for anchor in (6,8,18,20)}
    diagnostic={
        'kind':'train_counterfactual_motor_v13_offline_reclassification',
        'source_checkpoint_sha256':EXPECTED_BEST_SHA256,
        'v12_input_sha256':hashes,
        'v12_reproduction_status':'PASS',
        'input_candidates':len(candidates_out),'original_v11_candidates':len(v11),
        'explicit_v1_train20_candidates':len(extra),
        'safe_improvement_original_151':sum(r['safe_review_candidate'] for r in v11),
        'conditional_review_original_151':sum(r['conditional_review_candidate'] for r in v11),
        'qualified_all_including_extra':len(qualified),
        'qualified_train_anchors_original_151':sorted({int(r['anchor']) for r in v11
                                                    if r['safe_review_candidate']}),
        'unique_qualified_outcome_patterns':len(unique_success),
        'unique_qualified_outcome_patterns_original_151':len(unique_v11_success),
        'duplicate_qualified_candidates':len(qualified)-len(unique_success),
        'duplicate_qualified_candidates_original_151':(
            sum(r['safe_review_candidate'] for r in v11)-len(unique_v11_success)
        ),
        'category_counts_original_151':dict(Counter(r['new_category'] for r in v11)),
        'previous_category_counts_original_151':dict(Counter(r['old_category'] for r in v11)),
        'old_to_new_category_transitions_original_151':dict(Counter(
            f"{r['old_category']} -> {r['new_category']}" for r in v11
        )),
        'overload_rescue_original_151':sum(r['new_category']=='overload_rescue' for r in v11),
        'overload_rescue_review_original_151':sum(r['new_category']=='overload_rescue_review' for r in v11),
        'hit_gain_original_151':sum(r['new_category']=='hit_gain' for r in v11),
        'preserved_hit_early_reduction_original_151':sum(
            r['new_category']=='early_reduction_preserved_hits' for r in v11),
        'hit_suppression_original_151':sum(r['new_category']=='hit_suppression' for r in v11),
        'deferred_failure_original_151':sum(r['new_category']=='deferred_failure' for r in v11),
        'only_teacher_gate_prior_rejection_but_now_qualified':sum(
            r['safe_review_candidate'] and bool_value(r['old_teacher_only_rejection'],
                'old_teacher_only_rejection') for r in v11),
        'category_counts_all_152':dict(current_counter),
        'old_category_counts_all_152':dict(old_counter),
        'rejection_reason_counts':dict(Counter(reason for r in candidates_out
            for reason in r['reclassification']['rejection_reasons'])),
        'focus_anchors':comparison_detail,
        'focus_baselines':{str(anchor):baseline_map[anchor].as_json()
                           for anchor in (6,8,18,20)},
        'validation_evaluations':0,'final_used':False,
        'optimizer_updates':0,'checkpoint_writes':0,'teacher_adoptions':0,
        'notes':['Only recorded v1.2 measurements are used; no GPU replay.',
                 'Safe means eligible for manual review, NOT teacher adoption.',
                 'Early baseline termination makes later Misses incomparable.',
                 'Outcomes are grouped without motor contact timestamps.',
                 'Teacher threshold does not determine Student-relative classification.'],
    }
    outcome_groups=[]
    for (anchor,signature),group in sorted(patterns.items()):
        first=group[0]
        outcome_groups.append({'anchor':anchor,'outcome_group_signature':signature,
            'candidates':len(group),'safe_review_candidates':sum(r['safe_review_candidate'] for r in group),
            'conditional_review_candidates':sum(r['conditional_review_candidate'] for r in group),
            'category_breakdown':dict(Counter(r['new_category'] for r in group)),
            'hits_800':first['hits_800'],'misses_800':first['misses_800'],
            'early_800':first['early_800'],'overload_800':first['overload_800'],
            'keydown_count':len(parsed_json(first['keydowns_800'],'keydowns_800',list))})
    # Source file hashes checked again before publishing any output.
    require({key:file_hash(p) for key,p in paths.items()}==hashes,
            'source diagnostics changed while reading')
    output.mkdir(parents=True,exist_ok=False)
    try:
        columns=tuple(candidates_out[0].keys())
        csv_write_new(output/'reclassified_candidates.csv',candidates_out,columns=columns)
        csv_write_new(output/'qualified_corrections.csv',qualified,columns=columns)
        csv_write_new(output/'conditional_review.csv',review,columns=columns)
        csv_write_new(output/'outcome_groups.csv',outcome_groups,
                      columns=tuple(outcome_groups[0].keys()) if outcome_groups else
                      ('anchor','outcome_group_signature','candidates'))
        checks={'status':'PASS','v12_input_sha256':hashes,
                'source_checkpoint_sha256':EXPECTED_BEST_SHA256,
                'reproduced_v11_count':EXPECTED_ORIGINAL,
                'checked_candidates':len(validated),
                'preserved_fields':list(MEASUREMENT_FIELDS),
                'input_files_unchanged':True,
                'no_gpu_or_validation_used':True}
        (output/'reclassification_checks.json').write_text(
            json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (output/'diagnostic.json').write_text(
            json.dumps(diagnostic,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        require({key:file_hash(p) for key,p in paths.items()}==hashes,
                'source files changed while publishing')
    except Exception:
        # Never leave a plausible success report after failed publication.
        (output/'FAILED_NO_QUALIFICATION.txt').write_text('Fail Closed: no teacher adoption.\n',encoding='utf-8')
        (output/'diagnostic.json').unlink(missing_ok=True)
        (output/'qualified_corrections.csv').unlink(missing_ok=True)
        (output/'conditional_review.csv').unlink(missing_ok=True)
        raise
    return diagnostic
