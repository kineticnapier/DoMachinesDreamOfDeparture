from __future__ import annotations

"""Train-only v1.5: persistence, non-monotonic boundaries and rescue quality.

No training, privileged Student inputs, Validation/FINAL evaluation, optimizer,
or checkpoint writes. Pre-existing v1.2-v1.4 diagnostics are strictly read-only.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from dmdod.training.train_counterfactual_motor_v12 import (
    EXPECTED_BEST_SHA256, Snapshot, _environment_snapshot,
    _assert_snapshot_consistent, csv_write_new,
)
from dmdod.training.train_counterfactual_motor_v13 import file_hash
from dmdod.training.train_counterfactual_motor_v14 import (
    HORIZONS, TOL, Intervention, Trace, _hash_inputs, _load_inputs, _read_csv,
    assert_reproduced, compare_horizon, local_neighbors, select_cases,
)

GRID_06 = (-.24, -.23, -.22, -.21, -.20, -.19, -.18, -.17, -.16)
GRID_18 = (-.22, -.20, -.18)
RESCUE_ANCHORS = (8, 20)


@dataclass(frozen=True)
class FrameContext:
    time_s: float  # end of executed 10ms step
    visible_hud: tuple[float, ...]  # 263D, before step; no evaluator target time
    student_action: tuple[float, ...]  # raw 8D, before intervention
    executed_action: tuple[float, ...]  # 8D sent to motor
    new_keydowns: tuple[tuple[str, float], ...]


@dataclass
class DetailedReplay:
    trace: Trace
    context: tuple[FrameContext, ...]


def _required_layout(prepared, output: Path, inputs: tuple[Path, ...]):
    if (len(prepared.anchors) != 20 or prepared.key_count != 8
            or prepared.input_dim != 263
            or abs(prepared.control_dt_s - .010) > 1e-12
            or abs(prepared.physics_dt_s - .001) > 1e-12):
        raise ValueError('v1.5 requires Train20, 8-Key, 263D, control 10ms / physics 1ms')
    if output.exists():
        raise FileExistsError(f'v1.5 output path already exists: {output}')
    resolved = output.resolve()
    for directory in inputs:
        src = directory.resolve()
        if resolved == src or src in resolved.parents or resolved in src.parents:
            raise ValueError('v1.5 output overlaps a read-only prior diagnostic')
    best = prepared.source_checkpoint.resolve()
    if best == resolved or best in resolved.parents or resolved in best.parents:
        raise ValueError('v1.5 output overlaps Best checkpoint location')


def _snapshot(saved: dict, label: str) -> Snapshot:
    """Parse v1.4 snapshots to 3.2s (v1.3's parser caps time at 0.8s)."""
    if not isinstance(saved, dict):
        raise RuntimeError(f'Fail Closed: malformed {label}')
    try:
        def decode_array(key):
            raw = saved[key]
            return json.loads(raw) if isinstance(raw,str) else raw
        def decode_bool(key):
            raw=saved[key]
            if raw is True or raw=='True':return True
            if raw is False or raw=='False':return False
            raise ValueError('invalid bool '+key)
        raw_overload=saved['overload_time_s']
        overload_time=None if raw_overload in (None,'','None') else float(raw_overload)
        s=Snapshot(
            time_s=float(saved['time_s']),hits=int(saved['hits']),
            misses=int(saved['misses']),too_early=int(saved['too_early']),
            overloaded=decode_bool('overloaded'),overload_time_s=overload_time,
            terminated=decode_bool('terminated'),
            termination_reason=str(saved['termination_reason']),
            hit_ids=tuple(decode_array('hit_ids')),
            miss_ids=tuple(decode_array('miss_ids')),
            keydowns=tuple((str(k),float(t)) for k,t in decode_array('keydowns')),
        )
        if not (0<s.time_s<=3.2+TOL and
                s.hits==len(s.hit_ids)==len(set(s.hit_ids)) and
                s.misses==len(s.miss_ids)==len(set(s.miss_ids)) and
                not set(s.hit_ids)&set(s.miss_ids) and
                all(type(i) is int and i>=0 for i in (*s.hit_ids,*s.miss_ids)) and
                s.too_early>=0 and
                (not s.overloaded or s.overload_time_s is not None) and
                (s.overload_time_s is None or 0<=s.overload_time_s<=s.time_s+TOL) and
                all(math.isfinite(t) and 0<=t<=s.time_s+TOL and k for k,t in s.keydowns) and
                (not s.terminated or s.termination_reason!='Horizon cutoff')):
            raise ValueError('invalid physical result')
        return s
    except (KeyError,ValueError,TypeError,json.JSONDecodeError) as exc:
        raise RuntimeError('Fail Closed: malformed saved physical snapshot '+label) from exc


def case_from_dict(raw: dict) -> Intervention:
    try:
        keys = tuple(int(k) for k in raw['keys'])
        case = Intervention(int(raw['anchor']), keys, float(raw['delta']),
                            int(raw['start']), int(raw['frames']))
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError('Fail Closed: invalid v1.4 intervention') from exc
    if (not 1 <= case.anchor <= 20 or not keys or len(set(keys)) != len(keys)
            or any(not 0 <= k < 8 for k in keys) or case.start < 0
            or case.frames <= 0 or not math.isfinite(case.delta)
            or not -1 <= case.delta <= 1):
        raise RuntimeError('Fail Closed: impossible v1.4 intervention')
    return case



def _assert_v12_saved_candidate_matches(
    saved_checks: dict[tuple, Snapshot], case: Intervention, recorded: Snapshot,
) -> None:
    """Match v1.4 interventions to v1.2's persisted (anchor, keys, start, frames, delta)."""
    v12_key = (case.anchor, case.keys, case.start, case.frames, round(case.delta, 8))
    previous = saved_checks.get(v12_key)
    if previous is None:
        raise RuntimeError('Fail Closed: v1.2 recorded intervention missing')
    assert_reproduced(recorded, previous, label='v1.4/1.2 candidate')


def _saved_horizons(entry: dict, field: str, label: str) -> dict[float, Snapshot]:
    result = {}
    for horizon in HORIZONS:
        key = str(horizon)
        if key not in entry[field]:
            raise RuntimeError(f'Fail Closed: {label} missing {field}[{key}]')
        result[horizon] = _snapshot(entry[field][key], f'{label}/{key}')
    return result


def _compare_saved_metric(actual: dict, recorded: dict, label: str) -> None:
    """v1.4 neighborhood stores counts/IDs, not full physical keydown times."""
    important = (
        'horizon_s', 'reference_hits', 'candidate_hits', 'reference_misses',
        'candidate_misses', 'reference_too_early', 'candidate_too_early',
        'reference_overloaded', 'candidate_overloaded', 'reference_keydowns',
        'candidate_keydowns', 'hit_ids_lost_by_horizon', 'hit_ids_gained_by_horizon',
        'extension_miss_ids_unpaired', 'safe_observed_improvement',
        'candidate_reached_horizon', 'reference_reached_horizon',
    )
    for key in important:
        if key not in recorded or actual[key] != recorded[key]:
            raise RuntimeError(f'Fail Closed: v1.4 neighborhood mismatch at {label}/{key}')


def _load_previous(v12: Path, v13: Path, v14: Path, checkpoint: Path):
    files, v12v13_hashes, rows, baseline_800, saved_checks = _load_inputs(
        v12, v13, checkpoint
    )
    v14_files = {
        'v14_diagnostic': v14/'diagnostic.json',
        'v14_reproduction': v14/'reproduction_checks.json',
        'v14_baselines': v14/'baseline_continuation.csv',
        'v14_long': v14/'long_horizon_comparison.csv',
        'v14_neighbors': v14/'local_neighbors.csv',
        'v14_robustness': v14/'local_robustness.csv',
    }
    for key, path in v14_files.items():
        if not path.is_file():
            raise FileNotFoundError(f'Fail Closed: missing {key}: {path}')
    all_files = dict(files, **v14_files)
    hashes = _hash_inputs(all_files)
    report = json.loads(v14_files['v14_diagnostic'].read_text(encoding='utf-8'))
    checks = json.loads(v14_files['v14_reproduction'].read_text(encoding='utf-8'))
    if (report.get('kind') != 'train_counterfactual_motor_v14'
            or report.get('best_sha256') != EXPECTED_BEST_SHA256
            or report.get('input_hashes') != v12v13_hashes
            or report.get('train_anchors') != 20
            or report.get('validation_evaluations') != 0
            or report.get('final_used') is not False
            or report.get('optimizer_updates') != 0
            or report.get('teacher_adoptions') != 0
            or report.get('checkpoint_writes') != 0
            or report.get('physical_reproduction_status') != 'PASS'
            or report.get('original_student_state_unchanged') is not True
            or report.get('horizons_s') != list(HORIZONS)
            or checks.get('status') != 'PASS'
            or checks.get('best_sha256') != EXPECTED_BEST_SHA256
            or checks.get('prior_input_sha256') != v12v13_hashes
            or checks.get('baseline_reproduced') != 20
            or checks.get('candidate_reproduced') != len(report.get('stage_a', []))
            or checks.get('no_validation_or_final') is not True
            or checks.get('optimizer_updates') != 0):
        raise RuntimeError('Fail Closed: v1.4 provenance or reproduction mismatch')
    from dmdod.motor.n_key import n_key_names
    selected = select_cases(rows, n_key_names(8))
    stage_a = report['stage_a']
    if len(stage_a) != 9 or len(selected) != 9:
        raise RuntimeError('Fail Closed: expected 9 saved v1.4 interventions')
    saved_cases = {}
    for entry in stage_a:
        case = case_from_dict(entry['case'])
        key = case.identity()
        if key in saved_cases or entry.get('reproduced_800ms') is not True:
            raise RuntimeError('Fail Closed: repeated or unverified v1.4 case')
        saved_cases[key] = entry
    for case, _, role in selected:
        old = saved_cases.get(case.identity())
        if old is None or old.get('role') != role:
            raise RuntimeError('Fail Closed: saved representative/control identity changed')
        if case.anchor == 20 and role != 'representative':
            raise RuntimeError('Fail Closed: Train #20 separate case role mismatch')
    saved_base = {}
    rows_base = _read_csv(v14_files['v14_baselines'])
    if len(rows_base) != 60:
        raise RuntimeError('Fail Closed: missing v1.4 baseline horizons')
    for row in rows_base:
        anchor = int(row['anchor'])
        h = float(row['horizon_s'])
        if not (1 <= anchor <= 20 and h in HORIZONS) or (anchor, h) in saved_base:
            raise RuntimeError('Fail Closed: duplicate v1.4 baseline horizon')
        saved_base[anchor, h] = _snapshot(row, f'v1.4 baseline #{anchor}/{h}')
    if len(saved_base) != 60:
        raise RuntimeError('Fail Closed: incomplete baseline grid')
    for entry in stage_a:
        case = case_from_dict(entry['case'])
        for h, s in _saved_horizons(entry, 'base_snapshots', 'v1.4 base').items():
            assert_reproduced(s, saved_base[case.anchor, h], label='v1.4 saved baseline')
        recorded = _saved_horizons(entry, 'candidate_snapshots', 'v1.4 candidate')
        _assert_v12_saved_candidate_matches(saved_checks, case, recorded[.8])
    # Ensure all original near-neighbor records are present and unambiguous.
    original_neighbors = {}
    if (len(report.get('stage_b',[])) != 7 or
            len(_read_csv(v14_files['v14_robustness'])) != 7):
        raise RuntimeError('Fail Closed: expected seven v1.4 local robustness parents')
    for entry in report['stage_b']:
        parent = case_from_dict(entry['case'])
        if parent.identity() not in saved_cases:
            raise RuntimeError('Fail Closed: v1.4 neighborhood references unknown parent')
        actual_neighbors = local_neighbors(parent)
        saved = entry['trials']
        if {case_from_dict(x['case']).identity() for x in saved} != {
                c.identity() for c in actual_neighbors} or len(saved) != len(actual_neighbors):
            raise RuntimeError('Fail Closed: incomplete v1.4 neighborhood')
        for item in saved:
            c = case_from_dict(item['case'])
            if (c.anchor, c.identity()) in original_neighbors:
                raise RuntimeError('Fail Closed: duplicated saved neighbor')
            original_neighbors[c.anchor, c.identity()] = item
    if (len(_read_csv(v14_files['v14_long'])) != 9*3
            or len(_read_csv(v14_files['v14_neighbors'])) != len(original_neighbors)):
        raise RuntimeError('Fail Closed: v1.4 CSV/JSON row counts disagree')
    return all_files, hashes, selected, saved_base, saved_cases, original_neighbors


def run_detailed_trace(prepared, named, *, case: Intervention|None) -> DetailedReplay:
    """Same fixed Best, HUD encoding and step ordering as v1.4, with event context."""
    from dmdod.training.motor_sensitivity_audit import _environment
    from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
    from dmdod.motor.n_key import NKeyAction
    env = _environment(prepared, named)
    observation = env.reset()
    state = prepared.model.initial_state(prepared.device)
    prepared.model.eval()
    frames: list[Snapshot] = []
    context: list[FrameContext] = []
    at: dict[float, Snapshot] = {}
    with torch.no_grad():
        for frame in range(math.ceil(HORIZONS[-1]/prepared.control_dt_s)):
            hud = tuple(float(x) for x in encode_n_key_hud_real_chart_observation(observation))
            if len(hud) != 263:
                raise RuntimeError('Student visible 263D changed')
            inp = torch.tensor(hud, dtype=torch.float32, device=prepared.device)
            means, _, _, state = prepared.model.forward_step(inp, state)
            raw = tuple(float(x.item()) for x in torch.tanh(means))
            executed = list(raw)
            if case is not None and case.start <= frame < case.end:
                for key in case.keys:
                    executed[key] = max(-1., min(1., executed[key]+case.delta))
            before = len(env._trace_physical_keydowns)
            step = env.step(NKeyAction(tuple(executed)))
            observation = step.observation
            snap = _environment_snapshot(env, finished=bool(step.done))
            _assert_snapshot_consistent(snap)
            frames.append(snap)
            context.append(FrameContext(
                snap.time_s, hud, raw, tuple(executed),
                tuple((e.key, float(e.time_s)) for e in env._trace_physical_keydowns[before:]),
            ))
            for h in HORIZONS:
                if h not in at and snap.time_s >= h-TOL:
                    at[h] = snap
            if step.done:
                break
    if not frames:
        raise RuntimeError('Fail Closed: empty motor rollout')
    for h in HORIZONS:
        at.setdefault(h, frames[-1])  # terminal replayed as layout; not extrapolated
    return DetailedReplay(Trace(at, tuple(frames), ()), tuple(context))


class ReplayCache:
    def __init__(self, prepared):
        self.prepared = prepared
        self._cached: dict[tuple, DetailedReplay] = {}
        self.executions = 0
        self.cache_hits = 0

    def get(self, case: Intervention|None, *, anchor: int|None = None) -> DetailedReplay:
        selected_anchor = case.anchor if case is not None else anchor
        if selected_anchor is None or not 1 <= selected_anchor <= 20:
            raise ValueError('Train anchor must be in [1,20]')
        key = (selected_anchor, case.identity() if case is not None else 'baseline')
        if key not in self._cached:
            self._cached[key] = run_detailed_trace(
                self.prepared, self.prepared.anchors[selected_anchor-1], case=case
            )
            self.executions += 1
        else:
            self.cache_hits += 1
        return self._cached[key]


def _first_frame_of_id(frames: tuple[Snapshot,...], target: int, *, kind: str):
    field = 'hit_ids' if kind == 'hit' else 'miss_ids'
    return next((s.time_s for s in frames if target in getattr(s, field)), None)


def _context_brief(frame: FrameContext|None) -> dict|None:
    if frame is None:
        return None
    return {'time_s':frame.time_s,'visible_hud':list(frame.visible_hud),
            'student_action':list(frame.student_action),
            'executed_action':list(frame.executed_action),
            'physical_keydowns':list(frame.new_keydowns)}


def first_divergence(reference: DetailedReplay, candidate: DetailedReplay) -> dict:
    """First observed outcome/event difference; 10ms resolution, NOT causation."""
    count = min(len(reference.trace.frames), len(candidate.trace.frames))
    fields = ('hit_ids', 'miss_ids', 'too_early', 'overloaded', 'keydowns')
    for i in range(count):
        ref, alt = reference.trace.frames[i], candidate.trace.frames[i]
        if any(getattr(ref, key) != getattr(alt, key) for key in fields):
            a,b = reference.context[i], candidate.context[i]
            return {
                'first_divergence_s': min(ref.time_s, alt.time_s),
                'baseline_previous_step':_context_brief(reference.context[i-1] if i else None),
                'candidate_previous_step':_context_brief(candidate.context[i-1] if i else None),
                'baseline_next_step':_context_brief(reference.context[i+1] if i+1<len(reference.context) else None),
                'candidate_next_step':_context_brief(candidate.context[i+1] if i+1<len(candidate.context) else None),
                'time_resolution_s': .010,
                'observed_not_causal': True,
                'baseline_before_step_visible_hud': list(a.visible_hud),
                'candidate_before_step_visible_hud': list(b.visible_hud),
                'baseline_student_action': list(a.student_action),
                'candidate_student_action': list(b.student_action),
                'baseline_executed_action': list(a.executed_action),
                'candidate_executed_action': list(b.executed_action),
                'baseline_new_keydowns': list(a.new_keydowns),
                'candidate_new_keydowns': list(b.new_keydowns),
                'baseline_hit_ids': list(ref.hit_ids),
                'candidate_hit_ids': list(alt.hit_ids),
                'baseline_miss_ids': list(ref.miss_ids),
                'candidate_miss_ids': list(alt.miss_ids),
            }
    if len(reference.trace.frames) != len(candidate.trace.frames):
        # Differing episode length is observed, but future comparator is absent.
        return {'first_divergence_s': None,
                'different_termination_time_without_shared_result_divergence': True,
                'observed_not_causal': True}
    return {'first_divergence_s': None, 'no_observed_divergence': True,
            'observed_not_causal': True}


def persistence(reference: DetailedReplay, candidate: DetailedReplay) -> dict:
    baseline, alt = reference.trace.at, candidate.trace.at
    b8,c8 = baseline[.8],alt[.8]
    b32,c32 = baseline[3.2],alt[3.2]
    paired = all(x.time_s >= h-TOL for x,h in ((b8,.8),(c8,.8),(b32,3.2),(c32,3.2)))
    return {
        'paired_800_to_3200_observed': paired,
        'initial_hit_delta_800': c8.hits-b8.hits if b8.time_s>=.8-TOL and c8.time_s>=.8-TOL else None,
        'post800_hit_gain': ((c32.hits-c8.hits)-(b32.hits-b8.hits)) if paired else None,
        'post800_miss_gain': ((c32.misses-c8.misses)-(b32.misses-b8.misses)) if paired else None,
        'candidate_continuation_hits_after800': (c32.hits-c8.hits) if c8.time_s>=.8-TOL else None,
        'candidate_continuation_misses_after800': (c32.misses-c8.misses) if c8.time_s>=.8-TOL else None,
        'baseline_unobserved_future': b32.time_s < 3.2-TOL,
        'candidate_unobserved_future': c32.time_s < 3.2-TOL,
    }


def milestone_gate(reference: Trace, candidate: Trace, h: float) -> dict:
    b,c = reference.at[h],candidate.at[h]
    full = b.time_s >= h-TOL and c.time_s>=h-TOL
    kept = not (set(b.hit_ids)-set(c.hit_ids)) if full else None
    miss_ok = c.misses<=b.misses if full else None
    useful = (c.too_early<b.too_early or c.hits>b.hits) if full else None
    no_overload = not c.overloaded and c.time_s>=h-TOL
    pass_gate = full and kept and miss_ok and useful and no_overload and c.hits>0 and bool(c.keydowns)
    return {'horizon_s':h,'both_observed':full,'hit_ids_retained':kept,
            'misses_not_increased':miss_ok,'early_lower_or_hit_gain':useful,
            'no_overload_and_alive':no_overload,'passed':bool(pass_gate)}


def boundary_grid(cases: list[tuple[Intervention,DetailedReplay]],
                  baseline: DetailedReplay) -> list[dict]:
    rows=[]
    for case,record in cases:
        milestones={str(h):milestone_gate(baseline.trace,record.trace,h) for h in HORIZONS}
        s=record.trace.at[.8]
        rows.append({**asdict(case), 'source_800ms_5H_0M_1E':
                     (s.hits,s.misses,s.too_early)==(5,0,1) and s.time_s>=.8-TOL,
                     'milestones':milestones,'success_grid_point':all(x['passed'] for x in milestones.values()),
                     **persistence(baseline,record)})
    return rows


def first_hit_loss(reference: DetailedReplay, candidate: DetailedReplay) -> dict:
    """First observed lag by target ID, plus first persistent milestone loss.

    A lagging target is not called a permanent loss before 3200ms. The
    comparison stops at the shorter *actually observed* episode, never after.
    """
    first=None
    for b,c in zip(reference.trace.frames,candidate.trace.frames):
        lag=sorted(set(b.hit_ids)-set(c.hit_ids))
        if lag:
            first=(b.time_s,lag[0])
            break
    first_milestone=None
    for h in HORIZONS:
        b,c=reference.trace.at[h],candidate.trace.at[h]
        if b.time_s<h-TOL or c.time_s<h-TOL:
            continue
        lag=sorted(set(b.hit_ids)-set(c.hit_ids))
        if lag:
            first_milestone=(h,lag[0])
            break
    if first is None:
        return {'first_observed_hit_lag_s':None,
                'first_milestone_hit_loss_s':None,'lost_target_id':None,
                'no_loss_proven_at_all_horizons':all(
                    reference.trace.at[h].time_s>=h-TOL and
                    candidate.trace.at[h].time_s>=h-TOL for h in HORIZONS)}
    time,target=first
    return {
        'first_observed_hit_lag_s':time,
        'first_milestone_hit_loss_s':first_milestone[0] if first_milestone else None,
        'lost_target_id':target,
        'baseline_hit_confirmed_s':_first_frame_of_id(reference.trace.frames,target,kind='hit'),
        'candidate_hit_confirmed_s':_first_frame_of_id(candidate.trace.frames,target,kind='hit'),
        'candidate_miss_confirmed_s':_first_frame_of_id(candidate.trace.frames,target,kind='miss'),
        'candidate_target_outcome_at_3200':(
            'hit' if target in candidate.trace.at[3.2].hit_ids else
            'miss' if target in candidate.trace.at[3.2].miss_ids else 'unresolved'
        ),
        'recovered_by_3200':target in candidate.trace.at[3.2].hit_ids,
        'candidate_observed_3200':candidate.trace.at[3.2].time_s>=3.2-TOL,
    }


def horizon_target_deltas(reference: Trace,candidate: Trace,h:float) -> dict:
    b,c=reference.at[h],candidate.at[h]
    paired=b.time_s>=h-TOL and c.time_s>=h-TOL
    if not paired:
        return {'both_observed':False, 'baseline_hit_ids':list(b.hit_ids),
                'candidate_hit_ids':list(c.hit_ids),
                'baseline_miss_ids':list(b.miss_ids),
                'candidate_miss_ids':list(c.miss_ids),
                'paired_hit_gains':None,'paired_hit_losses':None,
                'paired_miss_gains':None,'paired_miss_losses':None}
    return {'both_observed':True,
            'baseline_hit_ids':list(b.hit_ids),'candidate_hit_ids':list(c.hit_ids),
            'baseline_miss_ids':list(b.miss_ids),'candidate_miss_ids':list(c.miss_ids),
            'paired_hit_gains':sorted(set(c.hit_ids)-set(b.hit_ids)),
            'paired_hit_losses':sorted(set(b.hit_ids)-set(c.hit_ids)),
            'paired_miss_gains':sorted(set(c.miss_ids)-set(b.miss_ids)),
            'paired_miss_losses':sorted(set(b.miss_ids)-set(c.miss_ids))}


def first_miss_id_difference(reference: DetailedReplay,candidate: DetailedReplay) -> dict:
    for b,c in zip(reference.trace.frames,candidate.trace.frames):
        added=sorted(set(c.miss_ids)-set(b.miss_ids))
        removed=sorted(set(b.miss_ids)-set(c.miss_ids))
        if added or removed:
            return {'time_s':min(b.time_s,c.time_s),
                    'candidate_new_miss_ids':added,'baseline_only_miss_ids':removed,
                    'observed_common_interval':True}
    return {'time_s':None,'candidate_new_miss_ids':[],
            'baseline_only_miss_ids':[], 'observed_common_interval':True}


def first_miss_excess(reference: DetailedReplay,candidate: DetailedReplay) -> float|None:
    for b,c in zip(reference.trace.frames,candidate.trace.frames):
        if c.misses>b.misses:
            return c.time_s
    return None


def too_early_reduction_duration(reference: DetailedReplay,candidate: DetailedReplay) -> dict:
    milestones=[]
    for h in HORIZONS:
        b,c=reference.trace.at[h],candidate.trace.at[h]
        milestones.append({'horizon_s':h,'comparable':b.time_s>=h-TOL and c.time_s>=h-TOL,
                           'reduced':c.too_early<b.too_early if b.time_s>=h-TOL and c.time_s>=h-TOL else None})
    continuous=None
    for m in milestones:
        if m['reduced'] is not True:
            break
        continuous=m['horizon_s']
    return {'milestones':milestones,'continuously_reduced_through_s':continuous,
            'last_observed_lower_s':next((m['horizon_s'] for m in reversed(milestones)
                                       if m['reduced'] is True),None)}


def rescue_quality(reference: DetailedReplay, candidate: DetailedReplay, case: Intervention) -> list[dict]:
    out=[]
    base_known=set(reference.trace.at[.8].hit_ids)
    for h in HORIZONS:
        s=candidate.trace.at[h]
        observed=s.time_s>=h-TOL
        # Survival and quality are *independent* Boolean assessments.
        survived=observed and not s.overloaded
        no_miss=observed and s.misses==0
        retained=base_known.issubset(s.hit_ids) if observed else False
        quality=observed and not s.overloaded and no_miss and s.hits>0 and bool(s.keydowns) and retained
        resolved=s.hits+s.misses
        out.append({**asdict(case),'horizon_s':h,
                    'reached_horizon':observed,
                    'overload_avoided':not s.overloaded,
                    'survival_success':bool(survived),
                    'quality_success':bool(quality),
                    'quality_definition':'reached horizon; overload-free; 0 Miss; >=1 Hit and KeyDown; retained known baseline Hits',
                    'known_baseline_hit_ids_retained':retained,
                    'hits':s.hits,'misses':s.misses,'too_early':s.too_early,
                    'keydowns':len(s.keydowns),'hit_ids':list(s.hit_ids),
                    'miss_ids':list(s.miss_ids),
                    'miss_fraction_of_resolved':s.misses/resolved if resolved else None,
                    'no_miss_at_horizon':bool(no_miss),
                    'overload':s.overloaded,'overload_time_s':s.overload_time_s,
                    'reached_time_s':s.time_s,
                    'new_misses_after800':(s.misses-candidate.trace.at[.8].misses)
                    if h>.8 and candidate.trace.at[.8].time_s>=.8-TOL else None})
    return out


def _json_field(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':')) if isinstance(value,(list,dict,tuple)) else value


def _csv(path:Path,rows:list[dict],fields:tuple[str,...]):
    csv_write_new(path,[{k:_json_field(r.get(k)) for k in fields} for r in rows],columns=fields)


def run_v15(prepared, *, v12_dir:Path, v13_dir:Path, v14_dir:Path, output:Path) -> dict:
    _required_layout(prepared,output,(v12_dir,v13_dir,v14_dir))
    files,input_hashes,selected,saved_base,saved_cases,saved_neighbors=_load_previous(
        v12_dir,v13_dir,v14_dir,prepared.source_checkpoint)
    from dmdod.training.fixed_state_action_audit import assert_state_equal
    from dmdod.training.real_chart import clone_model_state
    initial_state=clone_model_state(prepared.model)
    before_hash=file_hash(prepared.source_checkpoint)
    if before_hash!=EXPECTED_BEST_SHA256:
        raise RuntimeError('Fail Closed: Best SHA256 changed')
    cache=ReplayCache(prepared)
    grid06=[];drift18=[];rescue=[];persist_rows=[];events=[];checks=[]
    try:
        # Reproduce all Train20 baselines and all 9 v1.4 intervention traces first.
        for a in range(1,21):
            run=cache.get(None,anchor=a)
            for h in HORIZONS:
                assert_reproduced(run.trace.at[h],saved_base[a,h],label=f'Train #{a} {h}s')
        for case,row,role in selected:
            run=cache.get(case)
            saved=saved_cases[case.identity()]
            for h,reference in _saved_horizons(saved,'candidate_snapshots',f'Train #{case.anchor}').items():
                assert_reproduced(run.trace.at[h],reference,label=f'v1.4 {role} #{case.anchor}/{h}s')
            base=cache.get(None,anchor=case.anchor)
            record={'anchor':case.anchor,'role':role,**asdict(case),
                    **persistence(base,run)}
            record.update({f'cmp_{h}':compare_horizon(base.trace,run.trace,h) for h in HORIZONS})
            record['horizon_target_id_deltas']={str(h):horizon_target_deltas(base.trace,run.trace,h) for h in HORIZONS}
            persist_rows.append(record)
            events.append({'anchor':case.anchor,'role':role,**asdict(case),
                           **first_divergence(base,run)})
        checks.append({'reference':'v1.4_nine_interventions','result':'PASS','count':len(selected)})
        # Check ALL v1.4 saved neighborhoods before interpreting any new points.
        for (anchor,identity),previous in saved_neighbors.items():
            original=case_from_dict(previous['case'])
            actual=cache.get(original)
            base=cache.get(None,anchor=anchor)
            for h,key in ((.8,'at_800ms'),(1.6,'at_1600ms')):
                _compare_saved_metric(compare_horizon(base.trace,actual.trace,h),
                                      previous[key],f'Train #{anchor}/{identity}/{h}')
        checks.append({'reference':'v1.4_saved_neighbors','result':'PASS',
                       'count':len(saved_neighbors)})
        # Stage B: explicitly measured 9x2 finite grid. No interpolation.
        for start in (0,1):
            for delta in GRID_06:
                case=Intervention(6,(2,),delta,start,8)
                grid06.append((case,cache.get(case)))
        b06=cache.get(None,anchor=6)
        boundary=boundary_grid(grid06,b06)
        for case,record in grid06:
            events.append({'anchor':6,'role':'boundary_grid',**asdict(case),
                           **first_divergence(b06,record)})
        # Stage C: 2x3 starts x3 deltas, all to 3200ms.
        for keys,length,starts in (((5,),1,(3,4,5)),(tuple(range(8)),8,(7,8,9))):
            for start in starts:
                for delta in GRID_18:
                    case=Intervention(18,keys,delta,start,length)
                    current=cache.get(case);base=cache.get(None,anchor=18)
                    record={'anchor':18,**asdict(case),
                            **{f'hit_ids_retained_{int(h*1000)}':milestone_gate(base.trace,current.trace,h)['hit_ids_retained'] for h in HORIZONS},
                            **first_hit_loss(base,current),
                            'first_extra_miss_s':first_miss_excess(base,current),
                            'first_miss_id_difference':first_miss_id_difference(base,current),
                            'horizon_target_id_deltas':{str(h):horizon_target_deltas(base.trace,current.trace,h) for h in HORIZONS},
                            'too_early_reduction':too_early_reduction_duration(base,current),
                            'first_divergence':first_divergence(base,current),
                            **persistence(base,current)}
                    drift18.append(record)
                    events.append({'anchor':18,'role':'late_drift',**asdict(case),
                                   **first_divergence(base,current)})
        # Stage D: 2x(original + exactly five previously evaluated neighbors).
        for anchor in RESCUE_ANCHORS:
            candidate=[case for case,_,role in selected if case.anchor==anchor
                       and role in ('representative','conditional_08')]
            if len(candidate)!=1:
                raise RuntimeError(f'Fail Closed: rescue Train #{anchor} not uniquely identified')
            origin=candidate[0]
            cases=[origin,*local_neighbors(origin)]
            if len(cases)!=6 or len({c.identity() for c in cases})!=6:
                raise RuntimeError(f'Fail Closed: expected six rescue probes for #{anchor}')
            for case in cases:
                run=cache.get(case)
                rescue.extend(rescue_quality(cache.get(None,anchor=anchor),run,case))
                events.append({'anchor':anchor,'role':'rescue',**asdict(case),
                               **first_divergence(cache.get(None,anchor=anchor),run)})
        # Protect all pre-existing physical evidence and checkpoint from writes.
        assert_state_equal(clone_model_state(prepared.model),initial_state,
                           context='Train v1.5 frozen Student')
        if file_hash(prepared.source_checkpoint)!=before_hash or _hash_inputs(files)!=input_hashes:
            raise RuntimeError('Fail Closed: prior input or checkpoint changed')
        by_anchor={}
        for anchor in RESCUE_ANCHORS:
            entries=[r for r in rescue if r['anchor']==anchor]
            by_anchor[str(anchor)]={str(h):{
                'trials':6,'overload_avoided':sum(r['overload_avoided'] for r in entries if r['horizon_s']==h),
                'survived':sum(r['survival_success'] for r in entries if r['horizon_s']==h),
                'quality_success':sum(r['quality_success'] for r in entries if r['horizon_s']==h),
                'survival_rate':sum(r['survival_success'] for r in entries if r['horizon_s']==h)/6,
                'quality_rate':sum(r['quality_success'] for r in entries if r['horizon_s']==h)/6,
                'reached_with_zero_misses':sum(r['no_miss_at_horizon'] for r in entries if r['horizon_s']==h),
            } for h in HORIZONS}
        rescue_case_details={}
        for anchor in RESCUE_ANCHORS:
            short=[r for r in rescue if r['anchor']==anchor and r['horizon_s']==1.6]
            long=[r for r in rescue if r['anchor']==anchor and r['horizon_s']==3.2]
            non_overloaded=[r for r in long if r['survival_success']]
            rescue_case_details[str(anchor)]={
                'zero_miss_at_1600_cases':[
                    {k:r[k] for k in ('keys','delta','start','frames')}
                    for r in short if r['no_miss_at_horizon']],
                'survived_3200_cases':[
                    {k:r[k] for k in ('keys','delta','start','frames')}
                    for r in non_overloaded],
                'minimum_observed_misses_3200':min(
                    (r['misses'] for r in long if r['reached_horizon']),default=None),
            }
        # No model selection: conclusion is a factual descriptor of measured cases.
        grid_success={(r['start'],round(r['delta'],2)) for r in boundary if r['success_grid_point']}
        adjacent_robust=any((start,round(delta+.01,2)) in grid_success or
                            (start,round(delta-.01,2)) in grid_success
                            for start,delta in grid_success)
        if adjacent_robust:
            conclusion='long_term_locally_robust_correction_observed'
        elif grid_success:
            conclusion='long_term_correction_observed_but_grid_sensitive'
        elif any(any(x['survival_success'] for x in rescue if x['anchor']==a and x['horizon_s']==3.2)
                 and not all(x['quality_success'] for x in rescue if x['anchor']==a and x['horizon_s']==3.2)
                 for a in RESCUE_ANCHORS):
            conclusion='overload_survival_improved_but_quality_unstable'
        else:
            conclusion='no_long_term_safe_improvement_confirmed_in_measured_grid'
        summary={
            'kind':'train_counterfactual_motor_v15', 'best_sha256':before_hash,
            'measured_conclusion':conclusion,
            'input_sha256':input_hashes,'train_anchors':20,'validation_evaluations':0,
            'final_used':False,'optimizer_updates':0,'teacher_adoptions':0,
            'checkpoint_writes':0,'horizons_s':list(HORIZONS),
            'reference_v14_reproduced':True,'v14_representatives':9,
            'boundary_06':{'tested_grid_points':len(boundary),
                           'successful_grid_points':sum(r['success_grid_point'] for r in boundary),
                           'original_800ms_5H_0M_1E_count':sum(r['source_800ms_5H_0M_1E'] for r in boundary),
                           'no_interpolation':True},
            'late_drift_18':{'tested_grid_points':len(drift18),
                             'retained_all_horizons':sum(all(r[f'hit_ids_retained_{int(h*1000)}'] is True for h in HORIZONS) for r in drift18)},
            'rescue_survival_quality':by_anchor,
            'rescue_condition_details':rescue_case_details,
            'replay_cache':{'physical_replays':cache.executions,
                            'cache_hits':cache.cache_hits,'unique_cases':len(cache._cached)},
            'limitations':[
                'All interventions and origin states are Train only; no Validation/FINAL.',
                'A 10ms observed divergence does not identify its cause.',
                'No metrics are extrapolated past an episode termination.',
                'Paired post800 gains are null unless BOTH sides reached 800 and 3200ms.',
                'Overload-free survival and quality are independent; quality requires zero misses.',
                'Only measured finite grid points are labeled, no amplitude interpolation.',
                'No learning, optimization, teacher adoption, or checkpoint promotion.',
            ],
        }
        output.mkdir(parents=True,exist_ok=False)
        try:
            (output/'diagnostic.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            _csv(output/'gain_persistence.csv',persist_rows,
                 ('anchor','role','keys','delta','start','frames','paired_800_to_3200_observed',
                  'initial_hit_delta_800','post800_hit_gain','post800_miss_gain',
                  'candidate_continuation_hits_after800','candidate_continuation_misses_after800',
                  'baseline_unobserved_future','candidate_unobserved_future',
                  'horizon_target_id_deltas','cmp_0.8','cmp_1.6','cmp_3.2'))
            _csv(output/'boundary_grid_06.csv',boundary,
                 ('anchor','keys','delta','start','frames','source_800ms_5H_0M_1E',
                  'success_grid_point','milestones','paired_800_to_3200_observed',
                  'post800_hit_gain','post800_miss_gain'))
            _csv(output/'late_drift_18.csv',drift18,
                 ('anchor','keys','delta','start','frames','hit_ids_retained_800',
                  'hit_ids_retained_1600','hit_ids_retained_3200',
                  'first_observed_hit_lag_s','first_milestone_hit_loss_s','lost_target_id','baseline_hit_confirmed_s',
                  'candidate_hit_confirmed_s','candidate_miss_confirmed_s',
                  'candidate_target_outcome_at_3200','recovered_by_3200','candidate_observed_3200',
                  'first_extra_miss_s','first_miss_id_difference',
                  'horizon_target_id_deltas','too_early_reduction','first_divergence',
                  'paired_800_to_3200_observed','post800_hit_gain','post800_miss_gain'))
            _csv(output/'event_divergence.csv',events,
                 ('anchor','role','keys','delta','start','frames','first_divergence_s',
                  'time_resolution_s','observed_not_causal','no_observed_divergence',
                  'different_termination_time_without_shared_result_divergence',
                  'baseline_previous_step','candidate_previous_step',
                  'baseline_next_step','candidate_next_step',
                  'baseline_before_step_visible_hud','candidate_before_step_visible_hud',
                  'baseline_student_action','candidate_student_action',
                  'baseline_executed_action','candidate_executed_action',
                  'baseline_new_keydowns','candidate_new_keydowns',
                  'baseline_hit_ids','candidate_hit_ids','baseline_miss_ids','candidate_miss_ids'))
            _csv(output/'rescue_survival_quality.csv',rescue,
                 ('anchor','keys','delta','start','frames','horizon_s',
                  'overload_avoided','survival_success','quality_success','quality_definition',
                  'known_baseline_hit_ids_retained','hits','misses','too_early',
                  'keydowns','hit_ids','miss_ids','miss_fraction_of_resolved',
                  'no_miss_at_horizon','overload','overload_time_s','reached_time_s',
                  'new_misses_after800'))
            (output/'reproduction_checks.json').write_text(json.dumps({
                'status':'PASS','best_sha256':before_hash,
                'prior_input_sha256':input_hashes,
                'baseline_reproduced':20,'representative_reproduced':9,
                'v14_neighbor_reproduced':len(saved_neighbors),
                'physical_replays':cache.executions,
                'frozen_student':True,'validation_evaluations':0,
                'final_used':False,'optimizer_updates':0,
            },indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            if file_hash(prepared.source_checkpoint)!=before_hash or _hash_inputs(files)!=input_hashes:
                raise RuntimeError('Fail Closed: input changed during publishing')
        except Exception:
            (output/'diagnostic.json').unlink(missing_ok=True)
            (output/'FAILED_NO_TEACHER_ADOPTION.txt').write_text(
                'Fail Closed: diagnostic incomplete, no teacher adoption or training.\n',
                encoding='utf-8')
            raise
        return summary
    finally:
        prepared.model.load_state_dict(initial_state,strict=True)
        prepared.model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(prepared.model),initial_state,
                           context='v1.5 restores frozen Best Student')
