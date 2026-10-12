from __future__ import annotations

"""Train-only counterfactual gate/coverage/late-effect audit (v1.1).

This is a diagnostic, NOT a training or model-selection algorithm. No changes to
Student inputs, motor dynamics, original v1 pilot, checkpoint, or FINAL data.
Every candidate is evaluated against a fresh reset in closed-loop. Future target
information is evaluator-only, never fed to the Student.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random

import torch

from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.motor.n_key import NKeyAction, n_key_names
from dmdod.training.fixed_state_action_audit import assert_state_equal, file_sha256
from dmdod.training.human_visible_curriculum import _collect_dagger_trajectories, stable_seed
from dmdod.training.motor_sensitivity_audit import _environment
from dmdod.training.onset_ablation import _source_preflight
from dmdod.training.real_chart import clone_model_state
from dmdod.training.train_counterfactual_motor import (
    DELTA_GRID, DURATION_GRID, PRECONTACT_GRID, HORIZON_S, MAX_CASES_PER_ANCHOR,
    MotorRollout, assert_train_rollout_matches_dagger, rollout,
)

ONSET_S = 0.200
FOLLOWUP_S = 0.800
BEST_SOURCE_SHA256 = "476696a2a157c15a00d602845208a21049dcca94479cee9b91bfeb105cac62e0"


@dataclass(frozen=True)
class Probe:
    keys: tuple[int, ...]
    start: int
    duration: int
    delta: float
    lead: int

    @property
    def kind(self) -> str:
        return "single" if len(self.keys) == 1 else (
            "all_8" if len(self.keys) == 8 else "paired"
        )

    @property
    def end(self) -> int:
        return self.start + self.duration


def key_event_sources(base: MotorRollout, *, key_count: int):
    names = {name: i for i, name in enumerate(n_key_names(key_count))}
    found = {}
    for name, time_s in base.early:
        if name not in names:
            raise RuntimeError(f"Unknown physical key in Train TooEarly: {name}")
        if time_s <= ONSET_S + 1e-9:
            found.setdefault(names[name], float(time_s))
    return tuple(sorted(found.items(), key=lambda pair: (pair[1], pair[0])))


def candidate_pool(base: MotorRollout, *, dt: float, key_count: int,
                   include_coordination: bool = True) -> list[Probe]:
    """Enumerate all unique causal controls before budget selection.

    Unlike v1, duration is allowed to extend beyond the original failure.
    This changes only a new counterfactual rollout, never extrapolates actions.
    """
    if not 0 < dt <= ONSET_S:
        raise ValueError("invalid control timestep")
    sources = key_event_sources(base, key_count=key_count)
    if not sources:
        return []
    key_sets = [(key,) for key, _ in sources]
    if include_coordination:
        if len(sources) >= 2:
            key_sets.extend(combinations([key for key, _ in sources], 2))
        key_sets.append(tuple(range(key_count)))
    times = dict(sources)
    seen = set()
    for keys in key_sets:
        # For paired and all-8 interventions start before the earliest relevant
        # offending key. This is a diagnostic oracle on TRAIN only.
        onset_time = min(times[k] for k in keys if k in times)
        frame = max(0, math.ceil(onset_time / dt) - 1)
        for lead in PRECONTACT_GRID:
            for duration in DURATION_GRID:
                for delta in DELTA_GRID:
                    start = max(0, frame-lead)
                    if start >= base.raw_actions.shape[0]:
                        continue
                    seen.add(Probe(tuple(keys), start, duration, delta, frame-start))
    return sorted(seen, key=lambda c: (c.kind, c.keys, c.start, c.lead,
                                      c.duration, c.delta))


def balanced_subset(pool: list[Probe], *, budget: int, seed: int) -> list[Probe]:
    """Deterministic greedy stratification; no prefix preference for lead/delta.

    Counts are balanced across key groups, amplitude, duration and lead, with
    a seeded tie-breaker. Candidate generation is not adaptive to outcomes.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if len(pool) <= budget:
        return list(pool)
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    counts = [Counter() for _ in range(4)]
    remaining = list(shuffled)
    selected = []
    for _ in range(budget):
        def dimensions(c: Probe):
            return (c.keys, c.delta, c.duration, c.lead)
        # Normalize dimensions to keep groups with more possible values from
        # dominating; the maximum counter is a stronger fairness objective.
        winner = min(range(len(remaining)), key=lambda i: (
            max(counts[j][v] for j, v in enumerate(dimensions(remaining[i]))),
            sum(counts[j][v] for j, v in enumerate(dimensions(remaining[i]))),
            i,
        ))
        c = remaining.pop(winner)
        selected.append(c)
        for j, v in enumerate(dimensions(c)):
            counts[j][v] += 1
    return selected


def coverage(pool: list[Probe], sampled: list[Probe]) -> dict:
    axes = {"groups":lambda c: ",".join(str(k) for k in c.keys),
            "delta":lambda c: str(c.delta),
            "duration":lambda c: str(c.duration),
            "lead":lambda c: str(c.lead),
            "kind":lambda c: c.kind}
    return {name:{"available":dict(sorted(Counter(fn(p) for p in pool).items())),
                  "sampled":dict(sorted(Counter(fn(p) for p in sampled).items()))}
            for name, fn in axes.items()}


def altered_rollout(prepared, model, named, case: Probe, *, horizon_s: float) -> MotorRollout:
    """Re-infer Student actions after every changed motor observation."""
    env = _environment(prepared, named)
    observation = env.reset()
    state = model.initial_state(prepared.device)
    obs, raw, executed = [], [], []
    ended = False
    model.eval()
    with torch.no_grad():
        for t in range(math.ceil(horizon_s / prepared.control_dt_s)):
            encoded = encode_n_key_hud_real_chart_observation(observation)
            if len(encoded) != 263:
                raise RuntimeError("Student visible 263D contract violated")
            x = torch.tensor(encoded, device=prepared.device, dtype=torch.float32)
            mean, _, _, state = model.forward_step(x, state)
            original = tuple(float(v.item()) for v in torch.tanh(mean))
            action = list(original)
            if case.start <= t < case.end:
                for k in case.keys:
                    action[k] = max(-1., min(1., action[k] + case.delta))
            obs.append(encoded)
            raw.append(original)
            executed.append(tuple(action))
            result = env.step(NKeyAction(tuple(action)))
            observation = result.observation
            if result.done:
                ended = True
                break
    return MotorRollout(
        observations=torch.tensor(obs, dtype=torch.float32, device=prepared.device),
        raw_actions=torch.tensor(raw, dtype=torch.float32, device=prepared.device),
        executed_actions=torch.tensor(executed, dtype=torch.float32, device=prepared.device),
        used=tuple(i for i, b in enumerate(env._used) if b),
        missed=tuple(i for i, b in enumerate(env._missed) if b),
        keydowns=tuple((x.key,float(x.time_s)) for x in env._trace_physical_keydowns),
        early=tuple((x.key,float(x.time_s)) for x in env._trace_too_early),
        hits=int(env._hits), misses=int(env._misses), too_early=int(env._too_early),
        overloaded=bool(env._overload.overloaded),
        duration_s=float(env.privileged_episode_time_s()), terminated=ended,
    )


def expert_rollout(prepared, named, *, horizon_s: float) -> MotorRollout:
    """Privileged center-first teacher is scored on TRAIN, never shown to Student."""
    from dmdod.n_key_capacity import CenterFirstNKeyTeacher
    from dmdod.training.n_key import _teacher_action
    env = _environment(prepared, named)
    observation = env.reset()
    teacher = CenterFirstNKeyTeacher(prepared.key_count)
    for _ in range(math.ceil(horizon_s/prepared.control_dt_s)):
        action = _teacher_action(env, teacher, observation, lead_s=prepared.lead_s)
        result = env.step(action)
        observation = result.observation
        if result.done:
            break
    empty = torch.empty((0, prepared.key_count), device=prepared.device)
    return MotorRollout(
        observations=torch.empty((0,263), device=prepared.device),
        raw_actions=empty, executed_actions=empty,
        used=tuple(i for i, b in enumerate(env._used) if b),
        missed=tuple(i for i, b in enumerate(env._missed) if b),
        keydowns=tuple((x.key,float(x.time_s)) for x in env._trace_physical_keydowns),
        early=tuple((x.key,float(x.time_s)) for x in env._trace_too_early),
        hits=int(env._hits), misses=int(env._misses), too_early=int(env._too_early),
        overloaded=bool(env._overload.overloaded),
        duration_s=float(env.privileged_episode_time_s()), terminated=bool(env._done),
    )


def gate_checks(base: MotorRollout, candidate: MotorRollout,
                teacher: MotorRollout) -> dict[str,bool]:
    """Record ALL predicate results independently, including v1's hard filters."""
    return {
        "no_overload":not candidate.overloaded,
        "short_horizon_reached":candidate.duration_s >= HORIZON_S-1e-7,
        "has_keydown":len(candidate.keydowns)>0,
        "expert_valid":not (teacher.overloaded or
            (teacher.terminated and teacher.duration_s < ONSET_S)),
        "baseline_hit_ids_retained":set(base.used).issubset(candidate.used),
        "hits_not_lower":candidate.hits>=base.hits,
        "too_early_lower":candidate.too_early<base.too_early,
        "expert_relative_hits":candidate.hits>=max(base.hits, math.ceil(.75*teacher.hits)),
        "miss_guard":(candidate.misses<=base.misses if not base.overloaded
                       else candidate.misses<=teacher.misses+1),
        "safe_hit_gain":base.overloaded or candidate.hits>base.hits,
    }


HARD_GATES = ("no_overload", "short_horizon_reached", "has_keydown", "expert_valid",
              "baseline_hit_ids_retained", "hits_not_lower", "too_early_lower",
              "expert_relative_hits", "miss_guard")


def short_status(gates: dict[str,bool], *, baseline_overload: bool) -> str:
    if any(not gates[name] for name in HARD_GATES):
        return "rejected_short"
    if baseline_overload:
        return "overload_rescue_short"
    if gates["safe_hit_gain"]:
        return "safe_hit_gain_short"
    return "safe_early_reduction_review_only"


def followup_checks(short: MotorRollout, long: MotorRollout,
                    short_teacher: MotorRollout, long_teacher: MotorRollout,
                    *, horizon_s: float) -> dict[str,bool]:
    """Guard against deferred TooEarly, early termination and hit suppression."""
    incremental_early = long.too_early-short.too_early
    teacher_incremental = long_teacher.too_early-short_teacher.too_early
    return {
        "reached_horizon":long.duration_s >= horizon_s-1e-7,
        "no_overload":not long.overloaded,
        "short_hits_retained":set(short.used).issubset(long.used),
        "no_hit_regression":long.hits>=short.hits,
        "expert_relative_hits":long.hits>=math.ceil(.75*long_teacher.hits),
        "late_early_within_expert_plus_one":incremental_early<=teacher_incremental+1,
        "late_misses_within_expert_plus_one":long.misses<=long_teacher.misses+1,
    }


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def run_train_counterfactual_v11(prepared, config, *, output_dir: Path,
                                  max_cases_per_anchor: int = MAX_CASES_PER_ANCHOR,
                                  followup_s: float = FOLLOWUP_S) -> dict:
    """No optimizer, checkpoint writes or Validation physics evaluations."""
    if output_dir.exists():
        raise FileExistsError(f"output directory must not exist: {output_dir}")
    if not 0.4 < followup_s <= 1.0:
        raise ValueError("followup_s must be >0.4 and <=1.0")
    if not 1 <= max_cases_per_anchor <= 1024:
        raise ValueError("max_cases_per_anchor outside [1,1024]")
    _source_preflight(prepared, config, output_dir/'not_used.json', 10)
    if len(prepared.anchors)!=20 or len(prepared.validation)!=20 or prepared.key_count!=8 or prepared.input_dim!=263:
        raise ValueError("requires original 20 Train / 20 Validation, 8 keys and 263D")
    model = prepared.model
    source_state = clone_model_state(model)
    source_sha = file_sha256(prepared.source_checkpoint)
    if source_sha != BEST_SOURCE_SHA256:
        raise RuntimeError('Best checkpoint hash differs from fixed v1 audit')
    rows, anchors, qualified = [], [], []
    try:
        dagger, _, intervened = _collect_dagger_trajectories(
            prepared, model, beta=0.0, epoch=int(prepared.parent.get(
                'human_visible_best_epoch', prepared.parent['human_visible_epoch']))+1,
            base_seed=int(config.human_visible.seed),
        )
        if len(dagger)!=20 or intervened!=0:
            raise RuntimeError("unexpected DAgger Train collection")
        for i, named in enumerate(prepared.anchors):
            base = rollout(prepared, model, named, horizon_s=HORIZON_S)
            assert_train_rollout_matches_dagger(base, dagger[i])
            expert = expert_rollout(prepared, named, horizon_s=HORIZON_S)
            pool = candidate_pool(base, dt=prepared.control_dt_s,
                                  key_count=prepared.key_count)
            cases = balanced_subset(pool, budget=max_cases_per_anchor,
                                    seed=stable_seed(int(config.human_visible.seed),
                                                     'cf-motor-v11-coverage', i))
            shortlist = []
            row_list = []
            for case in cases:
                changed = altered_rollout(prepared, model, named, case, horizon_s=HORIZON_S)
                gates = gate_checks(base, changed, expert)
                status = short_status(gates, baseline_overload=base.overloaded)
                row = {
                    'anchor':i+1,'chart':named.chart_name,'kind':case.kind,
                    'keys':','.join(n_key_names(prepared.key_count)[k] for k in case.keys),
                    'start_frame':case.start,'lead_frames':case.lead,
                    'duration_frames':case.duration,'delta':case.delta,
                    'base_overload':base.overloaded,'base_hits':base.hits,
                    'base_early':base.too_early,
                    'short_hits':changed.hits,'short_early':changed.too_early,
                    'short_keydowns':len(changed.keydowns),
                    'short_overload':changed.overloaded,'short_status':status,
                    **{f'gate_{k}':v for k,v in gates.items()},
                    'followup_checked':False,'followup_qualified':False,
                    'followup_review_only':False,
                    'followup_hits':None,'followup_early':None,
                    'followup_misses':None,'followup_keydowns':None,
                    'followup_overload':None,'followup_duration_s':None,
                    'followup_new_hits':None,'followup_new_keydowns':None,
                    'followup_new_too_early':None,
                    **{f'followup_{k}':None for k in (
                        'reached_horizon','no_overload','short_hits_retained',
                        'no_hit_regression','expert_relative_hits',
                        'late_early_within_expert_plus_one',
                        'late_misses_within_expert_plus_one')},
                }
                row_list.append(row)
                if status!='rejected_short':
                    shortlist.append((case,changed,row))
            followup_teacher = expert_rollout(prepared,named,horizon_s=followup_s) if shortlist else None
            for case, short, row in shortlist:
                long = altered_rollout(prepared,model,named,case,horizon_s=followup_s)
                checks = followup_checks(short,long,expert,followup_teacher,horizon_s=followup_s)
                passed = all(checks.values())
                promotable_class = row['short_status']!='safe_early_reduction_review_only'
                qualified_case = passed and promotable_class and followup_teacher.hits > 0
                row.update({
                    'followup_checked':True,'followup_qualified':qualified_case,
                    'followup_review_only':passed and not qualified_case,
                    'followup_hits':long.hits,'followup_early':long.too_early,
                    'followup_misses':long.misses,'followup_keydowns':len(long.keydowns),
                    'followup_overload':long.overloaded,'followup_duration_s':long.duration_s,
                    'followup_new_hits':long.hits-short.hits,
                    'followup_new_keydowns':len(long.keydowns)-len(short.keydowns),
                    'followup_new_too_early':long.too_early-short.too_early,
                    **{f'followup_{k}':v for k,v in checks.items()},
                })
                if qualified_case:
                    qualified.append({key:row[key] for key in (
                        'anchor','chart','kind','keys','start_frame','lead_frames',
                        'duration_frames','delta','short_status','followup_hits',
                        'followup_early','followup_misses')})
            rows.extend(row_list)
            anchors.append({
                'anchor':i+1,'chart':named.chart_name,
                'base_overload_400ms':base.overloaded,
                'base_early_400ms':base.too_early,
                'pool_cases':len(pool),'evaluated':len(cases),
                'sampling_coverage':coverage(pool,cases),
                'shortlisted':len(shortlist),
                'qualified_followup':sum(bool(row['followup_qualified']) for row in row_list),
                'followup_review_only':sum(bool(row['followup_review_only']) for row in row_list),
                'too_early_source_keys':len(key_event_sources(base,key_count=prepared.key_count)),
                'short_status_counts':dict(Counter(row['short_status'] for row in row_list)),
                'failing_gate_counts':{key:sum(not row[f'gate_{key}'] for row in row_list)
                                       for key in (*HARD_GATES, 'safe_hit_gain')},
            })
            print(f'Train v1.1 {i+1:02d}/20: pool={len(pool)} '
                  f'cases={len(cases)} short={len(shortlist)} '
                  f'followup={anchors[-1]["qualified_followup"]}',flush=True)
        if file_sha256(prepared.source_checkpoint)!=source_sha:
            raise RuntimeError('Best checkpoint changed during v1.1 diagnostics')
        if output_dir.exists():
            raise FileExistsError('output appeared during audit')
        output_dir.mkdir(parents=True)
        fields=list(rows[0]) if rows else []
        if fields:
            write_rows(output_dir/'train_candidate_gates.csv',rows,fields)
        if qualified:
            write_rows(output_dir/'followup_qualified.csv',qualified,list(qualified[0]))
        report={
            'kind':'train_counterfactual_motor_diagnostic_v11',
            'source_sha256':source_sha,'uses_train_only_for_counterfactuals':True,
            'validation_evaluations':0,'optimizer_updates':0,
            'checkpoint_writes':0,'final_used':False,
            'short_horizon_s':HORIZON_S,'followup_horizon_s':followup_s,
            'max_cases_per_anchor':max_cases_per_anchor,
            'onset_sources':'first_200ms_Train_TooEarly_keys',
            'candidate_policy':'balanced_across_key_groups_amplitudes_durations_leads',
            'followup_policy':'all_short_passers_retested_closed_loop_from_reset',
            'no_teacher_promotion':True,
            'train_candidate_evaluations':len(rows),
            'train_followup_evaluations':sum(a['shortlisted'] for a in anchors),
            'train_anchors_with_followup_qualification':sum(a['qualified_followup']>0 for a in anchors),
            'qualified_followup_cases':len(qualified),
            'followup_review_only_cases':sum(a['followup_review_only'] for a in anchors),
            'short_status_counts':dict(Counter(row['short_status'] for row in rows)),
            'failing_gate_counts':{key:sum(not row[f'gate_{key}'] for row in rows)
                                   for key in (*HARD_GATES, 'safe_hit_gain')},
            'anchors':anchors,
            'limitations':[
                'Train-only evaluator oracle chooses offending keys, never Student input.',
                'A passing 800ms followup does not prove full-chart safety.',
                'Low 800ms teacher hit counts may still allow trivial suppression.',
                'No optimizer or checkpoint selection in this diagnostic.',
                'No validation used for intervention discovery or hyperparameter selection.'
            ],
        }
        with (output_dir/'diagnostic.json').open('x',encoding='utf-8') as f:
            json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
        return report
    finally:
        model.load_state_dict(source_state,strict=True)
        model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(model),source_state,
                           context='v1.1 restored original Best')
