from __future__ import annotations

"""Read-only, Motor-only full KeyDown attribution audit. Run from repository root."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_coupled_chart_audit import (
    ContactOracleTeacher, CONTROL_DT, PHYSICS_DT, MODES,
)

EXPECTED_U1_SHA = 'f023efe7b03dffaa6225c3927c368ff01fa910de218beb01e83ca906279e93ed'

def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

def summarize(rows, total, env, teacher):
    by_time = defaultdict(list)
    by_stroke = defaultdict(list)
    for row in rows:
        by_time[row['time_us']].append(row)
        by_stroke[(row['time_us'],row['axis'])].append(row)
    paired = [g for g in by_stroke.values() if len(g) > 1]
    if any(len(g) > 2 for g in paired):
        raise AssertionError('unexpected >2 contact stroke')
    matched = [r for r in rows if r['reserved'] is not None and r['reserved'] == r['credited']]
    mismatches = [r for r in rows if r['reserved'] is not None and r['reserved'] != r['credited']]
    unreserved = [r for r in rows if r['reserved'] is None]
    errors = [abs(r['reserved_error_ms']) for r in rows if r['reserved_error_ms'] is not None]
    error_bins = {'0-5':0,'5-10':0,'10-20':0,'20-40':0,'40+':0}
    for error in errors:
        error_bins['0-5' if error < 5 else '5-10' if error < 10 else '10-20' if error < 20 else '20-40' if error < 40 else '40+'] += 1
    multicredit = [g for g in by_time.values() if len(g) > 1]
    distinct_axis = [g for g in multicredit if len({r['axis'] for r in g}) > 1]
    return {
        'hits':env.stats.hits,'misses':env.stats.misses,'x_accuracy':env.stats.x_accuracy_percent,
        'too_early':env.stats.too_early_presses,'overload':env.stats.overloaded,
        'targets':total,'logical_keydowns':len(rows),'physical_axis_strokes':len(by_stroke),
        'distinct_physical_timestamps':len(by_time),
        'simultaneous_timestamp_groups':len(multicredit),
        'multi_contact_strokes':len(paired),
        'secondary_contact_hit_count':sum(len(g)-1 for g in paired),
        'same_time_multi_axis_groups':len(distinct_axis),
        'same_time_multi_axis_timestamps_ms':[t/1000 for t,g in by_time.items() if len(g)>1 and len({r['axis'] for r in g})>1],
        'matches':len(matched),'mismatches':len(mismatches),'unreserved':len(unreserved),
        'mismatch_type_counts':dict(Counter(r['classification'] for r in mismatches)),
        'mismatch_ordinal_displacement':dict(Counter(str(r['credited']-r['reserved']) if r['credited'] is not None else 'none' for r in mismatches)),
        'mismatch_rows':mismatches,'unreserved_rows':unreserved,
        'planned_error_mae_ms':mean(errors) if errors else None,
        'planned_error_max_abs_ms':max(errors) if errors else None,
        'planned_error_absolute_percentiles_ms':{str(p):percentile(errors,p) for p in (50,90,95,99,100)},
        'planned_error_abs_bins_ms':error_bins,
        'simultaneous_groups':[
            {'time_ms':t/1000,'rows':g} for t,g in sorted(by_time.items()) if len(g)>1
        ],
        'reservations_paired':teacher.two_contact_reservations,
    }

def audit(segment, mode):
    teacher = ContactOracleTeacher(mode)
    key_map = {key:(axis,i) for axis,keys in teacher.channels for i,key in enumerate(keys)}
    env = CoupledContactRealChartEnv(segment,control_dt_s=CONTROL_DT,physics_dt_s=PHYSICS_DT)
    obs=env.reset()
    rows=[]
    max_steps=int((segment.duration_s+1)/CONTROL_DT)+4
    for tick in range(max_steps):
        first=env.privileged_next_target()
        remaining=(tuple((int(t.ordinal),float(t.episode_time_s)) for t in segment.targets[int(first.ordinal):]) if first is not None else ())
        action=teacher.action(obs,env.privileged_episode_time_s(),remaining)
        plans={axis:tuple(plan.tokens) for axis,plan in teacher.reservations.items()}
        start=len(env.judgement_trace)
        step=env.step(action)
        judged=env.judgement_trace[start:]
        downs=[e for e in env.last_events if e.event is KeyEvent.DOWN]
        if len(judged)!=len(downs):
            raise AssertionError('trace incomplete: terminal event handling needs review')
        for event,info in zip(downs,judged):
            if info['key']!=event.key or abs(info['time_s']-event.time_s)>1e-8:
                raise AssertionError('scorer and Motor trace do not align')
            axis,contact_idx=key_map[event.key]
            tokens=plans.get(axis,())
            reserved=tokens[contact_idx] if contact_idx < len(tokens) else None
            credited=info['target_index']
            error=(event.time_s-segment.targets[reserved].episode_time_s)*1000 if reserved is not None else None
            classification=('unreserved' if reserved is None else 'matched' if reserved==credited
                            else 'next_target' if credited==reserved+1 else 'previous_target' if credited==reserved-1
                            else 'other_target' if credited is not None else 'not_credited')
            rows.append({'frame':tick,'time_us':round(event.time_s*1e6),'time_ms':round(event.time_s*1000,6),
                         'key':event.key,'axis':axis,'contact_index':contact_idx,
                         'paired_reserved':len(tokens)==2,'reserved':reserved,'credited':credited,
                         'margin':info['margin'],'reserved_error_ms':error,'classification':classification})
        obs=step.observation
        if step.done:
            break
    else:
        raise RuntimeError('chart did not finish')
    if env.stats.hits+env.stats.misses!=len(segment.targets):
        raise AssertionError('unresolved target')
    summary=summarize(rows,len(segment.targets),env,teacher)
    if summary['matches']+summary['mismatches']+summary['unreserved']!=len(rows):
        raise AssertionError('attribution accounting failed')
    if summary['physical_axis_strokes']+summary['secondary_contact_hit_count']!=len(rows):
        raise AssertionError('contact accounting failed')
    return {'summary':summary,'all_keydowns':rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('chart',type=Path)
    ap.add_argument('--out',type=Path,default=Path('u1-keydown-attribution.json'))
    ap.add_argument('--modes',type=int,nargs='+',default=[4,5,6,7,8])
    ap.add_argument('--allow-other-chart',action='store_true')
    args=ap.parse_args()
    if args.out.exists():
        ap.error(f'Output already exists: {args.out}')
    if any(mode not in MODES for mode in args.modes):
        ap.error('modes must be from 4 5 6 7 8')
    data=args.chart.read_bytes()
    digest=sha256(data).hexdigest()
    if digest!=EXPECTED_U1_SHA and not args.allow_other_chart:
        ap.error(f'U1 SHA256 mismatch: {digest}; use --allow-other-chart explicitly')
    segment=build_playable_segment(compile_adofai(parse_adofai_text(data.decode('utf-8-sig'))))
    report={'chart_sha256':digest,'oracle_motor_only':True,'no_learning':True,
            'not_actual_multipress':True,'modes':{}}
    for mode in args.modes:
        report['modes'][str(mode)]=audit(segment,mode)
        s=report['modes'][str(mode)]['summary']
        print(f'{mode}K: H={s["hits"]} strokes={s["physical_axis_strokes"]} '
              f'pairs={s["multi_contact_strokes"]} mismatch={s["mismatches"]} '
              f'unreserved={s["unreserved"]}',flush=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.out)
if __name__=='__main__':
    main()