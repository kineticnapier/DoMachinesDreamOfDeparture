from __future__ import annotations
"""Normalize actual H2.3 Motor event traces without concealing deferred replans.

Every Motor outcome is preserved, including out-of-order planning decisions.
The old H2.2 strict zero-replan exporter is intentionally NOT used here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FORMAT = 'dmdogo-h23-executed-motor-v1'
VARIANTS=('interval_only','right_after_long_rest')


def compile_report(summary: dict, telemetry: dict, chart_bytes: bytes, chart_name: str, *, variant: str) -> dict:
    if variant not in VARIANTS: raise ValueError('unsupported H2.3 variant')
    if summary.get('planning_mode')!='gesture_v08' or summary.get('chart')!=chart_name:
        raise ValueError('unexpected Motor planning mode/chart')
    digest=hashlib.sha256(chart_bytes).hexdigest()
    if summary.get('chart_sha256')!=digest:raise ValueError('chart checksum mismatch')
    n=summary.get('total_events')
    if type(n) is not int or n<1:raise ValueError('invalid event count')
    raw=telemetry.get('decisions')
    phys=telemetry.get('physical')
    if not isinstance(raw,list) or not isinstance(phys,dict) or len(raw)!=n:
        raise ValueError('incomplete Motor telemetry')
    selection_order=[d.get('event_index') for d in raw]
    if any(type(i) is not int for i in selection_order) or sorted(selection_order)!=list(range(n)):
        raise ValueError('duplicate/missing Motor event indices')
    decisions=sorted(raw,key=lambda d:d['event_index'])
    last_due=-float('inf')
    for d in decisions:
        keys=d.get('keydowns')
        due=d.get('due_ms')
        if not isinstance(keys,list) or len(keys)!=1 or not math.isfinite(float(due)) or due<=last_due:
            raise ValueError('unsupported chords, incomplete keydown associations or non-monotone due time')
        last_due=due
        axis=keys[0].get('axis')
        if (axis not in tuple(f'{h}_{i}' for h in ('left','right') for i in range(1,5))
            or d.get('chosen')!=axis or d.get('axes')!=[axis]
            or d.get('outcome') not in ('Hit','Miss','Unresolved')):
            raise ValueError('invalid/ambiguous Motor event result')
        if not math.isfinite(float(keys[0]['episode_s'])):
            raise ValueError('invalid actual keydown time')
    stats=summary.get('stats')
    if not isinstance(stats,dict):raise ValueError('missing stats')
    replans=sum(d.get('local_replan') is True for d in decisions)
    if phys.get('local_replans')!=replans:
        raise ValueError('local replan provenance inconsistent')
    all_hit=(summary.get('termination_reason')=='chart_complete'
             and stats.get('targets')==n and stats.get('hits')==n and stats.get('misses')==0
             and stats.get('too_early')==0 and stats.get('overloaded') is False
             and stats.get('physical_keydowns')==n
             and phys.get('unassigned_events')==0
             and all(d.get('outcome')=='Hit' for d in decisions))
    return {
        'format':FORMAT,'stage':'executed','candidate_id':'h23-'+variant,'h23_variant':variant,
        'chart':chart_name,'chart_sha256':digest,'chart_event_count':n,
        'termination_reason':summary.get('termination_reason'),
        'stats':stats,'physical':phys,'physical_local_replan_count':replans,
        'original_selection_event_order':selection_order,
        'out_of_event_order_selection_links':sum(x>y for x,y in zip(selection_order,selection_order[1:])),
        'decisions':decisions,
        'h23_all_hit_verified':all_hit,
        'h23_exact_original_plan_verified':all_hit and replans==0,
        'source':'H2.3 frozen H2.1 rule, original v0.8 physical Motor, Oracle timing',
        'human_target_log_used':False,
        'hud_only_student_verified':False,'human_likeness_verified':False,
        'limitations':['Motor event-keydown matching is diagnostic, not independent ADOFAI game judgment.',
                       'Event rows sorted for comparison; original selection order retained as evidence.',
                       'Local physical replans are preserved. All Hit is distinct from exact-plan execution.',
                       'Timing uses privileged Oracle, not HUD-only Student.'],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path)
    p.add_argument('--telemetry',required=True,type=Path)
    p.add_argument('--chart',required=True,type=Path)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('refusing to overwrite output')
    summary=json.loads(args.summary.read_text(encoding='utf-8-sig'))
    telemetry=json.loads(args.telemetry.read_text(encoding='utf-8-sig'))
    result=compile_report(summary,telemetry,args.chart.read_bytes(),args.chart.name,variant=args.variant)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'variant':args.variant,'all_hit':result['h23_all_hit_verified'],
                      'exact_plan':result['h23_exact_original_plan_verified'],
                      'local_replans':result['physical_local_replan_count'],
                      'event_selection_out_of_order_links':result['out_of_event_order_selection_links']},ensure_ascii=False))


if __name__=='__main__':main()
