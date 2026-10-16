"""Reproducible physical-Motor/JRP display-feedback audit (no trained Student).

Deliberately a small *fixed fixture* to isolate the causal display contract.
Actions from ScheduledMotorFixture are scripted using known nominal timing;
they are NOT predicted by a Student and are not claims of autonomous play.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.evaluation.jrp_closed_loop_v1 import (
    run_jrp_closed_loop, ScheduledMotorFixture, IdlePolicy,
)


PULSES = ((.30,.70),(2.24,2.64),(2.80,3.20),(3.55,3.95),(4.55,4.95))


def fixture():
    chart = parse_adofai_text(json.dumps({
        'angleData': [0,180,180,90,270,90,270],
        'settings': {'bpm':120,'pitch':100,'countdownTicks':0},
        'actions': [{'floor':1,'eventType':'Pause','duration':2}],
    }))
    return build_playable_segment(compile_adofai(chart))


def audit() -> dict:
    seg=fixture()
    scenarios={
        'scripted_motor_text_delay_0ms':(ScheduledMotorFixture(PULSES),0),
        'scripted_motor_text_delay_10ms':(ScheduledMotorFixture(PULSES),1),
        'scripted_motor_text_delay_30ms':(ScheduledMotorFixture(PULSES),3),
        'scripted_motor_skip_first_hit':(ScheduledMotorFixture(PULSES[1:]),0),
        'idle_all_miss':(IdlePolicy(),0),
    }
    rows={name:asdict(run_jrp_closed_loop(seg,policy,policy_label=name,
                   visible_delay_frames=delay)) for name,(policy,delay) in scenarios.items()}
    zero=rows['scripted_motor_text_delay_0ms']
    delayed=rows['scripted_motor_text_delay_10ms']
    slower=rows['scripted_motor_text_delay_30ms']
    skipped=rows['scripted_motor_skip_first_hit']
    idle=rows['idle_all_miss']
    good=(all(rows[name]['actual_hits']==5 and rows[name]['actual_misses']==2
              for name in ('scripted_motor_text_delay_0ms',
                           'scripted_motor_text_delay_10ms',
                           'scripted_motor_text_delay_30ms'))
          and zero['unique_jrp_candidates']==2
          and delayed['unique_jrp_candidates']==2
          and slower['unique_jrp_candidates']==0
          and all(row['false_jrp_candidates']==0 and row['false_jrp_forecasts']==0
                  for row in rows.values())
          and skipped['actual_hits']==4
          and skipped['unique_jrp_candidates']<zero['unique_jrp_candidates']
          and idle['actual_hits']==0 and idle['actual_display_hit_hooks']==0)
    return {
        'version': 'stage-t-jrp-physical-closed-loop-audit-v1',
        'status': 'PASS' if good else 'FAIL',
        'scope': 'DMDOD simulated 8-key physical motor and JRP text feedback',
        'trained_student_checkpoint_loaded': False,
        'motor_policy': 'fixed script (uses preselected timing), NOT Student',
        'source_chart': 'embedded 120 BPM fixture with 2-beat Pause',
        'scenarios': rows,
        'limitations': [
            'No trained Student weights available in this runtime; not an autonomous Student success benchmark.',
            'Accepted DMDOD Hit approximates native ADOFAI JRP OnHit; parity not proven.',
            'The script pulses were selected before rollout with chart timing knowledge, and are evaluator-side only.',
            'The observed CBPM candidates in this fixture are already covered by v1.4; JRP adds zero new target coverage.',
            '30ms delay causes conservative abstention on this fixture, not a general 30ms guarantee.',
            'Baseline v1.4 Stage A FAIL, 263D HUD and Student checkpoints remain unchanged.',
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('refusing to overwrite audit file')
    report=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps({'status':report['status'],
        'scenarios':{name:{k:row[k] for k in ('actual_hits','actual_misses',
            'actual_display_hit_hooks','unique_jrp_candidates','matched_jrp_candidates',
            'false_jrp_candidates','unique_jrp_forecasts')}
            for name,row in report['scenarios'].items()}},indent=2))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__': main()
