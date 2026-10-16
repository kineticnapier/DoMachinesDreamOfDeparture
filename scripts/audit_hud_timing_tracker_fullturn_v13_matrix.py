"""252 chart × tempo × Twirl HUD-only full-turn observational matrix.

Run on CPU. Build and simulate each generated chart at 1ms physics / 10ms
HUD intervals. All predictions use HUD only. Read exact targets *after*
forecasting/track reconciliation for posthoc scoring. Write-once output.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv,encode_n_key_hud_real_chart_observation
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_fullturn_v13 import CausalTimingTracker,forecast_visible_window

PATTERNS=(
    (0.,90.,180.,270.),(0.,30.,90.,180.),(90.,180.,270.,0.),
    (0.,0.,90.,90.,180.,180.),(0.,45.,180.,315.),(0.,0.,0.,0.),
    (0.,180.),(90.,270.),(0.,0.,180.,0.),(90.,90.,0.,180.),
    (0.,180.,0.,180.),(45.,225.),(0.,90.,0.,90.),(0.,180.,90.,270.),
)
BPM=(496.,992.,1984.,2380.,4000.,6000.,8000.,12000.,20000.)
TOL_S=50e-6
BOUNDARY_EPS_S=1e-6  # Strict frame-boundary events treated as occurring now.


def capture(pattern,bpm,twirl):
    actions=[{'floor':1,'eventType':'Twirl'}] if twirl else []
    txt=json.dumps({'angleData':list(pattern)*16,
                    'settings':{'bpm':bpm,'pitch':100.,'countdownTicks':0},
                    'actions':actions})
    segment=build_playable_segment(compile_adofai(parse_adofai_text(txt)))
    env=DiagnosticHudNKeyRealChartMotorEnv(segment,key_count=8,
             control_dt_s=.010,physics_dt_s=.001,
             behind_floors=FLOORS.behind_floors,ahead_floors=FLOORS.ahead_floors,
             tail_s=.04)
    idle=NKeyAction((0.,)*8)
    obs=env.reset();hud=[]
    for i in range(2000):
        hud.append(tuple(encode_n_key_hud_real_chart_observation(obs)))
        transition=env.step(idle);obs=transition.observation
        if transition.done: break
    else: raise RuntimeError('unbounded simulated chart')
    return segment,hud


def run_matrix():
    total=Counter(); failures=[]
    for pattern in PATTERNS:
        for bpm in BPM:
            for twirl in (False,True):
                segment,frames=capture(pattern,bpm,twirl)
                tracker=CausalTimingTracker(); predictions=[]; elapsed=[]
                prev=None
                for i,hud in enumerate(frames):
                    pred=forecast_visible_window(hud,previous_hud=prev)
                    predictions.append(pred)
                    update=tracker.observe(i,hud)
                    elapsed.extend(update.elapsed)
                    total['id_reuses']+=len(update.reused_ids)
                    prev=hud
                # ONLY POSTHOC: source-chart target times are never used
                # while calculating the above predictions or tracker IDs.
                truth=tuple(float(t.episode_time_s) for t in segment.targets)
                for i,p in enumerate(predictions):
                    if not p.eligible: continue
                    total['eligible_frames']+=1
                    start=i*.010
                    until=p.covered_until_s if p.covered_until_s is not None else .030
                    # An event within 1us of either window boundary is
                    # observationally tied to that boundary. Exclude this
                    # sliver ONLY from coverage counting, while checking
                    # every *emitted* event for truth correspondence.
                    lo=bisect_right(truth,start+BOUNDARY_EPS_S)
                    hi=bisect_right(truth,start+until-BOUNDARY_EPS_S)
                    expected=[t-start for t in truth[lo:hi]]
                    predicted=[t for t in p.times_s
                               if BOUNDARY_EPS_S<t<until-BOUNDARY_EPS_S]
                    emitted_valid=all(any(abs(t-(start+dt))<=TOL_S
                                          for t in truth) for dt in p.times_s)
                    if (len(predicted)!=len(expected) or
                        any(abs(x-y)>TOL_S for x,y in zip(expected,predicted)) or
                        not emitted_valid):
                        failures.append({'bpm':bpm,'pattern':list(pattern),'twirl':twirl,
                                         'frame':i,'expected':expected,'predicted':predicted,
                                         'emitted_valid':emitted_valid})
                for ev in elapsed:
                    total['elapsed_events']+=1
                    j=bisect_right(truth,ev.due_s)
                    nearest=truth[max(0,j-1):min(len(truth),j+1)]
                    if any(abs(t-ev.due_s)<=TOL_S for t in nearest):
                        total['matched_events']+=1
                    else:
                        failures.append({'bpm':bpm,'pattern':list(pattern),'twirl':twirl,
                                         'unmatched_due_s':ev.due_s})
                total['cases']+=1
                total['true_events']+=len(truth)
                total['observed_frames']+=len(frames)
    return {'format':'dmdod-stage-t-fullturn-v13-252case-matrix',
            'predictor_truth_input':False,'student_training':False,
            'status':'PASS' if not failures else 'FAIL',
            'counts':dict(total),'errors':failures[:20],
            'notes':'Events within 1us of either window boundary excluded from coverage count. '
                    'Scoring is posthoc; physical Hit/Miss not evaluated.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refuse overwrite')
    result=run_matrix()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
