"""Offline exploratory fixed matrix; evaluate eligible forecasts posthoc only."""
from __future__ import annotations
import json,argparse
from collections import Counter
from bisect import bisect_right
from pathlib import Path
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv,encode_n_key_hud_real_chart_observation
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker,forecast_visible_window
PATS=((0.,180.),(0.,90.,180.,270.),(0.,0.,180.,90.),(0.,30.,90.,180.))
BPMS=(496.,1984.,4000.,8000.,12000.)
CASES=('speed_double','speed_half','speed_offset','pause_1','pause_half','twirl_speed','twirl_pause','speed_pause')
TOL=50e-6;EDGE=1e-6

def generated(pattern,bpm,variant):
  p=5; a=[]
  speed=lambda off=0:{'floor':p,'eventType':'SetSpeed','speedType':'Bpm','beatsPerMinute':bpm*2,'angleOffset':off}
  pause={'floor':p,'eventType':'Pause','duration':1.}
  twirl={'floor':p,'eventType':'Twirl'}
  if variant=='speed_double':a=[speed()]
  elif variant=='speed_half':a=[{**speed(),'beatsPerMinute':bpm*.5}]
  elif variant=='speed_offset':a=[speed(90.)]
  elif variant=='pause_1':a=[pause]
  elif variant=='pause_half':a=[{**pause,'duration':.5}]
  elif variant=='twirl_speed':a=[twirl,speed()]
  elif variant=='twirl_pause':a=[twirl,pause]
  elif variant=='speed_pause':a=[speed(),pause]
  config={'angleData':list(pattern)*8,'settings':{'bpm':bpm,'pitch':100.,'countdownTicks':0},'actions':a}
  return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps(config))))

def run():
 counts=Counter();bad=[]
 for pat in PATS:
  for bpm in BPMS:
   for variant in CASES:
    seg=generated(pat,bpm,variant)
    env=DiagnosticHudNKeyRealChartMotorEnv(seg,key_count=8,control_dt_s=.01,physics_dt_s=.001,behind_floors=FLOORS.behind_floors,ahead_floors=FLOORS.ahead_floors,tail_s=.04)
    obs=env.reset();idle=NKeyAction((0.,)*8);tracker=CausalTimingTracker();preds=[];elapsed=[]
    for f in range(1500):
      hud=tuple(encode_n_key_hud_real_chart_observation(obs))
      pred=forecast_visible_window(hud,previous_hud=preds[-1][1] if preds else None)
      preds.append((pred,hud));up=tracker.observe(f,hud);elapsed.extend(up.elapsed)
      st=env.step(idle);obs=st.observation
      if st.done:break
    else:raise RuntimeError('unbounded')
    times=tuple(float(t.episode_time_s) for t in seg.targets)
    for i,(pred,_) in enumerate(preds):
      if not pred.eligible:continue
      counts['eligible']+=1
      t=i*.01;end=pred.covered_until_s if pred.covered_until_s is not None else .03
      lo=bisect_right(times,t+EDGE);hi=bisect_right(times,t+end-EDGE)
      expect=[x-t for x in times[lo:hi]]
      actual=[x for x in pred.times_s if EDGE<x<end-EDGE]
      if (len(expect)!=len(actual) or any(abs(x-y)>TOL for x,y in zip(expect,actual)) or
          any(not any(abs(t+dt-tt)<=TOL for tt in times) for dt in pred.times_s)):
        bad.append({'pattern':pat,'bpm':bpm,'variant':variant,'frame':i,'expected':expect,'predicted':actual,'status':pred.status})
    for ev in elapsed:
      counts['elapsed']+=1
      k=bisect_right(times,ev.due_s)
      if any(abs(ev.due_s-t)<=TOL for t in times[max(0,k-1):min(len(times),k+1)]):counts['matched']+=1
      else:bad.append({'pattern':pat,'bpm':bpm,'variant':variant,'bad_due':ev.due_s})
    counts['cases']+=1;counts['frames']+=len(preds)
    if len(bad)>=20:break
   if len(bad)>=20:break
  if len(bad)>=20:break
 return {'format':'dmdod-stage-t-v14-special-transition-160case-matrix',
         'predictor_oracle_inputs':False,'student_training':False,
         'status':'PASS' if not bad else 'FAIL','counts':dict(counts),'mismatches':bad[:20]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():p.error('no overwrite')
 result=run();args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));
 if result['status']!='PASS':raise SystemExit(1)
