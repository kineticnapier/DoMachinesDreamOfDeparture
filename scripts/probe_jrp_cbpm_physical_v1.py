"""Very small physical motor CPU probe for JRP CBPM guard behavior."""
from __future__ import annotations
import json
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.jrp_overlay_diagnostic import JrpOverlayDiagnosticNKeyEnv
from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
from dmdod.planning.jrp_cbpm_anchor_v1 import JrpCbpmAnchoredTracker
from dmdod.motor.n_key import NKeyAction


def probe(second_start):
 raw=json.dumps({'angleData':[0,180,180,90,270,90,270],
  'settings':{'bpm':120,'pitch':100,'countdownTicks':0},
  'actions':[{'floor':1,'eventType':'Pause','duration':2}]})
 seg=build_playable_segment(compile_adofai(parse_adofai_text(raw)))
 env=JrpOverlayDiagnosticNKeyEnv(seg,key_count=8,tail_s=.5)
 obs=env.reset();b=CausalTimingTracker();j=JrpCbpmAnchoredTracker();anchors=[];updates=[];prev=''
 for f in range(550):
  hud=tuple(encode_n_key_hud_real_chart_observation(obs))
  base=b.observe(f,hud)
  output=j.observe(f,hud,env.overlay_visible,base.elapsed)
  if output.due_s is not None and (not anchors or anchors[-1][1]!=output.due_s):
   anchors.append((f,output.due_s,env.overlay_visible))
  if env.overlay_visible!=prev:
   updates.append((f,env.overlay_visible,env.stats.hits))
   prev=env.overlay_visible
  t=f*.01
  enabled=any(lo<=t<hi for lo,hi in ((.30,.70),(second_start,second_start+.4),(2.80,3.20),(3.55,3.95),(4.55,4.95)))
  res=env.step(NKeyAction((1.,)+(-1.,)*7 if enabled else (-1.,)*8))
  obs=res.observation
  if res.done:break
 return {'second_start':second_start,'hits':env.stats.hits,'second_hook':next((x for x in updates if x[-1]==2),None),'anchors':anchors}

if __name__=='__main__':
 for start in (2.18,2.20,2.22,2.24,2.26,2.28,2.30):
  print(probe(start))
