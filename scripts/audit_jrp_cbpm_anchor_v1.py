"""JRP CBPM shadow on idle simulator; synthetic all-Hit hook scenario.

All chart target times enter the renderer/hook generator and the POSTHOC
validation only. Predictor inputs are 263D HUD, rendered strings, baseline
HUD-only v1.4 events, frame ordinal. This is *not* a gameplay success run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv,encode_n_key_hud_real_chart_observation
from dmdod.features.jrp_overlay_v2 import JrpOverlayRenderer
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
from dmdod.planning.jrp_cbpm_anchor_v1 import JrpCbpmAnchoredTracker,VERSION


def audit(chart:Path, end_s:float, *, mode:str='normal', miss_ordinals=())->dict:
    seg=build_playable_segment(compile_adofai(load_adofai(str(chart))),end_s=end_s)
    env=DiagnosticHudNKeyRealChartMotorEnv(seg,key_count=8,control_dt_s=.010,physics_dt_s=.001,tail_s=.01)
    obs=env.reset();idle=NKeyAction((-1.,)*8)
    render=JrpOverlayRenderer.from_segment(seg,mode=mode)
    render.on_show(0)
    baseline=CausalTimingTracker()
    anchor=JrpCbpmAnchoredTracker()
    targets=seg.targets
    ti=0;preds=[];baseline_ends=[];cnt=Counter();samples=[];horizons=[];anchor_logs=[];last_due=None
    for frame in range(int(end_s/.01)+5):
        now=frame*.01
        while ti<len(targets) and targets[ti].episode_time_s<=now+1e-9:
            if ti not in miss_ordinals:
                render.on_hit_hook(targets[ti].floor_index)
                cnt['synthetic_hit_hooks']+=1
            else:
                cnt['synthetic_skipped_hooks']+=1
            ti+=1
        hud=tuple(encode_n_key_hud_real_chart_observation(obs))
        b=baseline.observe(frame,hud)
        baseline_ends.extend(e.due_s for e in b.elapsed)
        a=anchor.observe(frame,hud,render.visible,b.elapsed)
        cnt['frames']+=1
        cnt['baseline_eligible']+=b.status in ('eligible','eligible_partial')
        cnt[f'anchor_{a.status}']+=1
        if a.due_s is not None and a.due_s != last_due:
            anchor_logs.append({'frame':frame,'due':a.due_s,'source_floor_posthoc':render.diagnostic.source_floor if render.diagnostic else None, 'pseudo_posthoc':render.diagnostic.pseudo if render.diagnostic else None,'baseline_elapsed_dues':[e.due_s for e in b.elapsed],'text':render.visible,'next_truth_ord':ti})
            last_due=a.due_s
        if a.eligible:
            preds.append((frame,a.due_s,a.uncertainty_s))
        if len(samples)<8 and a.eligible:
            samples.append({'frame':frame,'due':a.due_s,'uncertainty_ms':a.uncertainty_s*1000,
                            'text':render.visible})
        if frame==int(end_s/.01)+4:break
        step=env.step(idle)
        obs=step.observation
        if step.done:break
    truths=[t.episode_time_s for t in targets]
    due_unique=[]
    for frame,due,uncertainty in preds:
        if due > end_s:continue
        if not any(abs(p-due)<.00075 for p in due_unique):due_unique.append(due)
    def assign(predictions, truth, *, tolerance=.00005):
        # Greedy time-ordered one-to-one matching; a forecast cannot score
        # multiple adjacent notes, even if they are <0.75ms apart.
        remaining=set(range(len(truth)))
        matched=[]
        bad=[]
        for due in sorted(predictions):
            closest=min(remaining,key=lambda i:abs(truth[i]-due)) if remaining else None
            if closest is not None and abs(truth[closest]-due)<=tolerance:
                remaining.remove(closest)
                matched.append((closest,due))
            else:
                bad.append(due)
        return matched,bad

    base_matched,base_bad=assign(baseline_ends,truths)
    overlay_matched,overlay_bad=assign(due_unique,truths)
    supplemental=[truths[i] for i,_ in overlay_matched if i not in {k for k,_ in base_matched}]
    bad=[]
    for due in overlay_bad:
        close=min(truths,key=lambda t:abs(t-due))
        bad.append({'forecast_due':due,'nearest_truth':close,'error_ms':(due-close)*1000,
                    'source':next(({'frame':f,'quant_error_ms':e*1000} for f,d,e in preds if abs(d-due)<1e-8),None)})
    return {'version':VERSION,'chart':chart.name,'end_s':end_s,'mode':mode,
        'synthetic_skipped_ordinals':list(miss_ordinals),'true_targets':len(truths),
        'counts':dict(cnt),'baseline_elapsed':len(baseline_ends),
        'baseline_matched':len(base_matched),'cbpm_unique_forecasted':len(due_unique),
        'cbpm_matched':len(overlay_matched),'cbpm_unmatched':len(overlay_bad),
        'additional_targets_covered':len(supplemental),'additional_truth_times':supplemental,'bad_forecasts':bad,'combined_covered':len({k for k,_ in base_matched+overlay_matched}),
        'sample':samples,'example_anchors':anchor_logs[:5],
        'extra_target_anchors':[x for x in anchor_logs if any(abs(x['due']-t)<.00005 for t in supplemental)],
        'false_anchors': [x for x in anchor_logs if any(abs(x['due']-b['forecast_due'])<.001 for b in bad)],
        'limitations':[
            'Renderer is driven by oracle-scheduled hypothetical Hit hooks; not autonomous Student/real Motor play.',
            'Base Timing gets HUD only; CBPM receives only rendered text, baseline elapsed events and HUD.',
            'Causal eligibility is conditional on a text-changing hit and a matching visible floor transition.',
            'JRP game hook parity and pseudo angleLength parity are not established.',
        ]}


def main():
    p=argparse.ArgumentParser();p.add_argument('chart',type=Path);p.add_argument('--end-s',type=float,default=65.)
    p.add_argument('--mode',choices=('normal','jongyeol'),default='normal');p.add_argument('--skip',type=int,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.chart,a.end_s,mode=a.mode,miss_ordinals=tuple(a.skip))
    if a.output.exists():p.error('refusing to overwrite previous audit')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
