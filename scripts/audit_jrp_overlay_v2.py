"""Read-only JRP simulated overlay hook audit (hypothetical all-hit replay).

The renderer is fed every target as though a native OnHit hook fired. This
is *not* a HUD-only Timing forecast and *not* an actual JRP live replay.
Pseudo angleLength projection from compiled DMDOD is not Unity scrFloor parity.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.features.jrp_overlay_v2 import JrpOverlayRenderer,JRP_OVERLAY_FEATURE_VERSION


def audit(chart:Path, end_s:float)->dict:
    segment=build_playable_segment(compile_adofai(load_adofai(str(chart))), end_s=end_s)
    counters=Counter();samples=[]
    for mode in ('normal','jongyeol'):
        renderer=JrpOverlayRenderer.from_segment(segment,mode=mode)
        shown=renderer.on_show(0)
        counters[f'{mode}_initial_shows']+=1
        for target in segment.targets:
            old=shown
            shown=renderer.on_hit_hook(target.floor_index)
            diag=renderer.diagnostic
            counters[f'{mode}_hook_calls']+=1
            if diag and diag.source_floor==target.floor_index:
                counters[f'{mode}_floor_used_in_new_text']+=1
            if old==shown:
                counters[f'{mode}_unchanged_text']+=1
            else:
                counters[f'{mode}_changed_text']+=1
            if diag and diag.pseudo:
                counters[f'{mode}_pseudo_active_at_hook']+=1
                if len(samples)<8:
                    samples.append({'mode':mode,'at_floor':target.floor_index,
                                    'rendered_floor':diag.source_floor,
                                    'text':shown,'aggregate_count':diag.pseudo_count})
    return {'feature':JRP_OVERLAY_FEATURE_VERSION,'clip_s':end_s,
        'chart':chart.name,'hypothetical_accepted_hits':len(segment.targets),
        'counters':dict(counters),'sample_pseudo':samples,
        'limitations':['Every target triggers a hypothetical Hit hook; actual game/physics input not reproduced.',
        'Pseudo angleLength inferred from DMDOD compiled floor timing, not native Unity scrFloor.',
        'Source floor times used in renderer only; never policy inputs.',
        'The displayed numbers are not Student Timing accuracy evidence.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('chart',type=Path)
    p.add_argument('--end-s',type=float,default=65.);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.chart,a.end_s)
    if a.output.exists():p.error('refusing to overwrite old research')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(r,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
