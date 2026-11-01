from __future__ import annotations

"""H2.3 isolated, predeclared, frozen H2.1 hand-rule Motor interventions.

Never reads H2.2 human input log or fits on it. Original runner/planner are
not edited. All timing is offline Oracle; not HUD-only. Keep the no-reset
arm as control and the right-at-long-rest arm as a separate predeclared arm.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08, GesturePlan
import run_dmdogo_v06_pattern_first as runner

FORMAT = 'dmdogo-h23-frozen-h21-hand-choice-v1'
TRAIN_CHART = '02971 - ALL MY FELLAS - RetroBean.adofai'
TRAIN_SHA256 = 'e142af91b1784dc20c76fbde5a7320de234dbaebf9822f75e733d0b2db1924d5'
H21_REPORT_SHA256 = '9dd9ddcaae7213a55e5d07bbc5c313d6acc1bef6b04163c5378ecf8730acc92c'
BINS = ('<=20', '(20,50]', '(50,100]', '(100,150]', '(150,300]', '>300')
FROZEN_SWITCH = dict(zip(BINS, (False, False, True, True, False, False)))
VARIANTS = ('interval_only', 'right_after_long_rest')


def gap_bin(delta_ms: float) -> str:
    for max_ms, b in zip((20., 50., 100., 150., 300.), BINS):
        if delta_ms <= max_ms + 1e-5:
            return b
    return BINS[-1]


def propose_axes(events, *, minimum_reuse_s: float, variant: str):
    if variant not in VARIANTS:
        raise ValueError('unknown predeclared experimental arm')
    events = tuple(events)
    if not events or any(event.chord_size != 1 for event in events):
        raise ValueError('H2.3 first version supports nonempty single-contact charts only')
    if any(b.due_s <= a.due_s for a,b in zip(events, events[1:])):
        raise ValueError('H2.3 reference times must be strictly increasing')
    last={f'{hand}_{i}':float('-inf') for hand in ('left','right') for i in range(1,5)}
    axes=[];hands=[];reasons=[]
    for index,event in enumerate(events):
        due=event.due_s
        if index == 0:
            requested='right';reason='start_right'
        else:
            dt=due-events[index-1].due_s
            if variant == 'right_after_long_rest' and dt > .450:
                requested='right';reason='long_rest_training_side_prior'
            else:
                switch=FROZEN_SWITCH[gap_bin(dt*1000)]
                prev=hands[-1]
                requested=('left' if prev=='right' else 'right') if switch else prev
                reason='frozen_h21_interval_rule'
        hand=requested
        ready=[f'{hand}_{i}' for i in range(1,5)
               if due-last[f'{hand}_{i}'] + 1e-10 >= minimum_reuse_s]
        if not ready:
            hand='left' if hand=='right' else 'right'
            ready=[f'{hand}_{i}' for i in range(1,5)
                   if due-last[f'{hand}_{i}'] + 1e-10 >= minimum_reuse_s]
            reason='physical_cooldown_forced_other_hand'
        if not ready:
            raise RuntimeError(f'no safe axis at event {index}')
        axis=ready[0]
        last[axis]=due
        axes.append((axis,));hands.append(hand);reasons.append(reason)
    return tuple(axes), tuple(reasons)


def create_intervened_plan(events, original: GesturePlan, *, minimum_reuse_s: float, variant: str) -> GesturePlan:
    if not original.feasible or len(original.assignments) != len(events):
        raise ValueError('original v08 reference plan incomplete')
    axes,reasons=propose_axes(events,minimum_reuse_s=minimum_reuse_s,variant=variant)
    indices=[e.event_index for e in events]
    if indices!=list(range(len(events))):
        raise ValueError('events must be indexed sequentially')
    chunks=[]
    for chunk in original.chunks:
        s,e=chunk.start_event_index,chunk.end_event_index
        if (s<0 or e>=len(axes) or chunk.event_count!=e-s+1):
            raise ValueError('original v08 chunk segmentation inconsistent')
        chunks.append(replace(chunk,gesture_family='h23_frozen_interval',axes=axes[s:e+1]))
    if sum(c.event_count for c in chunks)!=len(axes):
        raise ValueError('original chunks do not cover all new events')
    counters={key:reasons.count(key) for key in sorted(set(reasons))}
    byhand={'left':sum(row[0].startswith('left_') for row in axes),
            'right':sum(row[0].startswith('right_') for row in axes)}
    return GesturePlan(feasible=True,assignments=axes,chunks=tuple(chunks),
        failure_reason=None,candidates_evaluated=original.candidates_evaluated,
        metrics={'H23':{'new_hand_usage':byhand,'event_count':len(events),
                        'reasons':counters,'rule_switch_by_interval':FROZEN_SWITCH,
                        'variant':variant},
                 'original_v08_metrics_for_reference_only':original.metrics},
        model_contract={'format':FORMAT,'source_h21_chart':TRAIN_CHART,
                        'source_h21_chart_sha256':TRAIN_SHA256,
                        'source_h21_report_sha256':H21_REPORT_SHA256,
                        'variant':variant,'right_rest_threshold_s':(.450 if variant==VARIANTS[1] else None),
                        'minimum_reuse_s':minimum_reuse_s,
                        'h21_switch_by_interval':FROZEN_SWITCH,
                        'human_data_on_target_chart_used':False,
                        'offline_oracle_timing_used':True,'hud_only':False,
                        'physical_motor_verified':False,
                        'original_planner_and_runner_unchanged':True})


def run_one(*,chart:Path,output_dir:Path,variant:str,wall_timeout_s:float):
    class IntervenedPlanner(GestureChunkPlannerV08):
        def plan(self, events):
            events=tuple(events)
            original=super().plan(events)
            return create_intervened_plan(events,original,minimum_reuse_s=self.minimum_reuse_s,variant=variant)
    with patch.object(runner,'GestureChunkPlannerV08',IntervenedPlanner):
        report=runner.run(chart,output_dir,
            minimum_reuse_s=.080,planning_mode='gesture_v08',wall_timeout_s=wall_timeout_s)
    data={'format':FORMAT,'variant':variant,
          'source_chart_sha256':sha256(chart.read_bytes()).hexdigest(),
          'source_h21_report_sha256':H21_REPORT_SHA256,
          'h21_rule':FROZEN_SWITCH,
          'run_termination_reason':report['termination_reason'],
          'motor_stats':report['stats'],'physical':report['physical'],
          'original_runner_modified':False,'human_target_log_used':False,
          'human_likeness_verified':False,
          'training_performed':False}
    with (output_dir/'h23-branch-metadata.json').open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chart',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--wall-timeout-s',type=float,default=900)
    args=p.parse_args()
    output_dir=args.output_root / args.variant
    if output_dir.exists():p.error('refusing to overwrite existing output')
    info=run_one(chart=args.chart,output_dir=output_dir,variant=args.variant,wall_timeout_s=args.wall_timeout_s)
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':
    main()
