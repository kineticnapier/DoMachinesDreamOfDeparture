#!/usr/bin/env python3
"""Post-hoc A001b physical KeyDown phase diagnosis.

Uses selected frozen A001b pulse parameters. Does not alter the original
experiment's tests, selection, scores, model, or planned inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from dmdod.motor.env import MotorEnv
from a001_periodic_motor_probe import Trial, target_times, periodic_command_at, CONTROL_DT_S, TAIL_S
from a001_startup_diagnostic import assign_events
from a001b_steady_motor_survey import WARMUP_S, CONFIRM_STARTS_S

CASES = (
    ("single_left", 140, .55, 50),
    ("single_left", 160, .55, 80),
    ("single_left", 170, .55, 50),
    ("alternating", 70, .55, 50),
    ("alternating", 80, .55, 80),
)
SHORT_STEADY_S = 4.0
WINDOW_S = .020


def diagnose(mode, interval_ms, duty, lead_ms, start_s):
    trial = Trial(mode, interval_ms, duty, lead_ms, start_s, WARMUP_S + SHORT_STEADY_S, True)
    targets = target_times(trial)
    motor = MotorEnv(same_hand=True, control_dt_s=CONTROL_DT_S)
    motor.reset()
    downs = []
    for step in range(int((targets[-1]+TAIL_S)/CONTROL_DT_S)+1):
        action = periodic_command_at(trial, step, targets)
        for evt in motor.step(action).evaluator_events:
            if evt.event.value == "down":
                downs.append((evt.time_s, evt.key))
    matches, extras = assign_events(targets, downs, window=WINDOW_S)
    steady_idx = next(i for i,t in enumerate(targets) if t >= start_s+WARMUP_S-1e-9)
    steady_targets = targets[steady_idx:]
    steady_downs = [
        (t,k) for t,k in downs
        if steady_targets[0]-.04 <= t <= steady_targets[-1]+.04
    ]
    diffs=[]
    for down,key in steady_downs:
        idx=min(range(len(steady_targets)),key=lambda i:abs(steady_targets[i]-down))
        diffs.append((down-steady_targets[idx])*1000)
    return {
        "mode": mode, "period_ms": interval_ms, "duty": duty,
        "lead_ms": lead_ms, "start_s": start_s,
        "steady_targets": len(steady_targets),
        "steady_hits": sum(m is not None for m in matches[steady_idx:]),
        "steady_physical_keydowns": len(steady_downs),
        "steady_nearest_target_offset_median_ms": (
            round(median(diffs),4) if diffs else None
        ),
        "steady_nearest_target_offset_range_ms": (
            [round(min(diffs),4),round(max(diffs),4)] if diffs else None
        ),
        "first_8_steady_offsets_ms": [round(t,4) for t in diffs[:8]],
        "first_5_steady_keydowns_s": [
            [round(t,5),key] for t,key in steady_downs[:5]
        ],
        "first_5_steady_targets_s": [
            round(t,5) for t in steady_targets[:5]
        ],
        "out_of_window_event_count": len([
            (t,k) for t,k in extras if steady_targets[0]-.02 <= t <= steady_targets[-1]+.02
        ]),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    records=[
        diagnose(mode,t,duty,lead,start)
        for mode,t,duty,lead in CASES
        for start in CONFIRM_STARTS_S
    ]
    result={
        "schema":"research-a001b-posthoc-physical-phase-v1",
        "status":"post hoc diagnostic; no original success criteria changed",
        "original_run":"41291858751",
        "rows":records,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
