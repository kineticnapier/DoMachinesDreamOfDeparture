#!/usr/bin/env python3
"""Research A001 *post hoc* start-transient diagnosis of frozen v1 selections.

Does not modify the original study results, scoring, controller candidate grid,
Motor constants, or study-002 inputs. Evaluates four preselected v1 rows.
The first four/eight targets are inspected as explicit post hoc sensitivity
checks, not as a retroactive change to the success criterion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dmdod.motor.env import MotorEnv
from a001_periodic_motor_probe import (
    CONTROL_DT_S, JUDGEMENT_WINDOW_S, TAIL_S, Trial,
    periodic_command_at, target_times,
)

# Post-hoc chosen from CI run 41287352794. Fixed before this diagnostic run.
SELECTIONS = (
    ("single_left", 180, .55, 80),
    ("single_left", 500, .40, 170),
    ("single_right", 500, .40, 170),
    ("alternating", 80, .55, 80),
    ("alternating", 220, .40, 170),
    ("alternating", 500, .25, 230),
)
START_S = 1.130
DURATION_S = 7.200


def assign_events(targets, downs, window=JUDGEMENT_WINDOW_S):
    """Return each target's hit/miss; count extras separately."""
    if not targets:
        raise ValueError("empty targets")
    first, last = targets[0] - window, targets[-1] + window
    relevant = sorted(
        ((when, key) for when, key in downs
         if first - 1e-9 <= when <= last + 1e-9),
        key=lambda p: p[0],
    )
    matches = [None] * len(targets)
    extra = []
    cursor = 0
    for time_s, key in relevant:
        while cursor < len(targets) and targets[cursor] < time_s - window - 1e-9:
            cursor += 1
        if cursor < len(targets) and abs(time_s - targets[cursor]) <= window + 1e-9:
            matches[cursor] = {
                "key": key,
                "error_ms": round((time_s - targets[cursor]) * 1000, 5),
            }
            cursor += 1
        else:
            extra.append((round(time_s, 5), key))
    return matches, extra


def run_diagnostic(trial):
    targets = target_times(trial)
    env = MotorEnv(same_hand=trial.same_hand, control_dt_s=CONTROL_DT_S)
    env.reset()
    downs, ups = [], []
    for step in range(int((targets[-1] + TAIL_S) / CONTROL_DT_S) + 1):
        action = periodic_command_at(trial, step, targets)
        r = env.step(action)
        for e in r.evaluator_events:
            (downs if e.event.value == "down" else ups).append((e.time_s, e.key))
    matches, extra = assign_events(targets, downs)
    missed = [i for i, m in enumerate(matches) if m is None]
    def suffix_cutoff(k):
        sub = matches[k:]
        return {
            "excluded_first_targets": k, "remaining_targets": len(sub),
            "hits": sum(x is not None for x in sub),
            "hit_rate": sum(x is not None for x in sub) / len(sub) if sub else None,
        }
    return {
        **asdict(trial), "target_count": len(targets),
        "hits": len(targets) - len(missed), "miss_count": len(missed),
        "extra_keydowns_in_scoring_range": len(extra),
        "missed_target_indices": missed,
        "first_12_assignments": [
            {"ordinal": i, "target_s": round(targets[i], 6), "hit": matches[i]}
            for i in range(min(len(matches), 12))
        ],
        "start_sensitivity": [suffix_cutoff(k) for k in (0, 4, 8)],
        "first_10_physical_keydowns": [
            {"time_s": round(t, 6), "key": k}
            for t, k in downs[:10]
        ],
        "first_10_keyups": [
            {"time_s": round(t, 6), "key": k}
            for t, k in ups[:10]
        ],
        "extra_keydowns_first_10": extra[:10],
        "last_fatigue": asdict(env.diagnostics()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [
        run_diagnostic(Trial(mode, interval, duty, lead, START_S, DURATION_S))
        for mode, interval, duty, lead in SELECTIONS
    ]
    report = {
        "schema": "research-a001-start-transient-post-hoc-v1",
        "source_pilot_run": "41287352794",
        "no_physical_or_pilot_parameter_changes": True,
        "status": "POST-HOC diagnostic; not an approved 001 completion",
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cases": [
            {"mode": r["mode"], "interval_ms": r["interval_ms"],
             "total_hits": r["hits"], "total_targets": r["target_count"],
             "missed_target_indices": r["missed_target_indices"],
             "start_sensitivity": r["start_sensitivity"]}
            for r in rows
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
