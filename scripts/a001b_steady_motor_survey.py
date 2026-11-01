#!/usr/bin/env python3
"""Research A001b: preregistered steady-state periodic Motor survey.

Protocol: docs/research-a-experiment-001b-fixed-protocol.md
THIS IS A BODY PROBE, NOT LEARNED FINGERING OR PHYSICAL IMPOSSIBILITY PROOF.

Research A001a observed startup misses; this new protocol was frozen BEFORE
the A001b workflow. The original A001a success labels are not altered.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from dmdod.motor.env import MotorEnv
from a001_periodic_motor_probe import (
    Trial, periodic_command_at, target_times, JUDGEMENT_WINDOW_S, TAIL_S,
    CONTROL_DT_S,
)
from a001_startup_diagnostic import assign_events

INTERVALS_MS = (60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
                180, 190, 200, 220, 260)
DUTIES = (.25, .40, .55, .70)
LEADS_MS = (20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350)
MODES = ("single_left", "single_right", "alternating")
WARMUP_S = 3.0
SCREEN_S = 6.0
CONFIRM_S = 18.0
SCREEN_START_S = 1.130
CONFIRM_STARTS_S = (1.137, 1.273, 1.419)
JUDGEMENT_S = .020
SAME_HAND = True


def score_trace(trial: Trial, targets: list[float], downs: list[tuple[float, str]],
                *, warmup_s: float = WARMUP_S) -> dict:
    """Score all targets exactly once; split startup/steady with no resets."""
    if not targets or not math.isfinite(warmup_s) or warmup_s <= 0:
        raise ValueError("invalid target schedule or warmup")
    matches, extras = assign_events(targets, downs, window=JUDGEMENT_S)
    start_score_s = trial.start_s + warmup_s
    first_steady = next((i for i, t in enumerate(targets)
                         if t >= start_score_s - 1e-9), len(targets))
    if first_steady == len(targets):
        raise ValueError("no steady targets")
    # Equal-time boundary cannot arise for tested intervals (minimum 60ms).
    bound = targets[first_steady] - JUDGEMENT_S
    warmup_extra = sum(t < bound for t, _ in extras)
    steady_extra = len(extras) - warmup_extra
    steady = matches[first_steady:]
    warm = matches[:first_steady]

    def part(rows: list, extra: int, *, ordinal_offset: int) -> dict:
        hits = sum(row is not None for row in rows)
        wrong = 0
        matched_keys = []
        errors = []
        for idx, item in enumerate(rows):
            if item is None:
                continue
            ordinal = idx + ordinal_offset
            expected = (
                "left" if trial.mode == "single_left"
                else "right" if trial.mode == "single_right"
                else ("left" if ordinal % 2 == 0 else "right")
            )
            wrong += item["key"] != expected
            matched_keys.append(item["key"])
            errors.append(abs(item["error_ms"]))
        return {
            "targets": len(rows),
            "hits": hits,
            "misses": len(rows) - hits,
            "hit_rate": hits / len(rows) if rows else None,
            "extra_keydowns": extra,
            "wrong_axis": wrong,
            "max_abs_error_ms": max(errors) if errors else None,
            "actual_matched_left": sum(k == "left" for k in matched_keys),
            "actual_matched_right": sum(k == "right" for k in matched_keys),
        }

    startup = part(warm, warmup_extra, ordinal_offset=0)
    steady_result = part(steady, steady_extra, ordinal_offset=first_steady)
    split_time = start_score_s + (trial.duration_s - warmup_s) / 2
    cut = next((i for i in range(first_steady, len(targets))
                if targets[i] >= split_time - 1e-9), len(targets))
    first_half_extra = sum(
        t < split_time for t, _ in extras if t >= bound - 1e-9
    )
    last_half_extra = steady_extra - first_half_extra
    first_half = part(matches[first_steady:cut], first_half_extra,
                      ordinal_offset=first_steady)
    last_half = part(matches[cut:], last_half_extra, ordinal_offset=cut)
    def no_extras_and_95(row: dict) -> bool:
        return (row["targets"] > 0 and row["hit_rate"] >= .95
                and row["extra_keydowns"] == 0 and row["wrong_axis"] == 0)
    passed = (no_extras_and_95(steady_result)
              and first_half["hit_rate"] is not None
              and last_half["hit_rate"] is not None
              and first_half["hit_rate"] >= .95
              and last_half["hit_rate"] >= .95)
    return {
        "warmup": startup, "steady": steady_result,
        "steady_first_half": first_half, "steady_second_half": last_half,
        "passed": passed,
        "first_steady_target_ordinal": first_steady,
    }


def run_trial(mode: str, interval_ms: int, duty: float, lead_ms: int,
              start_s: float, steady_s: float) -> dict:
    trial = Trial(mode, interval_ms, duty, lead_ms, start_s,
                  WARMUP_S + steady_s, SAME_HAND)
    targets = target_times(trial)
    motor = MotorEnv(same_hand=SAME_HAND, control_dt_s=CONTROL_DT_S)
    motor.reset()
    downs, ups = [], []
    stop_s = targets[-1] + TAIL_S
    for step in range(int(stop_s / CONTROL_DT_S) + 1):
        action = periodic_command_at(trial, step, targets)
        result = motor.step(action)
        for event in result.evaluator_events:
            (downs if event.event.value == "down" else ups).append(
                (event.time_s, event.key)
            )
    score = score_trace(trial, targets, downs)
    return {
        "mode": mode, "interval_ms": interval_ms,
        "duty": duty, "lead_ms": lead_ms,
        "start_s": start_s, "steady_duration_s": steady_s,
        "score": score,
        "physical_keydowns": len(downs), "physical_keyups": len(ups),
        "end_fatigue": {
            "left": motor.diagnostics().left_fatigue,
            "right": motor.diagnostics().right_fatigue,
            "hand": motor.diagnostics().hand_fatigue,
        },
    }


def candidate_order(row: dict) -> tuple:
    score = row["score"]
    steady = score["steady"]
    return (
        score["passed"],
        steady["hit_rate"],
        -steady["extra_keydowns"] / steady["targets"],
        -steady["wrong_axis"] / steady["targets"],
        -(steady["max_abs_error_ms"] if steady["max_abs_error_ms"] is not None else 1e6),
        -row["lead_ms"], -row["duty"],
    )


def survey(*, quick=False) -> dict:
    if quick:
        intervals = (80, 180)
        duties = (.40, .55)
        leads = (80, 170)
        modes = ("single_left", "alternating")
        confirm_starts = (1.137,)
        screen_s, confirm_s = 1.0, 1.2
    else:
        intervals, duties, leads = INTERVALS_MS, DUTIES, LEADS_MS
        modes, confirm_starts = MODES, CONFIRM_STARTS_S
        screen_s, confirm_s = SCREEN_S, CONFIRM_S

    rows = []
    for mode in modes:
        for interval in intervals:
            trials = [
                run_trial(mode, interval, duty, lead, SCREEN_START_S, screen_s)
                for duty in duties for lead in leads
            ]
            selected = max(trials, key=candidate_order)
            confirmations = [
                run_trial(mode, interval, selected["duty"], selected["lead_ms"],
                          start, confirm_s) for start in confirm_starts
            ]
            passes_all = all(t["score"]["passed"] for t in confirmations)
            rows.append({
                "mode": mode, "interval_ms": interval,
                "selected": selected,
                "screen_candidates": [
                    {
                        "duty": t["duty"], "lead_ms": t["lead_ms"],
                        "hit_rate": t["score"]["steady"]["hit_rate"],
                        "extra_keydowns": t["score"]["steady"]["extra_keydowns"],
                        "wrong_axis": t["score"]["steady"]["wrong_axis"],
                        "passed": t["score"]["passed"],
                    } for t in trials
                ],
                "confirmations": confirmations,
                "passed_all_confirmations": passes_all,
            })
    return {
        "schema": "research-a001b-steady-periodic-predeclared-v1",
        "status": "precommitted exploratory probe; NOT approved 001 physical limit",
        "precommit_protocol": "docs/research-a-experiment-001b-fixed-protocol.md",
        "conditions": {
            "motor": "MotorEnv personal_blue_switch_v0_1 same_hand=True",
            "control_dt_s": CONTROL_DT_S, "physics_dt_s": .001,
            "warmup_s": WARMUP_S, "screen_s": screen_s, "confirm_s": confirm_s,
            "window_s": JUDGEMENT_S, "intervals_ms": intervals,
            "duties": duties, "leads_ms": leads, "modes": modes,
            "screen_start_s": SCREEN_START_S, "confirm_starts_s": confirm_starts,
            "requires_all_confirmations": True,
            "required_hit_rate": .95, "allowed_extra_keydowns": 0,
            "max_wrong_axis": 0, "each_half_min_hit_rate": .95,
        },
        "rows": rows,
        "shortest_pass_ms_in_tested_fixed_command_family": {
            mode: min(
                (row["interval_ms"] for row in rows
                 if row["mode"] == mode and row["passed_all_confirmations"]),
                default=None,
            ) for mode in modes
        },
        "claims": {
            "body_physical_impossibility_proven": False,
            "human_equivalent": False,
            "motor_parameters_changed": False,
            "student_training_performed": False,
            "results_used_to_modify_original_001a": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    report = survey(quick=args.quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "quick": args.quick,
        "shortest_pass_ms_in_tested_fixed_command_family":
            report["shortest_pass_ms_in_tested_fixed_command_family"],
        "rows": [
            {"mode": r["mode"], "interval_ms": r["interval_ms"],
             "duty": r["selected"]["duty"], "lead_ms": r["selected"]["lead_ms"],
             "screen_steady_hit_rate": r["selected"]["score"]["steady"]["hit_rate"],
             "confirmed_hit_rates": [
                 c["score"]["steady"]["hit_rate"] for c in r["confirmations"]
             ],
             "confirmed_extras": [
                 c["score"]["steady"]["extra_keydowns"] for c in r["confirmations"]
             ],
             "passes": r["passed_all_confirmations"]}
            for r in report["rows"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
