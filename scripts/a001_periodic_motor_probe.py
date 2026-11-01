#!/usr/bin/env python3
"""Research A / Experiment 001 exploratory physical-Motor rate sweep.

PRE-REGISTRATION IS NOT APPROVED. This runs a fixed family of periodic
command schedules, not an optimal-control proof or any RL training. It must
not be used as a learned policy or fingering teacher in Experiment 002.

Use:
    python scripts/a001_periodic_motor_probe.py --out reports/research-a001-pilot.json

The command family, trial grids, timing tolerance and independent confirmation
are pinned below. This preliminary scan can fail to locate a feasible band.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from dmdod.motor.env import MotorAction, MotorEnv

CONTROL_DT_S = 0.010
PHYSICS_DT_S = 0.001
JUDGEMENT_WINDOW_S = 0.020  # pilot assumption; NOT an approved study threshold
INTERVALS_MS = (60, 80, 100, 120, 150, 180, 220, 260, 320, 400, 500, 650, 800)
DUTIES = (0.25, 0.40, 0.55, 0.70)
LEADS_MS = (20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350)
SCREEN_START_S = 0.800
SCREEN_DURATION_S = 3.600
CONFIRM_START_S = 1.130
CONFIRM_DURATION_S = 7.200
TAIL_S = 0.450
# For score comparisons, key-downs outside the playable target range are
# counted only in [first_target - window, last_target + window].
# Extra physical downs inside that interval count as unwanted contacts.


@dataclass(frozen=True)
class Trial:
    mode: str  # single_left, single_right, or alternating
    interval_ms: int
    duty: float
    lead_ms: int
    start_s: float
    duration_s: float
    same_hand: bool = True


def target_times(t: Trial) -> list[float]:
    interval_s = t.interval_ms / 1000.0
    # Exact rational index multiplications keep schedule independent of physics.
    count = int(round(t.duration_s / interval_s))
    if count < 2:
        raise ValueError("at least two note targets are required")
    return [t.start_s + i * interval_s for i in range(count)]


def periodic_command_at(t: Trial, step_index: int, targets: list[float]) -> MotorAction:
    """Diagnostic oracle command only; NEVER passed to experiment-002 Student.

    Periodic pulse starts ahead of a target. Relative phase is calculated
    separately for each physical key; no learned decision is made here.
    """
    time_s = step_index * CONTROL_DT_S
    lead = t.lead_ms / 1000.0
    interval = t.interval_ms / 1000.0
    if t.mode not in ("single_left", "single_right", "alternating"):
        raise ValueError(f"unrecognized mode {t.mode}")
    active = [False, False]
    keys = ("left", "right")
    for key_index, key in enumerate(keys):
        if t.mode == "single_left" and key != "left":
            continue
        if t.mode == "single_right" and key != "right":
            continue
        if t.mode == "alternating":
            first_due = targets[key_index]
            last_due = targets[-1] if (len(targets) - 1) % 2 == key_index else targets[-2]
            physical_period = 2.0 * interval
        else:
            first_due = targets[0]
            last_due = targets[-1]
            physical_period = interval
        first_press = first_due - lead
        last_press = last_due - lead
        if time_s < first_press - 1e-9 or time_s >= last_press + physical_period * t.duty - 1e-9:
            continue
        # Integer count avoids modulo float instability at phase boundaries.
        beat = int((time_s - first_press + 1e-9) / physical_period)
        pulse_start = first_press + beat * physical_period
        if pulse_start <= last_press + 1e-9 and pulse_start - 1e-9 <= time_s < pulse_start + physical_period * t.duty - 1e-9:
            active[key_index] = True
    return MotorAction(1.0 if active[0] else -1.0,
                       1.0 if active[1] else -1.0)


def match_keydowns(
    keydowns: list[tuple[float, str]],
    targets: list[float],
    *,
    window_s: float = JUDGEMENT_WINDOW_S,
) -> dict:
    """One-to-one ordered matching with extra keydowns explicitly counted.

    This is a synthetic score, NOT ADOFAI's full input-state machine.
    Events outside the score interval are ignored; they are kept in trace stats.
    """
    if window_s < 0 or not targets:
        raise ValueError("invalid scoring window or empty targets")
    start, end = targets[0] - window_s, targets[-1] + window_s
    relevant = [(time, key) for time, key in keydowns if start - 1e-9 <= time <= end + 1e-9]
    relevant.sort(key=lambda x: x[0])
    hits, extra, cursor, wrong_axis = 0, 0, 0, 0
    matched_errors_ms = []
    used = []
    for time_s, key in relevant:
        while cursor < len(targets) and targets[cursor] < time_s - window_s - 1e-9:
            cursor += 1
        if cursor < len(targets) and abs(time_s - targets[cursor]) <= window_s + 1e-9:
            expected = "left" if cursor % 2 == 0 else "right"
            if key != expected:
                wrong_axis += 1 if any(x == expected for x in ("left", "right")) else 0
            hits += 1
            used.append((cursor, key))
            matched_errors_ms.append((time_s - targets[cursor]) * 1000.0)
            cursor += 1
        else:
            extra += 1
    return {
        "targets": len(targets),
        "hits": hits,
        "misses": len(targets) - hits,
        "hit_rate": hits / len(targets),
        "extra_keydowns": extra,
        "wrong_axis_for_alternating_reference": wrong_axis,
        "max_absolute_timing_error_ms": max((abs(x) for x in matched_errors_ms), default=None),
        "matched_key_sequence": "".join("L" if key == "left" else "R" for _, key in used),
    }


def run_trial(t: Trial) -> dict:
    if not 0 < t.duty < 1 or t.interval_ms <= 0 or t.lead_ms < 0 or t.duration_s <= 0:
        raise ValueError("invalid trial settings")
    targets = target_times(t)
    env = MotorEnv(same_hand=t.same_hand, control_dt_s=CONTROL_DT_S)
    env.reset()
    all_downs, all_ups = [], []
    duration = targets[-1] + TAIL_S
    steps = int(duration / CONTROL_DT_S) + 1
    for i in range(steps):
        action = periodic_command_at(t, i, targets)
        transition = env.step(action)
        for event in transition.evaluator_events:
            if event.event.value == "down":
                all_downs.append((event.time_s, event.key))
            else:
                all_ups.append((event.time_s, event.key))
    score = match_keydowns(all_downs, targets)
    if t.mode != "alternating":
        # The reference alternation is not a target constraint for single-finger trials.
        score.pop("wrong_axis_for_alternating_reference")
    min_reuse = {}
    for key in ("left", "right"):
        times = [timestamp for timestamp, k in all_downs if k == key]
        min_reuse[key] = (min(b - a for a, b in zip(times, times[1:])) * 1000.0
                          if len(times) > 1 else None)
    # A false positive must not arise from arbitrary extra keydowns.
    clean = (
        score["hit_rate"] >= 0.95
        and score["extra_keydowns"] <= max(1, int(.02 * len(targets)))
    )
    return {
        **asdict(t),
        **score,
        "clean_95": clean,
        "physical_keydowns": len(all_downs),
        "physical_keyups": len(all_ups),
        "min_reuse_ms": min_reuse,
        "last_fatigue": {
            "left": env.diagnostics().left_fatigue,
            "right": env.diagnostics().right_fatigue,
            "hand": env.diagnostics().hand_fatigue,
        },
    }


def rank_trial(result: dict):
    # Fixed and specified before seeing results. Prevent a mass of extras
    # from improving the chosen trial at equal hit count.
    return (
        bool(result["clean_95"]),
        result["hits"] / result["targets"],
        -result["extra_keydowns"] / result["targets"],
        -result["lead_ms"],
        -result["duty"],
    )


def sweep(*, intervals_ms=INTERVALS_MS, duties=DUTIES, leads_ms=LEADS_MS,
          modes=("single_left", "single_right", "alternating"),
          same_hand=True, screen_duration_s=SCREEN_DURATION_S,
          confirm_duration_s=CONFIRM_DURATION_S) -> dict:
    rows = []
    for mode in modes:
        for interval in intervals_ms:
            trials = [
                run_trial(Trial(mode, interval, duty, lead, SCREEN_START_S,
                                screen_duration_s, same_hand))
                for duty in duties for lead in leads_ms
            ]
            best = max(trials, key=rank_trial)
            chosen = {
                "mode": mode, "interval_ms": interval,
                "screen": best, "screen_candidate_count": len(trials),
                "confirm": run_trial(Trial(
                    mode, interval, best["duty"], best["lead_ms"], CONFIRM_START_S,
                    confirm_duration_s, same_hand
                )),
            }
            rows.append(chosen)
    return {
        "schema": "research-a001-periodic-motor-pilot-v1",
        "status": "UNAPPROVED exploratory pilot; NOT physical limit certification",
        "physical_source": "DMDOD MotorEnv using personal_blue_switch_v0_1 unchanged",
        "study002_teacher_input": False,
        "same_hand": same_hand,
        "fixed_grid": {
            "interval_ms": list(intervals_ms), "duties": list(duties),
            "lead_ms": list(leads_ms), "modes": list(modes),
            "screen_start_s": SCREEN_START_S, "confirm_start_s": CONFIRM_START_S,
            "screen_duration_s": screen_duration_s,
            "confirm_duration_s": confirm_duration_s,
            "judgement_window_s": JUDGEMENT_WINDOW_S,
            "control_dt_s": CONTROL_DT_S, "physics_dt_s": PHYSICS_DT_S,
        },
        "rows": rows,
        "preliminary_clean_intervals_ms": {
            mode: [r["interval_ms"] for r in rows if r["mode"] == mode and r["confirm"]["clean_95"]]
            for mode in modes
        },
        "caveats": [
            "Fixed periodic controller family is not an optimal physical controller.",
            "Pilot judgment window and objective are provisional and unapproved.",
            "The screen tunes fixed controller parameters; confirmation uses a distinct start delay but not an independent unseen motor model.",
            "No claim of human-equivalent motor ability or learned alternation.",
            "No study 002 training, preregistration freeze, or state-dependent control.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="small smoke sweep; cannot establish any capability boundary")
    parser.add_argument("--different-hands", action="store_true")
    args = parser.parse_args()
    if args.quick:
        result = sweep(intervals_ms=(120, 260), duties=(.4, .7),
                       leads_ms=(80, 200), screen_duration_s=1.2,
                       confirm_duration_s=1.6, same_hand=not args.different_hands)
    else:
        result = sweep(same_hand=not args.different_hands)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "preliminary_clean_intervals_ms": result["preliminary_clean_intervals_ms"],
        "rows": [
            {"mode": r["mode"], "interval_ms": r["interval_ms"],
             "duty": r["screen"]["duty"], "lead_ms": r["screen"]["lead_ms"],
             "screen_hit": r["screen"]["hit_rate"], "confirm_hit": r["confirm"]["hit_rate"],
             "confirm_extras": r["confirm"]["extra_keydowns"],
             "clean_95": r["confirm"]["clean_95"]}
            for r in result["rows"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
