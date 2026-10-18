from __future__ import annotations

"""Reconstruct ADOFAI target consumption from saved *logical* switch traces.

Never reruns the Motor or its Fingering policy, and never supplies chart
truth to those policies. Reconstructs only the evaluator AFTER the completed
run. This distinguishes a KeyDown physically near target i that was actually
consumed by the earlier unresolved target j from a genuine delayed/no stroke.

Fail closed: all reconstructed per-target Hit/Miss flags and summary counts
must equal the original saved arm before the attribution is trusted.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_hud_v14_failed_cases_v02 import CASES, ARMS, get_case
from run_hud_v14_fingering_motor_integration_v02 import DEFAULT_TRAIN_CHART
from dmdod.envs.real_chart import RealChartMotorEnv
from dmdod.motor.env import TimedKeyEvent
from dmdod.motor.keyboard import KeyEvent

CONTROL_DT_S = .010
EPS_S = 1e-7


def replay_scorer(segment, arm: dict) -> dict:
    """Reconstruct exact target routing from saved finished logical events.

    All switch times and scorer outcome comparisons are from the completed
    evaluation, not causal policy inputs. No Motor commands are issued.
    """
    if "error" in arm:
        raise ValueError("cannot reconstruct errored arm")
    if "logical_switch_trace" not in arm or "target_outcomes_posthoc" not in arm:
        raise ValueError("requires enriched per-arm JSON from targeted crosscheck")
    frames = arm["frames"]
    if type(frames) is not int or frames <= 0:
        raise ValueError("invalid saved frame count")
    events = arm["logical_switch_trace"]
    if any(
        (not isinstance(e["time_s"], (float, int))
         or not isinstance(e["key"], str)
         or e["edge"] not in (KeyEvent.DOWN.value, KeyEvent.UP.value))
        for e in events
    ):
        raise ValueError("malformed saved logical KeyDown/KeyUp event")
    if any(float(b["time_s"]) + EPS_S < float(a["time_s"])
           for a, b in zip(events, events[1:])):
        raise ValueError("saved switch events must be time ordered")

    evaluator = RealChartMotorEnv(
        segment, control_dt_s=CONTROL_DT_S, tail_s=.040
    )
    evaluator.reset()
    routed = []
    expired = []
    cursor = 0
    for step in range(frames):
        frame_end = (step + 1) * CONTROL_DT_S
        while cursor < len(events) and events[cursor]["time_s"] <= frame_end + EPS_S:
            edge = events[cursor]
            cursor += 1
            at = float(edge["time_s"])
            key_event = KeyEvent(edge["edge"])
            evaluator._advance_overload(at)
            before_expire = set(i for i, m in enumerate(evaluator._missed) if m)
            evaluator._expire_misses(at)
            expired.extend({
                "target_ordinal": i, "at_s": at, "reason": "before_logical_event"
            } for i, m in enumerate(evaluator._missed)
                if m and i not in before_expire)
            if key_event is not KeyEvent.DOWN:
                continue
            target_index = evaluator._next_target_index()
            due_s = (
                float(segment.targets[target_index].episode_time_s)
                if target_index is not None else None
            )
            prior = len(evaluator._hit_margins)
            evaluator._score_event(TimedKeyEvent(at, edge["key"], key_event))
            margin = (
                evaluator._hit_margins[-1].value
                if len(evaluator._hit_margins) != prior else None
            )
            routed.append({
                "time_s": at, "key": edge["key"],
                "consumed_target_ordinal": target_index,
                "assigned_target_due_s": due_s,
                "assigned_error_ms": (
                    round((at - due_s) * 1000, 3)
                    if due_s is not None else None
                ),
                "margin": margin,
                "hit": bool(
                    target_index is not None and evaluator._used[target_index]
                ),
                "miss": bool(
                    target_index is not None and evaluator._missed[target_index]
                ),
            })
        prior_expired = set(i for i, m in enumerate(evaluator._missed) if m)
        evaluator._advance_overload(frame_end)
        evaluator._expire_misses(frame_end)
        expired.extend({
            "target_ordinal": i, "at_s": frame_end, "reason": "end_of_control_frame"
        } for i, m in enumerate(evaluator._missed)
            if m and i not in prior_expired)

    if cursor != len(events):
        raise ValueError("logical events extend beyond saved control-frame horizon")
    if frames * CONTROL_DT_S >= evaluator._episode_end_s - EPS_S:
        before = set(i for i, m in enumerate(evaluator._missed) if m)
        evaluator._expire_misses(float("inf"))
        expired.extend({
            "target_ordinal": i, "at_s": None, "reason": "episode_end"
        } for i, m in enumerate(evaluator._missed) if m and i not in before)

    expected = arm["target_outcomes_posthoc"]
    actual = tuple(
        (bool(evaluator._used[i]), bool(evaluator._missed[i]))
        for i in range(len(segment.targets))
    )
    wanted = tuple(
        (bool(x["hit"]), bool(x["miss"]))
        for x in expected
    )
    matched = (
        actual == wanted
        and evaluator._hits == arm["stats"]["hits"]
        and evaluator._misses == arm["stats"]["misses"]
        and evaluator._too_early == arm["stats"]["too_early"]
    )
    result = {
        "score_replay_exact_match": matched,
        "replayed_hits": evaluator._hits,
        "replayed_misses": evaluator._misses,
        "replayed_too_early": evaluator._too_early,
        "logical_down_target_routing": routed,
        "expiry_events": expired,
    }
    if not matched:
        result["mismatch"] = {
            "expected_flags": wanted, "actual_flags": actual,
            "expected_stats": arm["stats"],
        }
        return result

    # Summarize only after exact replay is established.
    missed = []
    for i, target in enumerate(segment.targets):
        if not evaluator._missed[i]:
            continue
        due = float(target.episode_time_s)
        near = sorted(
            (
                {
                    "time_s": row["time_s"],
                    "key": row["key"],
                    "error_to_missed_due_ms": round(
                        (row["time_s"] - due) * 1000, 3),
                    "actually_consumed_target": row["consumed_target_ordinal"],
                    "assigned_target_due_s": row["assigned_target_due_s"],
                    "assigned_error_ms": row["assigned_error_ms"],
                    "margin": row["margin"],
                }
                for row in routed
                if abs(row["time_s"] - due) <= .025
            ),
            key=lambda row: abs(row["error_to_missed_due_ms"]),
        )
        if any(x["actually_consumed_target"] is not None
               and x["actually_consumed_target"] < i for x in near):
            mechanism = "nearby_keydown_consumed_by_earlier_unresolved_target"
        elif near:
            mechanism = "nearby_keydown_other_assignment_or_evaluation_issue"
        else:
            mechanism = "no_logical_keydown_within_25ms"
        missed.append({
            "ordinal": i,
            "due_s": due,
            "mechanism": mechanism,
            "nearby_logical_keydowns": near,
            "expiry": tuple(x for x in expired if x["target_ordinal"] == i),
        })
    result["confirmed_missed_target_routing"] = missed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--train-chart", type=Path, default=DEFAULT_TRAIN_CHART)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("saved targeted crosscheck directory does not exist")
    invalid = []
    for case in CASES:
        segment = get_case(case, chart=args.train_chart)
        for arm_name in ARMS:
            path = args.input_dir / case / (arm_name + ".json")
            if not path.is_file():
                parser.error(f"missing saved arm JSON: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            result = replay_scorer(segment, record)
            print(f"=== {case} / {arm_name} ===", flush=True)
            print(f"exact_scorer_replay={result['score_replay_exact_match']}",
                  f"hits={result['replayed_hits']}",
                  f"misses={result['replayed_misses']}",
                  f"too_early={result['replayed_too_early']}", flush=True)
            if not result["score_replay_exact_match"]:
                invalid.append((case, arm_name))
                print("SCORE_REPLAY_MISMATCH:",
                      json.dumps(result["mismatch"], ensure_ascii=False),
                      flush=True)
                continue
            for item in result["confirmed_missed_target_routing"]:
                print("MISS_ROUTING:",
                      json.dumps(item, ensure_ascii=False), flush=True)
    if invalid:
        raise SystemExit(
            f"Reconstruction mismatch; no causal claim for {invalid}"
        )


if __name__ == "__main__":
    main()
