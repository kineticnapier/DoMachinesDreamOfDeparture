"""Stage C v0.6.1: scored closed-loop two-event motor rollout experiment.

Privileged, offline diagnostic only. The baseline uses the unmodified
TieAwareJointTeacher; the experiment substitutes at most N reservations and
then *continues actual physics and scorer*. No Student training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
from dmdod.motor.n_key import NKeyAction
from dmdod.motor.keyboard import KeyEvent
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.contact_16k import ContactAction, PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import (
    forecast_reservation_swap,
)
import compare_nerfed_tie_teacher as tie
import u1_keydown_attribution_audit as base

SUMMARY_KEYS = ("hits", "misses", "matches", "mismatches", "unreserved",
                "too_early", "overload", "planned_error_mae_ms")


def run_with_teacher(segment, teacher_class):
    """Execute scorer/physics normally, without altering the audit module after use."""
    created = []
    old_teacher, old_env = base.ContactOracleTeacher, base.CoupledContactRealChartEnv

    def make_teacher(mode):
        obj = teacher_class(mode, None)
        created.append(obj)
        return obj

    def make_env(*args, **kwargs):
        env = old_env(*args, **kwargs)
        created[-1].motor = env.motor
        return env

    base.ContactOracleTeacher, base.CoupledContactRealChartEnv = make_teacher, make_env
    try:
        result = base.audit(segment, 8)
    finally:
        base.ContactOracleTeacher, base.CoupledContactRealChartEnv = old_teacher, old_env
    return result, created[-1]


def next_note_rollout(motor, action, forecast, note_times, reservations,
                      *, horizon_frames=7):
    """Two-event diagnostic with explicitly timed control ticks.

    Preserve legacy v0.6 scoring interfaces; collect causes and timing for each
    trial. One NKeyMotorEnv.step advances 10ms (internally 1ms physics).
    This remains an *uncommitted* hypothetical second-note action.
    """
    occupied = {n for tokens in reservations.values() for n in tokens}
    future = sorted((due, ordinal) for ordinal, due in note_times.items()
                    if ordinal not in occupied)
    empty = {"next_ordinal": None, "next_axis": None,
             "second_error_ms": 0., "second_down_ms": None,
             "evaluated": 0, "failure_reason": "no_future_note",
             "attempts": []}
    if not future:
        return empty
    next_due, next_ordinal = future[0]
    physical = motor.motor
    dt = physical.control_dt_s
    first_axis = forecast.candidate_axis
    ix = PHYSICAL_AXES.index(first_axis)
    first_downs = [ts for ordinal, _, ts in forecast.projected
                   if ordinal in forecast.reserved_ordinals]
    if not first_downs:
        return {**empty, "next_ordinal": next_ordinal,
                "second_error_ms": None, "failure_reason": "first_down_missing"}
    first_down = max(first_downs)
    if next_due <= first_down:
        return {**empty, "next_ordinal": next_ordinal,
                "second_error_ms": None, "failure_reason": "next_due_before_first_down"}

    # Replay the same source-release/candidate-press command as the actual
    # reservation forecast, instead of leaving the source pressed.
    current = copy.deepcopy(physical)
    first = list(action.commands)
    source_ix = PHYSICAL_AXES.index(forecast.source_axis)
    first[source_ix] = -1.
    first[ix] = 1.
    first_down_seen = None
    max_first_ticks = max(horizon_frames, int((first_down - current.time_s)/dt) + 2)
    for _ in range(max_first_ticks):
        transition = current.step(NKeyAction(tuple(first)))
        if any(e.event is KeyEvent.DOWN and e.key == first_axis
               for e in transition.evaluator_events):
            first_down_seen = next(e.time_s for e in transition.evaluator_events
                                   if e.event is KeyEvent.DOWN and e.key == first_axis)
            break
    if first_down_seen is None:
        return {**empty, "next_ordinal": next_ordinal,
                "second_error_ms": None, "failure_reason": "first_replay_failed"}

    release_commands = list(first)
    release_commands[ix] = -1.
    # The first motor step may already have advanced beyond the predicted
    # 1ms event. This is intentional: commands change only on 10ms ticks.
    actual_first_tick_end = current.time_s
    observed = physical.observe()
    axes = [first_axis]
    for axis in PHYSICAL_AXES:
        if axis == first_axis or axis in reservations:
            continue
        axis_i = PHYSICAL_AXES.index(axis)
        if observed.pressed_flags[axis_i] or int(axis.split("_")[1]) > 4:
            continue
        axes.append(axis)

    attempts = []
    successes = []
    for axis in axes:
        branch = copy.deepcopy(current)
        target = PHYSICAL_AXES.index(axis)
        cmd = list(release_commands)
        # Lead-time assumption is a diagnostic probe, not optimized teacher timing.
        requested_start = max(actual_first_tick_end, next_due - .044)
        while branch.time_s + dt <= requested_start + 1e-8:
            branch.step(NKeyAction(tuple(cmd)))
        command_start = branch.time_s
        pre = branch.observe()
        cmd[target] = 1.
        down = None
        up_before_second = False
        events_seen = []
        # Search through next due + 70ms rather than 7 ticks from initiation;
        # record all physical UP/DOWN events on the tested axis.
        stop = max(command_start + horizon_frames*dt, next_due + .070)
        while branch.time_s < stop - 1e-8:
            transition = branch.step(NKeyAction(tuple(cmd)))
            for ev in transition.evaluator_events:
                if ev.key == axis:
                    events_seen.append({"event": ev.event.value, "time_ms": ev.time_s*1000})
                    if ev.event is KeyEvent.UP:
                        up_before_second = True
                    if ev.event is KeyEvent.DOWN and down is None:
                        down = ev.time_s
            if down is not None:
                break
        reason = ("keydown_observed" if down is not None
                  else "still_pressed" if branch.observe().pressed_flags[target]
                  else "pressed_at_command" if pre.pressed_flags[target]
                  else "no_keydown_before_deadline")
        row = {"axis": axis, "reason": reason,
               "command_start_ms": command_start*1000,
               "requested_start_ms": requested_start*1000,
               "actual_first_down_ms": first_down_seen*1000,
               "first_tick_end_ms": actual_first_tick_end*1000,
               "pressed_before_second": pre.pressed_flags[target],
               "up_during_second": up_before_second,
               "down_ms": None if down is None else down*1000,
               "stop_ms": branch.time_s*1000,
               "axis_events": events_seen}
        attempts.append(row)
        if down is not None:
            successes.append((abs(down-next_due)*1000, axis, down))
    if not successes:
        return {"next_ordinal": next_ordinal, "next_axis": None,
                "second_error_ms": None, "second_down_ms": None,
                "evaluated": len(axes), "failure_reason": "all_attempts_failed",
                "attempts": attempts}
    error, axis, down = min(successes)
    return {"next_ordinal": next_ordinal, "next_axis": axis,
            "second_error_ms": error, "second_down_ms": down*1000,
            "evaluated": len(axes), "failure_reason": None,
            "attempts": attempts}


def choose_swap(motor, action, reservations, note_times, source, *, horizon_frames=7, candidate_scores=None):
    """Rank swap by scored first-note forecast plus simulated next-note press."""
    results = []
    rejections = 0
    for target in PHYSICAL_AXES:
        if target == source or target in reservations:
            continue
        try:
            forecast = forecast_reservation_swap(
                motor, action, reservations, note_times, source, target,
                horizon_frames=horizon_frames,
            )
        except ValueError:
            rejections += 1
            continue
        results.append(forecast)
    ranked = []
    scores = []
    for r in results:
        if not r.feasible:
            scores.append({"axis": r.candidate_axis, "feasible": False,
                           "first_error_ms": r.timing_mae_ms, "reason": "local_forecast_failed"})
            continue
        future = next_note_rollout(motor, action, r, note_times, reservations,
                                  horizon_frames=horizon_frames)
        # The next physical note is evaluated explicitly, but does not replace
        # the unchanged oracle teacher's actual subsequent choice.
        combined = (float("inf") if r.timing_mae_ms is None else r.timing_mae_ms)
        combined += (1000. if future["second_error_ms"] is None else future["second_error_ms"])
        scores.append({"axis": r.candidate_axis, "feasible": True,
                       "first_error_ms": r.timing_mae_ms,
                       "second_error_ms": future["second_error_ms"],
                       "next_ordinal": future["next_ordinal"],
                       "next_axis": future["next_axis"],
                       "second_down_ms": future["second_down_ms"],
                       "next_axes_evaluated": future["evaluated"],
                       "rollout_failure_reason": future.get("failure_reason"),
                       "rollout_attempts": future.get("attempts", []),
                       "combined_error_ms": combined})
        ranked.append((combined, r.candidate_axis, r))
    ranked.sort(key=lambda x: (x[0], x[1]))
    if candidate_scores is not None:
        winner = ranked[0][1] if ranked else None
        candidate_scores.extend({**row, "selected": row["axis"] == winner,
                                 "reason": "minimum_combined_error" if row["axis"] == winner
                                           else row.get("reason", "higher_combined_error")}
                                for row in scores)
    return (ranked[0][2] if ranked else None), len(results), rejections


def compare(chart: Path, *, end_s: float = 15., after_s: float = 0.,
            max_swaps: int = 1, horizon_frames: int = 7) -> dict:
    if not 0 <= after_s < end_s or max_swaps < 0 or not 1 <= horizon_frames <= 100:
        raise ValueError("invalid experiment limits")
    raw = chart.read_bytes()
    segment = build_playable_segment(
        compile_adofai(parse_adofai_text(raw.decode("utf-8-sig"))),
        end_s=end_s,
    )
    baseline, _ = run_with_teacher(segment, tie.TieAwareJointTeacher)
    changes = []
    counts = {"candidate_evaluations": 0, "candidate_rejections": 0,
              "eligible_local": 0, "attempted_swaps": 0}

    class SwappedTeacher(tie.TieAwareJointTeacher):
        def action(self, observed, now_s, remaining):
            before = {a: tuple(p.tokens) for a, p in self.reservations.items()}
            actual_action = super().action(observed, now_s, remaining)
            if now_s < after_s or len(changes) >= max_swaps:
                return actual_action
            assigned = {a: tuple(p.tokens) for a, p in self.reservations.items()
                        if p.tokens}
            created = [a for a, tokens in assigned.items()
                       if before.get(a) != tokens]
            note_times = dict(remaining)
            for source in created:
                if len(changes) >= max_swaps:
                    break
                if source not in self.reservations:
                    continue
                ix = PHYSICAL_AXES.index(source)
                if actual_action.commands[ix] <= 0:
                    continue
                candidate_scores = []
                chosen, evaluated, rejected = choose_swap(
                    self.motor, actual_action, assigned, note_times, source,
                    horizon_frames=horizon_frames, candidate_scores=candidate_scores)
                counts["candidate_evaluations"] += evaluated
                counts["candidate_rejections"] += rejected
                if chosen is None:
                    continue
                counts["eligible_local"] += 1
                dest = chosen.candidate_axis
                plan = self.reservations.pop(source)
                plan.key = dest
                self.reservations[dest] = plan
                # Reuse the same state-aware teacher on the next 10ms tick;
                # only the current candidate command is explicitly substituted.
                commands = list(actual_action.commands)
                commands[ix] = -1.
                commands[PHYSICAL_AXES.index(dest)] = 1.
                actual_action = ContactAction(tuple(commands),
                    actual_action.thumb_modes, actual_action.edge_enabled)
                changes.append({
                    "at_ms": round(now_s*1000, 3),
                    "ordinals": list(plan.tokens),
                    "from": source, "to": dest,
                    "forecast": asdict(chosen),
                    "local_candidates_evaluated": evaluated,
                    "candidate_scores": candidate_scores,
                })
                counts["attempted_swaps"] += 1
                assigned.pop(source)
                assigned[dest] = tuple(plan.tokens)
            return actual_action

    alternative, modified = run_with_teacher(segment, SwappedTeacher)
    credited = {r["credited"]: r for r in alternative["all_keydowns"]
               if r["credited"] is not None}
    for item in changes:
        item["actual"] = [{
            "ordinal": n,
            "axis": credited[n]["axis"] if n in credited else None,
            "keydown_ms": credited[n]["time_ms"] if n in credited else None,
            "margin": credited[n]["margin"] if n in credited else None,
        } for n in item["ordinals"]]
    return {
        "format": "dmdod-stage-c-fingering-v061-timing-diagnostic",
        "chart_sha256": sha256(raw).hexdigest(),
        "segment_end_s": end_s, "swap_after_s": after_s,
        "max_swaps": max_swaps, "horizon_frames": horizon_frames,
        "oracle_teacher": True, "student_training": False,
        "alternative_actions_applied_to_real_run": True,
        "scored_alternative_fingering": True,
        "not_real_adofai_multipress": True,
        "baseline": {k: baseline["summary"][k] for k in SUMMARY_KEYS},
        "alternative": {k: alternative["summary"][k] for k in SUMMARY_KEYS},
        "counts": counts, "swaps": changes,
        "alternative_failure_rows": alternative["summary"]["mismatch_rows"][:20],
        "alternative_unreserved_rows": alternative["summary"]["unreserved_rows"][:20],
        "alternative_joint_diagnostics": {
            k: getattr(modified, k) for k in
            ("forecasts", "inversions", "tie_inversions", "delays")
        },
        "limitations": [
            "Privileged oracle timestamps are used for planning and local candidate selection.",
            "One selected reservation is altered at a time; future teacher reservations are recomputed.",
            "Real simulator scoring is used, but ADOFAI Multipress is not fully reproduced.",
            "Only explicitly sampled segment and candidate budget are evaluated.",
            "Two-event lookahead is only a local physical surrogate, not a committed next-note allocation.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", type=Path)
    ap.add_argument("--end-s", type=float, default=15.)
    ap.add_argument("--after-s", type=float, default=0.)
    ap.add_argument("--max-swaps", type=int, default=1)
    ap.add_argument("--horizon-frames", type=int, default=7)
    ap.add_argument("--output", type=Path, default=Path("fingering-v061-timing-diagnostic.json"))
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f"Refusing to overwrite {args.output}")
    result = compare(args.chart, end_s=args.end_s, after_s=args.after_s,
                     max_swaps=args.max_swaps, horizon_frames=args.horizon_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("baseline","alternative","counts","swaps")},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
