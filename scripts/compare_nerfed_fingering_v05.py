"""Stage C v0.5: scored closed-loop future-readiness ranking experiment.

Privileged, offline diagnostic only. The baseline uses the unmodified
TieAwareJointTeacher; the experiment substitutes at most N reservations and
then *continues actual physics and scorer*. No Student training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
from dmdod.motor.n_key import NKeyAction
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


def score_future_readiness(motor, action, forecast, note_times, reservations, *, frames=7):
    """Evaluate candidate-axis physical recovery after a forecasted press.

    Return a score penalty in milliseconds. This is a *local motor* surrogate,
    not a full alternative future-note assignment or a learned fatigue model.
    """
    occupied = {n for tokens in reservations.values() for n in tokens}
    upcoming = sorted(due for ordinal, due in note_times.items()
                      if ordinal not in occupied)
    if not upcoming:
        return 0.
    due = upcoming[0]
    last = max(note_times[n] for n in forecast.reserved_ordinals)
    if due <= last:
        return 20.  # overlapping upcoming note; reserve a lane conservatively
    gap_ms = (due-last)*1000.
    if gap_ms >= 110.:
        return 0.
    axis = forecast.candidate_axis
    i = PHYSICAL_AXES.index(axis)
    branch = copy.deepcopy(motor.motor)
    commands = list(action.commands)
    commands[i] = 1.
    # Reproduce the candidate-axis activation under the current concurrent
    # commands; then test whether the axis can physically reset by next due.
    for _ in range(frames):
        branch.step(NKeyAction(tuple(commands)))
    commands[i] = -1.
    reset_by_due = False
    remaining = max(0, min(110, int((due-branch.time_s)/branch.control_dt_s)))
    for _ in range(remaining):
        branch.step(NKeyAction(tuple(commands)))
        if not branch.observe().pressed_flags[i]:
            reset_by_due = True
            break
    # Worse physical recovery costs more. Do not claim a strict "safe" bound.
    return 0. if reset_by_due else (110.-gap_ms)*.25


def choose_swap(motor, action, reservations, note_times, source, *, horizon_frames=7):
    """Rank compatible alternatives by timing and axis-specific reset risk."""
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
    feasible = [r for r in results if r.feasible]
    feasible.sort(key=lambda r: (
        (float("inf") if r.timing_mae_ms is None else r.timing_mae_ms)
        + score_future_readiness(motor, action, r, note_times, reservations,
                                 frames=horizon_frames),
        r.candidate_axis,
    ))
    return (feasible[0] if feasible else None), len(results), rejections


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
                chosen, evaluated, rejected = choose_swap(
                    self.motor, actual_action, assigned, note_times, source,
                    horizon_frames=horizon_frames)
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
        "format": "dmdod-stage-c-fingering-closed-loop-v05",
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
            "Future readiness penalty tests axis reset only; not a full upcoming note allocation.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", type=Path)
    ap.add_argument("--end-s", type=float, default=15.)
    ap.add_argument("--after-s", type=float, default=0.)
    ap.add_argument("--max-swaps", type=int, default=1)
    ap.add_argument("--horizon-frames", type=int, default=7)
    ap.add_argument("--output", type=Path, default=Path("fingering-v05-closed-loop.json"))
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
