"""Read-only Stage C v0.3: sample alternate fingering on a real chart.

Runs the frozen tie-aware teacher normally; alternative press forecasts are
made on cloned physical states only. They never affect the actual play.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.contact_16k import PHYSICAL_AXES
from dmdod.planning.fingering_reservation_counterfactual import (
    forecast_reservation_swap, _contact_keys,
)
import compare_nerfed_tie_teacher as tie
import u1_keydown_attribution_audit as audit_module


def evaluate(chart: Path, *, end_s: float, max_reservations: int = 12,
             max_candidates: int = 2) -> dict:
    if end_s <= 0 or max_reservations < 0 or max_candidates < 0:
        raise ValueError("invalid diagnostic limits")
    raw = chart.read_bytes()
    segment = build_playable_segment(
        compile_adofai(parse_adofai_text(raw.decode("utf-8-sig"))),
        end_s=end_s,
    )
    collected = []
    captured = []
    counters = {"new_reservations": 0, "eligible": 0, "rejected": 0}

    class ObservedTeacher(tie.TieAwareJointTeacher):
        def action(self, observed, now_s, remaining):
            previous = {a: tuple(r.tokens) for a, r in self.reservations.items()}
            action = super().action(observed, now_s, remaining)
            if len(captured) >= max_reservations:
                return action
            reservations = {a: tuple(r.tokens) for a, r in self.reservations.items()}
            new_axes = [a for a, tokens in reservations.items()
                        if tokens and previous.get(a) != tokens]
            if not new_axes:
                return action
            note_times = dict(remaining)
            for source in new_axes:
                counters["new_reservations"] += 1
                if len(captured) >= max_reservations:
                    break
                # A new reservation may be planned in advance of the +1 force
                # command. The forecast deliberately uses that command.
                snapshots = []
                for candidate in PHYSICAL_AXES:
                    if candidate in reservations or candidate == source:
                        continue
                    if len(snapshots) >= max_candidates:
                        break
                    try:
                        result = forecast_reservation_swap(
                            self.motor, action, reservations, note_times,
                            source, candidate)
                    except ValueError:
                        counters["rejected"] += 1
                        continue
                    counters["eligible"] += 1
                    snapshots.append(asdict(result))
                captured.append({
                    "time_s": now_s, "source_axis": source,
                    "ordinals": list(reservations[source]),
                    "source_command": action.commands[PHYSICAL_AXES.index(source)],
                    "candidates": snapshots,
                })
            return action

    old_teacher, old_env = audit_module.ContactOracleTeacher, audit_module.CoupledContactRealChartEnv
    def factory(mode):
        teacher = ObservedTeacher(mode, None)
        collected.append(teacher)
        return teacher
    def environment(*args, **kwargs):
        env = old_env(*args, **kwargs)
        collected[-1].motor = env.motor
        return env

    audit_module.ContactOracleTeacher = factory
    audit_module.CoupledContactRealChartEnv = environment
    try:
        actual = audit_module.audit(segment, 8)
    finally:
        audit_module.ContactOracleTeacher = old_teacher
        audit_module.CoupledContactRealChartEnv = old_env

    actual_by_ordinal = {r["credited"]: r for r in actual["all_keydowns"]
                         if r["credited"] is not None}
    for row in captured:
        row["actual"] = [
            {"ordinal": n, "axis": actual_by_ordinal[n]["axis"],
             "keydown_ms": actual_by_ordinal[n]["time_ms"]}
            if n in actual_by_ordinal else {"ordinal": n, "missing": True}
            for n in row["ordinals"]
        ]
    s = actual["summary"]
    return {
        "format": "dmdod-fingering-reservation-counterfactual-v03",
        "chart_sha256": sha256(raw).hexdigest(),
        "segment_end_s": end_s,
        "oracle_teacher_fixed": True,
        "alternative_actions_applied_to_real_run": False,
        "student_training": False,
        "scored_alternative_fingering": False,
        "limitations": "Each alternative is a seven-frame open-loop clone only; actual results are from unchanged oracle play.",
        "reference": {k: s[k] for k in (
            "hits", "misses", "matches", "mismatches", "unreserved",
            "too_early", "overload", "planned_error_mae_ms"
        )},
        "diagnostics": counters,
        "samples": captured,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", type=Path)
    ap.add_argument("--end-s", type=float, default=15.0)
    ap.add_argument("--max-reservations", type=int, default=12)
    ap.add_argument("--max-candidates", type=int, default=2)
    ap.add_argument("--output", type=Path, default=Path("fingering-v03-diagnostic.json"))
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f"Refusing to overwrite {args.output}")
    report = evaluate(args.chart, end_s=args.end_s,
                      max_reservations=args.max_reservations,
                      max_candidates=args.max_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"reference": report["reference"],
                      "diagnostics": report["diagnostics"],
                      "samples": len(report["samples"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
