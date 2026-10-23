from __future__ import annotations

"""Offline research audit; v0.7 choices are NEVER credited with v0.6 Hits."""

import argparse
import json
from pathlib import Path

from dmdod.planning.human_phrase_planner_v07 import (
    HumanPhrasePlannerV07, PhraseEvent,
)

FORMAT = "dmdogo-human-phrase-hypothesis-v0.7"


def vars_of_phrase(phrase) -> dict:
    return {
        "start_event_index": phrase.start_event_index,
        "end_event_index": phrase.end_event_index,
        "event_count": phrase.event_count,
        "rhythm_class": phrase.rhythm_class,
        "gap_motif_s": list(phrase.gap_motif_s),
        "boundary_reason": phrase.boundary_reason,
        "gesture_template": phrase.template,
        "mirrored": phrase.mirrored,
        "phase": phrase.phase,
    }


def analyze_replay(payload: dict) -> dict:
    if payload.get("format") != "dmdogo-pattern-first-physical-v0.6":
        raise ValueError("expected v0.6 full-chart replay JSON")
    events = tuple(PhraseEvent(int(e["event_index"]), float(e["episode_s"]),
                               int(e["chord_size"])) for e in payload["events"])
    result = HumanPhrasePlannerV07().plan(events)
    if not result.feasible:
        raise RuntimeError(f"v0.7 offline planning failed: {result.failure_reason}")
    baseline = []
    for e in payload["events"]:
        row = (e.get("decisions") or {}).get("pattern_first_v06") or {}
        baseline.append(tuple(row.get("axes") or ()))
    known = [(a, b) for a, b in zip(result.assignments, baseline) if b]
    changes = sum(a != b for a, b in known)
    source_metrics = (payload.get("policies", {}).get("pattern_first_v06", {})
                      .get("saved_metrics", {}))
    return {
        "format": FORMAT,
        "chart": payload.get("chart"),
        "chart_sha256": payload.get("chart_sha256"),
        "events": len(events),
        "planner": result.model_contract,
        "summary": result.metrics | {
            "v06_physical_allocations_compared": len(known),
            "different_from_v06_physical_allocation": changes,
            "different_fraction_of_compared": (changes / len(known) if known else None),
            "v07_physical_motor_evaluated": False,
            "v07_hits": None,
            "v07_x_accuracy": None,
        },
        "v06_physical_scorer_metrics_only": source_metrics,
        "phrases": [vars_of_phrase(p) for p in result.phrases],
        "assignments": [
            {"event_index": e.event_index, "due_s": e.due_s,
             "axes": list(axes), "v06_physical_axes": list(previous),
             "match_v06": axes == previous if previous else None}
            for e, axes, previous in zip(events, result.assignments, baseline)
        ],
        "limitations": [
            "v0.7 is a hand-coded rhythm/gesture hypothesis, not trained on human demonstrations.",
            "The 16-event planning horizon is not a validated musical phrase boundary.",
            "v0.6 physical Hits must not be attributed to v0.7 offline assignments.",
            "Exact Oracle timing is privileged offline information; not HUD-only Student perception.",
            "The fixed ordinary Motor axes are not verified anatomical finger measurements.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh filename")
    value = json.loads(args.input_replay.read_text(encoding="utf-8"))
    report = analyze_replay(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
