from __future__ import annotations

"""Offline v0.8 gesture audit. Never attributes v0.6 Hits to v0.8."""

import argparse
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent,
)


def analyze(payload: dict) -> dict:
    if payload.get("format") != "dmdogo-pattern-first-physical-v0.6":
        raise ValueError("expected v0.6 physical replay as comparison source")
    rows = payload.get("events") or []
    sequence = tuple(GestureEvent(
        int(row["event_index"]), float(row["episode_s"]),
        int(row["chord_size"])) for row in rows)
    result = GestureChunkPlannerV08().plan(sequence)
    if not result.feasible:
        raise RuntimeError(f"v0.8 offline planning failed: {result.failure_reason}")
    if len(result.assignments) != len(sequence):
        raise RuntimeError("assignment count mismatches event count")
    v06 = [tuple(((row.get("decisions") or {}).get("pattern_first_v06")
                   or {}).get("axes") or ()) for row in rows]
    comparable = [(new, old) for new, old in zip(result.assignments, v06) if old]
    return {
        "format": "dmdogo-v08-offline-gesture-audit",
        "source_format": payload["format"],
        "chart": payload.get("chart"),
        "chart_sha256": payload.get("chart_sha256"),
        "event_count": len(rows),
        "planning_contract": result.model_contract,
        "planning_metrics": result.metrics,
        "v06_physical_allocations_compared": len(comparable),
        "different_from_v06_allocations": sum(new != old for new, old in comparable),
        "v08_physical_motor_evaluated": False,
        "v08_hits": None,
        "v08_x_accuracy_percent": None,
        "v06_saved_physical_stats_only": (
            (payload.get("policies") or {}).get("pattern_first_v06", {})
            .get("saved_metrics")),
        "chunks": [{
            "start_event_index": chunk.start_event_index,
            "end_event_index": chunk.end_event_index,
            "event_count": chunk.event_count,
            "rhythm_kind": chunk.rhythm_kind,
            "start_reason": chunk.start_reason,
            "gesture_family": chunk.gesture_family,
            "axes": [list(axes) for axes in chunk.axes],
        } for chunk in result.chunks],
        "limitations": [
            "A repetitive sequence can be physically valid without being human-like.",
            "Gesture vocabulary and exposure regularizer are untrained assumptions.",
            "Only the v0.6 saved physical arm has measured Hit/X-Accuracy.",
            "Full Oracle onset times are evaluator-only, not causal HUD observations.",
            "Live Motor eligibility, latency and judgement require a separate v0.8 run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    source = json.loads(args.input_replay.read_text(encoding="utf-8"))
    report = analyze(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["planning_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
