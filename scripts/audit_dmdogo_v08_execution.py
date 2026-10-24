from __future__ import annotations

"""Independent v0.8 declared-vs-executed fingering audit (no simulation).

Uses official scorer only for the source summary; diagnostic KeyDown matching
is one-to-one, still HEURISTIC, never presented as scorer attribution.
"""

import argparse
from bisect import bisect_left
import json
from pathlib import Path

FORMAT = "dmdogo-gesture-chunk-physical-v0.8"
MINIMUM_REUSE_MS = 80.
MATCH_WINDOW_MS = 35.


def _reuse(ordered: list[tuple[float, tuple[str, ...]]], minimum_ms: float) -> dict:
    last: dict[str, float] = {}
    smallest = float("inf")
    violations = []
    for when_ms, axes in ordered:
        for axis in axes:
            if axis in last:
                gap = when_ms - last[axis]
                smallest = min(smallest, gap)
                if gap + 1e-6 < minimum_ms:
                    violations.append({"axis": axis, "time_ms": when_ms,
                                       "reuse_gap_ms": round(gap, 6)})
            last[axis] = when_ms
    return {
        "minimum_reuse_ms": None if smallest == float("inf") else round(smallest, 6),
        "violations": violations, "violation_count": len(violations),
    }


def _match_unique_keydowns(decisions: list[dict], keydowns: list[dict]) -> dict:
    """One KeyDown cannot satisfy multiple events. Not canonical judging."""
    by_axis = {}
    for index, keydown in enumerate(keydowns):
        by_axis.setdefault(keydown["axis"], []).append((
            float(keydown["time_ms"]), index))
    for entries in by_axis.values():
        entries.sort()
    consumed = set()
    matches = []
    ambiguous = 0
    missing = []
    for row in decisions:
        for axis in row["axes"]:
            due = float(row["due_ms"])
            values = by_axis.get(axis, [])
            lo = bisect_left(values, (due - MATCH_WINDOW_MS, -1))
            candidates = []
            for time_ms, key_id in values[lo:]:
                if time_ms > due + MATCH_WINDOW_MS:
                    break
                if key_id not in consumed:
                    candidates.append((abs(time_ms - due), time_ms, key_id))
            if len(candidates) > 1:
                ambiguous += 1
            if not candidates:
                missing.append({"event_index": row["event_index"], "axis": axis})
                continue
            _, time_ms, key_id = min(candidates)
            consumed.add(key_id)
            matches.append({
                "event_index": row["event_index"], "axis": axis,
                "due_ms": due, "physical_keydown_ms": time_ms,
                "time_error_ms": round(time_ms - due, 6),
                "keydown_index": key_id,
            })
    return {
        "matched_contacts": len(matches),
        "unmatched_contacts": missing,
        "unmatched_contact_count": len(missing),
        "unused_keydowns": len(keydowns) - len(consumed),
        "ambiguous_contact_count": ambiguous,
        "matches": matches,
    }


def audit(summary: dict, telemetry: dict, replay: dict) -> dict:
    if (summary.get("format") != FORMAT
            or replay.get("format") != FORMAT
            or summary.get("chart_sha256") != replay.get("chart_sha256")
            or replay.get("default_policy") != "gesture_chunk_v08"):
        raise ValueError("not matching v0.8 physical summary and replay")
    events = replay.get("events")
    decisions = telemetry.get("decisions")
    chunks = summary.get("abstract_schedule", {}).get("chunks")
    if not isinstance(events, list) or not isinstance(decisions, list) or not isinstance(chunks, list):
        raise ValueError("missing v0.8 decisions, events, or abstract chunks")
    if summary.get("termination_reason") != "chart_complete":
        raise ValueError("this audit requires a completed full Train episode")
    if len(events) != summary["total_events"] or len(decisions) != len(events):
        raise ValueError("incomplete event/decision coverage")
    by_event = {int(e["event_index"]): e for e in events}
    selected = {int(d["event_index"]): d for d in decisions}
    if len(by_event) != len(events) or len(selected) != len(decisions):
        raise ValueError("duplicate event or decision index")
    if set(by_event) != set(selected):
        raise ValueError("event and decision IDs differ")
    planned = {}
    for chunk in chunks:
        ids = range(chunk["start_event_index"], chunk["end_event_index"] + 1)
        if len(ids) != chunk["event_count"] or len(chunk["axes"]) != len(ids):
            raise ValueError("invalid chunk length")
        for event_id, axes in zip(ids, chunk["axes"]):
            if event_id in planned:
                raise ValueError("overlapping chunks")
            planned[event_id] = tuple(axes)
    if set(planned) != set(selected):
        raise ValueError("chunk plan does not cover all decided events")
    ordered_ids = sorted(selected, key=lambda index: (
        by_event[index]["episode_s"], index))
    matched_ids = [
        event_id for event_id in ordered_ids
        if planned[event_id] == tuple(selected[event_id]["axes"])
    ]
    chunk_preserved = sum(
        all(planned[index] == tuple(selected[index]["axes"])
            for index in range(chunk["start_event_index"],
                               chunk["end_event_index"] + 1))
        for chunk in chunks
    )
    planned_timeline = []
    executed_timeline = []
    for event_id in ordered_ids:
        due_ms = float(by_event[event_id]["episode_s"]) * 1000.
        row = selected[event_id]
        if abs(due_ms - float(row["due_ms"])) > .005:
            raise ValueError("due times differ between replay and telemetry")
        planned_timeline.append((due_ms, planned[event_id]))
        executed_timeline.append((due_ms, tuple(row["axes"])))
    physical_keydowns = replay["policies"]["gesture_chunk_v08"]["keydowns"]
    matching = _match_unique_keydowns(
        [selected[index] for index in ordered_ids], physical_keydowns)
    actual_keydown_timeline = [
        (float(k["time_ms"]), (k["axis"],)) for k in physical_keydowns]
    actual_keydown_timeline.sort(key=lambda item: item[0])
    return {
        "format": "dmdogo-v08-gesture-execution-audit-v1",
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "events": len(events),
        "planned_gesture_chunks": len(chunks),
        "planned_axis_adoption_count": len(matched_ids),
        "planned_axis_adoption_fraction": len(matched_ids) / len(events),
        "preserved_chunks": chunk_preserved,
        "preserved_chunk_fraction": chunk_preserved / len(chunks) if chunks else None,
        "different_event_indices": [
            i for i in ordered_ids if i not in set(matched_ids)
        ],
        "declared_local_replans": sum(
            row.get("local_replan") is True for row in decisions),
        "declared_deferred_events": sum(
            row.get("deferral_frames", 0) > 0 for row in decisions),
        "planned_due_reuse": _reuse(planned_timeline, MINIMUM_REUSE_MS),
        "chosen_due_reuse": _reuse(executed_timeline, MINIMUM_REUSE_MS),
        "physical_keydown_reuse": _reuse(
            actual_keydown_timeline, MINIMUM_REUSE_MS),
        "unique_keydown_diagnostic": {
            key: value for key, value in matching.items() if key != "matches"
        },
        "scorer_reference_only": summary["stats"],
        "limitations": [
            "Unique ±35ms KeyDown pairing is heuristic, not canonical Scorer assignment.",
            "Missing KeyDown contacts can represent matching ambiguity; not auto-Misses.",
            "80ms reuse is a chosen abstract axis constraint, not a human physiology limit.",
            "A motor-legal fallback may alter the planned gesture without a Scorer miss.",
            "This audit compares axes, not the human-likeness of actual finger techniques.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite execution audit")
    source = [json.loads(path.read_text(encoding="utf-8"))
              for path in (args.summary, args.telemetry, args.replay)]
    report = audit(*source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "events": report["events"],
        "planned_axis_adoption_fraction": report["planned_axis_adoption_fraction"],
        "preserved_chunks": report["preserved_chunks"],
        "chosen_due_reuse": report["chosen_due_reuse"],
        "physical_keydown_reuse": report["physical_keydown_reuse"],
        "unique_keydown_diagnostic": report["unique_keydown_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
