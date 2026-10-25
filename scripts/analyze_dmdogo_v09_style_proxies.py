from __future__ import annotations

"""DMDOGO v0.9 H0: review-only fingering descriptors from frozen v0.8.

No human training data, calibration, learning or Motor execution. The flags
are explicitly UNVALIDATED heuristics: not ergonomic or skill judgments.
Use saved full-chart v0.8 files and create-once outputs only.
"""

import argparse
from collections import Counter, deque
import json
from math import isfinite
from pathlib import Path

FORMAT_V08 = "dmdogo-gesture-chunk-physical-v0.8"
OUTPUT_FORMAT = "dmdogo-v09-untrained-style-proxies-v1"
HAND_GAP_MS = 120.0
LONG_GAP_MS = 450.0
WINDOW_MS = 1000.0
MIN_HAND_WINDOW_CONTACTS = 12
HAND_WINDOW_SHARE = 0.9

AXES = {
    f"{hand}_{i}" for hand in ("left", "right") for i in range(1, 5)
}


def _number(value, field: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"invalid numeric field {field}")
    try:
        if not isfinite(value):
            raise ValueError(f"non-finite field {field}")
    except OverflowError as exc:
        raise ValueError(f"non-finite field {field}") from exc
    return float(value)


def _axes(value, field: str) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(axis, str) or axis not in AXES
                   for axis in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"invalid ordinary physical axes in {field}")
    return tuple(value)


def _hand(axis: str) -> str:
    return axis.split("_")[0]


def _depth(axis: str) -> int:
    return int(axis.split("_")[1])


def _flag(kind: str, rows: list[dict], detail: dict) -> dict:
    return {
        "kind": kind,
        "start_event_index": rows[0]["event_index"],
        "end_event_index": rows[-1]["event_index"],
        "start_due_ms": round(rows[0]["due_ms"], 3),
        "end_due_ms": round(rows[-1]["due_ms"], 3),
        "detail": detail,
    }


def _validate(summary: dict, telemetry: dict, replay: dict
              ) -> tuple[list[dict], list[dict], list[dict]]:
    if (summary.get("format") != FORMAT_V08
            or replay.get("format") != FORMAT_V08
            or summary.get("planning_mode") != "gesture_v08"
            or replay.get("default_policy") != "gesture_chunk_v08"
            or not summary.get("chart_sha256")
            or summary["chart_sha256"] != replay.get("chart_sha256")
            or summary.get("chart") != replay.get("chart")):
        raise ValueError("requires matching frozen v0.8 physical artifacts")
    if summary.get("termination_reason") != "chart_complete":
        raise ValueError("requires a completed full-chart physical episode")

    rows = telemetry.get("decisions")
    events = replay.get("events")
    chunks = summary.get("abstract_schedule", {}).get("chunks")
    if not all(isinstance(item, list) for item in (rows, events, chunks)):
        raise ValueError("missing decisions, replay events or gesture chunks")
    n = summary.get("total_events")
    if type(n) is not int or n < 1 or len(rows) != n or len(events) != n:
        raise ValueError("event count does not match full-chart decision coverage")

    by_event = {}
    for row in rows:
        index = row.get("event_index")
        if type(index) is not int or index in by_event:
            raise ValueError("invalid or duplicated event index in decisions")
        chosen = _axes(row.get("axes"), f"decision {index}")
        planned = _axes(row.get("desired_axes"), f"desired {index}")
        due = _number(row.get("due_ms"), f"due_ms {index}")
        by_event[index] = {
            "event_index": index, "due_ms": due,
            "axes": chosen, "desired_axes": planned,
            "local_replan": row.get("local_replan") is True,
            "deferral_frames": row.get("deferral_frames", 0),
        }
    ids = [event.get("event_index") for event in events]
    if len(set(ids)) != n or set(ids) != set(by_event):
        raise ValueError("event IDs do not match replay")
    for event in events:
        index = event["event_index"]
        due = 1000. * _number(event.get("episode_s"), f"episode_s {index}")
        if abs(by_event[index]["due_ms"] - due) > 0.005:
            raise ValueError(f"decision timestamp mismatch for {index}")
        chord = event.get("chord_size", 1)
        if type(chord) is not int or chord != len(by_event[index]["axes"]):
            raise ValueError(f"chord size mismatch for {index}")
    ordered = sorted(by_event.values(), key=lambda r: (r["due_ms"], r["event_index"]))
    if any(b["due_ms"] < a["due_ms"]
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("non-chronological decisions")

    plan = {}
    for chunk in chunks:
        a = chunk.get("start_event_index")
        b = chunk.get("end_event_index")
        members = chunk.get("axes")
        if (type(a) is not int or type(b) is not int
                or a > b or not isinstance(members, list)
                or len(members) != b - a + 1
                or chunk.get("event_count") != len(members)):
            raise ValueError("invalid gesture-chunk boundary")
        for i, raw in enumerate(members, a):
            if i in plan:
                raise ValueError("overlapping gesture chunks")
            plan[i] = _axes(raw, f"chunk {i}")
    if set(plan) != set(by_event):
        raise ValueError("gesture chunks do not cover decisions")
    for index, row in by_event.items():
        if row["desired_axes"] != plan[index]:
            raise ValueError(f"desired axis disagrees with chunk plan at {index}")

    keydowns = (replay.get("policies", {})
                .get("gesture_chunk_v08", {}).get("keydowns"))
    if not isinstance(keydowns, list):
        raise ValueError("missing physical KeyDown trace")
    return ordered, chunks, keydowns


def analyze(summary: dict, telemetry: dict, replay: dict, *,
            max_flags: int = 100) -> dict:
    if type(max_flags) is not int or max_flags < 0:
        raise ValueError("max_flags must be a nonnegative integer")
    events, chunks, keydowns = _validate(summary, telemetry, replay)

    strokes = []
    axis_counts = Counter()
    hand_counts = Counter()
    for row in events:
        for axis in row["axes"]:
            axis_counts[axis] += 1
            hand_counts[_hand(axis)] += 1
            strokes.append((row["due_ms"], row["event_index"], axis))
    strokes.sort()
    total_contacts = len(strokes)

    flags = []
    for previous, current in zip(events, events[1:]):
        gap = current["due_ms"] - previous["due_ms"]
        if (0 <= gap <= HAND_GAP_MS and len(previous["axes"]) == 1
                and len(current["axes"]) == 1
                and _hand(previous["axes"][0]) == _hand(current["axes"][0])
                and abs(_depth(previous["axes"][0]) -
                        _depth(current["axes"][0])) >= 2):
            flags.append(_flag("quick_same_hand_depth_jump", [previous, current], {
                "gap_ms": round(gap, 3),
                "from_axis": previous["axes"][0],
                "to_axis": current["axes"][0],
            }))

    run = [events[0]]
    def finish_run() -> None:
        if len(run) >= 4:
            flags.append(_flag("rapid_single_hand_sequence", run, {
                "hand": _hand(run[0]["axes"][0]),
                "contacts": len(run),
                "axes_sample": [r["axes"][0] for r in run[:12]],
            }))

    for row in events[1:]:
        previous = run[-1]
        same_hand = (len(previous["axes"]) == 1
                     and len(row["axes"]) == 1
                     and _hand(previous["axes"][0]) == _hand(row["axes"][0]))
        if same_hand and 0 <= row["due_ms"] - previous["due_ms"] <= HAND_GAP_MS:
            run.append(row)
        else:
            finish_run()
            run = [row]
    finish_run()

    window = deque()
    rolling_hand = Counter()
    peak_one_second = {"left": 0, "right": 0}
    last_flagged_window = {"left": float("-inf"), "right": float("-inf")}
    for at, index, axis in strokes:
        hand = _hand(axis)
        window.append((at, index, hand))
        rolling_hand[hand] += 1
        while window and at - window[0][0] > WINDOW_MS:
            _, _, old = window.popleft()
            rolling_hand[old] -= 1
        for candidate in ("left", "right"):
            peak_one_second[candidate] = max(
                peak_one_second[candidate], rolling_hand[candidate])
        if (len(window) >= MIN_HAND_WINDOW_CONTACTS
                and rolling_hand[hand] / len(window) >= HAND_WINDOW_SHARE
                and at - last_flagged_window[hand] >= WINDOW_MS):
            first = window[0][1]
            flags.append({
                "kind": "one_second_hand_concentration",
                "start_event_index": first,
                "end_event_index": index,
                "start_due_ms": round(window[0][0], 3),
                "end_due_ms": round(at, 3),
                "detail": {
                    "hand": hand,
                    "contacts_in_window": len(window),
                    "hand_share": round(rolling_hand[hand] / len(window), 4),
                },
            })
            last_flagged_window[hand] = at

    short_adjacent = 0
    short_same_hand = 0
    short_hand_switch = 0
    for prev, curr in zip(events, events[1:]):
        gap = curr["due_ms"] - prev["due_ms"]
        if not 0 <= gap <= LONG_GAP_MS:
            continue
        if len(prev["axes"]) == len(curr["axes"]) == 1:
            short_adjacent += 1
            if _hand(prev["axes"][0]) == _hand(curr["axes"][0]):
                short_same_hand += 1
            else:
                short_hand_switch += 1

    lag8_denominator = max(len(events) - 8, 0)
    lag8_equal = sum(
        events[i]["axes"] == events[i-8]["axes"]
        for i in range(8, len(events))
    )
    intact = sum(
        row["axes"] == row["desired_axes"] for row in events)
    chunk_counts = Counter(chunk.get("gesture_family", "unknown")
                           for chunk in chunks)

    flags.sort(key=lambda f: (
        f["start_due_ms"], f["end_due_ms"], f["kind"],
    ))
    flag_counts = dict(sorted(Counter(f["kind"] for f in flags).items()))
    return {
        "format": OUTPUT_FORMAT,
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "source_policy": "gesture_chunk_v08",
        "event_count": len(events),
        "planned_chunks": len(chunks),
        "total_contacts": total_contacts,
        "metrics": {
            "axis_contact_counts": dict(sorted(axis_counts.items())),
            "hand_contact_counts": {
                "left": hand_counts["left"], "right": hand_counts["right"],
            },
            "left_contact_fraction": round(
                hand_counts["left"] / total_contacts, 6),
            "short_adjacent_event_pairs": short_adjacent,
            "short_same_hand_pairs": short_same_hand,
            "short_hand_switch_pairs": short_hand_switch,
            "within_gap_hand_switch_fraction": (
                round(short_hand_switch / short_adjacent, 6)
                if short_adjacent else None),
            "lag8_same_axes_fraction": (
                round(lag8_equal / lag8_denominator, 6)
                if lag8_denominator else None),
            "max_hand_contacts_in_rolling_second": peak_one_second,
            "chunk_family_counts": dict(sorted(chunk_counts.items())),
            "planned_axis_adoption_fraction": round(
                intact / len(events), 6),
            "local_replan_count": sum(r["local_replan"] for r in events),
            "deferred_event_count": sum(
                r["deferral_frames"] > 0 for r in events),
        },
        "flag_counts": flag_counts,
        "total_flags": len(flags),
        "returned_flags": min(len(flags), max_flags),
        "review_flags": flags[:max_flags],
        "human_demonstrations_used": False,
        "human_likeness_verified": False,
        "scores_are_human_preferences": False,
        "flag_policy": {
            "rapid_run_min_contacts": 4,
            "rapid_run_max_gap_ms": HAND_GAP_MS,
            "depth_jump_min_indices": 2,
            "rolling_window_ms": WINDOW_MS,
            "rolling_window_min_contacts": MIN_HAND_WINDOW_CONTACTS,
            "rolling_window_hand_share": HAND_WINDOW_SHARE,
            "thresholds_calibrated_to_humans": False,
        },
        "limitations": [
            "Review flags are untrained qualitative proxies, not bad-fingering labels.",
            "Physical axes 1-4 are provisional finger proxies, not observed anatomy.",
            "Only scheduled/selected axes and due times are scored, not hand video.",
            "This does not verify HUD-only observation or imitation.",
            "The physical Scorer's Hit/X/MAE must be assessed separately.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-flags", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite style audit")
    source = [json.loads(path.read_text(encoding="utf-8"))
              for path in (args.summary, args.telemetry, args.replay)]
    report = analyze(*source, max_flags=args.max_flags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "chart": report["chart"],
        "events": report["event_count"],
        "hand_contact_counts": report["metrics"]["hand_contact_counts"],
        "chunk_family_counts": report["metrics"]["chunk_family_counts"],
        "flag_counts": report["flag_counts"],
        "flag_report": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
