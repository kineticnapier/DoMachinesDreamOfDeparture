from __future__ import annotations

"""Causal command-to-KeyDown diagnostics for frozen DMDOGO replays."""

from collections import defaultdict
from statistics import mean, median
from typing import Mapping, Sequence


def pair_fifo_events(
    events: Sequence[dict],
    keydowns: Sequence[dict],
    *,
    policy: str,
    stats: Mapping[str, object],
    bpm_by_source_floor: Mapping[int, float] | None = None,
) -> list[dict]:
    """Pair a complete all-Hit repository trace by scorer order.

    This deliberately refuses partial traces.  It does not use a nearest-time
    heuristic, which is ambiguous in the 31.245 ms bool sequences.
    """

    complete = (
        int(stats.get("hits", -1)) == len(events)
        and int(stats.get("misses", -1)) == 0
        and int(stats.get("too_early", -1)) == 0
        and not bool(stats.get("overloaded", True))
        and int(stats.get("physical_keydowns", -1)) == len(keydowns) == len(events)
        and all(int(event.get("chord_size", 0)) == 1 for event in events)
    )
    if not complete:
        raise ValueError("latency pairing requires a complete all-Hit one-contact trace")

    last_due_by_axis: dict[str, float] = {}
    rows = []
    for event, keydown in zip(events, keydowns):
        decision = event["decisions"][policy]
        axes = tuple(str(axis) for axis in decision.get("axes", ()))
        if len(axes) != 1:
            raise ValueError("latency pairing requires one selected physical axis")
        due_s = float(event["episode_s"])
        decision_s = float(decision["decision_s"])
        keydown_s = float(keydown["episode_s"])
        selected_axis = axes[0]
        physical_axis = str(keydown["axis"])
        previous_due = last_due_by_axis.get(selected_axis)
        source_floor = min(int(value) for value in event["floor_indices"]) - 1
        rows.append({
            "event_index": int(event["event_index"]),
            "due_s": due_s,
            "decision_s": decision_s,
            "keydown_s": keydown_s,
            "axis": selected_axis,
            "physical_axis": physical_axis,
            "axis_matches": selected_axis == physical_axis,
            "scheduled_lead_ms": 1000.0 * (due_s - decision_s),
            "command_to_keydown_ms": 1000.0 * (keydown_s - decision_s),
            "timing_error_ms": 1000.0 * (keydown_s - due_s),
            "axis_reuse_ms": (None if previous_due is None
                              else 1000.0 * (due_s - previous_due)),
            "bpm": (None if bpm_by_source_floor is None
                    else float(bpm_by_source_floor[source_floor])),
            "control_phase_ms": (1000.0 * due_s) % 10.0,
        })
        last_due_by_axis[selected_axis] = due_s
    return rows


def _distribution(values: Sequence[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"minimum": None, "q25": None, "median": None, "q75": None,
                "maximum": None, "mean": None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": min(ordered), "q25": percentile(.25),
        "median": median(ordered), "q75": percentile(.75),
        "maximum": max(ordered), "mean": mean(ordered),
    }


def _summary(rows: Sequence[dict]) -> dict:
    return {
        "count": len(rows),
        "latency_ms": _distribution([row["command_to_keydown_ms"] for row in rows]),
        "timing_error_ms": _distribution([row["timing_error_ms"] for row in rows]),
        "scheduled_lead_ms": _distribution([row["scheduled_lead_ms"] for row in rows]),
        "axis_mismatch_count": sum(not row["axis_matches"] for row in rows),
    }


def analyze_latency_rows(rows: Sequence[dict]) -> dict:
    by_axis: dict[str, list[dict]] = defaultdict(list)
    by_bpm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_axis[str(row["axis"])].append(row)
        if row.get("bpm") is not None:
            by_bpm[str(round(float(row["bpm"])))].append(row)
    short_reuse = [row for row in rows if row.get("axis_reuse_ms") is not None
                   and float(row["axis_reuse_ms"]) <= 125.0 + 1e-9]
    long_reuse = [row for row in rows if row.get("axis_reuse_ms") is not None
                  and float(row["axis_reuse_ms"]) > 125.0 + 1e-9]
    return {
        "overall": _summary(rows),
        "by_axis": {key: _summary(value) for key, value in sorted(by_axis.items())},
        "by_bpm": {key: _summary(value) for key, value in sorted(by_bpm.items())},
        "reuse_le_125ms": _summary(short_reuse),
        "reuse_gt_125ms": _summary(long_reuse),
        "evidence_contract": {
            "command_onset": "decision_s; verified for the frozen runner because selection and first action occur in the same control frame",
            "pairing": "repository FIFO order, only for complete one-contact all-Hit traces",
            "not_claimed": "isolated causal effect of one selection; each episode has coupled Motor state",
        },
    }
