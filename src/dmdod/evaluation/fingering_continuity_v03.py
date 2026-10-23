from __future__ import annotations

"""Offline diagnostics for DMDOGO fingering continuity experiments."""

from bisect import bisect_right
from collections import Counter
from statistics import mean, median
from typing import Mapping, Sequence

from dmdod.adofai.rules import classify_timing, x_accuracy_percent
from dmdod.motor.contact_16k import PHYSICAL_AXES
from dmdod.planning.fingering_continuity_v02 import continuity_metrics


def associate_event_keydowns(
    events: Sequence[dict],
    keydowns: Sequence[dict],
    *,
    policy: str,
    stats: Mapping[str, object] | None,
    window_ms: float = 35.0,
) -> dict:
    """Associate saved events and KeyDowns without hiding ambiguity.

    A complete one-contact, all-Hit repository trace has an exact FIFO order
    correspondence because the scorer consumes the oldest unresolved target.
    Outside that narrowly verified condition, all same-axis candidates in the
    time window are retained and the result is explicitly ambiguous when more
    than one exists.
    """

    complete_fifo = bool(
        stats is not None
        and int(stats.get("hits", -1)) == len(events)
        and int(stats.get("misses", -1)) == 0
        and int(stats.get("too_early", -1)) == 0
        and not bool(stats.get("overloaded", True))
        and int(stats.get("physical_keydowns", -1)) == len(keydowns) == len(events)
        and all(int(event.get("chord_size", 0)) == 1 for event in events)
    )
    rows = []
    if complete_fifo:
        for index, (event, keydown) in enumerate(zip(events, keydowns)):
            axes = tuple(event["decisions"][policy].get("axes", ()))
            axis_matches = len(axes) == 1 and axes[0] == keydown.get("axis")
            rows.append({
                "event_index": int(event["event_index"]),
                "keydown_index": index,
                "candidate_keydown_indices": [index],
                "status": "unique",
                "axis_matches": axis_matches,
                "error_ms": 1000.0 * (
                    float(keydown["episode_s"]) - float(event["episode_s"])
                ),
            })
        method = "repository_fifo_complete_hit_order"
    else:
        for event in events:
            axes = set(event["decisions"][policy].get("axes", ()))
            due_s = float(event["episode_s"])
            candidates = [
                index for index, keydown in enumerate(keydowns)
                if keydown.get("axis") in axes
                and abs(1000.0 * (float(keydown["episode_s"]) - due_s)) <= window_ms
            ]
            status = "unique" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched"
            rows.append({
                "event_index": int(event["event_index"]),
                "keydown_index": candidates[0] if len(candidates) == 1 else None,
                "candidate_keydown_indices": candidates,
                "status": status,
                "axis_matches": bool(candidates),
                "error_ms": (1000.0 * (float(keydowns[candidates[0]]["episode_s"]) - due_s)
                             if len(candidates) == 1 else None),
            })
        method = "axis_time_window_candidates"
    counts = Counter(row["status"] for row in rows)
    axis_mismatches = sum(not row["axis_matches"] for row in rows
                          if row["status"] == "unique")
    return {
        "method": method,
        "window_ms": window_ms,
        "rows": rows,
        "unique_count": counts["unique"],
        "ambiguous_count": counts["ambiguous"],
        "unmatched_count": counts["unmatched"],
        "inconsistent_count": counts["inconsistent"],
        "axis_mismatch_count": axis_mismatches,
    }


def summarize_error_rows(rows: Sequence[dict], *, pitch_ratio: float) -> dict:
    if not rows:
        return {"count": 0, "signed_error_ms": None, "timing_mae_ms": None,
                "x_accuracy_percent": None}
    errors = [float(row["error_ms"]) for row in rows]
    judgements = [
        classify_timing(float(row["error_ms"]) * .001, float(row["bpm"]),
                        pitch=pitch_ratio)
        for row in rows
    ]
    return {
        "count": len(rows),
        "signed_error_ms": mean(errors),
        "timing_mae_ms": mean(abs(value) for value in errors),
        "x_accuracy_percent": x_accuracy_percent(judgements),
        "judgements": dict(sorted(Counter(value.value for value in judgements).items())),
    }


def _fast_event_indices(event_times: Mapping[int, float], threshold_ms: float) -> list[int]:
    ordered = sorted(event_times.items(), key=lambda item: (item[1], item[0]))
    selected: set[int] = set()
    for (left_index, left_time), (right_index, right_time) in zip(ordered, ordered[1:]):
        if 1000.0 * (right_time - left_time) <= threshold_ms + 1e-9:
            selected.update((left_index, right_index))
    return sorted(selected)


def common_fast_metrics(
    decisions_by_policy: Mapping[str, Sequence[dict]],
    event_times: Mapping[int, float],
    *,
    threshold_ms: float = 35.0,
) -> dict:
    common_indices = set(event_times)
    for decisions in decisions_by_policy.values():
        common_indices &= {int(row["event_index"]) for row in decisions}
    common_event_times = {index: event_times[index] for index in common_indices}
    indices = _fast_event_indices(common_event_times, threshold_ms)
    selected = set(indices)
    policies = {}
    ordered_events = sorted(common_event_times.items(), key=lambda item: (item[1], item[0]))
    fixed_windows = [
        tuple(index for index, _ in ordered_events[start:start + 4])
        for start in range(max(0, len(ordered_events) - 3))
        if all(
            1000.0 * (ordered_events[offset + 1][1] - ordered_events[offset][1])
            <= threshold_ms + 1e-9
            for offset in range(start, start + 3)
        )
    ]
    for name, decisions in decisions_by_policy.items():
        rows = [dict(row) for row in decisions if int(row["event_index"]) in selected]
        metrics = continuity_metrics(rows)
        by_index = {int(row["event_index"]): row for row in decisions}
        stable = 0
        for window in fixed_windows:
            axes = [tuple(str(axis) for axis in by_index[index].get("axes", ()))
                    for index in window]
            if any(len(value) != 1 for value in axes):
                continue
            parts = [axis.split("_") for (axis,) in axes]
            if len({part[0] for part in parts}) != 1:
                continue
            depths = [int(part[1]) for part in parts]
            if any(depth > 4 for depth in depths):
                continue
            steps = [(right - left) % 4 for left, right in zip(depths, depths[1:])]
            stable += int(steps[-1] in steps[:-1])
        metrics["fixed_window_stability"] = (
            stable / len(fixed_windows) if fixed_windows else None
        )
        metrics["fixed_window_stable"] = stable
        metrics["fixed_window_comparisons"] = len(fixed_windows)
        policies[name] = metrics
    return {
        "threshold_ms": threshold_ms,
        "event_indices": indices,
        "fixed_windows": [list(window) for window in fixed_windows],
        "policies": policies,
    }


def _decision_rows(payload: dict, policy: str, association: dict) -> list[dict]:
    floors = {int(row["index"]): row for row in payload["floors"]}
    association_by = {row["event_index"]: row for row in association["rows"]}
    rows = []
    for event in payload["events"]:
        match = association_by[int(event["event_index"])]
        if match["status"] != "unique":
            continue
        source_floor = min(int(value) for value in event["floor_indices"]) - 1
        decision = event["decisions"][policy]
        rows.append({
            "event_index": int(event["event_index"]),
            "episode_s": float(event["episode_s"]),
            "bpm": float(floors[source_floor]["bpm"]),
            "error_ms": float(match["error_ms"]),
            "due_ms": 1000.0 * float(event["episode_s"]),
            "axes": list(decision.get("axes", ())),
            "chosen": decision.get("chosen"),
            "decision_s": float(decision["decision_s"]),
        })
    return rows


def _group_summaries(rows: Sequence[dict], key, *, pitch_ratio: float) -> list[dict]:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return [
        {"group": name, **summarize_error_rows(values, pitch_ratio=pitch_ratio)}
        for name, values in sorted(groups.items(), key=lambda item: item[0])
    ]


def _motor_state(trace: Sequence[list], at_s: float, axis: str) -> dict:
    times = [float(frame[0]) for frame in trace]
    frame_index = max(0, bisect_right(times, at_s) - 1)
    axis_index = PHYSICAL_AXES.index(axis)
    frame = trace[frame_index]
    return {
        "sample_s": float(frame[0]),
        "position_m": float(frame[1][axis_index]),
        "velocity_m_s": float(frame[2][axis_index]),
        "pressed": bool(frame[3][axis_index]),
    }


def analyze_saved_replay(payload: dict) -> dict:
    """Analyze a saved Free/v0.2 replay; no inference or physics is rerun."""

    pitch_ratio = float(payload["timebase"]["pitch_ratio"])
    policies = ("free", "continuity")
    associations = {}
    rows_by = {}
    decision_rows = {}
    event_times = {int(event["event_index"]): float(event["episode_s"])
                   for event in payload["events"]}
    for policy in policies:
        policy_data = payload["policies"][policy]
        associations[policy] = associate_event_keydowns(
            payload["events"], policy_data["keydowns"], policy=policy,
            stats=policy_data["saved_metrics"],
        )
        rows_by[policy] = _decision_rows(payload, policy, associations[policy])
        decision_rows[policy] = [
            {"event_index": row["event_index"], "due_ms": row["due_ms"],
             "axes": row["axes"]}
            for row in rows_by[policy]
        ]

    free_by = {row["event_index"]: row for row in rows_by["free"]}
    continuity_by = {row["event_index"]: row for row in rows_by["continuity"]}
    common = sorted(free_by.keys() & continuity_by.keys())
    changed_rows = []
    for index in common:
        left, right = free_by[index], continuity_by[index]
        free_axis = left["axes"][0]
        continuity_axis = right["axes"][0]
        free_depth = int(free_axis.split("_")[1])
        continuity_depth = int(continuity_axis.split("_")[1])
        category = (
            "unchanged" if free_axis == continuity_axis
            else "thumb_to_ordinary" if free_depth == 5 and continuity_depth <= 4
            else "ordinary_to_thumb" if free_depth <= 4 and continuity_depth == 5
            else "other_changed"
        )
        changed_rows.append({
            "event_index": index, "episode_s": left["episode_s"],
            "bpm": left["bpm"], "category": category,
            "free_axis": free_axis, "continuity_axis": continuity_axis,
            "free_error_ms": left["error_ms"],
            "continuity_error_ms": right["error_ms"],
            "absolute_error_delta_ms": abs(right["error_ms"]) - abs(left["error_ms"]),
        })

    category_summaries = []
    for category in sorted(set(row["category"] for row in changed_rows)):
        values = [row for row in changed_rows if row["category"] == category]
        category_summaries.append({
            "category": category, "count": len(values),
            "free_mae_ms": mean(abs(row["free_error_ms"]) for row in values),
            "continuity_mae_ms": mean(abs(row["continuity_error_ms"]) for row in values),
            "mean_absolute_error_delta_ms": mean(row["absolute_error_delta_ms"] for row in values),
        })

    axis_response = {}
    for policy in policies:
        trace = payload["policies"][policy]["trace"]
        groups: dict[str, list[dict]] = {}
        for row in rows_by[policy]:
            axis = row["axes"][0]
            state = _motor_state(trace, row["decision_s"], axis)
            groups.setdefault(axis, []).append({
                **row, **state,
                "response_delay_ms": 1000.0 * (row["episode_s"] + row["error_ms"] * .001
                                                - row["decision_s"]),
            })
        axis_response[policy] = [
            {
                "axis": axis, "count": len(values),
                "signed_error_ms": mean(value["error_ms"] for value in values),
                "timing_mae_ms": mean(abs(value["error_ms"]) for value in values),
                "mean_response_delay_ms": mean(value["response_delay_ms"] for value in values),
                "mean_decision_position_m": mean(value["position_m"] for value in values),
                "mean_decision_velocity_m_s": mean(value["velocity_m_s"] for value in values),
                "pressed_at_decision_count": sum(value["pressed"] for value in values),
            }
            for axis, values in sorted(groups.items())
        ]

    return {
        "format": "dmdogo-fingering-continuity-cause-audit-v0.3",
        "source_format": payload.get("format"),
        "chart": payload.get("chart"),
        "chart_sha256": payload.get("chart_sha256"),
        "association": {
            policy: {key: value for key, value in associations[policy].items() if key != "rows"}
            for policy in policies
        },
        "by_bpm": {
            policy: _group_summaries(rows_by[policy], lambda row: round(row["bpm"]),
                                     pitch_ratio=pitch_ratio)
            for policy in policies
        },
        "by_time_5s": {
            policy: _group_summaries(rows_by[policy],
                                     lambda row: int(row["episode_s"] // 5) * 5,
                                     pitch_ratio=pitch_ratio)
            for policy in policies
        },
        "change_categories": category_summaries,
        "motor_axis_response": axis_response,
        "common_fast": common_fast_metrics(decision_rows, event_times),
        "worst_absolute_error_deltas": sorted(
            changed_rows, key=lambda row: row["absolute_error_delta_ms"], reverse=True
        )[:50],
        "evidence_limits": {
            "causality": "Independent episodes make category differences causal candidates, not isolated-event effects.",
            "activation": "Saved v0.2 trace contains position, velocity and pressed state, but not per-axis activation.",
            "stock_scope": "Association uses the repository FIFO scorer contract, not a claim about live Stock ADOFAI.",
        },
    }
