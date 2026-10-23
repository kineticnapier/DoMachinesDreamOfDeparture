from __future__ import annotations

"""Posthoc causal diagnostics for Continuity-v0.2 plus Motor compensation."""

from collections import defaultdict
from statistics import mean, median
from typing import Mapping, Sequence


def _distribution(values: Sequence[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "minimum": None, "median": None,
                "maximum": None, "mean": None}
    return {"count": len(ordered), "minimum": ordered[0],
            "median": median(ordered), "maximum": ordered[-1],
            "mean": mean(ordered)}


def _thumb(row: Mapping[str, object]) -> bool:
    return any(str(axis).endswith("_5") for axis in row.get("axes", ()))


def analyze_continuity_compensation(
    continuity: Sequence[dict],
    compensated: Sequence[dict],
    *,
    bpm_by_event: Mapping[int, float] | None = None,
) -> dict:
    """Compare two independently executed causal histories by event ID.

    This is descriptive only. A changed event can alter every later Motor state,
    so the report never labels a per-event difference as an isolated effect.
    """

    left = {int(row["event_index"]): row for row in continuity}
    right = {int(row["event_index"]): row for row in compensated}
    common = sorted(left.keys() & right.keys())
    changed = [index for index in common
               if left[index].get("chosen") != right[index].get("chosen")]
    free_changed = [index for index in common
                    if left[index].get("free_chosen") != right[index].get("free_chosen")]
    decision_advance = [float(left[index]["decision_ms"])
                        - float(right[index]["decision_ms"]) for index in common]
    fast = []
    previous = None
    for index in common:
        due = float(right[index]["due_ms"])
        if previous is not None and due - previous <= 35.0 + 1e-9:
            fast.append(index)
        previous = due

    preferred_checked = preferred_unavailable = 0
    for index in changed:
        row = right[index]
        labels = list(row.get("candidate_labels", ()))
        if not labels:
            continue
        wanted = left[index].get("chosen")
        if wanted not in labels:
            continue
        preferred_checked += 1
        preferred_unavailable += not bool(row["eligible"][labels.index(wanted)])

    groups: dict[str, list[int]] = defaultdict(list)
    if bpm_by_event is not None:
        for index in common:
            if index in bpm_by_event:
                groups[str(round(float(bpm_by_event[index])))] .append(index)
    by_bpm = {}
    for name, indices in sorted(groups.items()):
        errors = [float(right[index]["actual_error_ms"])
                  for index in indices if right[index].get("actual_error_ms") is not None]
        by_bpm[name] = {
            "events": len(indices),
            "changed_choices": sum(index in changed for index in indices),
            "thumb_uses": sum(_thumb(right[index]) for index in indices),
            "timing_error_ms": _distribution(errors),
        }

    return {
        "matched_events": len(common),
        "choice_preserved": len(common) - len(changed),
        "choice_changed": len(changed),
        "choice_changed_rate": len(changed) / len(common) if common else None,
        "free_argmin_changed": len(free_changed),
        "decision_advance_ms": _distribution(decision_advance),
        "fast_events": len(fast),
        "fast_choice_preserved": sum(
            left[index].get("chosen") == right[index].get("chosen") for index in fast),
        "v02_preferred_live_eligibility": {
            "checked_changed_events": preferred_checked,
            "unavailable": preferred_unavailable,
            "available_but_not_selected": preferred_checked - preferred_unavailable,
        },
        "continuity_thumb_uses": sum(_thumb(left[index]) for index in common),
        "compensated_thumb_uses": sum(_thumb(right[index]) for index in common),
        "by_bpm": by_bpm,
        "evidence_limit": (
            "Independent episodes are coupled histories; per-event differences are "
            "causal candidates, not isolated treatment effects."
        ),
    }
