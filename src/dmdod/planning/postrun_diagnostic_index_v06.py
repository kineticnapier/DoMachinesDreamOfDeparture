"""Indexed, evaluator-only matching for v0.6–v0.8 physical replay reports.

These matches are diagnostic; they do not establish canonical scoring ownership.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import isfinite, isnan, ulp
from typing import Sequence


class PostrunDiagnosticIndex:
    """Preserve the legacy nearest-KeyDown tie break and early trace counts."""

    def __init__(self, keydowns: Sequence[dict], judgement_trace: Sequence[dict]):
        by_axis: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
        for order, value in enumerate(keydowns):
            timestamp = value["time_ms"]
            # Legacy abs(NaN - due) <= 35 is always false.
            if isfinite(timestamp):
                by_axis[value["axis"]].append((timestamp, order, value))
        self._keydowns = {}
        self._keydown_times = {}
        for axis, values in by_axis.items():
            values.sort(key=lambda item: (item[0], item[1]))
            self._keydowns[axis] = tuple(values)
            self._keydown_times[axis] = tuple(item[0] for item in values)

        early: dict[int, list[float]] = defaultdict(list)
        for trace in judgement_trace:
            if trace["margin"] is None:
                timestamp = trace["time_s"] * 1000.0
                if not isnan(timestamp):
                    early[trace["target_index"]].append(timestamp)
        self._early_times = {index: tuple(sorted(values))
                             for index, values in early.items()}

    def match_keydowns(self, axes: Sequence[str], due_ms: float) -> list[dict]:
        if not isfinite(due_ms):
            return []
        # Widen bisect bounds only to collect candidates: the final predicate
        # uses the legacy abs(time - due) <= 35.0 exactly.
        lower = due_ms - 35.0
        upper = due_ms + 35.0
        pad = 4.0 * max(ulp(due_ms), ulp(lower), ulp(upper), ulp(35.0))
        matches = []
        for axis in axes:
            times = self._keydown_times.get(axis, ())
            if not times:
                continue
            values = self._keydowns[axis]
            start = bisect_left(times, lower - pad)
            stop = bisect_right(times, upper + pad)
            eligible = (item for item in values[start:stop]
                        if abs(item[0] - due_ms) <= 35.0)
            closest = min(eligible,
                          key=lambda item: (abs(item[0] - due_ms), item[1]),
                          default=None)
            if closest is not None:
                matches.append(closest[2])
        return matches

    def count_too_early(self, ordinals: Sequence[int], due_ms: float) -> int:
        if isnan(due_ms):
            return 0
        return sum(bisect_left(self._early_times.get(index, ()), due_ms)
                   for index in set(ordinals))
