from __future__ import annotations

"""Offline pattern-first discrete key scheduler.

The model intentionally has no hand, fatigue, strength, Motor, Network or
judgement concepts.  It optimizes abstract logical-key cycles over known Oracle
event times.  Physical realization is a separate contract.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import mean, median
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PatternEvent:
    event_index: int
    due_s: float
    chord_size: int = 1


@dataclass(frozen=True, slots=True)
class PatternSpan:
    start_event_index: int
    end_event_index: int
    width: int
    cycle: tuple[str, ...]
    event_count: int


@dataclass(frozen=True, slots=True)
class PatternSchedule:
    feasible: bool
    assignments: tuple[tuple[str, ...], ...]
    event_times_s: tuple[float, ...]
    patterns: tuple[PatternSpan, ...]
    pattern_change_indices: tuple[int, ...]
    objective: tuple[int, int, tuple] | None
    failure_reason: str | None
    states_evaluated: int
    model_contract: dict


@dataclass(frozen=True, slots=True)
class _Candidate:
    width: int
    cycle: tuple[str, ...]
    assignments: tuple[tuple[str, ...], ...]
    first_use: tuple[tuple[str, float], ...]
    last_use: tuple[tuple[str, float], ...]
    signature: tuple


def _reuse_valid(previous: dict[str, float], current: dict[str, float], minimum: float) -> bool:
    return all(
        key not in previous
        or (due - previous[key] > 1e-12
            and due - previous[key] + 1e-12 >= minimum)
        for key, due in current.items()
    )


class PatternFirstScheduler:
    """Deterministic segment-DP over bounded periodic cycle candidates.

    For ``S`` detected timing segments, ``K`` logical keys and ``N`` strike
    tokens, candidate construction is O(N*K^2); adjacent-segment DP is
    O(S*K^4) in the loose bound and small in practice (at most K(K+1)/2
    rotations per segment).  No candidate graph grows with chart length.
    """

    def __init__(self, logical_keys: Sequence[str], *, minimum_reuse_s: float,
                 minimum_cycle_keys: int = 2, gap_change_ratio: float | None = None,
                 restart_gap_s: float = .500) -> None:
        keys = tuple(str(value) for value in logical_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("logical keys must be non-empty and unique")
        if not isfinite(minimum_reuse_s) or minimum_reuse_s < 0:
            raise ValueError("minimum reuse must be finite and non-negative")
        if minimum_cycle_keys < 1 or minimum_cycle_keys > len(keys):
            raise ValueError("minimum cycle width is outside the key set")
        if ((gap_change_ratio is not None
                 and (not isfinite(gap_change_ratio) or gap_change_ratio < 0))
                or not isfinite(restart_gap_s) or restart_gap_s <= 0):
            raise ValueError("invalid segmentation parameters")
        self.logical_keys = keys
        self.minimum_reuse_s = float(minimum_reuse_s)
        self.minimum_cycle_keys = int(minimum_cycle_keys)
        self.gap_change_ratio = (None if gap_change_ratio is None
                                 else float(gap_change_ratio))
        self.restart_gap_s = float(restart_gap_s)

    def _contract(self) -> dict:
        return {
            "minimum_reuse_s": self.minimum_reuse_s,
            "minimum_cycle_keys": self.minimum_cycle_keys,
            "gap_change_ratio": self.gap_change_ratio,
            "restart_gap_s": self.restart_gap_s,
            "cycle_floor_semantics": (
                "A declared anti-degenerate regularity constraint, not a physiological limit."
            ),
            "future_information": "Oracle event times and chord sizes only",
            "excluded": ["Motor state", "Network score", "future KeyDown", "judgement"],
        }

    def _segments(self, events: tuple[PatternEvent, ...]) -> tuple[tuple[PatternEvent, ...], ...]:
        if len(events) < 2:
            return (events,)
        boundaries = [0]
        gaps = [events[index].due_s - events[index - 1].due_s
                for index in range(1, len(events))]
        # The first gap has no preceding gap to compare with, but can still
        # be a genuine restarts-after-pause boundary.
        if gaps[0] >= self.restart_gap_s:
            boundaries.append(1)
        for gap_index in range(1, len(gaps)):
            previous, current = gaps[gap_index - 1], gaps[gap_index]
            scale = max(abs(previous), abs(current), 1e-12)
            changed_regime = (self.gap_change_ratio is not None
                              and abs(current - previous) / scale > self.gap_change_ratio)
            if current >= self.restart_gap_s or changed_regime:
                boundaries.append(gap_index + 1)
        boundaries.append(len(events))
        return tuple(events[left:right] for left, right in zip(boundaries, boundaries[1:]))

    def _candidates(self, events: tuple[PatternEvent, ...]) -> tuple[_Candidate, ...]:
        minimum_width = max(self.minimum_cycle_keys,
                            max(event.chord_size for event in events))
        result = []
        for width in range(minimum_width, len(self.logical_keys) + 1):
            base = self.logical_keys[:width]
            for phase in range(width):
                cycle = base[phase:] + base[:phase]
                cursor = 0
                assignments = []
                first_use: dict[str, float] = {}
                last_use: dict[str, float] = {}
                valid = True
                for event in events:
                    keys = tuple(cycle[(cursor + offset) % width]
                                 for offset in range(event.chord_size))
                    cursor += event.chord_size
                    if len(set(keys)) != len(keys):
                        valid = False
                        break
                    for key in keys:
                        if (key in last_use and (
                                event.due_s - last_use[key] <= 1e-12
                                or event.due_s - last_use[key] + 1e-12
                                < self.minimum_reuse_s)):
                            valid = False
                            break
                        first_use.setdefault(key, event.due_s)
                        last_use[key] = event.due_s
                    if not valid:
                        break
                    assignments.append(keys)
                if valid:
                    result.append(_Candidate(
                        width, cycle, tuple(assignments),
                        tuple(sorted(first_use.items())),
                        tuple(sorted(last_use.items())),
                        tuple(key for keys in assignments for key in keys),
                    ))
        return tuple(result)

    def schedule(self, events: Sequence[PatternEvent]) -> PatternSchedule:
        ordered = tuple(events)
        if any(not isfinite(event.due_s) for event in ordered):
            return PatternSchedule(False, (), (), (), (), None,
                                   "non_finite_event_time", 0, self._contract())
        if any(event.chord_size <= 0 for event in ordered):
            return PatternSchedule(False, (), (), (), (), None,
                                   "invalid_chord_size", 0, self._contract())
        if any(right.due_s < left.due_s
               for left, right in zip(ordered, ordered[1:])):
            return PatternSchedule(False, (), (), (), (), None,
                                   "non_monotonic_event_times", 0, self._contract())
        if any(event.chord_size > len(self.logical_keys) for event in ordered):
            return PatternSchedule(False, (), (), (), (), None,
                                   "chord_exceeds_logical_key_count", 0, self._contract())
        if not ordered:
            return PatternSchedule(True, (), (), (), (), (0, 0, ()), None, 0, self._contract())

        segments = self._segments(ordered)
        candidates = tuple(self._candidates(segment) for segment in segments)
        evaluated = sum(len(value) for value in candidates)
        if any(not value for value in candidates):
            return PatternSchedule(False, (), (), (), (), None,
                                   "no_periodic_assignment_satisfies_reuse", evaluated,
                                   self._contract())

        # DP must retain a Pareto frontier per current cycle. Two histories
        # ending in the same cycle can have different last-use times for keys
        # omitted by short intervening segments. Pruning solely by current
        # cycle loses feasible (and sometimes optimal) continuations.
        #
        # History A dominates B only if it has no worse objective AND every
        # physical key was last used no later than in B. A missing last use
        # counts as -infinity. Exact dominance preserves the optimum; the
        # explicit cap fails closed rather than silently pruning valid plans.
        max_frontier_states = 10000

        def dominates(left, right) -> bool:
            left_obj, _, left_last = left
            right_obj, _, right_last = right
            return (left_obj <= right_obj
                    and all(left_last.get(key, float("-inf"))
                            <= right_last.get(key, float("-inf"))
                            for key in self.logical_keys))

        states: dict[int, list[tuple[tuple[int, int, tuple], tuple[int, ...],
                                     dict[str, float]]]] = {}
        first_tokens = sum(event.chord_size for event in segments[0])
        for index, candidate in enumerate(candidates[0]):
            objective = (candidate.width * first_tokens, 0, (candidate.signature,))
            states[index] = [(objective, (index,), dict(candidate.last_use))]

        for segment_index in range(1, len(segments)):
            next_states = {}
            tokens = sum(event.chord_size for event in segments[segment_index])
            retained = 0
            for current_index, current in enumerate(candidates[segment_index]):
                frontier = []
                for previous_index, previous_states in states.items():
                    previous = candidates[segment_index - 1][previous_index]
                    for objective, path, global_last_use in previous_states:
                        evaluated += 1
                        if not _reuse_valid(global_last_use, dict(current.first_use),
                                            self.minimum_reuse_s):
                            continue
                        changed = previous.cycle != current.cycle
                        candidate_objective = (
                            objective[0] + current.width * tokens,
                            objective[1] + int(changed),
                            objective[2] + (current.signature,),
                        )
                        next_last_use = dict(global_last_use)
                        next_last_use.update(dict(current.last_use))
                        value = (candidate_objective, path + (current_index,),
                                 next_last_use)
                        if any(dominates(existing, value) for existing in frontier):
                            continue
                        frontier = [existing for existing in frontier
                                    if not dominates(value, existing)]
                        frontier.append(value)
                        if len(frontier) > max_frontier_states:
                            return PatternSchedule(
                                False, (), (), (), (), None,
                                "dp_frontier_state_limit_exceeded", evaluated, self._contract())
                if frontier:
                    next_states[current_index] = frontier
                    retained += len(frontier)
                    if retained > max_frontier_states:
                        return PatternSchedule(
                            False, (), (), (), (), None,
                            "dp_frontier_state_limit_exceeded", evaluated, self._contract())
            states = next_states
            if not states:
                return PatternSchedule(False, (), (), (), (), None,
                                       "no_cross_segment_reuse_safe_path", evaluated,
                                       self._contract())

        objective, path, _ = min(
            (state for frontier in states.values() for state in frontier),
            key=lambda state: state[0],
        )
        chosen = tuple(candidates[index][candidate_index]
                       for index, candidate_index in enumerate(path))
        assignments = tuple(keys for candidate in chosen for keys in candidate.assignments)
        patterns = []
        changes = []
        offset = 0
        previous_cycle = None
        for segment, candidate in zip(segments, chosen):
            if previous_cycle is not None and previous_cycle != candidate.cycle:
                changes.append(segment[0].event_index)
            patterns.append(PatternSpan(
                segment[0].event_index, segment[-1].event_index,
                candidate.width, candidate.cycle, len(segment)))
            previous_cycle = candidate.cycle
            offset += len(segment)
        return PatternSchedule(True, assignments,
                               tuple(event.due_s for event in ordered),
                               tuple(patterns), tuple(changes),
                               objective, None, evaluated, self._contract())


def schedule_metrics(schedule: PatternSchedule) -> dict:
    if not schedule.feasible:
        return {"feasible": False, "failure_reason": schedule.failure_reason}
    last_use: dict[str, float] = {}
    intervals = []
    for due_s, keys in zip(schedule.event_times_s, schedule.assignments):
        for key in keys:
            if key in last_use:
                intervals.append(1000.0 * (due_s - last_use[key]))
            last_use[key] = due_s
    run_lengths = [span.event_count for span in schedule.patterns]
    return {
        "feasible": True,
        "pattern_count": len(schedule.patterns),
        "pattern_changes": len(schedule.pattern_change_indices),
        "mean_pattern_run_events": mean(run_lengths) if run_lengths else 0.0,
        "median_pattern_run_events": median(run_lengths) if run_lengths else 0.0,
        "longest_pattern_run_events": max(run_lengths, default=0),
        "minimum_reuse_ms": min(intervals, default=None),
        "mean_reuse_ms": mean(intervals) if intervals else None,
        "reuse_violations": sum(
            value + 1e-9 < 1000.0 * float(schedule.model_contract["minimum_reuse_s"])
            for value in intervals),
        "states_evaluated": schedule.states_evaluated,
    }
