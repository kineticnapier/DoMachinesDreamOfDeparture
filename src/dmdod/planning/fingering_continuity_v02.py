from __future__ import annotations

"""Causal, post-network fingering continuity selector.

The selector never changes candidate eligibility and never inspects a future
physical rollout.  It combines the frozen network's within-event rank with a
small, scale-free continuity rank derived only from committed past choices.
The 35 ms fast-sequence boundary is shared with the existing fingering replay
diagnostics; it is a declared diagnostic convention, not a learned parameter.
"""

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence


FAST_SEQUENCE_S = 0.035
NETWORK_RANK_WEIGHT = 0.65
CONTINUITY_RANK_WEIGHT = 0.35


@dataclass(frozen=True)
class SelectionHistory:
    event_index: int
    due_s: float
    axes: tuple[str, ...]


@dataclass(frozen=True)
class SelectionDecision:
    selected_index: int | None
    free_index: int | None
    eligible: tuple[bool, ...]
    network_scores: tuple[float | None, ...]
    continuity_costs: tuple[float | None, ...]
    combined_rank_scores: tuple[float | None, ...]
    changed_from_free: bool
    reason: str
    fast_network_guard_applied: bool
    network_guard_indices: tuple[int, ...]


def _axis_parts(axis: str) -> tuple[str, int]:
    hand, separator, depth_text = str(axis).partition("_")
    if separator != "_" or hand not in {"left", "right"}:
        raise ValueError(f"unsupported motor axis: {axis!r}")
    depth = int(depth_text)
    if depth < 1 or depth > 6:
        raise ValueError(f"unsupported motor axis: {axis!r}")
    return hand, depth


def _axes(allocation: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(finger.axis) for finger in allocation)


def _rank(values: dict[int, float]) -> dict[int, float]:
    unique = sorted(set(values.values()))
    denominator = max(1, len(unique) - 1)
    by_value = {value: position / denominator
                for position, value in enumerate(unique)}
    return {index: by_value[value] for index, value in values.items()}


class ContinuitySelector:
    """Select an eligible allocation in ``free`` or ``continuity`` mode.

    Calling :meth:`select` commits the chosen allocation to causal history.
    Create one instance per replay arm and call :meth:`reset` between episodes.
    """

    def __init__(self, *, mode: str = "free") -> None:
        if mode not in {"free", "continuity", "continuity_v03"}:
            raise ValueError("mode must be 'free', 'continuity', or 'continuity_v03'")
        self.mode = mode
        self._history: list[SelectionHistory] = []

    @property
    def history(self) -> tuple[SelectionHistory, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def _single_axis_cost(self, axis: str, due_s: float) -> float:
        hand, depth = _axis_parts(axis)
        auxiliary = {5: 1.35, 6: 1.85}.get(depth, 0.0)
        if not self._history:
            return auxiliary

        last = self._history[-1]
        gap_s = max(0.0, due_s - last.due_s)
        last_axes = last.axes
        if len(last_axes) != 1:
            return auxiliary + (0.25 if axis not in last_axes else 0.75)
        last_hand, last_depth = _axis_parts(last_axes[0])
        fast = gap_s <= FAST_SEQUENCE_S + 1e-12
        if not fast:
            return auxiliary + (0.10 if hand != last_hand else 0.0)
        if axis == last_axes[0]:
            return auxiliary + 2.50
        if hand != last_hand:
            return auxiliary + 1.50
        if depth > 4 or last_depth > 4:
            return auxiliary + 0.80

        # With two recent ordinary fingers, continue the observed circular
        # rolling direction (1..4).  No direction is invented from one sample.
        if len(self._history) >= 2 and len(self._history[-2].axes) == 1:
            previous_hand, previous_depth = _axis_parts(self._history[-2].axes[0])
            previous_gap = last.due_s - self._history[-2].due_s
            if (previous_hand == hand and previous_depth <= 4
                    and previous_gap <= FAST_SEQUENCE_S + 1e-12):
                step = (last_depth - previous_depth) % 4
                candidate_step = (depth - last_depth) % 4
                if step in {1, 2, 3} and candidate_step == step:
                    return auxiliary

        distance = abs(depth - last_depth)
        return auxiliary + {1: 0.20, 2: 0.65, 3: 1.00}.get(distance, 1.20)

    def _continuity_cost(self, allocation: Sequence[object], due_s: float) -> float:
        axes = _axes(allocation)
        if not axes:
            return float("inf")
        costs = [self._single_axis_cost(axis, due_s) for axis in axes]
        # Chords remain atomic.  Reusing only part of the immediately preceding
        # chord is discouraged, while chord size itself is not penalized.
        overlap = 0.0
        if self._history and len(axes) > 1:
            shared = set(axes).intersection(self._history[-1].axes)
            overlap = 0.40 * len(shared) / len(axes)
        return mean(costs) + overlap

    def select(
        self,
        network_scores: Sequence[float | None],
        allocations: Sequence[Sequence[object]],
        eligible: Sequence[bool],
        event_index: int,
        due_s: float,
    ) -> SelectionDecision:
        if not (len(network_scores) == len(allocations) == len(eligible)):
            raise ValueError("network score, allocation, and eligibility length mismatch")
        if not allocations:
            raise ValueError("empty candidate set")
        eligible_indices = [index for index, allowed in enumerate(eligible) if allowed]
        for index in eligible_indices:
            if network_scores[index] is None:
                raise ValueError("eligible candidate is missing a network score")
        free_index = min(
            eligible_indices, key=lambda index: (float(network_scores[index]), index)
        ) if eligible_indices else None

        valid = list(eligible_indices)
        guard_applied = False
        guard_indices: tuple[int, ...] = ()
        if (self.mode == "continuity_v03" and self._history
                and due_s - self._history[-1].due_s <= FAST_SEQUENCE_S + 1e-12
                and len(valid) > 2):
            valid = sorted(
                valid, key=lambda index: (float(network_scores[index]), index)
            )[:2]
            guard_applied = True
            guard_indices = tuple(valid)

        continuity_values: tuple[float | None, ...] = tuple(None for _ in allocations)
        combined: tuple[float | None, ...] = tuple(None for _ in allocations)
        if free_index is None:
            selected = None
            reason = "no_eligible_candidate"
        elif self.mode == "free":
            selected = free_index
            reason = "free_network_argmin"
        else:
            costs = {index: self._continuity_cost(allocations[index], due_s)
                     for index in valid}
            network_rank = _rank({index: float(network_scores[index]) for index in valid})
            continuity_rank = _rank(costs)
            combined_values = {
                index: (NETWORK_RANK_WEIGHT * network_rank[index]
                        + CONTINUITY_RANK_WEIGHT * continuity_rank[index])
                for index in valid
            }
            continuity_values = tuple(costs.get(index) for index in range(len(allocations)))
            combined = tuple(combined_values.get(index) for index in range(len(allocations)))
            selected = min(
                valid,
                key=lambda index: (combined_values[index], network_rank[index], index),
            )
            reason = ("fast_network_top2_then_continuity_rank" if guard_applied
                      else "scale_free_network_continuity_rank")

        if selected is not None:
            self._history.append(SelectionHistory(
                event_index=int(event_index), due_s=float(due_s),
                axes=_axes(allocations[selected]),
            ))
        return SelectionDecision(
            selected_index=selected,
            free_index=free_index,
            eligible=tuple(bool(value) for value in eligible),
            network_scores=tuple(None if value is None else float(value)
                                 for value in network_scores),
            continuity_costs=continuity_values,
            combined_rank_scores=combined,
            changed_from_free=selected != free_index,
            reason=reason,
            fast_network_guard_applied=guard_applied,
            network_guard_indices=guard_indices,
        )


def continuity_metrics(decisions: Sequence[dict]) -> dict:
    """Compute descriptive continuity metrics from causal decision rows.

    ``fast_pattern_stability`` is the fraction of fast ordinary-finger
    transition steps that repeat either of the preceding two step classes.
    It rewards steady rolls and two-step alternation without treating fewer
    finger changes as intrinsically better.
    """

    axes_use: dict[str, int] = {}
    last_use_ms: dict[str, float] = {}
    reuse_intervals: list[float] = []
    short_reinputs = 0
    hand_switches = 0
    changed = 0
    fast_steps: list[int] = []
    stable_steps = 0
    eligible_fast_steps = 0

    previous: dict | None = None
    for row in decisions:
        axes = tuple(str(axis) for axis in row.get("axes", ()))
        due_ms = float(row["due_ms"])
        changed += int(bool(row.get("different_from_free", False)))
        for axis in axes:
            axes_use[axis] = axes_use.get(axis, 0) + 1
            if axis in last_use_ms:
                interval = due_ms - last_use_ms[axis]
                reuse_intervals.append(interval)
                short_reinputs += int(interval <= FAST_SEQUENCE_S * 1000.0 + 1e-9)
            last_use_ms[axis] = due_ms

        if previous is not None:
            previous_axes = tuple(str(axis) for axis in previous.get("axes", ()))
            gap_ms = due_ms - float(previous["due_ms"])
            if len(previous_axes) == len(axes) == 1:
                previous_hand, previous_depth = _axis_parts(previous_axes[0])
                hand, depth = _axis_parts(axes[0])
                hand_switches += int(previous_hand != hand)
                if (gap_ms <= FAST_SEQUENCE_S * 1000.0 + 1e-9
                        and previous_hand == hand
                        and previous_depth <= 4 and depth <= 4):
                    step = (depth - previous_depth) % 4
                    if len(fast_steps) >= 2:
                        eligible_fast_steps += 1
                        stable_steps += int(step in fast_steps[-2:])
                    fast_steps.append(step)
                else:
                    fast_steps.clear()
        previous = row

    interval_summary = {
        "count": len(reuse_intervals),
        "minimum": min(reuse_intervals) if reuse_intervals else None,
        "mean": mean(reuse_intervals) if reuse_intervals else None,
        "median": median(reuse_intervals) if reuse_intervals else None,
    }
    count = len(decisions)
    return {
        "definition_version": "dmdogo-fingering-continuity-metrics-v0.2",
        "decision_count": count,
        "axis_use": dict(sorted(axes_use.items())),
        "left_right_switches": hand_switches,
        "axis_reuse_interval_ms": interval_summary,
        "same_part_reinputs_le_35ms": short_reinputs,
        "thumb_uses": sum(value for axis, value in axes_use.items() if axis.endswith("_5")),
        "side_uses": sum(value for axis, value in axes_use.items() if axis.endswith("_6")),
        "fast_pattern_stability": (
            stable_steps / eligible_fast_steps if eligible_fast_steps else None
        ),
        "fast_pattern_comparisons": eligible_fast_steps,
        "changed_from_free_count": changed,
        "changed_from_free_rate": changed / count if count else 0.0,
    }
