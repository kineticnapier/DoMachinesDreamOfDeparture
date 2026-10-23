from __future__ import annotations

"""Bounded short-horizon fingering graph search for DMDOGO v0.4."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CandidateNode:
    event_index: int
    due_ms: float
    allocation_id: str
    axes: tuple[str, ...]
    network_rank: int
    eligible: bool


@dataclass(frozen=True, slots=True)
class EventCandidates:
    event_index: int
    due_ms: float
    candidates: tuple[CandidateNode, ...]


@dataclass(frozen=True, slots=True)
class FingeringPlan:
    allocations: tuple[CandidateNode, ...]
    objective: tuple[int, int, int, int]
    evaluated_candidates: int
    peak_beam: int


@dataclass(frozen=True, slots=True)
class _BeamState:
    nodes: tuple[CandidateNode, ...]
    pattern_breaks: int
    network_rank_sum: int
    thumb_uses: int
    hand_switches: int

    @property
    def objective(self) -> tuple[int, int, int, int, tuple[str, ...]]:
        return (self.pattern_breaks, self.network_rank_sum, self.thumb_uses,
                self.hand_switches, tuple(node.allocation_id for node in self.nodes))


def _axis_parts(axis: str) -> tuple[str, int]:
    hand, depth = axis.split("_")
    return hand, int(depth)


class ShortHorizonFingeringPlanner:
    """Beam search over Network-top candidates with structural safety checks.

    Future nodes are only *admissible predictions*, never claims that the live
    Motor has already found them eligible. The caller must recheck strict live
    eligibility before committing each event.
    """

    def __init__(self, *, horizon: int = 4, beam_width: int = 8,
                 candidates_per_event: int = 3,
                 fast_gap_ms: float = 35.0,
                 min_axis_reuse_ms: float = 80.0) -> None:
        if not 1 <= horizon <= 8 or beam_width <= 0 or candidates_per_event <= 0:
            raise ValueError("invalid bounded planner configuration")
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)
        self.candidates_per_event = int(candidates_per_event)
        self.fast_gap_ms = float(fast_gap_ms)
        self.min_axis_reuse_ms = float(min_axis_reuse_ms)
        self.reset()

    def reset(self) -> None:
        self._cache: dict[int, str] = {}
        self.replan_count = 0
        self.discard_count = 0

    def cached_choice(self, event_index: int,
                      candidates: Sequence[CandidateNode]) -> CandidateNode | None:
        allocation_id = self._cache.get(int(event_index))
        if allocation_id is None:
            return None
        for candidate in candidates:
            if candidate.allocation_id == allocation_id and candidate.eligible:
                return candidate
        self._cache.pop(int(event_index), None)
        self.discard_count += 1
        return None

    def commit(self, plan: FingeringPlan) -> None:
        for node in plan.allocations:
            previous = self._cache.get(node.event_index)
            if previous is not None and previous != node.allocation_id:
                self.replan_count += 1
            self._cache[node.event_index] = node.allocation_id

    def _reuse_safe(self, nodes: Sequence[CandidateNode],
                    candidate: CandidateNode) -> bool:
        for previous in nodes:
            if set(previous.axes) & set(candidate.axes):
                if candidate.due_ms - previous.due_ms < self.min_axis_reuse_ms - 1e-9:
                    return False
        return True

    def _transition_cost(self, nodes: Sequence[CandidateNode],
                         candidate: CandidateNode) -> tuple[int, int, int]:
        thumb = sum(axis.endswith("_5") for axis in candidate.axes)
        if not nodes or len(nodes[-1].axes) != len(candidate.axes) != 1:
            return 0, thumb, 0
        previous = nodes[-1]
        if len(previous.axes) != 1 or len(candidate.axes) != 1:
            return 0, thumb, 0
        gap = candidate.due_ms - previous.due_ms
        if gap > self.fast_gap_ms + 1e-9:
            return 0, thumb, 0
        previous_hand, previous_depth = _axis_parts(previous.axes[0])
        hand, depth = _axis_parts(candidate.axes[0])
        hand_switch = int(previous_hand != hand)
        if previous_depth > 4 or depth > 4:
            return 1, thumb, hand_switch
        if hand_switch:
            return 1, thumb, hand_switch
        if len(nodes) < 2 or len(nodes[-2].axes) != 1:
            return 0, thumb, hand_switch
        older_hand, older_depth = _axis_parts(nodes[-2].axes[0])
        if older_hand != hand or older_depth > 4:
            return 0, thumb, hand_switch
        previous_step = (previous_depth - older_depth) % 4
        current_step = (depth - previous_depth) % 4
        return int(current_step != previous_step), thumb, hand_switch

    def plan(self, events: Sequence[EventCandidates]) -> FingeringPlan:
        chosen_events = tuple(events[:self.horizon])
        if not chosen_events:
            return FingeringPlan((), (0, 0, 0, 0), 0, 0)
        beam = (_BeamState((), 0, 0, 0, 0),)
        evaluated = 0
        peak = 1
        for event in chosen_events:
            candidates = sorted(
                (candidate for candidate in event.candidates if candidate.eligible),
                key=lambda candidate: (candidate.network_rank, candidate.allocation_id),
            )[:self.candidates_per_event]
            expanded: list[_BeamState] = []
            for state in beam:
                for candidate in candidates:
                    evaluated += 1
                    if not self._reuse_safe(state.nodes, candidate):
                        continue
                    pattern, thumb, switch = self._transition_cost(state.nodes, candidate)
                    expanded.append(_BeamState(
                        state.nodes + (candidate,),
                        state.pattern_breaks + pattern,
                        state.network_rank_sum + candidate.network_rank,
                        state.thumb_uses + thumb,
                        state.hand_switches + switch,
                    ))
            if not expanded:
                break
            expanded.sort(key=lambda state: state.objective)
            beam = tuple(expanded[:self.beam_width])
            peak = max(peak, len(beam))
        if not beam:
            return FingeringPlan((), (0, 0, 0, 0), evaluated, peak)
        winner = min(beam, key=lambda state: (-len(state.nodes), state.objective))
        return FingeringPlan(
            winner.nodes,
            (winner.pattern_breaks, winner.network_rank_sum,
             winner.thumb_uses, winner.hand_switches),
            evaluated,
            peak,
        )
