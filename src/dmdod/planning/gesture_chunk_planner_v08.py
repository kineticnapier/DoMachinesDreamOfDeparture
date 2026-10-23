from __future__ import annotations

"""Auditable, untrained gesture-chunk fingering hypothesis.

Offline Oracle onset times and chord sizes are used to discover local
rhythmic attacks. Each attack group receives an independently planned
physical-axis gesture, joined by a bounded reuse-aware beam. This is NOT
human imitation, calibrated anatomy, live Motor prediction or HUD-only
observation. Motor eligibility remains the runner's responsibility.
"""

from dataclasses import dataclass
from itertools import combinations
from math import exp, isfinite, sqrt
from statistics import median
from typing import Sequence

AXES = (
    "left_1", "right_1", "left_2", "right_2",
    "left_3", "right_3", "left_4", "right_4",
)


@dataclass(frozen=True, slots=True)
class GestureEvent:
    event_index: int
    due_s: float
    chord_size: int = 1


@dataclass(frozen=True, slots=True)
class GestureChunk:
    start_event_index: int
    end_event_index: int
    event_count: int
    rhythm_kind: str
    start_reason: str
    gesture_family: str
    axes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class GesturePlan:
    feasible: bool
    assignments: tuple[tuple[str, ...], ...]
    chunks: tuple[GestureChunk, ...]
    failure_reason: str | None
    candidates_evaluated: int
    metrics: dict
    model_contract: dict


@dataclass(frozen=True, slots=True)
class _Window:
    events: tuple[GestureEvent, ...]
    reason: str
    rhythm: str


@dataclass(frozen=True, slots=True)
class _Gesture:
    axes: tuple[tuple[str, ...], ...]
    family: str
    preference: float


def _axis_parts(axis: str) -> tuple[str, int]:
    hand, number = axis.split("_")
    return hand, int(number)


def _candidate_gestures(window: _Window) -> tuple[_Gesture, ...]:
    """Local movement primitives; no global fixed eight-key sequence."""
    events = window.events
    count = len(events)
    output: dict[tuple[tuple[str, ...], ...], _Gesture] = {}

    def add(seq: tuple[str, ...], family: str, preference: float) -> None:
        axes = tuple((axis,) for axis in seq)
        value = _Gesture(axes, family, preference)
        old = output.get(axes)
        if old is None or (value.preference, value.family) < (old.preference, old.family):
            output[axes] = value

    if any(e.chord_size > 1 for e in events):
        if count != 1:
            raise RuntimeError("chord was not separated into its own chunk")
        for axes in combinations(AXES, events[0].chord_size):
            output[(axes,)] = _Gesture((axes,), "chord", 0.)
        return tuple(output.values())

    if count == 1:
        for axis in AXES:
            add((axis,), "single", 0.)
        return tuple(output.values())

    # Same-hand adjacent roll in either direction and one-step rebounds.
    for hand in ("left", "right"):
        for direction in (-1, 1):
            for start in range(1, 5):
                depths = tuple(start + direction * i for i in range(count))
                if all(1 <= d <= 4 for d in depths):
                    add(tuple(f"{hand}_{d}" for d in depths), "same_hand_roll", 0.)
        if count >= 3:
            for start in range(1, 4):
                seq = [start, start + 1]
                while len(seq) < count:
                    seq.append(start + 1 if len(seq) % 2 else start)
                add(tuple(f"{hand}_{d}" for d in seq), "rebound", .50)

    # Alternation, exchange and two-note handoff are chunk-level gestures.
    for first_hand, second_hand in (("left", "right"), ("right", "left")):
        for depth in (1, 2, 3, 4):
            if count == 2:
                add((f"{first_hand}_{depth}", f"{second_hand}_{depth}"),
                    "hand_exchange", .65)
            for start in range(1, 5):
                seq = []
                for i in range(count):
                    hand = first_hand if i % 2 == 0 else second_hand
                    d = start + i // 2
                    if d <= 4:
                        seq.append(f"{hand}_{d}")
                if len(seq) == count:
                    add(tuple(seq), "interleaved", .75)
            if count >= 3:
                seq = []
                for i in range(count):
                    hand = first_hand if i < 2 else second_hand
                    d = depth + (i % 2)
                    if d <= 4:
                        seq.append(f"{hand}_{d}")
                if len(seq) == count:
                    add(tuple(seq), "paired_exchange", .25)
    return tuple(sorted(output.values(),
                        key=lambda v: (v.preference, v.family, v.axes)))


class GestureChunkPlannerV08:
    """Time-gap chunk detection and reuse-safe, deterministic heuristic beam."""

    def __init__(self, *, minimum_reuse_s: float = .080,
                 max_chunk_events: int = 4, restart_gap_s: float = .450,
                 beam_width: int = 24) -> None:
        if not isfinite(minimum_reuse_s) or minimum_reuse_s < 0:
            raise ValueError("minimum_reuse_s must be finite and nonnegative")
        if not isfinite(restart_gap_s) or restart_gap_s <= 0:
            raise ValueError("restart_gap_s must be finite and positive")
        if not 2 <= max_chunk_events <= 4:
            raise ValueError("max_chunk_events must be between 2 and 4")
        if not 1 <= beam_width <= 256:
            raise ValueError("beam_width must be between 1 and 256")
        self.minimum_reuse_s = float(minimum_reuse_s)
        self.max_chunk_events = int(max_chunk_events)
        self.restart_gap_s = float(restart_gap_s)
        self.beam_width = int(beam_width)

    def _contract(self) -> dict:
        return {
            "source": "offline_exact_oracle_onsets_and_chord_sizes",
            "method": "adaptive_gap_segmentation_and_bounded_gesture_beam",
            "human_demonstrations_used": False,
            "human_likeness_verified": False,
            "physical_motor_evaluated": False,
            "hud_only_observable": False,
            "global_optimality": False,
            "minimum_reuse_s": self.minimum_reuse_s,
            "max_chunk_events": self.max_chunk_events,
            "beam_width": self.beam_width,
            "restart_gap_s": self.restart_gap_s,
            "topology": "eight ordinary physical axes only; live Motor checks required",
            "segmentation_warning": "gap thresholds and gesture grammar are untrained assumptions",
        }

    def _windows(self, events: tuple[GestureEvent, ...]) -> tuple[_Window, ...]:
        if not events:
            return ()
        gaps = tuple(b.due_s - a.due_s for a, b in zip(events, events[1:]))
        boundaries = []
        for i, gap in enumerate(gaps):
            if gap >= self.restart_gap_s:
                boundaries.append(True)
                continue
            near = sorted(g for g in gaps[max(0, i - 8):i + 9]
                          if g < self.restart_gap_s)
            if len(near) < 4:
                boundaries.append(False)
                continue
            low = near[(len(near) - 1) // 4]
            high = near[(3 * (len(near) - 1)) // 4]
            distinct = high > max(1e-9, low) * 1.8
            threshold = sqrt(max(low, 1e-12) * max(high, 1e-12))
            boundaries.append(distinct and gap > threshold + 1e-12)

        windows: list[_Window] = []
        start, reason = 0, "chart_start"

        def emit(end: int, next_reason: str) -> None:
            nonlocal start, reason
            if end > start:
                part = events[start:end]
                if any(e.chord_size > 1 for e in part):
                    rhythm = "chord"
                elif len(part) == 1:
                    rhythm = "isolated"
                else:
                    kind = [b.due_s - a.due_s for a, b in zip(part, part[1:])]
                    rhythm = ("stream" if max(kind) <= min(kind) * 1.2 + 1e-6
                              and len(kind) >= 2 else "burst")
                windows.append(_Window(part, reason, rhythm))
            start, reason = end, next_reason

        for i in range(1, len(events)):
            is_chord_boundary = (events[i].chord_size > 1
                                 or events[i - 1].chord_size > 1)
            if gaps[i - 1] >= self.restart_gap_s:
                emit(i, "pause")
            elif is_chord_boundary:
                emit(i, "chord_boundary")
            elif boundaries[i - 1]:
                emit(i, "salient_rhythm_gap")
            elif i - start >= self.max_chunk_events:
                emit(i, "stream_continuation")
        emit(len(events), "end")
        return tuple(windows)

    def plan(self, source: Sequence[GestureEvent]) -> GesturePlan:
        events = tuple(source)
        contract = self._contract()

        def failed(reason: str, evaluated: int = 0) -> GesturePlan:
            return GesturePlan(False, (), (), reason, evaluated, {}, contract)

        if any(not isfinite(e.due_s) for e in events):
            return failed("non_finite_due_time")
        if any(e.chord_size < 1 or e.chord_size > 8 for e in events):
            return failed("invalid_chord_size")
        if len({e.event_index for e in events}) != len(events):
            return failed("duplicate_event_index")
        if any(b.due_s < a.due_s for a, b in zip(events, events[1:])):
            return failed("nonmonotonic_event_times")
        if not events:
            return GesturePlan(True, (), (), None, 0, {"chunk_count": 0}, contract)

        chunks = self._windows(events)
        # State: objective, last axis times, previous axis, chosen chunks,
        # two-hand time-decayed exposure, last exposure update time.
        # The load prior is deliberately *not* a calibrated fatigue model.
        states = [(0.0, {}, None, (), (0.0, 0.0), events[0].due_s)]
        evaluated = 0
        for window_index, window in enumerate(chunks):
            previous_due = (chunks[window_index - 1].events[-1].due_s
                            if window_index else None)
            next_states = []
            candidates = _candidate_gestures(window)
            for score, previous, last_axis, chosen, exposure, exposure_t in states:
                for candidate in candidates:
                    evaluated += 1
                    last_use = previous.copy()
                    feasible = True
                    extra = candidate.preference
                    exposure_now = list(exposure)
                    last_time = exposure_t
                    for event, axes in zip(window.events, candidate.axes):
                        decay = exp(-max(0., event.due_s - last_time) / .75)
                        exposure_now = [value * decay for value in exposure_now]
                        last_time = event.due_s
                        for axis in axes:
                            hand_index = 0 if axis.startswith("left") else 1
                            extra += .45 * exposure_now[hand_index]
                            exposure_now[hand_index] += 1.0
                            prior = last_use.get(axis)
                            if prior is not None:
                                duration = event.due_s - prior
                                if duration <= 1e-12 or duration + 1e-12 < self.minimum_reuse_s:
                                    feasible = False
                                    break
                                extra += max(0., .150 - duration) * 2.
                            last_use[axis] = event.due_s
                        if not feasible:
                            break
                    if not feasible:
                        continue
                    flat = [axis for row in candidate.axes for axis in row]
                    hand_changes = sum(_axis_parts(a)[0] != _axis_parts(b)[0]
                                       for a, b in zip(flat, flat[1:]))
                    extra += .35 * hand_changes
                    for a, b in zip(flat, flat[1:]):
                        ha, da = _axis_parts(a)
                        hb, db = _axis_parts(b)
                        if ha == hb:
                            extra += .10 * max(0, abs(db - da) - 1)
                    if last_axis is not None and flat and previous_due is not None:
                        gap = window.events[0].due_s - previous_due
                        if gap <= .050:
                            ha, da = _axis_parts(last_axis)
                            hb, db = _axis_parts(flat[0])
                            extra += .12 if ha != hb else .05 * max(0, abs(db - da) - 1)
                    annotation = GestureChunk(
                        window.events[0].event_index,
                        window.events[-1].event_index,
                        len(window.events), window.rhythm, window.reason,
                        candidate.family, candidate.axes,
                    )
                    next_states.append((score + extra, last_use,
                                        flat[-1], chosen + (annotation,),
                                        tuple(exposure_now), last_time))
            if not next_states:
                return failed("no_reuse_safe_gesture_in_beam", evaluated)
            # Stable candidate enumeration guarantees deterministic tie order.
            next_states.sort(key=lambda item: item[0])
            states = next_states[:self.beam_width]

        score, _, _, selected, _, _ = states[0]
        assignments = tuple(axes for chunk in selected for axes in chunk.axes)
        min_gap = float("inf")
        previous_use: dict[str, float] = {}
        for event, axes in zip(events, assignments):
            for axis in axes:
                if axis in previous_use:
                    min_gap = min(min_gap, event.due_s - previous_use[axis])
                previous_use[axis] = event.due_s
        eight_lag = (sum(a == b for a, b in zip(assignments, assignments[8:]))
                     / max(1, len(assignments) - 8))
        lengths = sorted(set(chunk.event_count for chunk in selected))
        families = sorted(set(chunk.gesture_family for chunk in selected))
        metrics = {
            "chunk_count": len(selected),
            "chunk_sizes": {str(n): sum(c.event_count == n for c in selected)
                            for n in lengths},
            "gesture_families": {name: sum(c.gesture_family == name for c in selected)
                                 for name in families},
            "distinct_gesture_families": len(families),
            "axis_8_lag_repetition_fraction": round(eight_lag, 6),
            "minimum_abstract_axis_reuse_ms": (
                round(1000 * min_gap, 6) if min_gap != float("inf") else None),
            "abstract_reuse_violations": 0,
            "beam_cost": round(score, 6),
        }
        return GesturePlan(True, assignments, selected, None,
                           evaluated, metrics, contract)
