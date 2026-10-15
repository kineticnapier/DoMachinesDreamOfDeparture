from __future__ import annotations

"""Prestige I Autoplay-exact Timing Teacher (evaluator/label side ONLY).

The compiled playable chart, not the motor's KeyDown timestamps, supplies
the exact target times. No 10ms quantization, phase search, SA, hand/finger
assignment, or anticipation of physical actuator latency occurs here.

Do not feed this object, chart-relative truth, or its labels to Student
inference. Students still read only their causal 263D visible HUD history.
"""

from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite

from dmdod.adofai.playable import PlayableChartSegment

KEY_COUNT = 8
CONTROL_DT_S = .010
PHYSICS_DT_S = .001
NEAR_WINDOW_S = .200
CHORD_TOLERANCE_S = 1e-7


@dataclass(frozen=True, slots=True)
class AutoplayEvent:
    """One onset; equal-time input-bearing targets form one chord."""

    event_index: int
    target_ordinals: tuple[int, ...]
    floor_indices: tuple[int, ...]
    episode_time_s: float  # exact compiled chart truth, including pitch
    chart_time_s: float
    chord_size: int

    def __post_init__(self):
        if not (
            self.event_index >= 0
            and self.target_ordinals
            and len(self.target_ordinals) == len(self.floor_indices) == self.chord_size
            and 1 <= self.chord_size <= KEY_COUNT
            and isfinite(self.episode_time_s) and self.episode_time_s >= 0.
            and isfinite(self.chart_time_s) and self.chart_time_s >= 0.
        ):
            raise ValueError("Invalid exact Autoplay event")


@dataclass(frozen=True, slots=True)
class AutoplayTimingLabel:
    """Privileged evaluator truth: never a policy observation or prediction."""

    event_index: int
    next_target_ordinals: tuple[int, ...]
    relative_time_s: float
    absolute_episode_time_s: float
    chord_size: int
    near_200ms: bool
    visible_floor_window: bool


class AutoplayTimingTeacher:
    """Pure exact-timestamp source, with no motor simulation.

    events represent input-bearing playable targets, NOT midspins. The first
    target has zero-based ordinal in a frozen chart segment. The grouping
    tolerance is only for equivalent onset times; timestamps are never
    rounded to control frames. The teacher has no notion of 'actuate now'.
    """

    def __init__(self, events: tuple[AutoplayEvent, ...]):
        if not events:
            raise ValueError("Autoplay timeline must contain playable targets")
        self.events = tuple(events)
        for index, event in enumerate(self.events):
            if event.event_index != index:
                raise ValueError("Autoplay indices are not sequential")
            if index and event.episode_time_s <= self.events[index-1].episode_time_s + CHORD_TOLERANCE_S:
                raise ValueError("Equal-time targets must be grouped into one event")
        self._times = tuple(e.episode_time_s for e in self.events)

    @classmethod
    def from_segment(cls, segment: PlayableChartSegment) -> AutoplayTimingTeacher:
        if not isinstance(segment, PlayableChartSegment):
            raise TypeError("Expected a compiled PlayableChartSegment")
        if not isfinite(segment.pitch_ratio) or segment.pitch_ratio <= 0:
            raise ValueError("Invalid pitch ratio")
        groups: list[list] = []
        last_time = None
        for index, target in enumerate(segment.targets):
            time = float(target.episode_time_s)
            chart_time = float(target.chart_time_s)
            if (target.ordinal != index or not isfinite(time)
                    or time < 0 or not isfinite(chart_time)
                    or chart_time < 0):
                raise ValueError("Corrupt compiled playable target timeline")
            if abs(segment.episode_time_from_chart(chart_time) - time) > 1e-7:
                raise ValueError("Playable target chart/episode time disagreement")
            if last_time is not None and time < last_time - CHORD_TOLERANCE_S:
                raise ValueError("Unsorted playable target time")
            if groups and abs(time - groups[-1][0].episode_time_s) <= CHORD_TOLERANCE_S:
                groups[-1].append(target)
            else:
                groups.append([target])
            last_time = time
        events = tuple(
            AutoplayEvent(
                event_index=idx,
                target_ordinals=tuple(t.ordinal for t in group),
                floor_indices=tuple(t.floor_index for t in group),
                episode_time_s=float(group[0].episode_time_s),
                chart_time_s=float(group[0].chart_time_s),
                chord_size=len(group),
            )
            for idx, group in enumerate(groups)
        )
        if sum(event.chord_size for event in events) != len(segment.targets):
            raise RuntimeError("Autoplay events do not cover every playable target")
        return cls(events)

    @property
    def total_targets(self) -> int:
        return sum(e.chord_size for e in self.events)

    def future_event(self, now_s: float) -> AutoplayEvent | None:
        """Next scheduled onset at/after now, not next physically unresolved target.

        This is Autoplay clock semantics. Do NOT use it to implement the
        physical simulator's 'next unresolved' target selector.
        """
        if not isfinite(now_s):
            raise ValueError("Nonfinite evaluation time")
        idx = bisect_left(self._times, now_s)
        return self.events[idx] if idx < len(self.events) else None

    def signed_delta_s(self, event_index: int, now_s: float) -> float:
        """Exact scheduled time minus evaluator time, including overdue labels."""
        if type(event_index) is not int or not 0 <= event_index < len(self.events):
            raise IndexError("Autoplay event index outside timeline")
        if not isfinite(now_s):
            raise ValueError("Nonfinite evaluation time")
        return self.events[event_index].episode_time_s - now_s

    def label_at(
        self, now_s: float, *,
        current_floor: int, behind_floors: int, ahead_floors: int,
    ) -> AutoplayTimingLabel | None:
        """Exact forward label and visibility mask at one evaluator-supplied time.

        No 200ms truncation: a 700ms distant event retains its true +0.700s.
        The near flag is only a separate optional loss mask. Visibility
        tests target floor indices, not true future timing in Student input.
        """
        if (type(current_floor) is not int
                or type(behind_floors) is not int or type(ahead_floors) is not int
                or behind_floors < 0 or ahead_floors < 0):
            raise ValueError("Invalid evaluator floor visibility window")
        next_event = self.future_event(now_s)
        if next_event is None:
            return None
        dt = max(0.0, next_event.episode_time_s - now_s)
        visible = all(
            current_floor - behind_floors <= floor <= current_floor + ahead_floors
            for floor in next_event.floor_indices
        )
        return AutoplayTimingLabel(
            event_index=next_event.event_index,
            next_target_ordinals=next_event.target_ordinals,
            relative_time_s=dt,
            absolute_episode_time_s=next_event.episode_time_s,
            chord_size=next_event.chord_size,
            near_200ms=dt <= NEAR_WINDOW_S,
            visible_floor_window=visible,
        )
