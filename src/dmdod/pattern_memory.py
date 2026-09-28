from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, exp, pi
from typing import TypeAlias

from .motor_env import MotorAction
from .planet_perception import PlanetGeometryObservation


FrameToken: TypeAlias = tuple[int, int, int, int]
PatternKey: TypeAlias = tuple[FrameToken, ...]


@dataclass
class PatternMemoryEntry:
    """What tended to work after seeing one visual pattern.

    This deliberately stores no target timestamp or chart time. ``mean_error_ms``
    is post-attempt feedback (positive = the hit was late), while the action
    values remember the motor command that produced the hit.
    """

    mean_error_ms: float = 0.0
    mean_action_left: float = 0.0
    mean_action_right: float = 0.0
    count: int = 0


@dataclass(frozen=True)
class PatternMemoryFeatures:
    """Agent-visible temporal/pattern features.

    The delta fields are explicit two-frame motion cues. The shared fields are
    reusable knowledge accumulated across charts; the chart fields are practice
    memory accumulated only for the current chart identity.
    """

    delta_orbit_x: float
    delta_orbit_y: float
    delta_next_x: float
    delta_next_y: float
    shared_timing_correction: float
    shared_action_left: float
    shared_action_right: float
    shared_confidence: float
    chart_timing_correction: float
    chart_action_left: float
    chart_action_right: float
    chart_confidence: float


@dataclass(frozen=True)
class PatternLookup:
    features: PatternMemoryFeatures
    keys: tuple[PatternKey, ...]


class PatternMemory:
    """Hierarchical visual-pattern memory shared across repeated attempts.

    A pattern key is made only from visible planet/tile geometry over a short
    rolling window. Every successful hit updates all suffix lengths, so a long
    exact pattern can fall back to a shorter familiar motif. Two stores are kept:

    * shared: typical-pattern knowledge reusable on other charts;
    * per-chart: corrections learned while repeatedly practicing one chart.

    Exact chart timestamps, target angles and BPM are never keys or features.
    """

    def __init__(
        self,
        *,
        history_frames: int = 3,
        ema_alpha: float = 0.25,
        timing_scale_ms: float = 100.0,
        confidence_count: float = 4.0,
        max_shared_entries: int = 50_000,
        max_chart_entries: int = 20_000,
    ) -> None:
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if timing_scale_ms <= 0.0 or confidence_count <= 0.0:
            raise ValueError("timing/confidence scales must be positive")
        if max_shared_entries <= 0 or max_chart_entries <= 0:
            raise ValueError("memory limits must be positive")

        self.history_frames = int(history_frames)
        self.ema_alpha = float(ema_alpha)
        self.timing_scale_ms = float(timing_scale_ms)
        self.confidence_count = float(confidence_count)
        self.max_shared_entries = int(max_shared_entries)
        self.max_chart_entries = int(max_chart_entries)

        self._shared: dict[PatternKey, PatternMemoryEntry] = {}
        self._charts: dict[str, dict[PatternKey, PatternMemoryEntry]] = {}
        self._active_chart_id: str | None = None
        self._tokens: deque[FrameToken] = deque(maxlen=self.history_frames)
        self._previous_geometry: PlanetGeometryObservation | None = None

    @staticmethod
    def _wrap_angle(value: float) -> float:
        while value <= -pi:
            value += 2.0 * pi
        while value > pi:
            value -= 2.0 * pi
        return value

    @staticmethod
    def _circular_bin(angle: float, bins: int) -> int:
        wrapped = PatternMemory._wrap_angle(angle)
        unit = (wrapped + pi) / (2.0 * pi)
        return min(bins - 1, max(0, int(unit * bins)))

    @staticmethod
    def _motion_bin(value: float, *, span: float = 0.30, bins_each_side: int = 6) -> int:
        clipped = max(-span, min(span, value))
        scaled = clipped / span * bins_each_side
        return int(round(scaled))

    def _frame_token(
        self,
        geometry: PlanetGeometryObservation,
        previous: PlanetGeometryObservation | None,
    ) -> FrameToken:
        orbit_angle = atan2(geometry.orbit_y, geometry.orbit_x)
        next_angle = atan2(geometry.next_y, geometry.next_x)
        phase = self._wrap_angle(orbit_angle - next_angle)

        motion = 0.0
        next_turn = 0.0
        if previous is not None:
            previous_orbit = atan2(previous.orbit_y, previous.orbit_x)
            previous_next = atan2(previous.next_y, previous.next_x)
            previous_phase = self._wrap_angle(previous_orbit - previous_next)
            motion = self._wrap_angle(phase - previous_phase)
            next_turn = self._wrap_angle(next_angle - previous_next)

        return (
            self._circular_bin(phase, 24),
            self._motion_bin(motion),
            self._circular_bin(next_angle, 16),
            self._motion_bin(next_turn, span=pi, bins_each_side=8),
        )

    def begin_episode(self, chart_id: str | None) -> None:
        self._active_chart_id = chart_id
        self._tokens.clear()
        self._previous_geometry = None

    def _keys(self) -> tuple[PatternKey, ...]:
        values = tuple(self._tokens)
        return tuple(
            values[-length:]
            for length in range(min(len(values), self.history_frames), 0, -1)
        )

    @staticmethod
    def _lookup(
        table: dict[PatternKey, PatternMemoryEntry] | None,
        keys: tuple[PatternKey, ...],
    ) -> PatternMemoryEntry | None:
        if table is None:
            return None
        for key in keys:
            entry = table.get(key)
            if entry is not None:
                return entry
        return None

    def _confidence(self, entry: PatternMemoryEntry | None) -> float:
        if entry is None or entry.count <= 0:
            return 0.0
        return 1.0 - exp(-entry.count / self.confidence_count)

    def _correction(self, entry: PatternMemoryEntry | None) -> float:
        if entry is None:
            return 0.0
        # Positive hit error means late, so the recommended correction is earlier.
        correction_ms = -entry.mean_error_ms
        return max(-1.0, min(1.0, correction_ms / self.timing_scale_ms))

    @staticmethod
    def _action(entry: PatternMemoryEntry | None) -> tuple[float, float]:
        if entry is None:
            return 0.0, 0.0
        return (
            max(-1.0, min(1.0, entry.mean_action_left)),
            max(-1.0, min(1.0, entry.mean_action_right)),
        )

    def observe(
        self,
        geometry: PlanetGeometryObservation,
        *,
        use_memory: bool = True,
    ) -> PatternLookup:
        previous = self._previous_geometry
        if previous is None:
            delta_orbit_x = delta_orbit_y = delta_next_x = delta_next_y = 0.0
        else:
            delta_orbit_x = geometry.orbit_x - previous.orbit_x
            delta_orbit_y = geometry.orbit_y - previous.orbit_y
            delta_next_x = geometry.next_x - previous.next_x
            delta_next_y = geometry.next_y - previous.next_y

        self._tokens.append(self._frame_token(geometry, previous))
        self._previous_geometry = geometry
        keys = self._keys()

        shared = self._lookup(self._shared, keys) if use_memory else None
        chart_table = (
            self._charts.get(self._active_chart_id)
            if use_memory and self._active_chart_id is not None
            else None
        )
        chart = self._lookup(chart_table, keys)
        shared_left, shared_right = self._action(shared)
        chart_left, chart_right = self._action(chart)

        return PatternLookup(
            PatternMemoryFeatures(
                delta_orbit_x=delta_orbit_x,
                delta_orbit_y=delta_orbit_y,
                delta_next_x=delta_next_x,
                delta_next_y=delta_next_y,
                shared_timing_correction=self._correction(shared),
                shared_action_left=shared_left,
                shared_action_right=shared_right,
                shared_confidence=self._confidence(shared),
                chart_timing_correction=self._correction(chart),
                chart_action_left=chart_left,
                chart_action_right=chart_right,
                chart_confidence=self._confidence(chart),
            ),
            keys,
        )

    def _update_table(
        self,
        table: dict[PatternKey, PatternMemoryEntry],
        key: PatternKey,
        *,
        error_ms: float,
        action: MotorAction,
        limit: int,
    ) -> None:
        entry = table.get(key)
        if entry is None:
            if len(table) >= limit:
                table.pop(next(iter(table)))
            table[key] = PatternMemoryEntry(
                mean_error_ms=float(error_ms),
                mean_action_left=float(action.left),
                mean_action_right=float(action.right),
                count=1,
            )
            return

        alpha = self.ema_alpha
        entry.mean_error_ms = (1.0 - alpha) * entry.mean_error_ms + alpha * float(error_ms)
        entry.mean_action_left = (
            (1.0 - alpha) * entry.mean_action_left + alpha * float(action.left)
        )
        entry.mean_action_right = (
            (1.0 - alpha) * entry.mean_action_right + alpha * float(action.right)
        )
        entry.count += 1

    def learn(
        self,
        keys: tuple[PatternKey, ...],
        *,
        error_ms: float,
        action: MotorAction,
    ) -> None:
        """Learn from feedback after a hit, never from future target data."""

        for key in keys:
            self._update_table(
                self._shared,
                key,
                error_ms=error_ms,
                action=action,
                limit=self.max_shared_entries,
            )

        if self._active_chart_id is None:
            return
        chart = self._charts.setdefault(self._active_chart_id, {})
        for key in keys:
            self._update_table(
                chart,
                key,
                error_ms=error_ms,
                action=action,
                limit=self.max_chart_entries,
            )

    @staticmethod
    def _dump_table(table: dict[PatternKey, PatternMemoryEntry]) -> list[tuple[PatternKey, float, float, float, int]]:
        return [
            (
                key,
                entry.mean_error_ms,
                entry.mean_action_left,
                entry.mean_action_right,
                entry.count,
            )
            for key, entry in table.items()
        ]

    @staticmethod
    def _load_table(rows) -> dict[PatternKey, PatternMemoryEntry]:
        result: dict[PatternKey, PatternMemoryEntry] = {}
        for key, error, left, right, count in rows:
            normalized_key: PatternKey = tuple(tuple(int(x) for x in token) for token in key)
            result[normalized_key] = PatternMemoryEntry(
                mean_error_ms=float(error),
                mean_action_left=float(left),
                mean_action_right=float(right),
                count=int(count),
            )
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "shared": self._dump_table(self._shared),
            "charts": {
                chart_id: self._dump_table(table)
                for chart_id, table in self._charts.items()
            },
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        self._shared.clear()
        self._charts.clear()
        if not state:
            return
        self._shared.update(self._load_table(state.get("shared", [])))
        charts = state.get("charts", {})
        if isinstance(charts, dict):
            for chart_id, rows in charts.items():
                self._charts[str(chart_id)] = self._load_table(rows)

    @property
    def shared_entry_count(self) -> int:
        return len(self._shared)

    @property
    def chart_entry_count(self) -> int:
        return sum(len(table) for table in self._charts.values())
