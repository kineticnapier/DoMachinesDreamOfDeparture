from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .adofai_chart import AdoFaiAction, AdoFaiChart, load_adofai
from .adofai_geometry import AdoFaiFloorGeometry, build_floor_geometry


PI_STOCK = 3.1415927410125732
TWO_PI_STOCK = 6.2831854820251465
FIRST_ENTRY_ANGLE = 4.71238899230957
DEGREES_TO_RADIANS_STOCK = 0.017453292
RADIANS_TO_DEGREES_STOCK = 57.29578
VISIBLE_EVENT_TYPES = frozenset(
    {"Twirl", "SetSpeed", "MultiPlanet", "Pause", "Checkpoint", "SetFloorIcon"}
)


@dataclass(frozen=True, slots=True)
class AdoFaiFloorTiming:
    index: int
    entry_time_s: float
    exit_time_s: float
    entry_angle_rad: float
    exit_angle_rad: float
    angle_moved_rad: float
    bpm: float
    is_ccw: bool
    num_planets: int
    midspin: bool
    previous_midspin: bool
    pause_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.exit_time_s - self.entry_time_s


@dataclass(frozen=True, slots=True)
class VisibleChartFloor:
    """Policy-safe floor data: intentionally contains no exact time or BPM."""

    index: int
    x: float
    y: float
    entry_angle_rad: float
    exit_angle_rad: float
    midspin: bool
    is_ccw: bool
    num_planets: int
    event_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledChartFloor:
    index: int
    x: float
    y: float
    entry_angle_rad: float
    exit_angle_rad: float
    midspin: bool
    is_ccw: bool
    num_planets: int
    event_markers: tuple[str, ...]
    target_time_s: float
    exit_time_s: float
    bpm: float
    pause_s: float

    def visible(self) -> VisibleChartFloor:
        return VisibleChartFloor(
            index=self.index,
            x=self.x,
            y=self.y,
            entry_angle_rad=self.entry_angle_rad,
            exit_angle_rad=self.exit_angle_rad,
            midspin=self.midspin,
            is_ccw=self.is_ccw,
            num_planets=self.num_planets,
            event_markers=self.event_markers,
        )


@dataclass(frozen=True, slots=True)
class CompiledAdoFaiChart:
    source_path: str
    initial_bpm: float
    offset_ms: float
    pitch_percent: float
    countdown_ticks: int
    separate_countdown_time: bool
    floors: tuple[CompiledChartFloor, ...]

    @property
    def duration_s(self) -> float:
        return self.floors[-1].exit_time_s if self.floors else 0.0

    def floor_entries_between(self, start_s: float, end_s: float) -> tuple[CompiledChartFloor, ...]:
        if end_s < start_s:
            raise ValueError("end_s must be >= start_s")
        return tuple(
            floor
            for floor in self.floors[1:]
            if start_s <= floor.target_time_s <= end_s
        )

    def visible_window(self, current_floor: int, *, behind: int = 2, ahead: int = 12) -> tuple[VisibleChartFloor, ...]:
        start = max(0, current_floor - max(0, behind))
        end = min(len(self.floors), current_floor + max(0, ahead) + 1)
        return tuple(floor.visible() for floor in self.floors[start:end])

    def chart_to_audio_time(self, chart_seconds: float) -> float:
        pitch = max(1e-6, self.pitch_percent * 0.01)
        separate = self._separate_countdown_chart_seconds()
        return (chart_seconds + self.offset_ms * 0.001 - separate) / pitch

    def audio_to_chart_time(self, audio_seconds: float) -> float:
        pitch = max(1e-6, self.pitch_percent * 0.01)
        return audio_seconds * pitch + self._separate_countdown_chart_seconds() - self.offset_ms * 0.001

    def _separate_countdown_chart_seconds(self) -> float:
        if not self.separate_countdown_time or self.countdown_ticks <= 0:
            return 0.0
        bpm = self.initial_bpm if self.initial_bpm > 0.0 else 100.0
        return self.countdown_ticks * (60.0 / bpm)


@dataclass(slots=True)
class _StockFloorState:
    floor: int
    entry_angle: float
    exit_angle: float
    next_entry_angle: float
    speed: float = 1.0
    is_ccw: bool = False
    num_planets: int = 2
    midspin: bool = False
    previous_midspin: bool = False
    pause_s: float = 0.0


def load_compiled_adofai(path: str) -> CompiledAdoFaiChart:
    return compile_adofai(load_adofai(path))


def compile_adofai(chart: AdoFaiChart) -> CompiledAdoFaiChart:
    geometry = build_floor_geometry(chart)
    timing = build_stock_timing(chart)
    if len(geometry) != len(timing):
        raise RuntimeError("geometry/timing floor count mismatch")

    actions_by_floor: dict[int, list[AdoFaiAction]] = {}
    for action in chart.actions:
        actions_by_floor.setdefault(action.floor, []).append(action)

    floors = tuple(
        _compile_floor(geometry[index], timing[index], actions_by_floor.get(index, []))
        for index in range(len(timing))
    )
    return CompiledAdoFaiChart(
        source_path=chart.source_path,
        initial_bpm=chart.initial_bpm,
        offset_ms=chart.offset_ms,
        pitch_percent=chart.pitch_percent,
        countdown_ticks=chart.countdown_ticks,
        separate_countdown_time=chart.separate_countdown_time,
        floors=floors,
    )


def _compile_floor(
    geometry: AdoFaiFloorGeometry,
    timing: AdoFaiFloorTiming,
    actions: list[AdoFaiAction],
) -> CompiledChartFloor:
    markers = tuple(
        action.event_type
        for action in actions
        if action.active and action.event_type in VISIBLE_EVENT_TYPES
    )
    return CompiledChartFloor(
        index=timing.index,
        x=geometry.x,
        y=geometry.y,
        entry_angle_rad=geometry.entry_angle_rad,
        exit_angle_rad=geometry.exit_angle_rad,
        midspin=timing.midspin,
        is_ccw=timing.is_ccw,
        num_planets=timing.num_planets,
        event_markers=markers,
        target_time_s=timing.entry_time_s,
        exit_time_s=timing.exit_time_s,
        bpm=timing.bpm,
        pause_s=timing.pause_s,
    )


def build_stock_timing(chart: AdoFaiChart) -> tuple[AdoFaiFloorTiming, ...]:
    """Port the timing core used by ExtremeEditor's stock timing probe.

    The port keeps the delayed previous-floor finalization needed for the
    midspin -> MultiPlanet special case and supports SetSpeed angleOffset.
    Pause is layered on top using the runtime TimingMap semantics because the
    diagnostic StockTimingProbe itself did not include Pause.
    """

    if chart.floor_count <= 0:
        return ()

    base_bpm = _f32(chart.initial_bpm if chart.initial_bpm > 0.0 else 100.0)
    speed_mult = _f32(1.0)
    is_ccw = False
    num_planets = 2
    time_s = 0.0
    previous_midspin = False
    entry_angle = FIRST_ENTRY_ANGLE
    pending: _StockFloorState | None = None
    timings: list[AdoFaiFloorTiming] = []

    actions_by_floor: dict[int, list[AdoFaiAction]] = {}
    for action in chart.actions:
        actions_by_floor.setdefault(action.floor, []).append(action)

    for floor in range(chart.floor_count):
        current = _build_floor_state(chart, floor, entry_angle, previous_midspin)
        actions = actions_by_floor.get(floor, [])

        old_speed = speed_mult
        set_speeds = [
            action for action in actions if action.active and action.event_type == "SetSpeed"
        ]
        for action in actions:
            if not action.active:
                continue
            if action.event_type == "Twirl":
                is_ccw = not is_ccw
            elif action.event_type == "MultiPlanet":
                num_planets = _parse_planets(action.planets, num_planets)
                if pending is not None and pending.midspin:
                    pending.num_planets = num_planets

        if set_speeds:
            speed_average, speed_mult, has_mid_offset = _apply_set_speeds(
                set_speeds,
                current.entry_angle,
                current.exit_angle,
                is_ccw,
                base_bpm,
                speed_mult,
            )
            current.speed = speed_average if has_mid_offset else speed_mult
        else:
            current.speed = speed_mult

        current.is_ccw = is_ccw
        current.num_planets = num_planets
        current.pause_s = _pause_seconds(actions, base_bpm, old_speed)

        if pending is not None:
            timing, time_s = _finalize_floor(
                pending,
                time_s,
                base_bpm,
                chart.countdown_ticks if pending.floor == 0 else 0,
            )
            timings.append(timing)

        pending = current
        previous_midspin = current.midspin
        entry_angle = current.next_entry_angle

    if pending is not None:
        timing, time_s = _finalize_floor(pending, time_s, base_bpm, 0)
        timings.append(timing)

    return tuple(timings)


def _build_floor_state(
    chart: AdoFaiChart,
    floor: int,
    entry_angle: float,
    previous_midspin: bool,
) -> _StockFloorState:
    has_angle = floor < len(chart.angles)
    raw = _f32(chart.angles[floor]) if has_angle else 0.0
    midspin = has_angle and raw == _f32(999.0)
    if not has_angle:
        exit_angle = entry_angle + PI_STOCK
    elif midspin:
        exit_angle = float(_f32(entry_angle))
    else:
        exit_angle = (-float(raw) + 90.0) * _f32(DEGREES_TO_RADIANS_STOCK)
    next_entry = (
        _mod(exit_angle + PI_STOCK, TWO_PI_STOCK)
        if floor + 1 < chart.floor_count
        else entry_angle
    )
    return _StockFloorState(
        floor=floor,
        entry_angle=entry_angle,
        exit_angle=exit_angle,
        next_entry_angle=next_entry,
        midspin=midspin,
        previous_midspin=previous_midspin,
    )


def _apply_set_speeds(
    speeds: list[AdoFaiAction],
    entry_angle: float,
    exit_angle: float,
    is_ccw: bool,
    base_bpm: float,
    speed_mult: float,
) -> tuple[float, float, bool]:
    offsets = sorted({_f32(action.angle_offset or 0.0) for action in speeds})
    if not offsets:
        return speed_mult, speed_mult, False

    floor_angle_deg = (
        _f32(360.0)
        if abs(entry_angle - exit_angle) < 0.001
        else _f32(_get_angle_moved(entry_angle, exit_angle, not is_ccw) * _f32(RADIANS_TO_DEGREES_STOCK))
    )
    initial_bpm = _f32(base_bpm * speed_mult)
    initial_crotchet = _f32(60.0 / initial_bpm)
    total_time = _f32(offsets[0] / 180.0 * initial_crotchet)
    has_mid_floor_offset = False

    for offset_index, angle_offset in enumerate(offsets):
        at_offset = [
            action
            for action in speeds
            if _f32(action.angle_offset or 0.0) == angle_offset
        ]
        if not at_offset:
            continue
        last = at_offset[-1]
        next_offset = offsets[offset_index + 1] if offset_index + 1 < len(offsets) else floor_angle_deg

        if (last.speed_type or "").lower() == "bpm":
            new_bpm = _f32(last.beats_per_minute or (base_bpm * speed_mult))
        else:
            new_bpm = _f32(0.0)
            for same_index, speed in enumerate(at_offset):
                if (speed.speed_type or "").lower() == "bpm":
                    new_bpm = _f32(speed.beats_per_minute or (base_bpm * speed_mult))
                else:
                    multiplier = _f32(speed.bpm_multiplier or 1.0)
                    new_bpm = _f32(new_bpm * multiplier) if same_index > 0 else _f32(base_bpm * speed_mult * multiplier)

        if not new_bpm > 0.0:
            new_bpm = _f32(base_bpm * speed_mult)
        segment_crotchet = _f32(60.0 / new_bpm)
        total_time = _f32(total_time + (next_offset - angle_offset) / 180.0 * segment_crotchet)
        if angle_offset > 0.0 and angle_offset <= floor_angle_deg:
            has_mid_floor_offset = True
        speed_mult = _f32(new_bpm / base_bpm)

    if not has_mid_floor_offset or not total_time > 0.0:
        return speed_mult, speed_mult, False
    average = _f32(60.0 / total_time * (floor_angle_deg / 180.0) / base_bpm)
    return average, speed_mult, True


def _pause_seconds(actions: list[AdoFaiAction], base_bpm: float, speed_before: float) -> float:
    bpm = _f32(base_bpm * speed_before)
    total = 0.0
    for action in actions:
        if not action.active:
            continue
        if action.event_type == "SetSpeed" and _f32(action.angle_offset or 0.0) <= 0.0:
            if (action.speed_type or "").lower() == "bpm" and action.beats_per_minute and action.beats_per_minute > 0.0:
                bpm = _f32(action.beats_per_minute)
            elif action.bpm_multiplier and action.bpm_multiplier > 0.0:
                bpm = _f32(bpm * _f32(action.bpm_multiplier))
        elif action.event_type == "Pause" and action.duration and action.duration > 0.0:
            total += float(action.duration) * (60.0 / max(1e-9, bpm))
    return total


def _finalize_floor(
    floor: _StockFloorState,
    entry_time_s: float,
    base_bpm: float,
    countdown_ticks: int,
) -> tuple[AdoFaiFloorTiming, float]:
    rotation_seconds, angle_moved = _floor_rotation_seconds(
        floor, base_bpm, countdown_ticks
    )
    exit_time = entry_time_s + floor.pause_s + rotation_seconds
    timing = AdoFaiFloorTiming(
        index=floor.floor,
        entry_time_s=entry_time_s,
        exit_time_s=exit_time,
        entry_angle_rad=floor.entry_angle,
        exit_angle_rad=floor.exit_angle,
        angle_moved_rad=angle_moved,
        bpm=float(base_bpm * floor.speed),
        is_ccw=floor.is_ccw,
        num_planets=floor.num_planets,
        midspin=floor.midspin,
        previous_midspin=floor.previous_midspin,
        pause_s=floor.pause_s,
    )
    return timing, exit_time


def _floor_rotation_seconds(
    floor: _StockFloorState,
    base_bpm: float,
    countdown_ticks: int,
) -> tuple[float, float]:
    if floor.floor == 0:
        moved = _get_angle_moved(floor.entry_angle, floor.exit_angle, not floor.is_ccw)
        countdown_extra = max(0, countdown_ticks - 1)
        first_crotchet = float(_f32(60.0 / base_bpm))
        seconds = countdown_extra * first_crotchet + moved / PI_STOCK * (60.0 / base_bpm / floor.speed)
        return seconds, moved

    inverse = _inverse_angle_per_beat_multiplanet(floor.num_planets)
    offset = inverse * (-1.0 if floor.is_ccw else 1.0)
    if floor.midspin:
        offset = 0.0
    if floor.previous_midspin and floor.num_planets > 2:
        offset -= (TWO_PI_STOCK + inverse) * (-1.0 if floor.is_ccw else 1.0)

    moved = _get_angle_moved(
        floor.entry_angle + offset,
        floor.exit_angle + (offset if floor.midspin else 0.0),
        not floor.is_ccw,
    )
    seconds = moved / PI_STOCK * (60.0 / base_bpm / floor.speed)
    if moved <= 1e-6 or moved >= 6.283184482025146:
        if floor.midspin:
            moved = 0.0
            seconds = 0.0
        else:
            moved = float(_f32(6.2831855))
            seconds = 2.0 * (PI_STOCK / PI_STOCK * (60.0 / base_bpm / floor.speed))
    return seconds, moved


def _parse_planets(planets: str | None, fallback: int) -> int:
    text = (planets or "").lower()
    if text in {"threeplanets", "3"}:
        return 3
    if text in {"twoplanets", "2"}:
        return 2
    return fallback


def _inverse_angle_per_beat_multiplanet(planets: int) -> float:
    return 3.1415926 * (float(planets) - 2.0) / float(planets)


def _get_angle_moved(entry_angle: float, exit_angle: float, is_cw: bool) -> float:
    sign = 1.0 if is_cw else -1.0
    return _mod((exit_angle - entry_angle) * sign, TWO_PI_STOCK)


def _mod(value: float, modulus: float) -> float:
    result = math.fmod(value, modulus)
    return result + modulus if result < 0.0 else result


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]
