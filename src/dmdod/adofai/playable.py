from __future__ import annotations

from dataclasses import dataclass

from .timing import CompiledAdoFaiChart


@dataclass(frozen=True, slots=True)
class PlayableChartTarget:
    """Privileged truth for one input-bearing ADOFAI floor.

    ``chart_time_s`` and ``bpm`` belong to the evaluator side only.  Policies
    must receive geometry from ``RealChartObservation`` instead of this object.
    ``episode_time_s`` is wall-clock time relative to the selected segment and
    therefore includes the chart pitch multiplier.
    """

    ordinal: int
    floor_index: int
    chart_time_s: float
    episode_time_s: float
    bpm: float


@dataclass(frozen=True, slots=True)
class PlayableHoldInterval:
    """Evaluator-side Hold interval, separate from ordinary press targets."""

    ordinal: int
    start_floor_index: int
    end_floor_index: int
    chart_start_s: float
    chart_end_s: float
    episode_start_s: float
    episode_end_s: float
    duration_loops: int


@dataclass(frozen=True, slots=True)
class PlayableChartSegment:
    chart: CompiledAdoFaiChart
    start_chart_s: float
    end_chart_s: float
    pitch_ratio: float
    targets: tuple[PlayableChartTarget, ...]
    holds: tuple[PlayableHoldInterval, ...] = ()

    @property
    def duration_s(self) -> float:
        return (self.end_chart_s - self.start_chart_s) / self.pitch_ratio

    def chart_time_from_episode(self, episode_time_s: float) -> float:
        return self.start_chart_s + episode_time_s * self.pitch_ratio

    def episode_time_from_chart(self, chart_time_s: float) -> float:
        return (chart_time_s - self.start_chart_s) / self.pitch_ratio


def build_playable_segment(
    chart: CompiledAdoFaiChart,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> PlayableChartSegment:
    """Convert compiled floors into actual press targets for a chart segment.

    Floor 0 is the starting floor and never requires an input.  Midspin floors
    are visual/geometry transitions and likewise do not require an input.  This
    is intentionally a separate layer from compiled floors because a Midspin
    has a real floor index and can share its exact entry time with the following
    playable floor.

    Remaining equal-time playable floors are preserved rather than silently
    deduplicated.  If a real chart uses such a pattern, the input state-machine
    should decide its semantics explicitly instead of the importer guessing.
    """

    if start_s < 0.0:
        raise ValueError("start_s must be non-negative")
    if end_s is None:
        end_s = chart.duration_s
    if end_s < start_s:
        raise ValueError("end_s must be >= start_s")

    pitch_ratio = max(1e-6, chart.pitch_percent * 0.01)
    selected = [
        floor
        for floor in chart.floors[1:]
        if not floor.midspin and start_s <= floor.target_time_s <= end_s
    ]
    targets = tuple(
        PlayableChartTarget(
            ordinal=ordinal,
            floor_index=floor.index,
            chart_time_s=floor.target_time_s,
            episode_time_s=(floor.target_time_s - start_s) / pitch_ratio,
            bpm=floor.bpm,
        )
        for ordinal, floor in enumerate(selected)
    )
    selected_holds = [
        floor for floor in chart.floors[:-1]
        if floor.hold is not None
        and start_s <= floor.hold.start_time_s <= end_s
    ]
    holds = tuple(
        PlayableHoldInterval(
            ordinal=ordinal,
            start_floor_index=floor.index,
            end_floor_index=floor.index + 1,
            chart_start_s=floor.hold.start_time_s,
            chart_end_s=floor.hold.end_time_s,
            episode_start_s=(floor.hold.start_time_s - start_s) / pitch_ratio,
            episode_end_s=(floor.hold.end_time_s - start_s) / pitch_ratio,
            duration_loops=floor.hold.duration_loops,
        )
        for ordinal, floor in enumerate(selected_holds)
    )
    return PlayableChartSegment(
        chart=chart,
        start_chart_s=start_s,
        end_chart_s=end_s,
        pitch_ratio=pitch_ratio,
        targets=targets,
        holds=holds,
    )
