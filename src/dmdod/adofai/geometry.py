from __future__ import annotations

import math
from dataclasses import dataclass

from .chart import AdoFaiChart


DEFAULT_LONG_TILE_SIZE = 1.5
INITIAL_ENTRY_ANGLE = 4.71238899230957
TWO_PI = math.tau


@dataclass(frozen=True, slots=True)
class AdoFaiFloorGeometry:
    index: int
    x: float
    y: float
    entry_angle_rad: float
    exit_angle_rad: float
    midspin: bool


def build_floor_geometry(chart: AdoFaiChart) -> tuple[AdoFaiFloorGeometry, ...]:
    """Build stock-style floor centers and entry/exit angles from angleData.

    This is the small geometry slice ported from ExtremeEditor's PathBuilder.
    Track/decorative transforms are intentionally not part of the gameplay
    geometry foundation yet.
    """

    floor_count = chart.floor_count
    if floor_count <= 0:
        return ()

    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    entry_angles: list[float] = [0.0] * floor_count
    exit_angles: list[float] = [0.0] * floor_count
    midspins: list[bool] = [False] * floor_count

    entry_angle = INITIAL_ENTRY_ANGLE
    x = 0.0
    y = 0.0
    for floor in range(floor_count):
        entry_angles[floor] = entry_angle
        if floor < len(chart.angles):
            raw = chart.angles[floor]
            midspin = abs(raw - 999.0) < 1e-6
            exit_angle = entry_angle if midspin else math.radians(-raw + 90.0)
            midspins[floor] = midspin
        else:
            exit_angle = entry_angle + math.pi
        exit_angles[floor] = exit_angle

        if floor < len(chart.angles):
            x += math.sin(exit_angle) * DEFAULT_LONG_TILE_SIZE
            y += math.cos(exit_angle) * DEFAULT_LONG_TILE_SIZE
            positions.append((x, y))
            entry_angle = _positive_mod(exit_angle + math.pi, TWO_PI)

    return tuple(
        AdoFaiFloorGeometry(
            index=floor,
            x=positions[floor][0],
            y=positions[floor][1],
            entry_angle_rad=entry_angles[floor],
            exit_angle_rad=exit_angles[floor],
            midspin=midspins[floor],
        )
        for floor in range(floor_count)
    )


def _positive_mod(value: float, modulus: float) -> float:
    return (value % modulus + modulus) % modulus
