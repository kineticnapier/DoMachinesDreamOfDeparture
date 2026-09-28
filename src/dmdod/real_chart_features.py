from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from .real_chart_env import RealChartObservation, RelativeVisibleFloor


REAL_CHART_EVENT_MARKERS = (
    "Twirl",
    "SetSpeed",
    "MultiPlanet",
    "Pause",
    "Checkpoint",
    "SetFloorIcon",
)
REAL_CHART_FLOOR_FEATURE_DIM = 15


@dataclass(frozen=True, slots=True)
class RealChartFeatureConfig:
    """Fixed-layout encoder configuration for real-chart student policies.

    The encoder consumes only ``RealChartObservation``.  In particular it never
    sees chart/evaluator time, BPM, target timestamps, absolute floor indices,
    timing error, or the privileged next-target object.
    """

    behind_floors: int = 2
    ahead_floors: int = 12
    tile_size: float = 1.5
    position_span_tiles: float = 8.0

    def __post_init__(self) -> None:
        if self.behind_floors < 0 or self.ahead_floors < 0:
            raise ValueError("visible floor counts must be non-negative")
        if self.tile_size <= 0.0:
            raise ValueError("tile_size must be positive")
        if self.position_span_tiles <= 0.0:
            raise ValueError("position_span_tiles must be positive")

    @property
    def floor_slots(self) -> int:
        return self.behind_floors + self.ahead_floors + 1

    @property
    def input_dim(self) -> int:
        # MotorObservation=6, orbiting planet xy=2, then fixed floor slots.
        return 8 + self.floor_slots * REAL_CHART_FLOOR_FEATURE_DIM


DEFAULT_REAL_CHART_FEATURE_CONFIG = RealChartFeatureConfig()
REAL_CHART_INPUT_DIM = DEFAULT_REAL_CHART_FEATURE_CONFIG.input_dim


def encode_real_chart_observation(
    observation: RealChartObservation,
    *,
    config: RealChartFeatureConfig = DEFAULT_REAL_CHART_FEATURE_CONFIG,
) -> tuple[float, ...]:
    """Encode one policy-visible real-chart frame into a fixed-size vector."""

    motor = observation.motor
    values: list[float] = [
        motor.left_position_m / 0.006,
        motor.right_position_m / 0.006,
        motor.left_velocity_m_s / 1.0,
        motor.right_velocity_m_s / 1.0,
        1.0 if motor.left_pressed else 0.0,
        1.0 if motor.right_pressed else 0.0,
        observation.orbiting_x / config.tile_size,
        observation.orbiting_y / config.tile_size,
    ]

    # relative_index is used only to place already-visible floors into stable
    # slots.  Its numeric value is not itself emitted as a feature, avoiding an
    # accidental absolute/chart-progress surrogate.
    by_relative_index = {floor.relative_index: floor for floor in observation.floors}
    for relative_index in range(-config.behind_floors, config.ahead_floors + 1):
        floor = by_relative_index.get(relative_index)
        if floor is None:
            values.extend((0.0,) * REAL_CHART_FLOOR_FEATURE_DIM)
        else:
            values.extend(_floor_features(floor, config))

    if len(values) != config.input_dim:
        raise RuntimeError(
            f"real-chart feature size mismatch: expected {config.input_dim}, got {len(values)}"
        )
    return tuple(values)


def _floor_features(
    floor: RelativeVisibleFloor,
    config: RealChartFeatureConfig,
) -> tuple[float, ...]:
    scale = config.tile_size * config.position_span_tiles
    event_set = set(floor.event_markers)
    result = (
        _clip(floor.x / scale, -2.0, 2.0),
        _clip(floor.y / scale, -2.0, 2.0),
        sin(floor.entry_angle_rad),
        cos(floor.entry_angle_rad),
        sin(floor.exit_angle_rad),
        cos(floor.exit_angle_rad),
        1.0 if floor.midspin else 0.0,
        1.0 if floor.num_planets >= 3 else 0.0,
        *(1.0 if marker in event_set else 0.0 for marker in REAL_CHART_EVENT_MARKERS),
        1.0,  # slot mask
    )
    if len(result) != REAL_CHART_FLOOR_FEATURE_DIM:
        raise RuntimeError("internal real-chart floor feature size mismatch")
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
