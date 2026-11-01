from math import pi

import pytest

from dmdod.motor_env import MotorObservation
from dmdod.real_chart_env import RealChartObservation, RelativeVisibleFloor
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    RealChartFeatureConfig,
    REAL_CHART_FLOOR_FEATURE_DIM,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)


def _motor() -> MotorObservation:
    return MotorObservation(
        left_position_m=0.003,
        right_position_m=0.006,
        left_velocity_m_s=0.25,
        right_velocity_m_s=-0.50,
        left_pressed=True,
        right_pressed=False,
    )


def _floor(relative_index: int, *, events: tuple[str, ...] = ()) -> RelativeVisibleFloor:
    return RelativeVisibleFloor(
        relative_index=relative_index,
        x=1.5 * relative_index,
        y=-0.75 * relative_index,
        entry_angle_rad=pi / 2,
        exit_angle_rad=pi,
        midspin=relative_index == 1,
        is_ccw=True,
        num_planets=3 if relative_index == 2 else 2,
        event_markers=events,
    )


def test_real_chart_encoder_has_fixed_dimension_and_visible_only_layout():
    observation = RealChartObservation(
        motor=_motor(),
        orbiting_x=1.5,
        orbiting_y=-1.5,
        floors=(
            _floor(-2),
            _floor(0, events=("Twirl", "SetSpeed")),
            _floor(1),
            _floor(2, events=("Pause",)),
        ),
    )
    values = encode_real_chart_observation(observation)

    assert len(values) == REAL_CHART_INPUT_DIM
    assert REAL_CHART_INPUT_DIM == (
        8
        + DEFAULT_REAL_CHART_FEATURE_CONFIG.floor_slots
        * REAL_CHART_FLOOR_FEATURE_DIM
    )
    assert values[:8] == pytest.approx(
        (0.5, 1.0, 0.25, -0.5, 1.0, 0.0, 1.0, -1.0)
    )


def test_real_chart_encoder_uses_relative_index_only_for_slot_placement():
    config = DEFAULT_REAL_CHART_FEATURE_CONFIG
    current = RealChartObservation(
        motor=_motor(),
        orbiting_x=0.0,
        orbiting_y=0.0,
        floors=(_floor(0),),
    )
    values = encode_real_chart_observation(current)

    slot_size = REAL_CHART_FLOOR_FEATURE_DIM
    current_offset = 8 + config.behind_floors * slot_size
    current_slot = values[current_offset : current_offset + slot_size]
    assert current_slot[-1] == 1.0

    # A missing neighboring floor is all-zero padded; no absolute floor number,
    # chart time, target time, BPM, or timing error is synthesized into the vector.
    next_offset = current_offset + slot_size
    assert values[next_offset : next_offset + slot_size] == (0.0,) * slot_size


def test_real_chart_encoder_maps_visible_event_markers_without_event_payloads():
    observation = RealChartObservation(
        motor=_motor(),
        orbiting_x=0.0,
        orbiting_y=0.0,
        floors=(_floor(0, events=("Twirl", "SetSpeed", "Checkpoint")),),
    )
    values = encode_real_chart_observation(observation)
    slot_size = REAL_CHART_FLOOR_FEATURE_DIM
    offset = 8 + DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors * slot_size
    slot = values[offset : offset + slot_size]

    # floor layout: xy(2), entry sin/cos(2), exit sin/cos(2), mid, 3p,
    # six event-presence bits, mask.
    assert slot[8:14] == (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    assert slot[14] == 1.0


@pytest.mark.parametrize("field", ("behind_floors", "ahead_floors"))
@pytest.mark.parametrize("bad", (True, False, 1.5, "2", None, object(), -1))
def test_real_chart_feature_config_rejects_noninteger_floor_counts(field, bad):
    with pytest.raises(ValueError, match=rf"{field} must be a non-negative integer"):
        RealChartFeatureConfig(**{field: bad})


@pytest.mark.parametrize("field", ("tile_size", "position_span_tiles"))
@pytest.mark.parametrize(
    "bad",
    (True, False, "1.5", None, object(), float("nan"), float("inf"),
     float("-inf"), 10**1000, 0.0, -1.0),
)
def test_real_chart_feature_config_rejects_invalid_scales(field, bad):
    with pytest.raises(ValueError, match=rf"{field} must be finite and positive"):
        RealChartFeatureConfig(**{field: bad})


@pytest.mark.parametrize(
    "kwargs",
    (
        {"tile_size": 1e-300, "position_span_tiles": 1e-300},
        {"tile_size": 1e308, "position_span_tiles": 1e308},
    ),
)
def test_real_chart_feature_config_rejects_nonfinite_derived_scale(kwargs):
    with pytest.raises(
        ValueError, match=r"tile_size \* position_span_tiles must be finite and positive"
    ):
        RealChartFeatureConfig(**kwargs)


def test_real_chart_feature_config_preserves_valid_custom_layout_and_defaults():
    config = RealChartFeatureConfig(
        behind_floors=0, ahead_floors=3, tile_size=2.0, position_span_tiles=4.0
    )
    assert config.floor_slots == 4
    assert config.input_dim == 8 + 4 * REAL_CHART_FLOOR_FEATURE_DIM
    assert DEFAULT_REAL_CHART_FEATURE_CONFIG.input_dim == REAL_CHART_INPUT_DIM
    observation = RealChartObservation(
        motor=_motor(), orbiting_x=2.0, orbiting_y=-2.0,
        floors=(_floor(0),),
    )
    encoded = encode_real_chart_observation(observation, config=config)
    assert len(encoded) == config.input_dim
    assert encoded[6:8] == (1.0, -1.0)
