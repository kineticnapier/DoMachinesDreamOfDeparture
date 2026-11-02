import math

import pytest

from dmdod.evaluator import TargetHit
from dmdod.planet_perception import PlanetVisionConfig, StraightPlanetGeometryEncoder


def test_straight_geometry_reaches_next_tile_at_target_time():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    observation = encoder.observe(1.0, active_target_index=0)
    assert observation.orbit_x == pytest.approx(1.0)
    assert observation.orbit_y == pytest.approx(0.0, abs=1e-9)
    assert observation.next_x == pytest.approx(1.0)
    assert observation.next_y == pytest.approx(0.0)


def test_straight_geometry_is_opposite_one_beat_before_target():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    # 120 BPM => one beat = 0.5 s; straight geometry moves pi radians/beat.
    observation = encoder.observe(0.5, active_target_index=0)
    assert observation.orbit_x == pytest.approx(-1.0)
    assert observation.orbit_y == pytest.approx(0.0, abs=1e-9)


def test_visual_sample_is_held_until_next_frame():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=180.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=1.0 / 60.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    first = encoder.observe(0.900, active_target_index=0)
    held = encoder.observe(0.905, active_target_index=0)
    refreshed = encoder.observe(0.920, active_target_index=0)

    assert held == first
    assert refreshed != first


def test_target_change_forces_fresh_sample():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left"), TargetHit(1.5, "left")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=1.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    first = encoder.observe(1.0, active_target_index=0)
    second = encoder.observe(1.0, active_target_index=1)
    assert first != second


def test_no_active_target_returns_blank_geometry():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    observation = encoder.observe(0.0, active_target_index=None)
    assert observation.orbit_x == 0.0
    assert observation.orbit_y == 0.0
    assert observation.next_x == 0.0
    assert observation.next_y == 0.0


@pytest.mark.parametrize(
    "field",
    (
        "latency_s",
        "latency_jitter_s",
        "sample_period_s",
        "position_noise_std",
        "dropout_probability",
    ),
)
@pytest.mark.parametrize(
    "invalid",
    (float("nan"), float("inf"), float("-inf"), True, False, None, "0.1", 10**400),
)
def test_planet_vision_config_rejects_invalid_numeric_values(field, invalid):
    with pytest.raises(ValueError):
        PlanetVisionConfig(**{field: invalid})


@pytest.mark.parametrize(
    "invalid",
    (float("nan"), float("inf"), float("-inf"), True, False, None, "120", 10**400),
)
def test_straight_geometry_rejects_invalid_bpm(invalid):
    with pytest.raises(ValueError):
        StraightPlanetGeometryEncoder([TargetHit(1.0, "left")], bpm=invalid)


@pytest.mark.parametrize(
    "invalid",
    (float("nan"), float("inf"), float("-inf"), True, False, None, "1.0", 10**400),
)
def test_straight_geometry_rejects_invalid_target_time(invalid):
    with pytest.raises(ValueError):
        StraightPlanetGeometryEncoder([TargetHit(invalid, "left")], bpm=120.0)


@pytest.mark.parametrize("invalid", ("middle", None, 1))
def test_straight_geometry_rejects_invalid_target_key(invalid):
    with pytest.raises(ValueError):
        StraightPlanetGeometryEncoder([TargetHit(1.0, invalid)], bpm=120.0)


@pytest.mark.parametrize(
    "invalid",
    (float("nan"), float("inf"), float("-inf"), True, False, None, "0.1", 10**400),
)
def test_invalid_observation_time_does_not_corrupt_held_frame(invalid):
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.5,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.0, active_target_index=0)
    next_sample_time = encoder._next_sample_time_s
    with pytest.raises(ValueError):
        encoder.observe(invalid, active_target_index=0)
    assert encoder._next_sample_time_s == next_sample_time
    assert encoder._held_observation == first
    assert encoder._held_target_index == 0
    assert encoder.observe(0.1, active_target_index=0) == first


def test_planet_vision_config_preserves_finite_zero_and_dropout_one():
    config = PlanetVisionConfig(
        latency_s=0.0,
        latency_jitter_s=0.0,
        sample_period_s=0.0,
        position_noise_std=0.0,
        dropout_probability=1.0,
    )
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(-0.5, "right")], bpm=120.0, config=config,
    )
    assert encoder.observe(-0.5, active_target_index=0).orbit_x == 0.0
