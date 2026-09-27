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
