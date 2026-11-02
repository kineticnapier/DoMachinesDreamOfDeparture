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


@pytest.mark.parametrize("sample_period_s", (0.0, 0.5))
@pytest.mark.parametrize(
    "invalid",
    (False, True, 0.0, 1.0, 0.5, "0", b"0", object(), -1, 2,
     float("nan"), float("inf")),
)
def test_planet_rejects_invalid_active_index_before_held_frame_without_mutation(
    sample_period_s, invalid,
):
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left"), TargetHit(1.5, "right")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=sample_period_s,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.9, active_target_index=0)
    clock = encoder._next_sample_time_s
    with pytest.raises(IndexError, match="active target index"):
        encoder.observe(0.91, active_target_index=invalid)
    assert encoder._held_observation == first
    assert encoder._held_target_index == 0
    assert encoder._next_sample_time_s == clock
    if sample_period_s:
        assert encoder.observe(0.91, active_target_index=0) == first


def test_planet_allows_none_and_integer_indices_after_validation():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left"), TargetHit(1.5, "right")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0, latency_jitter_s=0.0,
            sample_period_s=0.5, position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.9, active_target_index=0)
    assert encoder.observe(0.91, active_target_index=0) == first
    assert encoder.observe(0.91, active_target_index=1) != first
    blank = encoder.observe(0.91, active_target_index=None)
    assert blank.orbit_x == blank.orbit_y == blank.next_x == blank.next_y == 0.0


class _FloatOnly:
    """A finite numeric input without ordering or arithmetic operators."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


@pytest.mark.parametrize(
    "field",
    (
        "latency_s", "latency_jitter_s", "sample_period_s",
        "position_noise_std", "dropout_probability",
    ),
)
def test_planet_config_normalizes_float_only_sensor_values(field):
    config = PlanetVisionConfig(**{field: _FloatOnly(0.1)})
    assert type(getattr(config, field)) is float
    assert getattr(config, field) == pytest.approx(0.1)
    # The normalized field must also be usable by the sensor at runtime.
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")], bpm=120.0, config=config, seed=1,
    )
    assert math.isfinite(encoder.observe(0.9, active_target_index=0).orbit_x)


@pytest.mark.parametrize(
    "field",
    (
        "latency_s", "latency_jitter_s", "sample_period_s",
        "position_noise_std", "dropout_probability",
    ),
)
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_planet_config_rejects_nonfinite_float_only_values(field, bad):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        PlanetVisionConfig(**{field: _FloatOnly(bad)})


def test_planet_encoder_accepts_float_only_bpm_and_observation_time():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")],
        bpm=_FloatOnly(120.0),
        config=PlanetVisionConfig(
            latency_s=0.0, latency_jitter_s=0.0,
            sample_period_s=0.5, position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )
    assert encoder.bpm == 120.0
    first = encoder.observe(_FloatOnly(0.9), active_target_index=0)
    assert encoder.observe(_FloatOnly(0.91), active_target_index=0) == first


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_planet_encoder_rejects_nonfinite_float_only_time_atomically(bad):
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left")], bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0, latency_jitter_s=0.0,
            sample_period_s=0.5, position_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.9, active_target_index=0)
    before = (encoder._held_observation, encoder._held_target_index,
              encoder._next_sample_time_s)
    with pytest.raises(ValueError, match="simulation_time_s must be finite"):
        encoder.observe(_FloatOnly(bad), active_target_index=0)
    assert (encoder._held_observation, encoder._held_target_index,
            encoder._next_sample_time_s) == before
    assert encoder.observe(0.91, active_target_index=0) == first


def test_planet_normalizes_float_only_target_times_without_mutating_input():
    original = [
        TargetHit(_FloatOnly(1.5), "right"),
        TargetHit(_FloatOnly(1.0), "left"),
    ]
    config = PlanetVisionConfig(
        latency_s=0.0, latency_jitter_s=0.0, sample_period_s=0.0,
        position_noise_std=0.0, dropout_probability=0.0,
    )
    encoder = StraightPlanetGeometryEncoder(original, bpm=120.0, config=config)
    baseline = StraightPlanetGeometryEncoder(
        [TargetHit(1.5, "right"), TargetHit(1.0, "left")],
        bpm=120.0, config=config,
    )
    assert [target.time_s for target in encoder._targets] == [1.0, 1.5]
    assert all(type(target.time_s) is float for target in encoder._targets)
    assert isinstance(original[0].time_s, _FloatOnly)
    for time_s in (0.9, 1.0, 1.5):
        for index in (0, 1):
            assert encoder.observe(time_s, active_target_index=index) == baseline.observe(
                time_s, active_target_index=index
            )


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_planet_rejects_nonfinite_float_only_target_times(bad):
    with pytest.raises(ValueError, match="target time_s must be finite"):
        StraightPlanetGeometryEncoder(
            [TargetHit(_FloatOnly(bad), "left")], bpm=120.0
        )


class _UnorderablePlanetIndex(int):
    def __lt__(self, other):
        raise TypeError("custom ordering unavailable")

    def __ge__(self, other):
        raise TypeError("custom ordering unavailable")


class _AlwaysEqualPlanetIndex(int):
    def __eq__(self, other):
        return True


def test_planet_normalizes_integral_target_indices_before_range_checks():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left"), TargetHit(1.5, "right")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0, latency_jitter_s=0.0, sample_period_s=0.5,
            position_noise_std=0.0, dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.9, active_target_index=_UnorderablePlanetIndex(0))
    assert encoder._held_target_index == 0
    assert type(encoder._held_target_index) is int
    clock = encoder._next_sample_time_s
    with pytest.raises(IndexError, match="out of range"):
        encoder.observe(0.91, active_target_index=_UnorderablePlanetIndex(2))
    assert encoder._held_observation == first
    assert encoder._held_target_index == 0
    assert encoder._next_sample_time_s == clock


def test_planet_normalizes_integral_index_before_held_frame_comparison():
    encoder = StraightPlanetGeometryEncoder(
        [TargetHit(1.0, "left"), TargetHit(1.5, "right")],
        bpm=120.0,
        config=PlanetVisionConfig(
            latency_s=0.0, latency_jitter_s=0.0, sample_period_s=0.5,
            position_noise_std=0.0, dropout_probability=0.0,
        ),
    )
    first = encoder.observe(0.9, active_target_index=0)
    second = encoder.observe(0.91, active_target_index=_AlwaysEqualPlanetIndex(1))
    assert second != first
    assert encoder._held_target_index == 1
    assert type(encoder._held_target_index) is int
