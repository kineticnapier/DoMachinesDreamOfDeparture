import pytest

from dmdod import TargetHit, VisualCueConfig, VisualCueEncoder


def test_visual_cue_can_select_only_the_active_target():
    config = VisualCueConfig(latency_s=0.0, width_s=0.050, horizon_s=1.0)
    encoder = VisualCueEncoder(
        [
            TargetHit(0.50, "left"),
            TargetHit(0.80, "right"),
        ],
        config=config,
    )

    both = encoder.observe(0.50)
    first_only = encoder.observe(0.50, active_target_indices=(0,))
    second_only = encoder.observe(0.50, active_target_indices=(1,))
    none = encoder.observe(0.50, active_target_indices=())

    assert both.left > 0.99
    assert first_only.left > 0.99
    assert first_only.right == 0.0
    assert second_only.left == 0.0
    assert second_only.right > 0.0
    assert none.left == 0.0
    assert none.right == 0.0


def test_visual_cue_rejects_invalid_active_target_index():
    encoder = VisualCueEncoder([TargetHit(0.50, "left")])

    with pytest.raises(IndexError):
        encoder.observe(0.0, active_target_indices=(1,))


def test_visual_cue_latency_jitter_is_reproducible_by_seed():
    config = VisualCueConfig(latency_s=0.050, latency_jitter_s=0.015)
    a = VisualCueEncoder([TargetHit(0.50, "left")], config=config, seed=123)
    b = VisualCueEncoder([TargetHit(0.50, "left")], config=config, seed=123)
    c = VisualCueEncoder([TargetHit(0.50, "left")], config=config, seed=456)

    assert a.episode_latency_s == pytest.approx(b.episode_latency_s)
    assert a.episode_latency_s != pytest.approx(c.episode_latency_s)
    assert 0.035 <= a.episode_latency_s <= 0.065


def test_visual_cue_sample_period_holds_frame_until_next_sample():
    config = VisualCueConfig(
        latency_s=0.0,
        width_s=0.050,
        horizon_s=1.0,
        sample_period_s=0.020,
    )
    encoder = VisualCueEncoder([TargetHit(0.50, "left")], config=config, seed=1)

    first = encoder.observe(0.45, active_target_indices=(0,))
    held = encoder.observe(0.46, active_target_indices=(0,))
    refreshed = encoder.observe(0.471, active_target_indices=(0,))

    assert held == first
    assert refreshed != first


def test_visual_cue_target_change_forces_fresh_frame():
    config = VisualCueConfig(
        latency_s=0.0,
        width_s=0.050,
        horizon_s=1.0,
        sample_period_s=0.100,
    )
    encoder = VisualCueEncoder(
        [TargetHit(0.50, "left"), TargetHit(0.80, "right")],
        config=config,
        seed=1,
    )

    first = encoder.observe(0.50, active_target_indices=(0,))
    second = encoder.observe(0.50, active_target_indices=(1,))

    assert first.left > 0.99 and first.right == 0.0
    assert second.left == 0.0 and second.right > 0.0

@pytest.mark.parametrize(
    "field",
    (
        "latency_s", "width_s", "horizon_s", "latency_jitter_s",
        "sample_period_s", "amplitude_noise_std", "dropout_probability",
    ),
)
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_visual_cue_config_rejects_nonfinite_sensor_parameters(field, bad):
    # NaN previously bypassed every range check; positive infinity bypassed
    # most of them. Neither is a meaningful observable sensor parameter.
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        VisualCueConfig(**{field: bad})


def test_visual_cue_config_preserves_finite_boundary_values():
    config = VisualCueConfig(
        latency_s=0.0,
        width_s=1e-6,
        horizon_s=1e-6,
        latency_jitter_s=0.0,
        sample_period_s=0.0,
        amplitude_noise_std=0.0,
        dropout_probability=1.0,
    )
    assert config.dropout_probability == 1.0
    assert config.width_s == 1e-6
    assert VisualCueConfig().latency_s == 0.050


@pytest.mark.parametrize(
    "field",
    (
        "latency_s", "width_s", "horizon_s", "latency_jitter_s",
        "sample_period_s", "amplitude_noise_std", "dropout_probability",
    ),
)
@pytest.mark.parametrize("bad", (True, False, "0.1", b"0.1", None, 10**1000, object()))
def test_visual_cue_config_rejects_malformed_sensor_parameters(field, bad):
    # Each invalid field must raise the same public ValueError instead of
    # silently accepting bool or leaking TypeError/OverflowError.
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        VisualCueConfig(**{field: bad})


def test_visual_cue_config_preserves_integer_compatible_finite_values():
    config = VisualCueConfig(
        latency_s=0, width_s=1, horizon_s=2,
        latency_jitter_s=0, sample_period_s=0,
        amplitude_noise_std=0, dropout_probability=1,
    )
    assert config.width_s == 1
    assert config.dropout_probability == 1


def test_visual_cue_sample_clock_skips_large_observation_gap():
    # An unbounded per-frame loop would require 50 million iterations.
    config = VisualCueConfig(
        latency_s=0.0, sample_period_s=0.020,
    )
    encoder = VisualCueEncoder([TargetHit(0.5, "left")], config=config)
    encoder.observe(0.45, active_target_indices=(0,))
    result = encoder.observe(1_000_000.0, active_target_indices=(0,))
    assert result.left == result.right == 0.0
    assert 1_000_000.0 < encoder._next_sample_time_s <= 1_000_000.021


def test_visual_cue_sample_clock_sub_ulp_period_makes_progress():
    # At 0.45 seconds, adding the smallest subnormal does not advance a float.
    config = VisualCueConfig(
        latency_s=0.0, sample_period_s=5e-324,
    )
    encoder = VisualCueEncoder([TargetHit(0.5, "left")], config=config)
    encoder.observe(0.45, active_target_indices=(0,))
    encoder.observe(0.46, active_target_indices=(0,))
    assert encoder._next_sample_time_s > 0.46


def test_visual_cue_sample_clock_preserves_ordinary_repeated_addition():
    period = 0.0001
    config = VisualCueConfig(latency_s=0.0, sample_period_s=period)
    encoder = VisualCueEncoder([TargetHit(0.5, "left")], config=config)
    encoder.observe(0.45, active_target_indices=(0,))
    encoder.observe(0.50, active_target_indices=(0,))
    expected = 0.45 + period
    while expected <= 0.50 + 1e-12:
        expected += period
    assert encoder._next_sample_time_s == expected
