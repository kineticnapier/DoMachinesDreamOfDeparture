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
