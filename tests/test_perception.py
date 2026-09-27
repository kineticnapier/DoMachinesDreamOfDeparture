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

    try:
        encoder.observe(0.0, active_target_indices=(1,))
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for invalid active target index")
