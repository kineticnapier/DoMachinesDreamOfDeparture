from dmdod.ablation import CueAblationMode, CueAblator
from dmdod.perception import VisualCueObservation


def test_zero_ablation_removes_visual_signal():
    ablator = CueAblator(CueAblationMode.ZERO)
    assert ablator.transform(VisualCueObservation(0.8, 0.2)) == VisualCueObservation(0.0, 0.0)


def test_random_ablation_is_reproducible_for_seed():
    a = CueAblator(CueAblationMode.RANDOM, seed=123)
    b = CueAblator(CueAblationMode.RANDOM, seed=123)
    cue = VisualCueObservation(0.8, 0.2)

    assert a.transform(cue) == b.transform(cue)
    assert a.transform(cue) == b.transform(cue)


def test_delay_ablation_delays_by_control_steps():
    ablator = CueAblator(
        CueAblationMode.DELAY,
        control_dt_s=0.010,
        delay_s=0.020,
    )
    first = VisualCueObservation(0.1, 0.2)
    second = VisualCueObservation(0.3, 0.4)
    third = VisualCueObservation(0.5, 0.6)

    assert ablator.transform(first) == VisualCueObservation(0.0, 0.0)
    assert ablator.transform(second) == VisualCueObservation(0.0, 0.0)
    assert ablator.transform(third) == first


def test_zero_delay_is_identity():
    ablator = CueAblator(CueAblationMode.DELAY, delay_s=0.0)
    cue = VisualCueObservation(0.7, 0.1)
    assert ablator.transform(cue) == cue
