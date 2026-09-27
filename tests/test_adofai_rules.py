import pytest

from dmdod import (
    OverloadCounter,
    TimingJudgement,
    classify_normal_timing,
    normal_timing_windows,
)


def test_normal_timing_windows_scale_until_310_bpm():
    w180 = normal_timing_windows(180.0)
    assert w180.perfect_s * 1000.0 == pytest.approx(55.5555556)
    assert w180.early_late_perfect_s * 1000.0 == pytest.approx(83.3333333)
    assert w180.pass_s * 1000.0 == pytest.approx(111.1111111)

    w310 = normal_timing_windows(310.0)
    w620 = normal_timing_windows(620.0)
    assert w310 == w620
    assert w310.perfect_s * 1000.0 == pytest.approx(32.2580645)
    assert w310.pass_s * 1000.0 == pytest.approx(64.5161290)


def test_normal_timing_judgement_boundaries():
    bpm = 180.0
    assert classify_normal_timing(0.0, bpm) is TimingJudgement.PERFECT
    assert classify_normal_timing(-0.070, bpm) is TimingJudgement.EARLY_PERFECT
    assert classify_normal_timing(0.070, bpm) is TimingJudgement.LATE_PERFECT
    assert classify_normal_timing(-0.100, bpm) is TimingJudgement.EARLY
    assert classify_normal_timing(0.100, bpm) is TimingJudgement.LATE
    assert classify_normal_timing(-0.120, bpm) is TimingJudgement.TOO_EARLY
    assert classify_normal_timing(0.120, bpm) is TimingJudgement.TOO_LATE


def test_overload_counter_matches_too_early_rule():
    overload = OverloadCounter()
    assert not overload.record_too_early()
    assert overload.value == 2
    assert not overload.record_too_early()
    assert overload.value == 4

    overload.record_valid_hit()
    assert overload.value == 3

    assert not overload.record_too_early()
    assert overload.value == 5
    assert overload.record_too_early()
    assert overload.value == 7
    assert overload.overloaded

    overload.record_valid_hit()
    assert overload.value == 6
