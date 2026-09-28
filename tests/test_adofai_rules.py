import pytest

from dmdod import (
    OverloadCounter,
    TimingDifficulty,
    TimingJudgement,
    classify_normal_timing,
    classify_timing,
    normal_accuracy_percent,
    normal_timing_windows,
    timing_windows,
    x_accuracy_components,
    x_accuracy_percent,
    x_accuracy_weight,
)


def test_normal_timing_windows_use_angle_then_time_floor():
    w180 = normal_timing_windows(180.0)
    assert w180.perfect_s * 1000.0 == pytest.approx(55.5555556)
    assert w180.early_late_perfect_s * 1000.0 == pytest.approx(83.3333333)
    assert w180.pass_s * 1000.0 == pytest.approx(111.1111111)
    assert w180.perfect_deg == pytest.approx(30.0)
    assert w180.early_late_perfect_deg == pytest.approx(45.0)
    assert w180.pass_deg == pytest.approx(60.0)

    # Normal's outer window reaches its 65 ms floor around 310 BPM, but the
    # inner boundaries continue tightening until their own 30/25 ms floors.
    w310 = normal_timing_windows(310.0)
    assert w310.pass_s * 1000.0 == pytest.approx(65.0)
    assert w310.early_late_perfect_s * 1000.0 == pytest.approx(48.3870968)
    assert w310.perfect_s * 1000.0 == pytest.approx(32.2580645)

    w620 = normal_timing_windows(620.0)
    assert w620.pass_s * 1000.0 == pytest.approx(65.0)
    assert w620.early_late_perfect_s * 1000.0 == pytest.approx(30.0)
    assert w620.perfect_s * 1000.0 == pytest.approx(25.0)


def test_lenient_normal_strict_change_outer_minimum():
    bpm = 400.0
    lenient = timing_windows(bpm, difficulty=TimingDifficulty.LENIENT)
    normal = timing_windows(bpm, difficulty=TimingDifficulty.NORMAL)
    strict = timing_windows(bpm, difficulty=TimingDifficulty.STRICT)

    assert lenient.pass_s * 1000.0 == pytest.approx(91.0)
    assert normal.pass_s * 1000.0 == pytest.approx(65.0)
    # At 400 BPM, Strict is still limited by the 60-degree angular boundary.
    assert strict.pass_s * 1000.0 == pytest.approx(50.0)

    strict500 = timing_windows(500.0, difficulty=TimingDifficulty.STRICT)
    assert strict500.pass_s * 1000.0 == pytest.approx(40.0)


def test_timing_scale_cannot_shrink_below_dll_time_minima():
    w120 = timing_windows(120.0, timing_scale=0.5)
    assert w120.perfect_s * 1000.0 == pytest.approx(41.6666667)
    assert w120.early_late_perfect_s * 1000.0 == pytest.approx(62.5)
    assert w120.pass_s * 1000.0 == pytest.approx(83.3333333)

    w300 = timing_windows(300.0, timing_scale=0.5)
    assert w300.perfect_s * 1000.0 == pytest.approx(25.0)
    assert w300.early_late_perfect_s * 1000.0 == pytest.approx(30.0)
    assert w300.pass_s * 1000.0 == pytest.approx(65.0)


def test_speed_trial_adjustment_respects_absolute_25ms_floor():
    windows = timing_windows(1000.0, speed_trial=4.0)
    assert windows.perfect_s * 1000.0 == pytest.approx(25.0)
    assert windows.early_late_perfect_s * 1000.0 == pytest.approx(25.0)
    assert windows.pass_s * 1000.0 == pytest.approx(25.0)


def test_mobile_uses_mobile_specific_minima():
    windows = timing_windows(1000.0, mobile=True)
    assert windows.perfect_s * 1000.0 == pytest.approx(50.0)
    assert windows.early_late_perfect_s * 1000.0 == pytest.approx(70.0)
    assert windows.pass_s * 1000.0 == pytest.approx(90.0)


def test_normal_timing_judgement_boundaries():
    bpm = 180.0
    assert classify_normal_timing(0.0, bpm) is TimingJudgement.PERFECT
    assert classify_normal_timing(-0.070, bpm) is TimingJudgement.EARLY_PERFECT
    assert classify_normal_timing(0.070, bpm) is TimingJudgement.LATE_PERFECT
    assert classify_normal_timing(-0.100, bpm) is TimingJudgement.VERY_EARLY
    assert classify_normal_timing(0.100, bpm) is TimingJudgement.VERY_LATE
    assert classify_normal_timing(-0.120, bpm) is TimingJudgement.TOO_EARLY
    assert classify_normal_timing(0.120, bpm) is TimingJudgement.TOO_LATE


def test_difficulty_changes_classification_at_high_bpm():
    # 55 ms early at 400 BPM: Strict is outside 50 ms, Normal is inside 65 ms.
    assert (
        classify_timing(-0.055, 400.0, difficulty=TimingDifficulty.STRICT)
        is TimingJudgement.TOO_EARLY
    )
    assert (
        classify_timing(-0.055, 400.0, difficulty=TimingDifficulty.NORMAL)
        is TimingJudgement.VERY_EARLY
    )


def test_x_accuracy_weights_match_dll_values():
    assert x_accuracy_weight(TimingJudgement.PERFECT) == pytest.approx(1.0)
    assert x_accuracy_weight(TimingJudgement.AUTO) == pytest.approx(1.0)
    assert x_accuracy_weight(TimingJudgement.EARLY_PERFECT) == pytest.approx(0.75)
    assert x_accuracy_weight(TimingJudgement.LATE_PERFECT) == pytest.approx(0.75)
    assert x_accuracy_weight(TimingJudgement.VERY_EARLY) == pytest.approx(0.40)
    assert x_accuracy_weight(TimingJudgement.VERY_LATE) == pytest.approx(0.40)
    assert x_accuracy_weight(TimingJudgement.TOO_EARLY) == pytest.approx(0.20)
    assert x_accuracy_weight(TimingJudgement.TOO_LATE) == pytest.approx(0.20)
    assert x_accuracy_weight(TimingJudgement.FAIL_MISS) == pytest.approx(0.0)
    assert x_accuracy_weight(TimingJudgement.FAIL_OVERLOAD) == pytest.approx(0.0)

    value = x_accuracy_percent(
        [
            TimingJudgement.PERFECT,
            TimingJudgement.EARLY_PERFECT,
            TimingJudgement.VERY_LATE,
            TimingJudgement.TOO_EARLY,
        ]
    )
    assert value == pytest.approx(58.75)


def test_x_accuracy_fail_is_one_zero_weight_denominator_entry():
    margins = [TimingJudgement.PERFECT, TimingJudgement.FAIL_MISS]
    points, denominator = x_accuracy_components(margins)
    assert points == pytest.approx(1.0)
    assert denominator == 2
    assert x_accuracy_percent(margins) == pytest.approx(50.0)

    # Checkpoint penalty is applied after the weighted fraction.
    assert x_accuracy_percent(margins, checkpoints_used=1) == pytest.approx(49.375)


def test_normal_accuracy_double_counts_fail_in_denominator():
    margins = [TimingJudgement.PERFECT, TimingJudgement.FAIL_OVERLOAD]
    # N=2 and F=1, so base=1/3. One pure Perfect also adds +0.0001.
    assert normal_accuracy_percent(margins) == pytest.approx(
        100.0 * (1.0 / 3.0 + 0.0001)
    )


def test_overload_counter_matches_dll_damage_strict_limit_and_decay():
    overload = OverloadCounter()

    assert not overload.record_too_early()
    assert overload.value == pytest.approx(0.5)
    assert not overload.record_too_early()
    assert overload.value == pytest.approx(1.0)
    assert not overload.overloaded  # DLL uses > 1.0, not >= 1.0.

    overload.record_valid_hit()
    assert overload.value == pytest.approx(1.0)  # valid hits do not heal it

    overload.advance_beats(0.5)
    assert overload.value == pytest.approx(0.8)
    assert overload.record_too_early()
    assert overload.value == pytest.approx(1.3)
    assert overload.overloaded


def test_overload_decay_can_prevent_a_later_too_early_from_failing():
    overload = OverloadCounter()
    overload.record_too_early()
    overload.record_too_early()
    overload.advance_beats(2.0)
    assert overload.value == pytest.approx(0.2)
    assert not overload.record_too_early()
    assert overload.value == pytest.approx(0.7)
