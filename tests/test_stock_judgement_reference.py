from __future__ import annotations

import math

import pytest

from dmdod.adofai.rules import TimingJudgement, classify_timing, timing_windows
from dmdod.adofai.stock_judgement_reference import (
    StockAngleBoundaries,
    StockHitMargin,
    StockTimingDifficulty,
    classify_stock_angle,
    stock_angle_boundaries,
)


def test_default_angle_boundaries_use_base_degrees() -> None:
    assert stock_angle_boundaries(180.0) == StockAngleBoundaries(30.0, 45.0, 60.0)


@pytest.mark.parametrize(
    ("difficulty", "expected_counted_deg"),
    [
        (StockTimingDifficulty.LENIENT, 273.0),
        (StockTimingDifficulty.NORMAL, 195.0),
        (StockTimingDifficulty.STRICT, 120.0),
    ],
)
def test_desktop_time_minima_expand_with_difficulty(
    difficulty: StockTimingDifficulty,
    expected_counted_deg: float,
) -> None:
    boundaries = stock_angle_boundaries(1000.0, difficulty=difficulty)

    assert boundaries.perfect_deg == pytest.approx(75.0)
    assert boundaries.early_late_perfect_deg == pytest.approx(90.0)
    assert boundaries.counted_deg == pytest.approx(expected_counted_deg)


def test_speed_pitch_margin_scale_and_desktop_speed_trial_are_applied() -> None:
    assert stock_angle_boundaries(1000.0, margin_scale=0.0) == StockAngleBoundaries(
        75.0, 90.0, 195.0
    )
    assert stock_angle_boundaries(
        1000.0, speed=2.0, margin_scale=0.0
    ) == StockAngleBoundaries(150.0, 180.0, 390.0)
    assert stock_angle_boundaries(
        1000.0, pitch=0.5, margin_scale=0.0
    ) == StockAngleBoundaries(37.5, 45.0, 97.5)
    assert stock_angle_boundaries(
        1000.0, speed_trial=2.0, margin_scale=0.0
    ) == StockAngleBoundaries(75.0, 75.0, 97.5)
    assert stock_angle_boundaries(180.0, margin_scale=2.0) == StockAngleBoundaries(
        60.0, 90.0, 120.0
    )


def test_mobile_minima_do_not_use_speed_trial_division() -> None:
    boundaries = stock_angle_boundaries(1000.0, speed_trial=4.0, mobile=True)

    assert boundaries.perfect_deg == pytest.approx(150.0)
    assert boundaries.early_late_perfect_deg == pytest.approx(210.0)
    assert boundaries.counted_deg == pytest.approx(270.0)


def test_stock_classification_at_center_and_every_boundary() -> None:
    boundaries = StockAngleBoundaries(30.0, 45.0, 60.0)
    cases = {
        0.0: StockHitMargin.PERFECT,
        math.nextafter(-60.0, -math.inf): StockHitMargin.TOO_EARLY,
        -60.0: StockHitMargin.TOO_EARLY,
        math.nextafter(-60.0, math.inf): StockHitMargin.VERY_EARLY,
        math.nextafter(-45.0, -math.inf): StockHitMargin.VERY_EARLY,
        -45.0: StockHitMargin.VERY_EARLY,
        math.nextafter(-45.0, math.inf): StockHitMargin.EARLY_PERFECT,
        math.nextafter(-30.0, -math.inf): StockHitMargin.EARLY_PERFECT,
        -30.0: StockHitMargin.EARLY_PERFECT,
        math.nextafter(-30.0, math.inf): StockHitMargin.PERFECT,
        math.nextafter(30.0, -math.inf): StockHitMargin.PERFECT,
        30.0: StockHitMargin.PERFECT,
        math.nextafter(30.0, math.inf): StockHitMargin.LATE_PERFECT,
        math.nextafter(45.0, -math.inf): StockHitMargin.LATE_PERFECT,
        45.0: StockHitMargin.LATE_PERFECT,
        math.nextafter(45.0, math.inf): StockHitMargin.VERY_LATE,
        math.nextafter(60.0, -math.inf): StockHitMargin.VERY_LATE,
        60.0: StockHitMargin.VERY_LATE,
        math.nextafter(60.0, math.inf): StockHitMargin.TOO_LATE,
    }

    assert {
        error: classify_stock_angle(error, boundaries) for error in cases
    } == cases


@pytest.mark.parametrize(
    ("boundary_name", "stock_expected", "repository_expected"),
    [
        ("perfect", StockHitMargin.EARLY_PERFECT, TimingJudgement.PERFECT),
        (
            "early_late_perfect",
            StockHitMargin.VERY_EARLY,
            TimingJudgement.EARLY_PERFECT,
        ),
        ("pass", StockHitMargin.TOO_EARLY, TimingJudgement.VERY_EARLY),
    ],
)
def test_negative_exact_boundaries_differ_from_repository_classifier(
    boundary_name: str,
    stock_expected: StockHitMargin,
    repository_expected: TimingJudgement,
) -> None:
    repository_windows = timing_windows(180.0)
    stock_boundaries = stock_angle_boundaries(180.0)
    repository_seconds = getattr(repository_windows, f"{boundary_name}_s")
    stock_degrees_name = (
        "counted_deg" if boundary_name == "pass" else f"{boundary_name}_deg"
    )
    stock_degrees = getattr(stock_boundaries, stock_degrees_name)

    assert classify_stock_angle(-stock_degrees, stock_boundaries) is stock_expected
    assert classify_timing(-repository_seconds, 180.0) is repository_expected


def test_mobile_window_difference_changes_classification() -> None:
    bpm = 1000.0
    error_s = 0.050
    angular_speed_deg_s = 3.0 * bpm
    stock_boundaries = stock_angle_boundaries(
        bpm,
        speed_trial=4.0,
        mobile=True,
    )

    assert (
        classify_stock_angle(error_s * angular_speed_deg_s, stock_boundaries)
        is StockHitMargin.PERFECT
    )
    assert (
        classify_timing(error_s, bpm, speed_trial=4.0, mobile=True)
        is TimingJudgement.TOO_LATE
    )


def test_boundaries_and_errors_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        StockAngleBoundaries(45.0, 30.0, 60.0)
    with pytest.raises(ValueError):
        stock_angle_boundaries(0.0)
    with pytest.raises(ValueError):
        stock_angle_boundaries(180.0, margin_scale=math.inf)
    with pytest.raises(ValueError):
        classify_stock_angle(math.nan, StockAngleBoundaries(30.0, 45.0, 60.0))
