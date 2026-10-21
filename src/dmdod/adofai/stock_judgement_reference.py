"""Small, stateless reference for audited Stock ADOFAI hit categories.

This module intentionally models only the angle-boundary calculation in
``scrMisc.GetAdjustedAngleBoundaryInDeg`` and the ordered comparisons in
``scrMisc.GetHitMargin`` from the DLL snapshot audited in Phase 19.  It does
not model floor selection, automatic misses, fail state, or input queues.

Signed errors and boundaries are in degrees.  Negative errors are early and
positive errors are late.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


PERFECT_BASE_DEG = 30.0
EARLY_LATE_PERFECT_BASE_DEG = 45.0
COUNTED_BASE_DEG = 60.0

LENIENT_COUNTED_MIN_S = 0.091
NORMAL_COUNTED_MIN_S = 0.065
STRICT_COUNTED_MIN_S = 0.040
EARLY_LATE_PERFECT_MIN_S = 0.030
PURE_PERFECT_MIN_S = 0.020
ABSOLUTE_MIN_S = 0.025

MOBILE_COUNTED_MIN_S = 0.090
MOBILE_EARLY_LATE_PERFECT_MIN_S = 0.070
MOBILE_PERFECT_MIN_S = 0.050


class StockTimingDifficulty(str, Enum):
    LENIENT = "lenient"
    NORMAL = "normal"
    STRICT = "strict"


class StockHitMargin(str, Enum):
    """Ordinary timing categories produced by the audited comparison chain."""

    TOO_EARLY = "too_early"
    VERY_EARLY = "very_early"
    EARLY_PERFECT = "early_perfect"
    PERFECT = "perfect"
    LATE_PERFECT = "late_perfect"
    VERY_LATE = "very_late"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class StockAngleBoundaries:
    """Positive one-sided radii, in degrees, from inner to outer."""

    perfect_deg: float
    early_late_perfect_deg: float
    counted_deg: float

    def __post_init__(self) -> None:
        values = (
            self.perfect_deg,
            self.early_late_perfect_deg,
            self.counted_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("angle boundaries must be finite")
        if not 0.0 <= values[0] <= values[1] <= values[2]:
            raise ValueError("angle boundaries must be non-negative and ordered")


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _coerce_difficulty(
    value: StockTimingDifficulty | str,
) -> StockTimingDifficulty:
    if isinstance(value, StockTimingDifficulty):
        return value
    return StockTimingDifficulty(str(value).lower())


def _desktop_counted_min_s(difficulty: StockTimingDifficulty) -> float:
    if difficulty is StockTimingDifficulty.LENIENT:
        return LENIENT_COUNTED_MIN_S
    if difficulty is StockTimingDifficulty.STRICT:
        return STRICT_COUNTED_MIN_S
    return NORMAL_COUNTED_MIN_S


def stock_angle_boundaries(
    bpm: float,
    *,
    speed: float = 1.0,
    pitch: float = 1.0,
    difficulty: StockTimingDifficulty | str = StockTimingDifficulty.NORMAL,
    speed_trial: float = 1.0,
    margin_scale: float = 1.0,
    mobile: bool = False,
) -> StockAngleBoundaries:
    """Build audited Stock angle radii for one tile.

    ``bpm``, ``speed``, ``pitch``, and ``speed_trial`` are positive scalar
    multipliers. ``margin_scale`` is the normalized tile margin scale (1.0 is
    100%).  Desktop minimum times are divided by ``speed_trial`` and then
    clamped to 25 ms.  The audited mobile branch selects fixed 90/70/50 ms
    minima instead of applying the speed-trial division.
    """

    _positive_finite("bpm", bpm)
    _positive_finite("speed", speed)
    _positive_finite("pitch", pitch)
    _positive_finite("speed_trial", speed_trial)
    if not math.isfinite(margin_scale) or margin_scale < 0.0:
        raise ValueError("margin_scale must be finite and non-negative")

    difficulty = _coerce_difficulty(difficulty)
    angular_speed_deg_s = 3.0 * bpm * speed * pitch

    if mobile:
        counted_min_s = max(MOBILE_COUNTED_MIN_S, ABSOLUTE_MIN_S)
        early_late_perfect_min_s = max(
            MOBILE_EARLY_LATE_PERFECT_MIN_S, ABSOLUTE_MIN_S
        )
        perfect_min_s = max(MOBILE_PERFECT_MIN_S, ABSOLUTE_MIN_S)
    else:
        counted_min_s = max(
            _desktop_counted_min_s(difficulty) / speed_trial,
            ABSOLUTE_MIN_S,
        )
        early_late_perfect_min_s = max(
            EARLY_LATE_PERFECT_MIN_S / speed_trial,
            ABSOLUTE_MIN_S,
        )
        perfect_min_s = max(
            PURE_PERFECT_MIN_S / speed_trial,
            ABSOLUTE_MIN_S,
        )

    return StockAngleBoundaries(
        perfect_deg=max(
            PERFECT_BASE_DEG * margin_scale,
            angular_speed_deg_s * perfect_min_s,
        ),
        early_late_perfect_deg=max(
            EARLY_LATE_PERFECT_BASE_DEG * margin_scale,
            angular_speed_deg_s * early_late_perfect_min_s,
        ),
        counted_deg=max(
            COUNTED_BASE_DEG * margin_scale,
            angular_speed_deg_s * counted_min_s,
        ),
    )


def classify_stock_angle(
    signed_error_deg: float,
    boundaries: StockAngleBoundaries,
) -> StockHitMargin:
    """Classify a signed angle error using the DLL's ordered strict ``>`` tests."""

    if not math.isfinite(signed_error_deg):
        raise ValueError("signed_error_deg must be finite")

    result = StockHitMargin.TOO_EARLY
    if signed_error_deg > -boundaries.counted_deg:
        result = StockHitMargin.VERY_EARLY
    if signed_error_deg > -boundaries.early_late_perfect_deg:
        result = StockHitMargin.EARLY_PERFECT
    if signed_error_deg > -boundaries.perfect_deg:
        result = StockHitMargin.PERFECT
    if signed_error_deg > boundaries.perfect_deg:
        result = StockHitMargin.LATE_PERFECT
    if signed_error_deg > boundaries.early_late_perfect_deg:
        result = StockHitMargin.VERY_LATE
    if signed_error_deg > boundaries.counted_deg:
        result = StockHitMargin.TOO_LATE
    return result
