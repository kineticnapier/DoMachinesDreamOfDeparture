from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


# Custom-level difficulty UI thresholds found in the decompiled game.
LENIENT_OPTION_MINIMUM_BPM_CUSTOM = 220.0
STRICT_OPTION_MINIMUM_BPM_CUSTOM = 310.0
# Backward-compatible name from the first toy implementation. 310 BPM is not a
# universal timing-window freeze point in the real game.
NORMAL_THRESHOLD_BPM = STRICT_OPTION_MINIMUM_BPM_CUSTOM

OVERLOAD_LIMIT = 6

# Base angular boundaries used by ADOFAI before minimum-time clamping.
PERFECT_BASE_DEG = 30.0
EARLY_LATE_PERFECT_BASE_DEG = 45.0
COUNTED_BASE_DEG = 60.0

# Desktop minimum timing radii from the DLL. The inner/Pure value is 20 ms,
# but the game subsequently applies a 25 ms absolute minimum, so the effective
# default Perfect floor is 25 ms at 1x.
LENIENT_COUNTED_MIN_S = 0.091
NORMAL_COUNTED_MIN_S = 0.065
STRICT_COUNTED_MIN_S = 0.040
EARLY_LATE_PERFECT_MIN_S = 0.030
PURE_PERFECT_MIN_S = 0.020
ABSOLUTE_MIN_S = 0.025

# Mobile uses its own timing minima.
MOBILE_COUNTED_MIN_S = 0.090
MOBILE_EARLY_LATE_PERFECT_MIN_S = 0.070
MOBILE_PERFECT_MIN_S = 0.050


class TimingDifficulty(str, Enum):
    LENIENT = "lenient"
    NORMAL = "normal"
    STRICT = "strict"


class TimingJudgement(str, Enum):
    """ADOFAI timing categories relevant to ordinary tile hits.

    The DLL calls the outer counted-hit categories VeryEarly/VeryLate. EARLY
    and LATE remain aliases so old simulator code keeps working.
    """

    PERFECT = "perfect"
    EARLY_PERFECT = "early_perfect"
    LATE_PERFECT = "late_perfect"
    VERY_EARLY = "very_early"
    VERY_LATE = "very_late"
    EARLY = "very_early"
    LATE = "very_late"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class TimingWindows:
    """One-sided timing-window radii after all ADOFAI clamps.

    ``*_deg`` are the final angular radii. ``*_s`` are the same boundaries in
    seconds for the supplied effective BPM/pitch.
    """

    perfect_s: float
    early_late_perfect_s: float
    pass_s: float
    perfect_deg: float
    early_late_perfect_deg: float
    pass_deg: float


def _coerce_difficulty(value: TimingDifficulty | str) -> TimingDifficulty:
    if isinstance(value, TimingDifficulty):
        return value
    return TimingDifficulty(str(value).lower())


def _desktop_counted_min_s(difficulty: TimingDifficulty) -> float:
    if difficulty is TimingDifficulty.LENIENT:
        return LENIENT_COUNTED_MIN_S
    if difficulty is TimingDifficulty.STRICT:
        return STRICT_COUNTED_MIN_S
    return NORMAL_COUNTED_MIN_S


def _speed_adjusted_minimum(base_s: float, speed_trial: float) -> float:
    """Mirror the DLL's speed-trial adjustment plus absolute 25 ms floor."""

    if speed_trial <= 0.0:
        raise ValueError("speed_trial must be positive")
    return max(base_s / speed_trial, ABSOLUTE_MIN_S)


def timing_windows(
    bpm: float,
    *,
    difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
    timing_scale: float = 1.0,
    controller_speed: float = 1.0,
    pitch: float = 1.0,
    speed_trial: float = 1.0,
    mobile: bool = False,
) -> TimingWindows:
    """Return DLL-style timing windows for a tile.

    ADOFAI starts with angular boundaries 30/45/60 degrees, scales those by
    the tile's ScaleMargin value, and then widens each boundary as necessary to
    satisfy a minimum time radius. The minimum-time part is intentionally *not*
    multiplied by ScaleMargin.

    ``timing_scale`` is the already-normalized ScaleMargin value (1.0 = 100%).
    ``controller_speed`` and ``pitch`` affect the angular velocity. The DLL's
    speed-trial timing minima are additionally divided by ``speed_trial`` before
    the absolute 25 ms floor is applied.
    """

    if bpm <= 0.0:
        raise ValueError("bpm must be positive")
    if timing_scale < 0.0:
        raise ValueError("timing_scale must be non-negative")
    if controller_speed <= 0.0:
        raise ValueError("controller_speed must be positive")
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")

    difficulty = _coerce_difficulty(difficulty)
    angular_speed_deg_s = 3.0 * bpm * controller_speed * pitch

    if mobile:
        counted_min = _speed_adjusted_minimum(MOBILE_COUNTED_MIN_S, speed_trial)
        ep_min = _speed_adjusted_minimum(MOBILE_EARLY_LATE_PERFECT_MIN_S, speed_trial)
        perfect_min = _speed_adjusted_minimum(MOBILE_PERFECT_MIN_S, speed_trial)
    else:
        counted_min = _speed_adjusted_minimum(
            _desktop_counted_min_s(difficulty), speed_trial
        )
        ep_min = _speed_adjusted_minimum(EARLY_LATE_PERFECT_MIN_S, speed_trial)
        perfect_min = _speed_adjusted_minimum(PURE_PERFECT_MIN_S, speed_trial)

    perfect_deg = max(
        PERFECT_BASE_DEG * timing_scale,
        angular_speed_deg_s * perfect_min,
    )
    early_late_perfect_deg = max(
        EARLY_LATE_PERFECT_BASE_DEG * timing_scale,
        angular_speed_deg_s * ep_min,
    )
    pass_deg = max(
        COUNTED_BASE_DEG * timing_scale,
        angular_speed_deg_s * counted_min,
    )

    return TimingWindows(
        perfect_s=perfect_deg / angular_speed_deg_s,
        early_late_perfect_s=early_late_perfect_deg / angular_speed_deg_s,
        pass_s=pass_deg / angular_speed_deg_s,
        perfect_deg=perfect_deg,
        early_late_perfect_deg=early_late_perfect_deg,
        pass_deg=pass_deg,
    )


def normal_timing_windows(bpm: float, *, timing_scale: float = 1.0) -> TimingWindows:
    """Backward-compatible Normal/1x/desktop timing-window helper."""

    return timing_windows(
        bpm,
        difficulty=TimingDifficulty.NORMAL,
        timing_scale=timing_scale,
    )


def classify_timing(
    error_s: float,
    bpm: float,
    *,
    difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
    timing_scale: float = 1.0,
    controller_speed: float = 1.0,
    pitch: float = 1.0,
    speed_trial: float = 1.0,
    mobile: bool = False,
) -> TimingJudgement:
    """Classify a signed timing error with the DLL-style timing windows."""

    windows = timing_windows(
        bpm,
        difficulty=difficulty,
        timing_scale=timing_scale,
        controller_speed=controller_speed,
        pitch=pitch,
        speed_trial=speed_trial,
        mobile=mobile,
    )
    abs_error = abs(error_s)
    if abs_error <= windows.perfect_s:
        return TimingJudgement.PERFECT
    if abs_error <= windows.early_late_perfect_s:
        return (
            TimingJudgement.EARLY_PERFECT
            if error_s < 0.0
            else TimingJudgement.LATE_PERFECT
        )
    if abs_error <= windows.pass_s:
        return (
            TimingJudgement.VERY_EARLY
            if error_s < 0.0
            else TimingJudgement.VERY_LATE
        )
    return TimingJudgement.TOO_EARLY if error_s < 0.0 else TimingJudgement.TOO_LATE


def classify_normal_timing(
    error_s: float,
    bpm: float,
    *,
    timing_scale: float = 1.0,
) -> TimingJudgement:
    """Backward-compatible Normal/1x/desktop judgement helper."""

    return classify_timing(
        error_s,
        bpm,
        difficulty=TimingDifficulty.NORMAL,
        timing_scale=timing_scale,
    )


def x_accuracy_weight(judgement: TimingJudgement) -> float:
    """Per-judgement X-Accuracy weight from scrMistakesManager."""

    if judgement is TimingJudgement.PERFECT:
        return 1.0
    if judgement in {TimingJudgement.EARLY_PERFECT, TimingJudgement.LATE_PERFECT}:
        return 0.75
    if judgement in {TimingJudgement.VERY_EARLY, TimingJudgement.VERY_LATE}:
        return 0.40
    if judgement in {TimingJudgement.TOO_EARLY, TimingJudgement.TOO_LATE}:
        return 0.20
    raise ValueError(f"unsupported timing judgement: {judgement!r}")


def x_accuracy_percent(judgements: Iterable[TimingJudgement]) -> float:
    """Calculate X-Accuracy for a supplied sequence of timing judgements."""

    values = [x_accuracy_weight(judgement) for judgement in judgements]
    if not values:
        return 0.0
    return 100.0 * sum(values) / len(values)


@dataclass
class OverloadCounter:
    """Toy ADOFAI Too Early overload counter retained by the simulator."""

    value: int = 0
    limit: int = OVERLOAD_LIMIT

    def record_too_early(self) -> bool:
        self.value += 2
        return self.value >= self.limit

    def record_valid_hit(self) -> None:
        self.value = max(0, self.value - 1)

    @property
    def overloaded(self) -> bool:
        return self.value >= self.limit
