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

# DLL fail-bar values. Multipress detection itself lives outside the fail bar;
# these constants model the counters once damage is applied.
OVERLOAD_LIMIT = 1.0
OVERLOAD_DAMAGE = 0.5
OVERLOAD_RECOVERY_PER_BEAT = 0.4
MULTIPRESS_DAMAGE = 0.35
MULTIPRESS_RECOVERY_PER_BEAT = 0.2
MULTIPRESS_RESET_AFTER_BEATS = 6.0

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
    """ADOFAI HitMargin values used by the simulator's score history.

    ``classify_timing`` only produces the ordinary timing categories. Fail
    margins are added by the environment when a miss expires or an ordinary
    TooEarly overload crosses the DLL fail-bar threshold.

    Multipress/OverPress are deliberately absent from score history here: in
    the inspected DLL's ordinary path they are display states rather than
    ``marginTracker.AddHit`` entries.
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
    FAIL_MISS = "fail_miss"
    FAIL_OVERLOAD = "fail_overload"
    AUTO = "auto"


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
    """Per-HitMargin X-Accuracy weight from the inspected DLL."""

    if judgement in {TimingJudgement.PERFECT, TimingJudgement.AUTO}:
        return 1.0
    if judgement in {TimingJudgement.EARLY_PERFECT, TimingJudgement.LATE_PERFECT}:
        return 0.75
    if judgement in {TimingJudgement.VERY_EARLY, TimingJudgement.VERY_LATE}:
        return 0.40
    if judgement in {TimingJudgement.TOO_EARLY, TimingJudgement.TOO_LATE}:
        return 0.20
    if judgement in {TimingJudgement.FAIL_MISS, TimingJudgement.FAIL_OVERLOAD}:
        return 0.0
    raise ValueError(f"unsupported timing judgement: {judgement!r}")


def x_accuracy_components(
    judgements: Iterable[TimingJudgement],
    *,
    dead_tiles: int = 0,
) -> tuple[float, int]:
    """Return the DLL X-Accuracy weighted sum and denominator before checkpoints.

    Dead tiles contribute 0.20 to the numerator and one denominator entry each.
    FailMiss/FailOverload remain ordinary denominator entries with zero weight.
    """

    if dead_tiles < 0:
        raise ValueError("dead_tiles must be non-negative")
    values = [x_accuracy_weight(judgement) for judgement in judgements]
    return sum(values) + 0.20 * dead_tiles, len(values) + dead_tiles


def x_accuracy_percent(
    judgements: Iterable[TimingJudgement],
    *,
    dead_tiles: int = 0,
    checkpoints_used: int = 0,
) -> float:
    """Calculate DLL-style X-Accuracy for a supplied HitMargin history."""

    if checkpoints_used < 0:
        raise ValueError("checkpoints_used must be non-negative")
    weighted_sum, denominator = x_accuracy_components(judgements, dead_tiles=dead_tiles)
    if denominator == 0:
        return 0.0
    return 100.0 * weighted_sum / denominator * (0.9875**checkpoints_used)


def normal_accuracy_percent(
    judgements: Iterable[TimingJudgement],
    *,
    dead_tiles: int = 0,
) -> float:
    """Calculate the inspected DLL's ordinary Accuracy percentage.

    FailMiss and FailOverload are already present in ``hitMargins.Count`` and
    are then added once more to the denominator through the fail count. This
    intentionally reproduces that double denominator effect.
    """

    if dead_tiles < 0:
        raise ValueError("dead_tiles must be non-negative")
    margins = list(judgements)
    fail_count = sum(
        judgement in {TimingJudgement.FAIL_MISS, TimingJudgement.FAIL_OVERLOAD}
        for judgement in margins
    )
    numerator = sum(
        judgement
        in {
            TimingJudgement.PERFECT,
            TimingJudgement.EARLY_PERFECT,
            TimingJudgement.LATE_PERFECT,
            TimingJudgement.AUTO,
        }
        for judgement in margins
    )
    denominator = len(margins) + fail_count
    base = numerator / denominator if denominator else 0.0
    pure_bonus_count = sum(
        judgement in {TimingJudgement.PERFECT, TimingJudgement.AUTO}
        for judgement in margins
    )
    return 100.0 * (
        base + 0.0001 * pure_bonus_count - 0.0001 * dead_tiles
    )


@dataclass
class OverloadCounter:
    """DLL fail-bar counters used by ordinary overload and future Multipress logic.

    The environment currently applies ordinary TooEarly damage. Multipress
    detection/queue semantics are intentionally left for the separate input
    state-machine work, but its fail-bar counter, decay, and six-beat reset are
    already represented here so the fail-bar model matches the DLL.
    """

    value: float = 0.0
    multipress_value: float = 0.0
    multipress_reset_beats: float = 0.0
    limit: float = OVERLOAD_LIMIT
    damage: float = OVERLOAD_DAMAGE
    recovery_per_beat: float = OVERLOAD_RECOVERY_PER_BEAT
    multipress_damage: float = MULTIPRESS_DAMAGE
    multipress_recovery_per_beat: float = MULTIPRESS_RECOVERY_PER_BEAT
    multipress_reset_after_beats: float = MULTIPRESS_RESET_AFTER_BEATS

    def record_too_early(self) -> bool:
        self.value += self.damage
        return self.overloaded

    def record_multipress(self) -> bool:
        self.multipress_value += self.multipress_damage
        self.multipress_reset_beats = 0.0
        return self.overloaded

    def record_valid_hit(self) -> None:
        """Valid hits do not directly heal either DLL fail-bar counter."""

    def advance_beats(self, beat_delta: float) -> None:
        if beat_delta < 0.0:
            raise ValueError("beat_delta must be non-negative")
        self.value = max(0.0, self.value - self.recovery_per_beat * beat_delta)
        self.multipress_value = max(
            0.0,
            self.multipress_value - self.multipress_recovery_per_beat * beat_delta,
        )
        self.multipress_reset_beats += beat_delta
        if self.multipress_reset_beats > self.multipress_reset_after_beats:
            self.multipress_value = 0.0
            self.multipress_reset_beats = 0.0

    def rewind(self) -> None:
        self.value = 0.0
        self.multipress_value = 0.0
        self.multipress_reset_beats = 0.0

    @property
    def overloaded(self) -> bool:
        # The DLL uses strict > 1.0 comparisons, not >=.
        return self.value > self.limit or self.multipress_value > self.limit
