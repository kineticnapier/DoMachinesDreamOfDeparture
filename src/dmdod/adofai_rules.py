from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


NORMAL_THRESHOLD_BPM = 310.0
OVERLOAD_LIMIT = 6


class TimingJudgement(str, Enum):
    """Timing categories used by the toy ADOFAI rules layer."""

    PERFECT = "perfect"
    EARLY_PERFECT = "early_perfect"
    LATE_PERFECT = "late_perfect"
    EARLY = "early"
    LATE = "late"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class TimingWindows:
    """One-sided timing-window radii in seconds.

    For Normal timing, the ADOFAI wiki gives Pass = 20000/BPM ms and
    Perfect = 10000/BPM ms until 310 BPM.  Above that threshold the strictest
    Normal windows stay fixed.  The 45 degree E/L Perfect boundary is halfway
    between the 30 degree Perfect and 60 degree Pass boundaries.
    """

    perfect_s: float
    early_late_perfect_s: float
    pass_s: float


def normal_timing_windows(bpm: float) -> TimingWindows:
    if bpm <= 0.0:
        raise ValueError("bpm must be positive")

    effective_bpm = min(bpm, NORMAL_THRESHOLD_BPM)
    perfect_s = 10.0 / effective_bpm
    pass_s = 20.0 / effective_bpm
    early_late_perfect_s = 15.0 / effective_bpm
    return TimingWindows(perfect_s, early_late_perfect_s, pass_s)


def classify_normal_timing(error_s: float, bpm: float) -> TimingJudgement:
    """Classify signed timing error using the Normal timing option.

    Negative error is early and positive error is late.
    """

    windows = normal_timing_windows(bpm)
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
        return TimingJudgement.EARLY if error_s < 0.0 else TimingJudgement.LATE
    return TimingJudgement.TOO_EARLY if error_s < 0.0 else TimingJudgement.TOO_LATE


@dataclass
class OverloadCounter:
    """ADOFAI Too Early overload counter.

    The referenced game-mechanics description states that Too Early adds 2,
    each valid tile hit removes 1 without going below zero, and OVERLOAD occurs
    when the counter reaches 6.
    """

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
