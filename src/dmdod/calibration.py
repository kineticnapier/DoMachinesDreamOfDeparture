from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateTarget:
    """Observed human tapping-rate target used for model calibration.

    Values are provisional observations from the initial calibration session.
    They are targets for the simulator, not universal human limits.
    """

    name: str
    fingers: tuple[str, ...]
    duration_s: float
    rate_hz: float
    notes: str = ""


# Initial measurements. Keep these separate from BodyConfig so that changing the
# physical model never silently changes the empirical targets it is fitted to.
INITIAL_RATE_TARGETS: tuple[RateTarget, ...] = (
    RateTarget(
        name="RI single, short",
        fingers=("RI",),
        duration_s=6.0,
        rate_hz=8.9,
        notes="Short sustained single-finger rate, approximately 9 KPS.",
    ),
    RateTarget(
        name="RI single, sustained",
        fingers=("RI",),
        duration_s=20.0,
        rate_hz=8.2,
        notes="20 s mean; rate fell from roughly 9.1 KPS toward 7.8 KPS.",
    ),
    RateTarget(
        name="RI/RM same-hand alternation",
        fingers=("RI", "RM"),
        duration_s=10.0,
        rate_hz=12.5,
        notes="Strong short/long interval asymmetry appeared during alternation.",
    ),
    RateTarget(
        name="RI/LI bilateral alternation",
        fingers=("RI", "LI"),
        duration_s=12.0,
        rate_hz=16.0,
        notes="Approximate practical ceiling in the initial session.",
    ),
)

# The multi-finger recording suggests that adding fingers must not be modeled as
# independent rate addition.  This is intentionally a calibration constraint,
# not a hard-coded input limiter.
INITIAL_GLOBAL_RATE_CEILING_HZ = 16.0
