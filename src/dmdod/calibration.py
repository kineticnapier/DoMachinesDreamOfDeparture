from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateTarget:
    """Observed tapping-rate target used for model calibration.

    Values are provisional observations from the initial personal blue-switch
    calibration session.  They are simulator targets, not universal human
    limits and not hard-coded rate ceilings.
    """

    name: str
    fingers: tuple[str, ...]
    duration_s: float
    rate_hz: float
    notes: str = ""


# Keep empirical targets separate from BodyConfig so later changes to the model
# cannot silently rewrite the observations it was fitted against.
INITIAL_RATE_TARGETS: tuple[RateTarget, ...] = (
    RateTarget(
        name="RI single, short",
        fingers=("RI",),
        duration_s=6.0,
        rate_hz=8.9,
        notes="Personal short single-finger observation on the initial blue-switch setup.",
    ),
    RateTarget(
        name="RI single, sustained",
        fingers=("RI",),
        duration_s=20.0,
        rate_hz=8.2,
        notes="20 s personal mean; observed rate fell from roughly 9.1 KPS toward 7.8 KPS.",
    ),
    RateTarget(
        name="RI/RM same-hand alternation",
        fingers=("RI", "RM"),
        duration_s=10.0,
        rate_hz=12.5,
        notes="Personal observation; strong short/long interval asymmetry appeared during alternation.",
    ),
    RateTarget(
        name="RI/LI bilateral alternation",
        fingers=("RI", "LI"),
        duration_s=12.0,
        rate_hz=16.0,
        notes="Personal practical observation from the initial session; not a global ceiling.",
    ),
)
