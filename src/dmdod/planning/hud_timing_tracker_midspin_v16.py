from __future__ import annotations

"""Opt-in causal recovery for one visible stock Midspin transition.

The stock Midspin floor is a visible, zero-duration, non-input geometry
transition.  If the HUD also shows the immediately following ordinary floor,
arrival at the Midspin boundary is therefore the input time of that following
floor.  This module exposes that rule to the frozen v1.4 geometry estimator by
normalizing only the visible Midspin flag for the calculation.  It never reads
chart indices, target times, evaluator state, or future actions.

The rule intentionally fails closed for missing lookahead, consecutive
Midspins, MultiPlanet, and every visible action marker.  The original v1.4
path remains the default.
"""

from dataclasses import dataclass
from math import atan2, pi, remainder

from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    ANG_EPS,
    CURRENT_SLOT,
    FLOOR_DIM,
    FLOORS,
    HUD_DIM,
    SKIP_MARKER_FROM,
    SKIP_MARKER_TO,
    SLOT_START,
    WindowForecast,
    forecast_visible_window,
    visible_slot,
)


@dataclass(frozen=True, slots=True)
class MidspinNormalization:
    """Result of a truth-free visible-HUD normalization."""

    hud: tuple[float, ...]
    recovered_floor_slots: tuple[int, ...]


def _same_visible_angle(floor: tuple[float, ...]) -> bool:
    entry = atan2(floor[2], floor[3])
    exit_angle = atan2(floor[4], floor[5])
    return abs(remainder(exit_angle - entry, 2 * pi)) <= ANG_EPS


def _has_marker(floor: tuple[float, ...]) -> bool:
    return any(floor[index] > .5 for index in range(
        SKIP_MARKER_FROM, SKIP_MARKER_TO
    ))


def normalize_visible_midspin_transition(
    hud: tuple[float, ...],
) -> MidspinNormalization:
    """License at most one nearest visible Midspin-to-ordinary transition.

    Clearing the bit does not claim that the Midspin itself is playable.  It
    lets v1.4 emit the incoming boundary once; the following zero arc then
    stops its forecast as ``eligible_partial``.  The emitted event represents
    the separately visible ordinary floor immediately after the Midspin.
    """

    if len(hud) != HUD_DIM:
        return MidspinNormalization(hud, ())
    for slot in range(
        CURRENT_SLOT + 1,
        CURRENT_SLOT + FLOORS.ahead_floors,
    ):
        midspin = visible_slot(hud, slot)
        following = visible_slot(hud, slot + 1)
        if not (
            midspin[-1] > .5
            and midspin[6] > .5
            and midspin[7] < .5
            and _same_visible_angle(midspin)
            and not _has_marker(midspin)
            and following[-1] > .5
            and following[6] < .5
            and following[7] < .5
            and not _has_marker(following)
        ):
            continue
        normalized = list(hud)
        normalized[SLOT_START + slot * FLOOR_DIM + 6] = 0.0
        return MidspinNormalization(tuple(normalized), (slot,))
    return MidspinNormalization(hud, ())


class CausalMidspinForecastPredictor:
    """Stateful v1.4 predictor front-end using only current/past HUD263."""

    def __init__(self, *, horizon_s: float = .080) -> None:
        self.horizon_s = horizon_s
        self.reset()

    def reset(self) -> None:
        self._previous_hud: tuple[float, ...] | None = None
        self.last_recovered_floor_slots: tuple[int, ...] = ()

    def __call__(self, hud: tuple[float, ...]) -> WindowForecast:
        normalized = normalize_visible_midspin_transition(hud)
        result = forecast_visible_window(
            normalized.hud,
            horizon_s=self.horizon_s,
            previous_hud=self._previous_hud,
        )
        self._previous_hud = hud
        self.last_recovered_floor_slots = (
            normalized.recovered_floor_slots if result.eligible else ()
        )
        return result


class MidspinRecoveryLifecycleCausalTimingTracker(
    QuantizedLifecycleCausalTimingTracker
):
    """Lifecycle v1.5 with the conservative Midspin rule explicitly enabled."""

    def __init__(self, *, horizon_s: float = .080,
                 control_dt_s: float = .010,
                 match_tolerance_s: float = .00075) -> None:
        self.midspin_predictor = CausalMidspinForecastPredictor(
            horizon_s=horizon_s
        )
        super().__init__(
            horizon_s=horizon_s,
            control_dt_s=control_dt_s,
            match_tolerance_s=match_tolerance_s,
            predictor=self.midspin_predictor,
        )

    def reset(self) -> None:
        if hasattr(self, "midspin_predictor"):
            self.midspin_predictor.reset()
        super().reset()
