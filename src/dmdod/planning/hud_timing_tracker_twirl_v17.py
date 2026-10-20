from __future__ import annotations

"""Opt-in causal recovery across one visible stock Twirl marker.

Twirl is rendered in the HUD before arrival and reverses rotation on entry to
its floor.  The frozen v1.4 estimator assumes one direction across its visible
window.  Once current direction is established from two HUD frames, swapping
the visible entry/exit angles from the Twirl floor onward is geometrically
equivalent to evaluating those arcs in the reversed direction.  This module
performs only that representation change; it does not read chart actions,
target times, floor ordinals, or evaluator state.

The rule fails closed for aliased/missing motion history, a mixed-action Twirl
floor, Midspin/MultiPlanet on the Twirl floor, or missing lookahead. A later
Twirl remains a marker gate and bounds the normalized region. Other future
markers remain untouched, so the frozen
SetSpeed/Pause/Checkpoint gates still apply.  The default v1.4/v1.5 paths are
unchanged unless this predictor is explicitly selected.
"""

from dataclasses import dataclass
from math import atan2, isfinite, pi, remainder

from dmdod.features.hud_features import HUD_BPM_REFERENCE
from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)
from dmdod.planning.hud_timing_tracker_midspin_v16 import (
    normalize_visible_midspin_transition,
)
from dmdod.planning.hud_timing_tracker_checkpoint_v18 import (
    normalize_visible_checkpoint_transition,
    stop_after_checkpoint,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    FLOOR_DIM,
    FLOORS,
    HUD_DIM,
    HUD_OFFSET,
    MOTOR_FEATURES,
    SKIP_MARKER_FROM,
    SKIP_MARKER_TO,
    SLOT_START,
    WindowForecast,
    forecast_visible_window,
    visible_slot,
)


@dataclass(frozen=True, slots=True)
class TwirlNormalization:
    """Result of a truth-free visible-HUD normalization."""

    hud: tuple[float, ...]
    recovered_floor_slots: tuple[int, ...]
    observed_direction: int | None
    reason: str


def _markers(floor: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(
        index for index in range(SKIP_MARKER_FROM, SKIP_MARKER_TO)
        if floor[index] > .5
    )


def _observed_direction(
    hud: tuple[float, ...], previous_hud: tuple[float, ...] | None,
) -> tuple[int | None, str]:
    if previous_hud is None or len(previous_hud) != HUD_DIM:
        return None, "twirl_no_motion_history"
    current = visible_slot(hud, CURRENT_SLOT)
    previous = visible_slot(previous_hud, CURRENT_SLOT)
    if max(abs(a - b) for a, b in zip(current, previous)) > 1e-7:
        return None, "twirl_changed_current_floor"
    if abs(hud[HUD_OFFSET] - previous_hud[HUD_OFFSET]) > 1e-9:
        return None, "twirl_changed_visible_bpm"
    try:
        bpm = HUD_BPM_REFERENCE * (2.0 ** hud[HUD_OFFSET])
    except OverflowError:
        return None, "twirl_invalid_bpm"
    if not isfinite(bpm) or bpm <= 0.0:
        return None, "twirl_invalid_bpm"
    expected = pi * bpm / 60.0 * .010
    if expected >= pi / 2.0 - .05:
        return None, "twirl_motion_alias"
    phase = atan2(hud[MOTOR_FEATURES], hud[MOTOR_FEATURES + 1])
    old_phase = atan2(
        previous_hud[MOTOR_FEATURES], previous_hud[MOTOR_FEATURES + 1]
    )
    step = remainder(phase - old_phase, 2.0 * pi)
    tolerance = max(.02, expected * .025)
    candidates = tuple(
        sign for sign in (1, -1)
        if abs(step - sign * expected) <= tolerance
    )
    if len(candidates) != 1:
        return None, "twirl_direction_unobservable"
    return candidates[0], "eligible"


def normalize_visible_twirl_transition(
    hud: tuple[float, ...],
    previous_hud: tuple[float, ...] | None,
) -> TwirlNormalization:
    """Normalize exactly one future visible, otherwise ordinary Twirl floor."""

    if len(hud) != HUD_DIM:
        return TwirlNormalization(hud, (), None, "invalid_hud_dimension")
    future_twirl_slots = tuple(
        slot for slot in range(
            CURRENT_SLOT + 1, CURRENT_SLOT + FLOORS.ahead_floors + 1
        )
        if visible_slot(hud, slot)[-1] > .5
        and visible_slot(hud, slot)[8] > .5
    )
    if not future_twirl_slots:
        return TwirlNormalization(hud, (), None, "no_future_twirl")
    slot = future_twirl_slots[0]
    floor = visible_slot(hud, slot)
    if floor[6] > .5 or floor[7] > .5 or _markers(floor) != (8,):
        return TwirlNormalization(hud, (), None, "unsafe_twirl_floor")
    if slot >= CURRENT_SLOT + FLOORS.ahead_floors:
        return TwirlNormalization(hud, (), None, "twirl_missing_lookahead")
    if visible_slot(hud, slot + 1)[-1] < .5:
        return TwirlNormalization(hud, (), None, "twirl_missing_lookahead")
    direction, reason = _observed_direction(hud, previous_hud)
    if direction is None:
        return TwirlNormalization(hud, (), None, reason)

    normalized = list(hud)
    normalized[SLOT_START + slot * FLOOR_DIM + 8] = 0.0
    # Under a fixed sign, swapping endpoints gives exactly the arc that the
    # opposite sign assigns to the original endpoints. Apply it to every
    # visible floor whose rotation occurs after the Twirl takes effect.
    stop_slot = (
        future_twirl_slots[1] if len(future_twirl_slots) > 1
        else CURRENT_SLOT + FLOORS.ahead_floors + 1
    )
    for changed_slot in range(slot, stop_slot):
        item = visible_slot(hud, changed_slot)
        if item[-1] < .5:
            break
        start = SLOT_START + changed_slot * FLOOR_DIM
        normalized[start + 2:start + 4] = item[4:6]
        normalized[start + 4:start + 6] = item[2:4]
    return TwirlNormalization(tuple(normalized), (slot,), direction, "eligible")


class CausalTwirlForecastPredictor:
    """Stateful v1.4 front-end using current/past HUD263 only."""

    def __init__(self, *, horizon_s: float = .080,
                 midspin_recovery: bool = False,
                 enable_twirl_recovery: bool = True,
                 checkpoint_recovery: bool = False) -> None:
        self.horizon_s = horizon_s
        self.midspin_recovery = midspin_recovery
        self.enable_twirl_recovery = enable_twirl_recovery
        self.checkpoint_recovery = checkpoint_recovery
        self.reset()

    def reset(self) -> None:
        self._previous_raw_hud: tuple[float, ...] | None = None
        self._previous_forecast_hud: tuple[float, ...] | None = None
        self.last_twirl_normalization = TwirlNormalization(
            (), (), None, "not_observed"
        )
        self.last_recovered_midspin_floor_slots: tuple[int, ...] = ()
        self.last_recovered_checkpoint_floor_slots: tuple[int, ...] = ()

    def __call__(self, hud: tuple[float, ...]) -> WindowForecast:
        twirl = (
            normalize_visible_twirl_transition(hud, self._previous_raw_hud)
            if self.enable_twirl_recovery
            else TwirlNormalization(hud, (), None, "twirl_disabled")
        )
        normalized_hud = twirl.hud
        midspin_slots: tuple[int, ...] = ()
        if self.midspin_recovery:
            midspin = normalize_visible_midspin_transition(normalized_hud)
            normalized_hud = midspin.hud
            midspin_slots = midspin.recovered_floor_slots
        checkpoint_slots: tuple[int, ...] = ()
        if self.checkpoint_recovery:
            checkpoint = normalize_visible_checkpoint_transition(normalized_hud)
            normalized_hud = checkpoint.hud
            checkpoint_slots = checkpoint.recovered_floor_slots
        result = forecast_visible_window(
            normalized_hud,
            horizon_s=self.horizon_s,
            previous_hud=self._previous_forecast_hud,
        )
        if self.checkpoint_recovery:
            result = stop_after_checkpoint(result, checkpoint_slots)
        # Raw history establishes observed motion direction. Forecast history
        # must use the same causal representation as the current frame or the
        # frozen full-turn check would compare transformed angles with raw
        # angles and report a false floor change.
        self._previous_raw_hud = hud
        self._previous_forecast_hud = normalized_hud
        self.last_twirl_normalization = twirl
        self.last_recovered_midspin_floor_slots = (
            midspin_slots if result.eligible else ()
        )
        self.last_recovered_checkpoint_floor_slots = (
            checkpoint_slots if result.eligible else ()
        )
        return result


class TwirlRecoveryLifecycleCausalTimingTracker(
    QuantizedLifecycleCausalTimingTracker
):
    """Lifecycle v1.5 with conservative Twirl recovery explicitly enabled."""

    def __init__(self, *, horizon_s: float = .080,
                 control_dt_s: float = .010,
                 match_tolerance_s: float = .00075,
                 midspin_recovery: bool = False,
                 enable_twirl_recovery: bool = True,
                 checkpoint_recovery: bool = False) -> None:
        self.twirl_predictor = CausalTwirlForecastPredictor(
            horizon_s=horizon_s,
            midspin_recovery=midspin_recovery,
            enable_twirl_recovery=enable_twirl_recovery,
            checkpoint_recovery=checkpoint_recovery,
        )
        super().__init__(
            horizon_s=horizon_s,
            control_dt_s=control_dt_s,
            match_tolerance_s=match_tolerance_s,
            predictor=self.twirl_predictor,
        )

    def reset(self) -> None:
        if hasattr(self, "twirl_predictor"):
            self.twirl_predictor.reset()
        super().reset()
