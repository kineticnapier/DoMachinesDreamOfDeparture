from __future__ import annotations

"""HUD-only interval forecast for the first event after stock countdown."""

from dataclasses import dataclass
from math import atan2, isfinite, log2, pi, remainder

from dmdod.features.hud_features import HUD_BPM_REFERENCE
from dmdod.planning.countdown_phase_tracker_v01 import (
    CountdownPhaseUpdate,
    TimeInterval,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    FLOOR_DIM,
    HUD_DIM,
    HUD_OFFSET,
    MOTOR_FEATURES,
    SLOT_START,
    visible_slot,
)


FIRST_EVENT_ESTIMATOR_VERSION = "countdown-first-event-v0.1"
MOTOR_LEAD_S = .044
MAX_DUE_UNCERTAINTY_S = .010
EPS = 1e-9


@dataclass(frozen=True, slots=True)
class CountdownFirstEventEstimate:
    step: int
    observed_at_s: float
    status: str
    due_interval_s: TimeInterval | None
    lead_interval_s: TimeInterval | None
    lead_guaranteed: bool
    source: str = FIRST_EVENT_ESTIMATOR_VERSION

    @property
    def eligible(self) -> bool:
        return self.due_interval_s is not None


def encode_visible_bpm(bpm: float) -> float:
    """Test/helper encoding for the already-visible v2 BPM field."""
    if not isfinite(bpm) or bpm <= 0.0:
        raise ValueError("visible BPM must be positive and finite")
    return log2(bpm / HUD_BPM_REFERENCE)


class CountdownFirstEventEstimator:
    """Combine observed countdown phase with visible current-floor geometry."""

    def __init__(self, *, required_lead_s: float = MOTOR_LEAD_S,
                 max_due_uncertainty_s: float = MAX_DUE_UNCERTAINTY_S) -> None:
        if not isfinite(required_lead_s) or required_lead_s < 0.0:
            raise ValueError("required lead must be nonnegative and finite")
        if not isfinite(max_due_uncertainty_s) or max_due_uncertainty_s < 0.0:
            raise ValueError("maximum due uncertainty must be nonnegative and finite")
        self.required_lead_s = float(required_lead_s)
        self.max_due_uncertainty_s = float(max_due_uncertainty_s)
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._previous_hud: tuple[float, ...] | None = None
        self._previous_time_s: float | None = None
        self._numeric_anchor: TimeInterval | None = None
        self._numeric_digit: int | None = None
        self._go_anchor: TimeInterval | None = None

    def _abstain(self, phase: CountdownPhaseUpdate, status: str):
        return CountdownFirstEventEstimate(
            step=phase.step, observed_at_s=phase.observed_at_s,
            status=status, due_interval_s=None, lead_interval_s=None,
            lead_guaranteed=False,
        )

    def _visible_motion(self, phase: CountdownPhaseUpdate,
                        hud: tuple[float, ...]):
        try:
            bpm = HUD_BPM_REFERENCE * (2.0 ** hud[HUD_OFFSET])
        except OverflowError:
            return None, None, "invalid_visible_bpm"
        if not isfinite(bpm) or bpm <= 0.0:
            return None, None, "invalid_visible_bpm"
        if self._previous_hud is None or self._previous_time_s is None:
            return None, None, "direction_history_required"
        dt = phase.observed_at_s - self._previous_time_s
        if dt <= EPS:
            return None, None, "motion_interval_required"
        expected = pi * bpm / 60.0 * dt
        if expected >= pi / 2.0 - .05:
            return None, None, "direction_alias"
        old = atan2(
            self._previous_hud[MOTOR_FEATURES],
            self._previous_hud[MOTOR_FEATURES + 1],
        )
        current = atan2(hud[MOTOR_FEATURES], hud[MOTOR_FEATURES + 1])
        delta = remainder(current - old, 2.0 * pi)
        if abs(delta) < max(.02, expected * .025):
            return None, None, "direction_unobservable"
        if abs(abs(delta) - expected) > max(.02, expected * .025):
            return None, None, "motion_bpm_mismatch"
        return (1 if delta > 0 else -1), bpm, None

    @staticmethod
    def _arc_duration(hud: tuple[float, ...], sign: int, bpm: float):
        floor = visible_slot(hud, CURRENT_SLOT)
        if len(floor) != FLOOR_DIM or floor[-1] < .5:
            return None, "current_floor_not_visible"
        if any(floor[index] > .5 for index in range(6, 14)):
            return None, "unsupported_countdown_floor"
        entry = atan2(floor[2], floor[3])
        exit_angle = atan2(floor[4], floor[5])
        arc = (sign * (exit_angle - entry)) % (2.0 * pi)
        if arc < 1e-5 or 2.0 * pi - arc < 1e-5:
            arc = 2.0 * pi
        return arc / (pi * bpm / 60.0), None

    def observe(self, phase: CountdownPhaseUpdate,
                legacy_hud263: tuple[float, ...]) -> CountdownFirstEventEstimate:
        if not isinstance(phase, CountdownPhaseUpdate):
            raise TypeError("expected CountdownPhaseUpdate")
        if phase.step != self._next_step:
            raise ValueError("first-event observations must be consecutive")
        if len(legacy_hud263) != HUD_DIM or not all(
            isfinite(value) for value in legacy_hud263
        ):
            result = self._abstain(phase, "invalid_hud263")
        elif not phase.valid:
            result = self._abstain(phase, f"phase_{phase.status}")
        else:
            if phase.displayed_digit is not None and (
                phase.transition_interval_s is not None
            ):
                self._numeric_anchor = phase.transition_interval_s
                self._numeric_digit = phase.displayed_digit
            if phase.status == "go" and phase.transition_interval_s is not None:
                self._go_anchor = phase.transition_interval_s

            sign, bpm, motion_error = self._visible_motion(
                phase, legacy_hud263
            )
            if motion_error is not None:
                result = self._abstain(phase, motion_error)
            else:
                arc_s, arc_error = self._arc_duration(
                    legacy_hud263, sign, bpm
                )
                if arc_error is not None:
                    result = self._abstain(phase, arc_error)
                else:
                    period = phase.measured_tick_period_s
                    if self._go_anchor is not None:
                        go = self._go_anchor
                    elif (
                        self._numeric_anchor is not None
                        and self._numeric_digit is not None
                        and period is not None
                    ):
                        go = TimeInterval(
                            self._numeric_anchor.lower_s
                            + self._numeric_digit * period.lower_s,
                            self._numeric_anchor.upper_s
                            + self._numeric_digit * period.upper_s,
                        )
                    else:
                        go = None
                    if go is None:
                        result = self._abstain(phase, "tick_period_unconfirmed")
                    else:
                        due = TimeInterval(
                            go.lower_s + arc_s, go.upper_s + arc_s
                        )
                        if due.width_s * .5 > self.max_due_uncertainty_s + EPS:
                            result = self._abstain(
                                phase, "due_uncertainty_exceeds_10ms"
                            )
                        else:
                            lead = TimeInterval(
                                due.lower_s - phase.observed_at_s,
                                due.upper_s - phase.observed_at_s,
                            )
                            guaranteed = lead.lower_s + EPS >= self.required_lead_s
                            result = CountdownFirstEventEstimate(
                                step=phase.step,
                                observed_at_s=phase.observed_at_s,
                                status=(
                                    "eligible_lead_guaranteed" if guaranteed
                                    else "eligible_insufficient_lead"
                                ),
                                due_interval_s=due,
                                lead_interval_s=lead,
                                lead_guaranteed=guaranteed,
                            )

        self._previous_hud = tuple(legacy_hud263) if len(legacy_hud263) == HUD_DIM else None
        self._previous_time_s = phase.observed_at_s
        self._next_step += 1
        return result
