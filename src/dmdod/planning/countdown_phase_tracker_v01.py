from __future__ import annotations

"""Causal countdown phase tracking from observed stock HUD text only."""

from dataclasses import dataclass
from math import isfinite

from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind,
    CountdownStudentObservation,
)


PHASE_TRACKER_VERSION = "countdown-phase-tracker-v0.1"
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class TimeInterval:
    lower_s: float
    upper_s: float

    def __post_init__(self) -> None:
        if not (isfinite(self.lower_s) and isfinite(self.upper_s)):
            raise ValueError("interval bounds must be finite")
        if self.lower_s > self.upper_s + EPS:
            raise ValueError("interval lower bound exceeds upper bound")

    @property
    def width_s(self) -> float:
        return self.upper_s - self.lower_s


@dataclass(frozen=True, slots=True)
class CountdownPhaseUpdate:
    step: int
    observed_at_s: float
    status: str
    displayed_digit: int | None
    transition_interval_s: TimeInterval | None
    measured_tick_period_s: TimeInterval | None
    remaining_transitions_to_go: int | None
    transition_confirmations: int
    evidence_age_s: float
    valid: bool
    version: str = PHASE_TRACKER_VERSION

    def __post_init__(self) -> None:
        if self.version != PHASE_TRACKER_VERSION:
            raise ValueError("unknown phase tracker version")
        if self.step < 0 or not isfinite(self.observed_at_s):
            raise ValueError("invalid phase update clock")
        if self.displayed_digit is not None and not 0 <= self.displayed_digit <= 9:
            raise ValueError("displayed digit must be 0..9")
        if self.remaining_transitions_to_go is not None and (
            self.remaining_transitions_to_go < 0
        ):
            raise ValueError("remaining transition count cannot be negative")
        if self.transition_confirmations < 0 or self.evidence_age_s < -EPS:
            raise ValueError("invalid evidence metadata")


class CountdownPhaseTracker:
    """Track observed cue transitions without a countdown setting or schedule.

    Each visible transition is bounded by the previous and current rendered
    frame timestamps. A tick-period interval becomes available only after two
    numeric onsets have been observed. Any missing/stale/contradictory frame
    makes the current countdown sequence unusable until reset.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._last: CountdownStudentObservation | None = None
        self._last_evidence_at_s: float | None = None
        self._last_kind: CountdownCueKind | None = None
        self._last_digit: int | None = None
        self._numeric_onsets: list[TimeInterval] = []
        self._periods: list[TimeInterval] = []
        self._failed_reason: str | None = None

    def _fail(self, reason: str) -> None:
        if self._failed_reason is None:
            self._failed_reason = reason

    def _transition_interval(
        self, observation: CountdownStudentObservation
    ) -> TimeInterval | None:
        if self._last is None:
            return None
        return TimeInterval(self._last.rendered_at_s, observation.rendered_at_s)

    def _add_numeric_onset(self, interval: TimeInterval) -> None:
        if self._numeric_onsets:
            previous = self._numeric_onsets[-1]
            period = TimeInterval(
                max(0.0, interval.lower_s - previous.upper_s),
                interval.upper_s - previous.lower_s,
            )
            if period.upper_s <= EPS:
                self._fail("inconsistent_transition_timing")
                return
            self._periods.append(period)
            lower = max(item.lower_s for item in self._periods)
            upper = min(item.upper_s for item in self._periods)
            if lower > upper + EPS:
                self._fail("inconsistent_tick_period")
        self._numeric_onsets.append(interval)

    def _period_intersection(self) -> TimeInterval | None:
        if len(self._numeric_onsets) < 2 or not self._periods:
            return None
        lower = max(item.lower_s for item in self._periods)
        upper = min(item.upper_s for item in self._periods)
        if lower > upper + EPS:
            return None
        return TimeInterval(lower, upper)

    def observe(
        self, observation: CountdownStudentObservation
    ) -> CountdownPhaseUpdate:
        if not isinstance(observation, CountdownStudentObservation):
            raise TypeError("expected CountdownStudentObservation")
        if observation.step != self._next_step:
            raise ValueError("phase observations must be consecutive")
        if self._last is not None and (
            observation.observed_at_s < self._last.observed_at_s - EPS
            or observation.rendered_at_s < self._last.rendered_at_s - EPS
        ):
            raise ValueError("phase observation clock cannot move backward")

        transition = None
        if not observation.valid:
            self._fail("invalid_or_stale_frame")
        elif self._failed_reason is None:
            changed = (
                observation.kind != self._last_kind
                or observation.digit != self._last_digit
            )
            if changed:
                transition = self._transition_interval(observation)
                if observation.kind is CountdownCueKind.NUMBER:
                    if self._last_kind not in (
                        CountdownCueKind.READY, CountdownCueKind.NUMBER
                    ):
                        self._fail("unanchored_numeric_cue")
                    elif (
                        self._last_kind is CountdownCueKind.NUMBER
                        and observation.digit != self._last_digit - 1
                    ):
                        self._fail("contradictory_digit_sequence")
                    elif transition is None:
                        self._fail("unbounded_numeric_transition")
                    else:
                        self._add_numeric_onset(transition)
                elif observation.kind is CountdownCueKind.GO:
                    if self._last_kind is not CountdownCueKind.NUMBER:
                        self._fail("go_without_numeric_predecessor")
                    elif self._last_digit != 1:
                        self._fail("go_before_digit_one")
                elif observation.kind is CountdownCueKind.READY:
                    if self._last_kind is not None:
                        self._fail("countdown_restarted")
                elif observation.kind is CountdownCueKind.HIDDEN:
                    if self._last_kind is not CountdownCueKind.GO:
                        self._fail("countdown_hidden_before_go")
                else:
                    self._fail("unknown_countdown_cue")

        if observation.valid and observation.kind in {
            CountdownCueKind.READY,
            CountdownCueKind.NUMBER,
            CountdownCueKind.GO,
        }:
            self._last_evidence_at_s = observation.observed_at_s

        if self._failed_reason is not None:
            status = self._failed_reason
            valid = False
        else:
            status = observation.kind.value
            valid = observation.valid
        digit = observation.digit if valid else None
        remaining = digit if valid and digit is not None else (
            0 if valid and observation.kind is CountdownCueKind.GO else None
        )
        age = (
            0.0 if self._last_evidence_at_s is None
            else max(0.0, observation.observed_at_s - self._last_evidence_at_s)
        )
        update = CountdownPhaseUpdate(
            step=observation.step,
            observed_at_s=observation.observed_at_s,
            status=status,
            displayed_digit=digit,
            transition_interval_s=transition if valid else None,
            measured_tick_period_s=self._period_intersection() if valid else None,
            remaining_transitions_to_go=remaining,
            transition_confirmations=len(self._numeric_onsets),
            evidence_age_s=age,
            valid=valid,
        )
        self._last = observation
        if observation.valid:
            self._last_kind = observation.kind
            self._last_digit = observation.digit
        self._next_step += 1
        return update
