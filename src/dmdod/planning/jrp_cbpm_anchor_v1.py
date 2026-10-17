from __future__ import annotations

"""Conservative JRP-text Timing continuation, separate from frozen Stage T v1.4.

Inputs to `observe` are ONLY a current 263D HUD, the three actual rendered
JRP strings (possibly stale), and the previous HUD tracker's elapsed events.
The caller must not supply renderer floor IDs, unrounded BPM or truth times.

The displayed CBPM gives a time *interval*, not a time origin. An origin is
accepted only when an existing HUD-only tracker independently establishes
exactly one elapsed event, a visible floor change, and a simultaneous textual
JRP update. No update (Miss or coincident identical text) => no new anchor.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS, REAL_CHART_FLOOR_FEATURE_DIM,
)

CURRENT_SLOT = FLOORS.behind_floors
FLOOR_DIM = REAL_CHART_FLOOR_FEATURE_DIM
SLOT_START = 26   # N-key 8K: 24 physical + 2 orbit
HUD_DIM = 263
EPS = 1e-7
VERSION = 'stage-t-jrp-cbpm-anchored-v1'


@dataclass(frozen=True)
class CbpmForecast:
    frame: int
    status: str
    due_s: float | None
    remaining_s: float | None
    uncertainty_s: float | None
    confidence: float = 0.0

    @property
    def eligible(self) -> bool:
        return self.status == 'eligible'


def _evidence_confidence(error_s: float | None, limit_s: float) -> float:
    """Conservative causal evidence score, not a calibrated probability."""
    if error_s is None or error_s < 0 or not isfinite(error_s):
        return 0.0
    if limit_s <= 0:
        return 1.0 if error_s == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - error_s / limit_s))


@dataclass(frozen=True)
class ElapsedVisibleEvent:
    """Contract for baseline CausalTimingTracker elapsed events (structural)."""
    due_s: float


def _floor(hud: Sequence[float]) -> tuple[float, ...]:
    offset = SLOT_START + CURRENT_SLOT * FLOOR_DIM
    return tuple(hud[offset:offset+FLOOR_DIM])


def _quantized_interval(text: tuple[str, str, str] | None, decimals: int,
                        max_error_s: float) -> tuple[float, float] | None:
    if text is None or len(text) != 3:
        return None
    try:
        tbpm, cbpm, kps = [float(s) for s in text]
    except (ValueError, TypeError, OverflowError):
        return None
    if not all(isfinite(x) and x > 0 for x in (tbpm, cbpm, kps)):
        return None
    q = .5 * 10.**(-decimals)
    # In JRP's pseudo display, KPS=CBPM/60 * group_count (count>1).
    # Only rounded *normal* text with count=1 is eligible. This classifier
    # is conservative and NOT a source-provided mode bit.
    if abs(kps - cbpm/60.) > q * (1 + 1/60.) + 1e-7:
        return None
    if cbpm <= q:
        return None
    interval = 60. / cbpm
    # Printed CBPM is at most half an LSD from internal raw CBPM;
    # the asymmetric bound is important for low BPM/long Pause.
    error = max(abs(60./(cbpm-q)-interval), abs(60./(cbpm+q)-interval))
    if error > max_error_s:
        return None
    return interval, error


class JrpCbpmAnchoredTracker:
    """Read-only temporal continuation, requiring a strong visible epoch.

    Does not alter v1.4's own predictions. One event interval per anchor;
    after its forecasted due passes, wait for fresh visible evidence.
    The tracker's `now` clock comes from 10ms frame count, never chart time.
    """

    def __init__(self, *, dt_s=.010, horizon_s=.030, decimals=2,
                 max_text_uncertainty_s=.00075, max_anchor_lag_s=.010001):
        if (dt_s != .010 or not .010 <= horizon_s <= .080 or
            type(decimals) is not int or not 0 <= decimals <= 4 or
            max_text_uncertainty_s < 0 or not 0 < max_anchor_lag_s <= .011):
            raise ValueError('unsupported JRP causal observation settings')
        self.dt = dt_s
        self.horizon = horizon_s
        self.decimals = decimals
        self.max_error = max_text_uncertainty_s
        self.max_lag = max_anchor_lag_s
        self.reset()

    def reset(self):
        self._frame = 0
        self._previous_hud = None
        self._previous_text = None
        self._due = None
        self._error = None
        self._pending_floor_event = None  # (frame, due, visible floor signature)
        self._recent_text_change = None    # (frame, exact newly visible text)
        self._recent_elapsed = None        # (frame, uniquely identified HUD event due)

    def observe(self, frame: int, hud: Sequence[float],
                jrp_text: tuple[str, str, str] | None,
                baseline_elapsed: Sequence[ElapsedVisibleEvent] = ()) -> CbpmForecast:
        if type(frame) is not int or frame != self._frame:
            raise ValueError('frames must be sequential from zero')
        self._frame += 1
        now = frame * self.dt
        invalid = len(hud) != HUD_DIM or not all(isfinite(x) for x in hud)
        if invalid:
            self._previous_hud = None
            self._previous_text = jrp_text
            self._due = None
            self._error = None
            self._pending_floor_event = None
            self._recent_text_change = None
            self._recent_elapsed = None
            return CbpmForecast(frame, 'invalid_hud', None, None, None)
        curfloor = _floor(hud)
        previous = _floor(self._previous_hud) if self._previous_hud is not None else None
        changed_floor = previous is not None and curfloor != previous
        valid_floor = curfloor[-1] >= .5
        text_changed = self._previous_text is not None and jrp_text != self._previous_text
        if changed_floor or not valid_floor:
            # If a floor changed without an authenticated display update,
            # all earlier CBPM intervals are stale even if due is later.
            self._due = self._error = None
        elapsed = [e.due_s for e in baseline_elapsed
                   if isfinite(e.due_s) and -EPS <= now-e.due_s <= self.max_lag]
        just_elapsed = (elapsed[0] if len(elapsed) == 1 else
            self._recent_elapsed[1] if (not elapsed and self._recent_elapsed is not None and
                self._recent_elapsed[0] == frame - 1 and
                now - self._recent_elapsed[1] <= self.max_lag) else None)
        anchor = None
        anchor_due = None
        previous_pending = self._pending_floor_event
        # One-frame late text from a real Motor Hit is admissible only when
        # the rendered current floor still matches the previously validated
        # single-floor transition. This is NOT an OnHit callback to Student.
        if (not changed_floor and text_changed and previous_pending is not None and
            previous_pending[0] == frame - 1 and valid_floor and
            previous_pending[2] == curfloor):
            anchor = _quantized_interval(jrp_text, self.decimals, self.max_error)
            anchor_due = previous_pending[1]
        self._pending_floor_event = None
        if changed_floor and valid_floor and just_elapsed is not None:
            # The *same* visible upcoming floor must be the new current floor.
            # Two events inside a single 10ms tick can otherwise produce a
            # stale baseline due even when only one baseline event elapsed.
            def old_slot(slot):
                st = SLOT_START + slot * FLOOR_DIM
                return tuple(self._previous_hud[st:st+FLOOR_DIM])
            def new_slot(slot):
                st = SLOT_START + slot * FLOOR_DIM
                return tuple(hud[st:st+FLOOR_DIM])
            def same_visible_geometry(a,b):
                # The x/y fields are RELATIVE to the current tile and
                # necessarily change when the visual window scrolls.
                return (a[-1] >= .5 and b[-1] >= .5 and
                        max(abs(x-y) for x,y in zip(a[2:-1],b[2:-1])) < 1e-8)
            exactly_one_visible_step = (
                same_visible_geometry(old_slot(CURRENT_SLOT),new_slot(CURRENT_SLOT-1))
                and same_visible_geometry(old_slot(CURRENT_SLOT+1),curfloor)
                and same_visible_geometry(old_slot(CURRENT_SLOT+2),new_slot(CURRENT_SLOT+1))
                # If the shifted windows are visually consistent with both
                # 1 step and 2+ steps, an elapsed v1.4 event is ambiguous.
                and not same_visible_geometry(old_slot(CURRENT_SLOT+2),curfloor))
            if previous is not None and previous[-1] >= .5 and exactly_one_visible_step:
                recent_early_change = (self._recent_text_change is not None and
                    self._recent_text_change[0] == frame - 1 and
                    self._recent_text_change[1] == jrp_text)
                if text_changed or recent_early_change:
                    anchor = _quantized_interval(jrp_text, self.decimals, self.max_error)
                    anchor_due = just_elapsed
                else:
                    self._pending_floor_event = (frame, just_elapsed, curfloor)
        if anchor is not None:
            interval, error = anchor
            self._due = anchor_due + interval
            self._error = error
        self._previous_hud = tuple(hud)
        self._previous_text = jrp_text
        self._recent_text_change = (frame, jrp_text) if text_changed else None
        self._recent_elapsed = (frame, elapsed[0]) if len(elapsed) == 1 else None
        if self._due is None:
            return CbpmForecast(frame, 'no_confirmed_display_anchor', None, None, None)
        remaining = self._due - now
        if remaining <= -EPS:
            self._due = self._error = None
            return CbpmForecast(frame, 'anchor_expired', None, None, None)
        if remaining <= 0:
            return CbpmForecast(frame, 'at_boundary', self._due, 0., self._error,
                                _evidence_confidence(self._error, self.max_error))
        return CbpmForecast(frame, 'eligible' if remaining <= self.horizon else 'outside_horizon',
                            self._due, remaining, self._error,
                            _evidence_confidence(self._error, self.max_error))


@dataclass(frozen=True)
class JrpJointTimingUpdate:
    """Research-only combined *pending forecast*, never an executed action."""
    frame: int
    base_update: object
    cbpm_update: CbpmForecast
    future_due_s: tuple[float, ...]
    jrp_added: bool


class CausalJrpTimingBridge:
    """Versioned two-stream read-only Timing interface.

    Input: 263D visible HUD + JRP *rendered text* only. The v1.4 base
    tracker remains unmodified, and JRP can add at most one next event.
    No chart target, floor ordinal, evaluator time or native callback leaks
    through the predictor call. Merge is purely provisional for Timing.
    """

    def __init__(self, *, baseline=None, overlay=None, tolerance_s=.00075):
        if baseline is None:
            from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
            baseline = CausalTimingTracker()
        self.baseline = baseline
        self.overlay = overlay or JrpCbpmAnchoredTracker()
        self.tolerance = tolerance_s

    def reset(self):
        self.baseline.reset()
        self.overlay.reset()

    def observe(self, frame: int, hud: Sequence[float],
                rendered_jrp_text: tuple[str, str, str] | None) -> JrpJointTimingUpdate:
        base = self.baseline.observe(frame, tuple(hud))
        derived = self.overlay.observe(frame, hud, rendered_jrp_text, base.elapsed)
        now = frame * self.overlay.dt
        pending = sorted(p.due_s for p in base.pending if p.due_s > now)
        jrp_added = False
        if derived.eligible and derived.due_s is not None:
            if not any(abs(t - derived.due_s) <= self.tolerance for t in pending):
                pending.append(derived.due_s)
                pending.sort()
                jrp_added = True
        return JrpJointTimingUpdate(frame, base, derived, tuple(pending), jrp_added)
