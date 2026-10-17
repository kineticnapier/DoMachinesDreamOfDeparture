from __future__ import annotations

"""Hit-epoch causal JRP continuation with ambiguous-frame abstention.

The visible overlay is sampled before each Motor step.  Therefore a text
change first observed at frame F can only have been produced by a Hit during
frame F-1, never by an event that first elapsed at F.  This version encodes
that ordering directly and refuses coincident/high-density associations.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from dmdod.planning.jrp_cbpm_anchor_v1 import (
    CbpmForecast,
    JrpJointTimingUpdate,
    _evidence_confidence,
    _floor,
    _quantized_interval,
    HUD_DIM,
    EPS,
)


VERSION = 'stage-t-jrp-hit-epoch-v2'


class JrpCbpmHitEpochTracker:
    """Associate changed text only with the uniquely prior elapsed event."""

    def __init__(self, *, dt_s=.010, horizon_s=.030, decimals=2,
                 max_text_uncertainty_s=.00075):
        if (dt_s != .010 or not .010 <= horizon_s <= .080 or
            type(decimals) is not int or not 0 <= decimals <= 4 or
            max_text_uncertainty_s < 0):
            raise ValueError('unsupported JRP causal observation settings')
        self.dt = dt_s
        self.horizon = horizon_s
        self.decimals = decimals
        self.max_error = max_text_uncertainty_s
        self.reset()

    def reset(self):
        self._frame = 0
        self._previous_hud = None
        self._previous_text = None
        self._prior_unique_elapsed = None  # (observation frame, exact due)
        self._due = None
        self._error = None
        self._confidence = 0.0

    def observe(self, frame: int, hud: Sequence[float],
                jrp_text: tuple[str, str, str] | None,
                baseline_elapsed=()) -> CbpmForecast:
        if type(frame) is not int or frame != self._frame:
            raise ValueError('frames must be sequential from zero')
        self._frame += 1
        now = frame * self.dt
        invalid = len(hud) != HUD_DIM or not all(isfinite(x) for x in hud)
        if invalid:
            self._previous_hud = None
            self._previous_text = jrp_text
            self._prior_unique_elapsed = None
            self._due = self._error = None
            self._confidence = 0.
            return CbpmForecast(frame, 'invalid_hud', None, None, None, 0.)

        current_floor = _floor(hud)
        previous_floor = (_floor(self._previous_hud)
                          if self._previous_hud is not None else None)
        changed_floor = previous_floor is not None and current_floor != previous_floor
        valid_floor = current_floor[-1] >= .5
        text_changed = self._previous_text is not None and jrp_text != self._previous_text
        if changed_floor or not valid_floor:
            self._due = self._error = None
            self._confidence = 0.

        elapsed = [float(e.due_s) for e in baseline_elapsed
                   if isfinite(e.due_s) and e.due_s <= now + EPS]
        status = 'no_confirmed_display_anchor'
        if text_changed:
            prior = self._prior_unique_elapsed
            # A text update visible now was produced after the previous
            # observation.  If another event elapsed now, the source epoch is
            # observationally ambiguous and must not be guessed.
            if (not elapsed and prior is not None and prior[0] == frame - 1 and
                    0. <= now - prior[1] <= 2. * self.dt + EPS):
                anchor = _quantized_interval(jrp_text, self.decimals, self.max_error)
                if anchor is not None:
                    interval, error = anchor
                    self._due = prior[1] + interval
                    self._error = error
                    quantized = _evidence_confidence(error, self.max_error)
                    lag = max(0., now - prior[1])
                    self._confidence = quantized * max(0., 1. - lag / (2. * self.dt))
            elif elapsed:
                status = 'ambiguous_coincident_event'

        self._previous_hud = tuple(hud)
        self._previous_text = jrp_text
        self._prior_unique_elapsed = ((frame, elapsed[0]) if len(elapsed) == 1 else None)

        if self._due is None:
            return CbpmForecast(frame, status, None, None, None, 0.)
        remaining = self._due - now
        if remaining <= -EPS:
            self._due = self._error = None
            self._confidence = 0.
            return CbpmForecast(frame, 'anchor_expired', None, None, None, 0.)
        if remaining <= 0:
            return CbpmForecast(frame, 'at_boundary', self._due, 0., self._error,
                                self._confidence)
        return CbpmForecast(frame,
            'eligible' if remaining <= self.horizon else 'outside_horizon',
            self._due, remaining, self._error, self._confidence)


class CausalJrpTimingBridgeV2:
    """Timing v1.4 plus hit-epoch JRP forecasts, without changing v1.4."""

    def __init__(self, *, baseline=None, overlay=None, tolerance_s=.00075):
        if baseline is None:
            from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
            baseline = CausalTimingTracker()
        self.baseline = baseline
        self.overlay = overlay or JrpCbpmHitEpochTracker()
        self.tolerance = tolerance_s

    def reset(self):
        self.baseline.reset()
        self.overlay.reset()

    def observe(self, frame: int, hud: Sequence[float], rendered_jrp_text):
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
