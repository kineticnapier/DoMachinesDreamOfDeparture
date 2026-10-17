from __future__ import annotations

"""One-step phase continuation from causal elapsed history and visible CBPM."""

from math import isfinite
from typing import Sequence

from dmdod.planning.jrp_cbpm_anchor_v1 import (
    CbpmForecast, JrpJointTimingUpdate, _evidence_confidence,
    _quantized_interval, HUD_DIM, EPS,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT, visible_slot,
)


VERSION='stage-t-jrp-cbpm-phase-v4'


def _ordinary_current_and_next(hud: Sequence[float]) -> bool:
    data=tuple(hud)
    for slot in (CURRENT_SLOT,CURRENT_SLOT+1):
        floor=visible_slot(data,slot)
        if len(floor)!=15 or floor[-1]<.5:
            return False
        # midspin, MultiPlanet and all six visible event markers.
        if any(value>.5 for value in floor[6:14]):
            return False
    return True


class JrpCbpmPhaseTracker:
    """Predict at most one event beyond the latest baseline elapsed event.

    A new phase step requires two distinct, sufficiently reconfirmed baseline
    events whose observed spacing agrees with the currently displayed normal
    CBPM.  If the extrapolated event is not subsequently observed by baseline,
    history is not advanced and no further event can be synthesized.
    """

    def __init__(self,*,dt_s=.010,horizon_s=.030,decimals=2,
                 max_text_uncertainty_s=.00075,match_tolerance_s=.00075,
                 min_confirmations=2):
        if (dt_s!=.010 or not .010<=horizon_s<=.080 or
            type(decimals) is not int or not 0<=decimals<=4 or
            max_text_uncertainty_s<0 or not 0<=match_tolerance_s<=.002 or
            type(min_confirmations) is not int or min_confirmations<2):
            raise ValueError('unsupported JRP phase settings')
        self.dt=dt_s;self.horizon=horizon_s;self.decimals=decimals
        self.max_error=max_text_uncertainty_s;self.tolerance=match_tolerance_s
        self.min_confirmations=min_confirmations
        self.reset()

    def reset(self):
        self._frame=0
        self._elapsed=[]
        self._due=None;self._error=None;self._confidence=0.

    def observe(self,frame:int,hud:Sequence[float],jrp_text,
                baseline_elapsed=()) -> CbpmForecast:
        if type(frame) is not int or frame!=self._frame:
            raise ValueError('frames must be sequential from zero')
        self._frame+=1;now=frame*self.dt
        if len(hud)!=HUD_DIM or not all(isfinite(x) for x in hud):
            self.reset();self._frame=frame+1
            return CbpmForecast(frame,'invalid_hud',None,None,None,0.)
        if not _ordinary_current_and_next(hud):
            self._elapsed.clear();self._due=self._error=None;self._confidence=0.
            return CbpmForecast(frame,'special_or_unseen_floor',None,None,None,0.)

        elapsed=[e for e in baseline_elapsed if isfinite(e.due_s)]
        if len(elapsed)>1:
            self._elapsed.clear();self._due=self._error=None;self._confidence=0.
            return CbpmForecast(frame,'ambiguous_elapsed_events',None,None,None,0.)
        if elapsed:
            event=elapsed[0]
            if (getattr(event,'confirmations',0)<self.min_confirmations or
                    not getattr(event,'confirmed_now',False)):
                self._elapsed.clear();self._due=self._error=None;self._confidence=0.
                return CbpmForecast(frame,'unconfirmed_elapsed_event',None,None,None,0.)
            due=float(event.due_s)
            if not self._elapsed or abs(due-self._elapsed[-1])>EPS:
                self._elapsed.append(due)
                self._elapsed=self._elapsed[-2:]
            self._due=self._error=None;self._confidence=0.
            if len(self._elapsed)==2:
                parsed=_quantized_interval(jrp_text,self.decimals,self.max_error)
                if parsed is not None:
                    interval,error=parsed
                    observed=self._elapsed[1]-self._elapsed[0]
                    allowed=self.tolerance+error
                    disagreement=abs(observed-interval)
                    if observed>0 and disagreement<=allowed:
                        self._due=self._elapsed[1]+interval
                        self._error=error
                        agreement=max(0.,1.-disagreement/max(allowed,EPS))
                        self._confidence=_evidence_confidence(error,self.max_error)*agreement

        if self._due is None:
            return CbpmForecast(frame,'no_confirmed_phase',None,None,None,0.)
        remaining=self._due-now
        if remaining<=-EPS:
            self._due=self._error=None;self._confidence=0.
            return CbpmForecast(frame,'phase_expired',None,None,None,0.)
        if remaining<=0:
            return CbpmForecast(frame,'at_boundary',self._due,0.,self._error,self._confidence)
        return CbpmForecast(frame,
            'eligible' if remaining<=self.horizon else 'outside_horizon',
            self._due,remaining,self._error,self._confidence)


class CausalJrpTimingBridgeV4:
    def __init__(self,*,baseline=None,overlay=None,tolerance_s=.00075,
                 min_confidence=.5):
        if baseline is None:
            from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
            baseline=CausalTimingTracker()
        self.baseline=baseline;self.overlay=overlay or JrpCbpmPhaseTracker()
        self.tolerance=tolerance_s;self.min_confidence=min_confidence
        if not 0<=min_confidence<=1:raise ValueError('invalid confidence threshold')

    def reset(self):self.baseline.reset();self.overlay.reset()

    def observe(self,frame:int,hud:Sequence[float],rendered_jrp_text):
        base=self.baseline.observe(frame,tuple(hud))
        derived=self.overlay.observe(frame,hud,rendered_jrp_text,base.elapsed)
        now=frame*self.overlay.dt
        pending=sorted(p.due_s for p in base.pending if p.due_s>now)
        added=False
        if (derived.eligible and derived.due_s is not None and
                derived.confidence>=self.min_confidence and
                not any(abs(t-derived.due_s)<=self.tolerance for t in pending)):
            pending.append(derived.due_s);pending.sort();added=True
        return JrpJointTimingUpdate(frame,base,derived,tuple(pending),added)
