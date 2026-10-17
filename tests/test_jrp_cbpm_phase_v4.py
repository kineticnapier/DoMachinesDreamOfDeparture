from types import SimpleNamespace

import pytest

from dmdod.planning.jrp_cbpm_phase_v4 import JrpCbpmPhaseTracker,CausalJrpTimingBridgeV4
from dmdod.planning.hud_timing_tracker_transition_v14 import TimingUpdate,TrackedEvent


def frame(floor=1):
    from dmdod.features.real_chart import (
        DEFAULT_REAL_CHART_FEATURE_CONFIG as cfg, REAL_CHART_FLOOR_FEATURE_DIM as dim)
    h=[0.]*263;start=26+cfg.behind_floors*dim
    def slot(n):return [n*.01,0.,1.+n*.01,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
    h[start-dim:start]=slot(floor-1);h[start:start+dim]=slot(floor)
    h[start+dim:start+2*dim]=slot(floor+1);h[start+2*dim:start+3*dim]=slot(floor+2)
    return tuple(h)


def event(due,confirmations=2):
    return TrackedEvent(1,due,confirmations,True)


def test_two_confirmed_intervals_matching_cbpm_predict_one_step():
    t=JrpCbpmPhaseTracker()
    t.observe(0,frame(1),('120','3000','50'))
    t.observe(1,frame(1),('120','3000','50'),[event(.01)])
    t.observe(2,frame(1),('120','3000','50'))
    v=t.observe(3,frame(1),('120','3000','50'),[event(.03)])
    assert v.status=='eligible' and v.due_s==pytest.approx(.05)
    assert v.confidence>.5


def test_phase_rejects_mismatch_special_marker_and_unconfirmed_event():
    t=JrpCbpmPhaseTracker()
    t.observe(0,frame(1),('120','3000','50'),[event(0.,1)])
    assert t.observe(1,frame(1),('120','3000','50')).due_s is None
    t.reset();t.observe(0,frame(1),('120','3000','50'),[event(0.)])
    assert t.observe(1,frame(1),('120','1500','25'),[event(.01)]).due_s is None
    marked=list(frame(1)); marked[26+2*15+8]=1.
    assert t.observe(2,marked,('120','3000','50')).status=='special_or_unseen_floor'


class Baseline:
    def __init__(self):self.n=0
    def reset(self):self.n=0
    def observe(self,frame,hud):
        elapsed=(event(.01),) if frame==1 else ((event(.03),) if frame==3 else ())
        return TimingUpdate(frame,'eligible',(),elapsed,(),(),())


def test_bridge_adds_phase_only_when_baseline_lacks_due():
    b=CausalJrpTimingBridgeV4(baseline=Baseline())
    for i in range(3):assert not b.observe(i,frame(1),('120','3000','50')).jrp_added
    assert b.observe(3,frame(1),('120','3000','50')).jrp_added
