from __future__ import annotations

import pytest
from dmdod.planning.jrp_cbpm_anchor_v1 import (
    JrpCbpmAnchoredTracker, ElapsedVisibleEvent, _quantized_interval,
)


def frame(floor=1, *, nonfinite=False):
    from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as cfg, REAL_CHART_FLOOR_FEATURE_DIM as dim
    h=[0.] * 263
    start=26+cfg.behind_floors*dim
    def slot(n):
        return [n*.01, 0., 1. + n*.01, 0., 0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
    h[start-dim:start]=slot(floor-1)
    h[start:start+dim]=slot(floor)
    h[start+dim:start+2*dim]=slot(floor+1)
    h[start+2*dim:start+3*dim]=slot(floor+2)
    if nonfinite:h[0]=float('nan')
    return tuple(h)


def step(tracker, i, which, bpm=('120','30','0.5'), anchors=()):
    return tracker.observe(i,frame(which),bpm,anchors)


def test_requires_floor_change_and_text_change_and_elapsed_anchor():
    t=JrpCbpmAnchoredTracker()
    assert step(t,0,1,('120','60','1')).status=='no_confirmed_display_anchor'
    assert step(t,1,2,('120','60','1'),[ElapsedVisibleEvent(.01)]).status=='no_confirmed_display_anchor'
    assert step(t,2,2,('120','30','0.5')).status=='outside_horizon'
    assert step(t,3,3,('120','60','1'),[]).status=='no_confirmed_display_anchor'
    v=step(t,4,4,('120','30','0.5'),[ElapsedVisibleEvent(.04)])
    assert v.status=='outside_horizon' and v.due_s==pytest.approx(2.04)


def test_missed_note_floor_change_invalidates_previous_forecast():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','6000','100'))
    v=step(t,1,2,('120','3000','50'),[ElapsedVisibleEvent(.01)])
    assert v.status=='eligible'
    assert v.due_s==pytest.approx(.03)
    assert step(t,2,2,('120','3000','50')).status=='eligible'
    # Floor advances, but no new Hit; JRP text stays stale.
    assert step(t,3,3,('120','3000','50')).status=='no_confirmed_display_anchor'


def test_reject_pseudo_aggregation():
    assert _quantized_interval(('992','1488','49.6'),2,.001) is None
    assert _quantized_interval(('496','744','24.8'),2,.001) is None
    assert _quantized_interval(('992','1488','24.8'),2,.001) is not None


def test_rounding_error_rejects_slow_long_pause():
    assert _quantized_interval(('120','0.1','0'),2,.001) is None
    assert _quantized_interval(('120','20','0.33'),2,.0005) is None
    assert _quantized_interval(('120','20','0.33'),2,.1) is not None


def test_rounding_and_epoch_anchor_avoid_early_or_late_hit_time_origins():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','120','2'))
    v=step(t,1,2,('120','3000','50'),[ElapsedVisibleEvent(.005)])
    assert v.due_s==pytest.approx(.025)
    # Derived from *baseline scheduled event*, not from frame timestamp .010
    assert v.remaining_s==pytest.approx(.015)


def test_missing_or_nonsequential_frame_is_rejected():
    t=JrpCbpmAnchoredTracker()
    with pytest.raises(ValueError):step(t,1,1)
    assert t.observe(0,frame(1,nonfinite=True),None).status=='invalid_hud'
    assert t.observe(1,frame(1),None).status=='no_confirmed_display_anchor'


def test_multiple_elapsed_events_do_not_pick_arbitrary_anchor():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','1000','16.67'))
    v=step(t,1,2,('120','2000','33.33'),[ElapsedVisibleEvent(.003),ElapsedVisibleEvent(.007)])
    assert v.status=='no_confirmed_display_anchor'


def test_rounded_visible_text_is_only_cbpm_input():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','1000','16.67'))
    v=step(t,1,2,('120','2000','33.33'),[ElapsedVisibleEvent(.01)])
    assert v.due_s==pytest.approx(.04)
    assert v.uncertainty_s < .00075


def test_skipped_intermediate_floor_rejects_stale_baseline_event():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('100','600','10'))
    # Two floor advances inside one 10ms tick, but only an old baseline
    # elapsed event survives; source display belongs to the *later* floor.
    v=step(t,1,3,('100','1200','20'),[ElapsedVisibleEvent(.003)])
    assert v.status=='no_confirmed_display_anchor'


def test_one_frame_late_jrp_update_can_match_prior_causal_event():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','60','1'))
    step(t,1,2,('120','60','1'),[ElapsedVisibleEvent(.01)])
    v=step(t,2,2,('120','3000','50'))
    assert v.status=='eligible' and v.due_s==pytest.approx(.03)

def test_one_frame_early_jrp_update_can_match_upcoming_causal_event():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','60','1'))
    assert step(t,1,1,('120','3000','50')).status=='no_confirmed_display_anchor'
    v=step(t,2,2,('120','3000','50'),[ElapsedVisibleEvent(.02)])
    assert v.status=='eligible' and v.due_s==pytest.approx(.04)

def test_two_frames_late_jrp_update_abstains():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','60','1'))
    step(t,1,2,('120','60','1'),[ElapsedVisibleEvent(.01)])
    step(t,2,2,('120','60','1'))
    assert step(t,3,2,('120','3000','50')).status=='no_confirmed_display_anchor'


def test_v14_elapsed_event_one_frame_before_visible_floor_change():
    t=JrpCbpmAnchoredTracker()
    step(t,0,1,('120','60','1'))
    step(t,1,1,('120','60','1'),[ElapsedVisibleEvent(.01)])
    v=step(t,2,2,('120','3000','50'))
    assert v.status=='eligible' and v.due_s==pytest.approx(.03)


def test_combined_bridge_inserts_one_next_event_without_duplicate():
    from dmdod.planning.jrp_cbpm_anchor_v1 import CausalJrpTimingBridge
    from dmdod.planning.hud_timing_tracker_transition_v14 import TimingUpdate, TrackedEvent
    class BaselineStub:
        def __init__(self):self.n=0
        def reset(self):self.n=0
        def observe(self,frame,hud):
            assert frame==self.n
            self.n+=1
            elapsed=() if frame==0 else (TrackedEvent(1,.01,2,True),)
            return TimingUpdate(frame,'eligible',(),elapsed,(),(),())
    bridge=CausalJrpTimingBridge(baseline=BaselineStub())
    a=bridge.observe(0,frame(1),('120','1000','16.67'))
    assert not a.jrp_added
    b=bridge.observe(1,frame(2),('120','3000','50'))
    assert b.jrp_added and b.future_due_s==pytest.approx((.03,))
    bridge.reset()
    assert not bridge.observe(0,frame(1),('120','1000','16.67')).jrp_added
