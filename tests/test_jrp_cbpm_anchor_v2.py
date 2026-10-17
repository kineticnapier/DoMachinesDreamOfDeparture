from __future__ import annotations

import pytest

from dmdod.planning.jrp_cbpm_anchor_v1 import ElapsedVisibleEvent
from dmdod.planning.jrp_cbpm_anchor_v2 import JrpCbpmHitEpochTracker


def frame(floor=1):
    from dmdod.features.real_chart import (
        DEFAULT_REAL_CHART_FEATURE_CONFIG as cfg, REAL_CHART_FLOOR_FEATURE_DIM as dim)
    h=[0.] * 263
    start=26+cfg.behind_floors*dim
    def slot(n):
        return [n*.01,0.,1.+n*.01,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
    h[start-dim:start]=slot(floor-1)
    h[start:start+dim]=slot(floor)
    h[start+dim:start+2*dim]=slot(floor+1)
    h[start+2*dim:start+3*dim]=slot(floor+2)
    return tuple(h)


def test_text_change_anchors_only_to_unique_prior_frame_event():
    t=JrpCbpmHitEpochTracker()
    t.observe(0,frame(1),('120','1000','16.67'),[ElapsedVisibleEvent(0.)])
    v=t.observe(1,frame(1),('120','3000','50'),())
    assert v.status=='eligible'
    assert v.due_s==pytest.approx(.02)
    assert 0.<v.confidence<=1.


def test_coincident_new_event_makes_changed_text_ambiguous():
    t=JrpCbpmHitEpochTracker()
    t.observe(0,frame(1),('120','1000','16.67'),[ElapsedVisibleEvent(0.)])
    v=t.observe(1,frame(2),('120','3000','50'),[ElapsedVisibleEvent(.01)])
    assert v.status=='ambiguous_coincident_event'
    assert v.due_s is None and v.confidence==0.


def test_early_text_is_never_reused_for_future_event():
    t=JrpCbpmHitEpochTracker()
    t.observe(0,frame(1),('120','1000','16.67'),())
    assert t.observe(1,frame(1),('120','3000','50'),()).due_s is None
    assert t.observe(2,frame(2),('120','3000','50'),
                     [ElapsedVisibleEvent(.02)]).due_s is None


def test_more_than_one_frame_display_delay_abstains():
    t=JrpCbpmHitEpochTracker()
    t.observe(0,frame(1),('120','1000','16.67'),[ElapsedVisibleEvent(0.)])
    t.observe(1,frame(1),('120','1000','16.67'),())
    assert t.observe(2,frame(1),('120','3000','50'),()).due_s is None


def test_invalid_and_nonsequential_inputs_fail_closed():
    t=JrpCbpmHitEpochTracker()
    with pytest.raises(ValueError):
        t.observe(1,frame(1),None)
    bad=list(frame(1));bad[0]=float('nan')
    assert t.observe(0,bad,None).status=='invalid_hud'
