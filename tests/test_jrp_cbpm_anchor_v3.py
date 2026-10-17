from types import SimpleNamespace

from dmdod.planning.jrp_cbpm_anchor_v1 import CbpmForecast
from dmdod.planning.jrp_cbpm_anchor_v3 import CausalJrpTimingBridgeV3


class Baseline:
    def reset(self):pass
    def observe(self,frame,hud):
        return SimpleNamespace(pending=(),elapsed=())


class Overlay:
    dt=.01
    def __init__(self,confidence):self.confidence=confidence
    def reset(self):pass
    def observe(self,frame,hud,text,elapsed):
        return CbpmForecast(frame,'eligible',.02,.02,.0001,self.confidence)


def test_v3_rejects_development_low_confidence_and_accepts_boundary():
    low=CausalJrpTimingBridgeV3(baseline=Baseline(),overlay=Overlay(.499999))
    assert not low.observe(0,(0.,)*263,None).jrp_added
    boundary=CausalJrpTimingBridgeV3(baseline=Baseline(),overlay=Overlay(.5))
    assert boundary.observe(0,(0.,)*263,None).jrp_added


def test_v3_threshold_cannot_be_overridden():
    try:
        CausalJrpTimingBridgeV3(min_confidence=.3)
    except ValueError as exc:
        assert 'frozen' in str(exc)
    else:
        raise AssertionError('v3 threshold override was accepted')
