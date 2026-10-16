from __future__ import annotations

from bisect import bisect_right
import json
import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.hud_timing_tracker_partial_v11 import (
    CausalTimingTracker, WindowForecast, forecast_visible_window,
)
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as sim


def chart(angles,bpm,actions):
    raw=json.dumps({"angleData":angles*20,"settings":{"bpm":bpm,"pitch":100.,"countdownTicks":0},"actions":actions})
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


@pytest.mark.parametrize("bpm",[4000.,8000.,12000.,20000.])
def test_twirl_partial_forecast_is_safe_prefix_at_four_speeds(bpm):
    seg=chart([0.,90.,180.,270.],bpm,[{"floor":1,"eventType":"Twirl"}])
    samples,_=sim._capture(seg)
    truth=tuple(float(t.episode_time_s) for t in seg.targets)
    partial=0
    for sample in samples[::10]:
        f=forecast_visible_window(sample.hud)
        if not f.eligible:continue
        if f.status=="eligible_partial":
            partial+=1
            assert f.covered_until_s is not None
            assert f.times_s and abs(f.times_s[-1]-f.covered_until_s)<1e-7
        window=f.covered_until_s if f.covered_until_s is not None else .030
        lo=bisect_right(truth,sample.seconds+1e-7)
        hi=bisect_right(truth,sample.seconds+window+1e-7)
        exp=tuple(t-sample.seconds for t in truth[lo:hi])
        assert len(exp)==len(f.times_s), (sample.seconds,exp,f)
        assert all(abs(a-b)<=5e-5 for a,b in zip(exp,f.times_s))
    assert partial>0


@pytest.mark.parametrize("event,extra",[
    ("SetSpeed",{"speedType":"Bpm","beatsPerMinute":6000}),
    ("Pause",{"duration":1}),
])
def test_speed_and_pause_future_events_are_not_predicted_as_safe_prefix(event,extra):
    seg=chart([0.]*25,4000.,[{"floor":1,"eventType":event,**extra}])
    samples,_=sim._capture(seg)
    assert forecast_visible_window(samples[0].hud).status in (
        "special_future_floor","special_event","full_turn_or_zero_arc")
    assert forecast_visible_window(samples[0].hud).status!="eligible_partial"


def test_reconciliation_uses_partial_coverage_to_keep_out_of_window_events():
    forecasts=iter([
        WindowForecast((1,2),(.015,.025),"eligible"),
        WindowForecast((1,),(.005,),"eligible_partial",.005),
        WindowForecast((),(),"special_event"),
    ])
    tracker=CausalTimingTracker(predictor=lambda _:next(forecasts))
    a=tracker.observe(0,())
    b=tracker.observe(1,())
    assert b.reused_ids==(0,)
    assert not b.canceled
    assert {x.local_id for x in b.pending}=={0,1}
    c=tracker.observe(2,())
    assert {x.local_id for x in c.elapsed}=={0}
    assert {x.local_id for x in c.pending}=={1}


def test_malformed_partial_is_rejected():
    tracker=CausalTimingTracker(predictor=lambda _:WindowForecast((),(),"eligible_partial",.01))
    with pytest.raises(ValueError):
        tracker.observe(0,())
