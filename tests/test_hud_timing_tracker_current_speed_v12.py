from __future__ import annotations

from bisect import bisect_right
import json
import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning import hud_timing_tracker_current_speed_v12 as current
from dmdod.planning import hud_timing_tracker_partial_v11 as prior
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as sim


@pytest.mark.parametrize("initial,changed",[
    (2000.,6000.),(4000.,1200.),(8000.,3000.),(16000.,12000.),
])
def test_current_setspeed_visible_bpm_reconstructs_safe_times(initial,changed):
    raw=json.dumps({"angleData":[0.,90.,180.,270.]*20,
                    "settings":{"bpm":initial,"pitch":100.,"countdownTicks":0},
                    "actions":[{"floor":5,"eventType":"SetSpeed","speedType":"Bpm",
                                "beatsPerMinute":changed}]})
    seg=build_playable_segment(compile_adofai(parse_adofai_text(raw)))
    samples,_=sim._capture(seg)
    truth=tuple(float(t.episode_time_s) for t in seg.targets)
    new_eligibility=old_eligibility=0
    for sm in samples[::10]:
        forecast=current.forecast_visible_window(sm.hud)
        new_eligibility+=forecast.eligible
        old_eligibility+=prior.forecast_visible_window(sm.hud).eligible
        if not forecast.eligible:continue
        end=forecast.covered_until_s if forecast.covered_until_s is not None else .030
        lo=bisect_right(truth,sm.seconds+1e-7)
        hi=bisect_right(truth,sm.seconds+end+1e-7)
        expected=tuple(t-sm.seconds for t in truth[lo:hi])
        assert len(expected)==len(forecast.times_s),(initial,changed,sm.seconds,expected,forecast)
        assert all(abs(a-b)<5e-5 for a,b in zip(expected,forecast.times_s))
    assert new_eligibility>=old_eligibility


def test_future_speed_transition_still_abstains():
    raw=json.dumps({"angleData":[0.]*30,
                    "settings":{"bpm":4000.,"pitch":100.,"countdownTicks":0},
                    "actions":[{"floor":1,"eventType":"SetSpeed",
                                "speedType":"Bpm","beatsPerMinute":8000.}]})
    seg=build_playable_segment(compile_adofai(parse_adofai_text(raw)))
    samples,_=sim._capture(seg)
    assert not current.forecast_visible_window(samples[0].hud).eligible
