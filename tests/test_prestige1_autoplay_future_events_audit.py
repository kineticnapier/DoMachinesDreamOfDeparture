from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_future_events_audit as future
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as simulator


def _chart(angles:list[float],*,bpm:float=8000., actions:list[dict]|None=None):
    compiled=compile_adofai(parse_adofai_text(json.dumps({
        "angleData":angles,
        "settings":{"bpm":bpm,"pitch":100.,"countdownTicks":0},
        "actions":actions or [],
    })))
    return build_playable_segment(compiled)


def _capture(segment):
    samples,done=simulator._capture(segment)
    assert done
    return samples


@pytest.mark.parametrize("bpm,reverse",[
    (4000.,False),(8000.,True),(12000.,False),(12000.,True),
    (20000.,False),(20000.,True),
])
def test_real_simulator_forecast_order_and_times_for_10ms(bpm,reverse):
    segment,_=simulator._fixture(bpm,reverse=reverse)
    sampled=_capture(segment)
    # Offset+1ms avoids exact floor-boundary ties for harmonics of 10ms.
    for offset in (0,1,2,5):
        result=future.evaluate_chart(segment,sampled[offset::10])
        assert result["incorrect"]==0, result["examples"]
        assert result["eligible"]>=0
    probe=future.evaluate_chart(segment,sampled[1::10])
    assert probe["eligible"]>0
    if bpm>=8000.:
        assert probe["correct_multi_target"]>0
        assert probe["correct_multi_target"]==probe["eligible_multi_target"]


def test_20000_bpm_three_events_on_original_10ms_clock():
    segment,_=simulator._fixture(20000.,reverse=False)
    frames=_capture(segment)
    prediction=future.forecast(frames[10].hud)
    assert prediction.status=="eligible"
    assert prediction.floor_offsets==(1,2,3)
    assert prediction.times_s==pytest.approx((.002,.005,.008),abs=2e-6)
    # The planner has not changed the physics/control clock.
    assert p.CONTROL_DT==.010 and p.PHYSICS_DT==.001


@pytest.mark.parametrize("angles,bpm,twirl",[
    ([0.,90.,180.,270.]*20,4000.,False),
    ([0.,90.,180.,270.]*20,20000.,True),
    ([0.,30.,90.,180.]*20,20000.,False),
    ([90.,180.,270.,0.]*20,4000.,True),
    ([0.,0.,90.,90.,180.,180.]*12,20000.,False),
])
def test_varied_real_chart_geometry_never_wrongly_claims_timing(angles,bpm,twirl):
    segment=_chart(angles,bpm=bpm,actions=[{"floor":1,"eventType":"Twirl"}] if twirl else [])
    samples=_capture(segment)
    for offset in (0,1,2,5):
        r=future.evaluate_chart(segment,samples[offset::10])
        assert r["incorrect"]==0,r["examples"]


def test_current_phase_does_not_choose_direction_by_future_failure():
    # CW future geometry is a full turn: this is NOT evidence for CCW.
    segment=_chart([0.,30.,90.,180.]*20,bpm=20000.)
    first=_capture(segment)[0]
    assert future.forecast(first.hud).status=="full_turn_or_zero_arc"
    # Future Twirl encountered under one direction is also not proof of
    # the opposite, slower rotation mode.
    segment2=_chart([90.,180.,270.,0.]*20,bpm=4000.,actions=[
        {"floor":1,"eventType":"Twirl"},
    ])
    assert future.forecast(_capture(segment2)[0].hud).status=="special_future_floor"


@pytest.mark.parametrize("event,fields",[
    ("Twirl",{}),
    ("SetSpeed",{"speedType":"Bpm","beatsPerMinute":6000}),
    ("Pause",{"duration":1}),
    ("MultiPlanet",{"planets":"ThreePlanets"}),
])
def test_visible_special_future_floor_abstains(event,fields):
    segment=_chart([0.]*25,bpm=8000.,actions=[{"floor":1,"eventType":event,**fields}])
    result=future.forecast(_capture(segment)[0].hud)
    assert not result.eligible
    assert not result.times_s
    assert result.status in ("special_future_floor","special_event", "midspin_or_multiplanet")


def test_too_few_visible_future_floors_abstains():
    segment=_chart([0.]*2,bpm=20000.)
    frame=_capture(segment)[0]
    result=future.forecast(frame.hud)
    assert result.status=="insufficient_lookahead"


def test_corrupted_input_fails_closed():
    segment,_=simulator._fixture(4000.,reverse=False)
    hud=_capture(segment)[0].hud
    assert future.forecast(hud[:30]).status=="invalid_hud_dimension"
    h=list(hud);h[20]=float('nan')
    assert future.forecast(tuple(h)).status=="nonfinite_hud"
    h=list(hud);h[240]=float('inf')
    assert future.forecast(tuple(h)).status=="nonfinite_hud"


def test_report_refuses_overwrite_and_keeps_frozen_config(tmp_path):
    cfg=replace(p.PrestigeConfig.load("configs/training/prestige1.toml"),output_dir=str(tmp_path))
    report=future.run_audit(cfg)
    assert report["input_dim"]==263 and report["horizon_ms"]==10
    assert len(report["cases"])==26
    assert report["no_training"] is True
    assert report["validation_evaluated"] is False
    assert report["final_evaluated"] is False
    assert any(x["bpm"]==20000. and x["direction"]=="ccw" and
               x["phase_offset_10ms_probes"]["1"]["correct_multi_target"]>0
               for x in report["cases"])
    assert all(
        result["incorrect"]==0
        for case in report["cases"]
        for result in case["phase_offset_10ms_probes"].values()
    )
    path=tmp_path/"diagnostics"/"prestige1"/future.REPORT
    original=path.read_bytes()
    with pytest.raises(FileExistsError):
        future.run_audit(cfg)
    assert path.read_bytes()==original
