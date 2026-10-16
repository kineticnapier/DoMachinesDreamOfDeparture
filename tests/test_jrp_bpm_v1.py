from __future__ import annotations

import json
from math import isfinite

import pytest
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.features.hud import hud_bpms_for_floor
from dmdod.features.jrp_bpm_v1 import (
    JrpNormalBpmDisplay, jrp_normal_bpm_for_floor, JRP_SOURCE_REVISION,
)


def segment(pause_beats=0., pitch=100., bpm=120., *, angle_data=None, extra_actions=()):
    actions = list(extra_actions)
    if pause_beats:
        actions.append({'floor': 1, 'eventType': 'Pause', 'duration': pause_beats})
    raw = json.dumps({'angleData': angle_data or [0., 180., 0., 180., 0.],
        'settings': {'bpm': bpm, 'pitch': pitch, 'countdownTicks': 0},
        'actions': actions})
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


@pytest.mark.parametrize('beats, expected', [(0,60.), (1,40.), (2,30.), (4,20.)])
def test_pause_duration_affects_cbpm_not_tbpm(beats,expected):
    seg = segment(pause_beats=beats)
    floor = seg.chart.floors[1]
    visible = jrp_normal_bpm_for_floor(seg,1)
    assert visible.tbpm == pytest.approx(120.)
    assert visible.cbpm == pytest.approx(expected)
    assert visible.kps == pytest.approx(expected/60.)
    # Historical DMDOD 'Real BPM' EXCLUDES pause and hence differs.
    old_tbpm, old_real = hud_bpms_for_floor(seg,floor)
    assert old_tbpm == pytest.approx(120.)
    if beats:
        assert old_real != pytest.approx(visible.cbpm)


def test_pitch_changes_both_jrp_numbers_in_wall_clock_units():
    slow = segment(pause_beats=2, pitch=100)
    fast = segment(pause_beats=2, pitch=150)
    a = jrp_normal_bpm_for_floor(slow,1)
    b = jrp_normal_bpm_for_floor(fast,1)
    assert b.tbpm == pytest.approx(a.tbpm * 1.5)
    assert b.cbpm == pytest.approx(a.cbpm * 1.5)


def test_setspeed_cbpm_from_actual_next_entry_not_angle_shortcut():
    speed = {'floor': 1, 'eventType': 'SetSpeed', 'speedType': 'Bpm',
             'beatsPerMinute': 240., 'angleOffset': 90.}
    seg=segment(extra_actions=[speed])
    floor=seg.chart.floors[1]; nxt=seg.chart.floors[2]
    bpm=jrp_normal_bpm_for_floor(seg,1)
    assert bpm.cbpm == pytest.approx(60. / (nxt.target_time_s-floor.target_time_s))
    assert bpm.cbpm > 0


def test_jrp_display_only_changes_when_on_hit_or_show_called():
    seg = segment(pause_beats=2.)
    display = JrpNormalBpmDisplay(seg)
    assert display.visible is None
    first = display.on_show(0)
    assert display.visible is first
    # Floor changes do not directly change the rendered text in this model.
    assert display.visible is first
    second = display.on_hit_hook(1)
    assert second is display.visible and second is not first
    assert second.cbpm == pytest.approx(30.)
    assert second.printed == ('120','30','0.5')
    assert second.visible_numeric == (120.,30.,.5)
    display.reset()
    assert display.visible is None


def test_last_floor_fallback_and_invalid_floor_index():
    seg=segment()
    last=jrp_normal_bpm_for_floor(seg,len(seg.chart.floors)-1)
    assert last.cbpm==last.tbpm
    for i in (-1,len(seg.chart.floors)):
        with pytest.raises(ValueError): jrp_normal_bpm_for_floor(seg,i)


def test_source_revision_and_reading_nonfinite_guard():
    assert len(JRP_SOURCE_REVISION)==40
    seg=segment()
    for i in range(len(seg.chart.floors)):
        a=jrp_normal_bpm_for_floor(seg,i)
        assert isfinite(a.tbpm)
        assert a.cbpm is None or isfinite(a.cbpm)


def test_jrp_write_rounded_trims_zeros_and_observation_is_display_quantized():
    from dmdod.features.jrp_bpm_v1 import JrpBpmReading
    reading = JrpBpmReading(120., 12.34567, 12.34567/60.)
    assert reading.printed == ('120','12.35','0.21')
    assert reading.visible_numeric == (120.,12.35,.21)
    assert reading.cbpm != reading.visible_numeric[1]
