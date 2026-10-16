from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.features.jrp_overlay_v2 import (
    JrpFloorState, JrpOverlayRenderer, JrpDisplayedHud,
    SOURCE_90_RAD, _pseudo,
)


def floors(angles, *, dt=.04, speeds=None):
    return tuple(JrpFloorState(i,i*dt,a,(speeds[i] if speeds else 1.))
                 for i,a in enumerate(angles))


def chart(*, bpm=120., pause=0., speed=None):
    actions=[]
    if pause:
        actions.append({'floor':1,'eventType':'Pause','duration':pause})
    if speed:
        actions.append({'floor':2,'eventType':'SetSpeed','speedType':'Bpm',
                        'beatsPerMinute':speed})
    raw=json.dumps({'angleData':[0.,180.,0.,180.,0.,180.],
                    'settings':{'bpm':bpm,'pitch':100.,'countdownTicks':0},
                    'actions':actions})
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


def test_normal_mode_update_on_show_and_hit_but_not_time_or_floor_transition():
    seg=chart(pause=2.,speed=240.)
    r=JrpOverlayRenderer.from_segment(seg,mode='normal')
    assert r.visible is None
    first=r.on_show(0)
    assert first[0]=='120'
    assert r.on_floor_advance(1)==first
    assert r.on_floor_advance(2)==first
    after_hit=r.on_hit_hook(1)
    assert after_hit==('120','30','0.5')
    assert r.on_floor_advance(2)==after_hit
    after_second_hit=r.on_hit_hook(2)
    assert after_second_hit[0]=='240'
    assert r.diagnostic.source_floor==2
    r.reset()
    assert r.visible is None


def test_pseudo_aggregation_and_cached_text_until_next_group():
    f=floors((.1,.2,.35,.5,.45,.3,.2,.1))
    r=JrpOverlayRenderer(f,base_bpm=1000.,mode='jongyeol')
    assert r.on_show(0)==('1000','375','25')
    assert r.diagnostic.pseudo is True
    assert r.diagnostic.pseudo_count==4
    first=r.diagnostic
    for i in (1,2,3):
        assert r.on_hit_hook(i)==first.visible_text
        assert r.diagnostic is first
    assert r.on_hit_hook(4)!=first.visible_text
    assert r.diagnostic.source_floor==4


def test_same_floor_sample_normal_vs_pseudo_have_distinct_cbpm_kps():
    f=floors((.1,.2,.35,.5,.45,.3,.2,.1))
    normal=JrpOverlayRenderer(f,base_bpm=1000.,mode='normal').on_show(0)
    pseudo=JrpOverlayRenderer(f,base_bpm=1000.,mode='jongyeol').on_show(0)
    assert normal==('1000','1500','25')
    assert pseudo==('1000','375','25')
    # The apparent 60/CBPM horizon is .16s rather than one .04s floor:
    assert 60./float(pseudo[1])==pytest.approx(.16)
    assert 60./float(normal[1])==pytest.approx(.04)


def test_pseudo_disable_or_low_bpm_or_not_game_scene_falls_back_to_normal():
    f=floors((.1,.2,.35,.5,.45,.3,.2,.1))
    for kwargs in ({'base_bpm':199.,'mode':'jongyeol'},
                   {'base_bpm':1000.,'mode':'jongyeol','check_pseudo':False},
                   {'base_bpm':1000.,'mode':'jongyeol','is_game_scene':False}):
        r=JrpOverlayRenderer(f,**kwargs)
        assert r.on_show(0)[1]=='1500'
        assert r.diagnostic.pseudo is False


def test_repeated_short_angle_branch_suppresses_repeated_updates():
    # JOverlay.CheckPseudo fast repeated same angle branch caches the skip end
    f=floors((.3,.3,.3,.3,.3,.6,.9,.2))
    r=JrpOverlayRenderer(f,base_bpm=1000,mode='jongyeol')
    first=r.on_show(1)
    assert r.diagnostic.pseudo is False
    assert r._pseudo_floor>=2
    initial=r.diagnostic
    for i in range(2,r._pseudo_floor+1):
        assert r.on_hit_hook(i)==first
        assert r.diagnostic==initial


def test_show_clears_pseudo_skip_cache():
    f=floors((.1,.2,.35,.5,.45,.3,.2,.1))
    r=JrpOverlayRenderer(f,base_bpm=1000,mode='jongyeol')
    r.on_show(0)
    assert r._pseudo_floor==3
    r.on_show(1)
    assert r.diagnostic.source_floor==1


def test_90degree_exception_and_edge_floor_midspin_cases_are_guarded():
    angles=(SOURCE_90_RAD,)*5
    f=floors(angles)
    p=_pseudo(f,0,1000.,1.,is_game_scene=True)
    assert p.cbpm is None
    assert p.count==0
    f2=(JrpFloorState(0,0.,.1,1.,False),
        JrpFloorState(1,.02,.4,1.,True),
        JrpFloorState(2,.04,.7,1.,False),
        JrpFloorState(3,.06,.9,1.,False))
    r=JrpOverlayRenderer(f2,base_bpm=1000,mode='jongyeol')
    assert len(r.on_show(0))==3


def test_immutable_visible_snapshot_and_hidden_metadata_separation():
    r=JrpOverlayRenderer(floors((.1,.2,.35,.5,.45)),base_bpm=1000,mode='jongyeol')
    text=r.on_show(0)
    assert isinstance(text,tuple)
    assert all(isinstance(v,str) for v in text)
    assert all('floor' not in v for v in text)
    assert isinstance(r.diagnostic,JrpDisplayedHud)
    with pytest.raises(FrozenInstanceError):
        r.diagnostic.source_floor=9


@pytest.mark.parametrize('bad',[-1,4,True,'1'])
def test_rejects_invalid_floor_hook_index(bad):
    r=JrpOverlayRenderer(floors((.1,.2,.35,.5)),base_bpm=1000)
    with pytest.raises(ValueError):r.on_hit_hook(bad)


def test_live_projection_matches_previous_normal_bpm_renderer_on_standard_pause():
    from dmdod.features.jrp_bpm_v1 import JrpNormalBpmDisplay
    seg=chart(pause=2.)
    old=JrpNormalBpmDisplay(seg)
    current=JrpOverlayRenderer.from_segment(seg,mode='normal')
    for i in (0,1,2):
        assert old.on_show(i).printed==current.on_show(i)


def test_pseudo_uses_compiled_chart_only_on_renderer_side():
    seg=chart(bpm=600.,pause=2.)
    r=JrpOverlayRenderer.from_segment(seg,mode='jongyeol')
    strings=r.on_show(0)
    assert strings is not None and len(strings)==3
    assert all(float(x)>0 for x in strings)
