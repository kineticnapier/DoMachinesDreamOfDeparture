from __future__ import annotations

import json

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.jrp_overlay_diagnostic import JrpOverlayDiagnosticNKeyEnv
from dmdod.motor.n_key import NKeyAction


def fixture():
    raw=json.dumps({'angleData':[0,180,180,90,270],
         'settings':{'bpm':120,'pitch':100,'countdownTicks':0},
         'actions':[{'floor':1,'eventType':'Pause','duration':2}]})
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


def test_floor_advance_without_hit_does_not_change_jrp_text():
    env=JrpOverlayDiagnosticNKeyEnv(fixture(),key_count=8,tail_s=.5)
    env.reset()
    initial=env.overlay_visible
    for i in range(100):
        state=env.step(NKeyAction((-1.,)*8))
        assert env.overlay_visible == initial
        if state.done: break
    assert env.overlay_hit_hook_count==0
    assert env.stats.hits==0


def test_real_physical_keydown_scoring_updates_renderer_at_hit_event():
    env=JrpOverlayDiagnosticNKeyEnv(fixture(),key_count=8,tail_s=.5)
    env.reset()
    initial=env.overlay_visible
    changed=[]
    for i in range(100):
        controls=(-1.,)*8
        if .30<=i*.01<.70:
            controls=(1.,)+(-1.,)*7
        result=env.step(NKeyAction(controls))
        if env.overlay_visible!=initial:
            changed.append((i,env.overlay_visible,env.stats.hits))
            initial=env.overlay_visible
        if result.done:break
    assert env.stats.hits==1
    assert env.overlay_hit_hook_count==1
    assert changed and changed[0][1]==('120','30','0.5')
    assert changed[0][2]==1


def test_reset_restores_initial_display_and_hook_count():
    env=JrpOverlayDiagnosticNKeyEnv(fixture(),key_count=8,tail_s=.5)
    env.reset()
    original=env.overlay_visible
    env._jrp_display.on_hit_hook(1)  # renderer-only test insertion
    assert env.overlay_visible!=original
    env.reset()
    assert env.overlay_visible==original
    assert env.overlay_hit_hook_count==0


def test_physical_actions_and_scores_identical_to_original_n_key_env():
    from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv
    segment=fixture()
    normal=DiagnosticHudNKeyRealChartMotorEnv(segment,key_count=8,tail_s=.5)
    hooked=JrpOverlayDiagnosticNKeyEnv(segment,key_count=8,tail_s=.5)
    normal.reset();hooked.reset()
    for i in range(125):
        cmd=(1.,)+(-1.,)*7 if .30<=i*.01<.70 else (-1.,)*8
        left=normal.step(NKeyAction(cmd))
        right=hooked.step(NKeyAction(cmd))
        assert left.observation==right.observation
        assert left.reward==right.reward
        assert left.done==right.done
        if left.done:break
    assert normal.stats==hooked.stats
