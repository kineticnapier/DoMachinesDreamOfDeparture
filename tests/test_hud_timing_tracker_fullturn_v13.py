"""Stage T: signed HUD motion may identify a 360° floor; aliases must abstain."""
import json
from bisect import bisect_right

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv, encode_n_key_hud_real_chart_observation
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_fullturn_v13 import (
    CausalTimingTracker, forecast_visible_window, visible_slot, CURRENT_SLOT,
)


def capture(pattern, bpm, event=None, floors=32):
    actions = ([{'floor':1, 'eventType':'Twirl'}] if event=='Twirl' else [])
    if event=='SetSpeed':
        actions.append({'floor':4, 'eventType':'SetSpeed','speedType':'Bpm','beatsPerMinute':bpm*2})
    if event=='Pause':
        actions.append({'floor':4,'eventType':'Pause','duration':1})
    text=json.dumps({'angleData':(pattern*((floors+len(pattern)-1)//len(pattern)))[:floors],
                     'settings':{'bpm':bpm,'pitch':100.,'countdownTicks':0},
                     'actions':actions})
    chart=build_playable_segment(compile_adofai(parse_adofai_text(text)))
    env=DiagnosticHudNKeyRealChartMotorEnv(
        chart,key_count=8,control_dt_s=.010,physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,ahead_floors=FLOORS.ahead_floors,tail_s=.04)
    obs=env.reset();idle=NKeyAction((0.,)*8)
    rows=[]
    for i in range(2200):
        rows.append((i*.01,tuple(encode_n_key_hud_real_chart_observation(obs))))
        r=env.step(idle);obs=r.observation
        if r.done:break
    else:
        raise AssertionError('fixture exceeded deterministic horizon')
    return chart,rows


def check_chart(chart,rows):
    times=tuple(float(t.episode_time_s) for t in chart.targets)
    tracker=CausalTimingTracker();prior=None
    matched_ids=set();elapsed=0; single_predictions=0; partial=0
    for frame,(now,hud) in enumerate(rows):
        prediction=forecast_visible_window(hud,previous_hud=prior)
        if prediction.eligible:
            single_predictions+=1
            partial+=prediction.status=='eligible_partial'
            end=prediction.covered_until_s if prediction.covered_until_s is not None else .030
            # Scores exclude sub-microsecond boundary events, which are
            # physically indistinguishable from the current HUD frame.
            lo=bisect_right(times,now+1e-6)
            hi=bisect_right(times,now+end+1e-6)
            expected=tuple(t-now for t in times[lo:hi])
            predicted=tuple(t for t in prediction.times_s if t>1e-6)
            assert len(expected)==len(predicted),(now,expected,predicted,prediction)
            assert all(abs(a-b)<=5e-5 for a,b in zip(expected,predicted))
        update=tracker.observe(frame,hud)
        for target in update.elapsed:
            assert target.local_id not in matched_ids
            matched_ids.add(target.local_id)
            elapsed+=1
            assert any(abs(t-target.due_s)<5e-5 for t in times),('phantom-event',target)
        prior=hud
    return elapsed,single_predictions,partial


@pytest.mark.parametrize('bpm,twirl',[
    (496.,False),(992.,True),(1984.,False),(2380.,True),
    (4000.,False),(6000.,True),(8000.,False),(12000.,True),(20000.,False),
])
def test_fullturn_scores_match_actual_hud_simulator(bpm,twirl):
    chart, rows=capture([0.,180.,90.,270.],bpm,'Twirl' if twirl else None)
    elapsed,single,partial=check_chart(chart,rows)
    assert elapsed>0 and single>0


def test_fullturn_real_history_recovers_additional_events():
    chart,rows=capture([0.,180.],992.,floors=32)
    tracked=CausalTimingTracker(); IDs=[]
    for step,(_,hud) in enumerate(rows):
        IDs.extend(x.local_id for x in tracked.observe(step,hud).elapsed)
    assert len(set(IDs))==len(IDs)
    assert len(IDs)>6


@pytest.mark.parametrize('event', ['SetSpeed','Pause'])
def test_unverified_future_speed_and_pause_are_not_extrapolated(event):
    chart,rows=capture([0.,180.,90.,270.],992.,event)
    elapsed,_,_=check_chart(chart,rows)
    assert elapsed>0


def test_a_full_turn_without_direction_history_does_not_hallucinate():
    _,rows=capture([0.,180.],496.)
    candidates=[]
    for _,hud in rows:
        fl=visible_slot(hud,CURRENT_SLOT)
        if fl[-1]>.5 and abs(fl[2]-fl[4])<1e-8 and abs(fl[3]-fl[5])<1e-8:
            candidates.append(hud)
    assert candidates
    assert any(forecast_visible_window(hud,previous_hud=None).status=='full_turn_no_motion_history'
               for hud in candidates)


def test_high_bpm_full_turn_signed_step_does_not_guess_alias():
    _,rows=capture([0.,180.],12000.)
    assert any(forecast_visible_window(rows[i][1],previous_hud=rows[i-1][1]).status
               in ('full_turn_motion_alias','full_turn_changed_floor') for i in range(1,len(rows)))


def test_tracker_only_uses_hud_not_oracle_labels():
    import ast,inspect
    from dmdod.planning import hud_timing_tracker_fullturn_v13 as module
    tree=ast.parse(inspect.getsource(module))
    assert all(not (node.module or '').startswith('dmdod.training')
               for node in ast.walk(tree) if isinstance(node,ast.ImportFrom))
