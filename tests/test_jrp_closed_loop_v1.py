"""JRP Timing's effects on real CPU Motor scorer; no trained Student supplied."""
from __future__ import annotations
import json
from dataclasses import fields
import pytest
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.evaluation.jrp_closed_loop_v1 import (
    run_jrp_closed_loop, ScheduledMotorFixture, IdlePolicy,
    CausalJrpPolicyInput,
)
from dmdod.motor.n_key import NKeyAction


def fixture():
    return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps({
        'angleData':[0,180,180,90,270,90,270],
        'settings':{'bpm':120,'pitch':100,'countdownTicks':0},
        'actions':[{'floor':1,'eventType':'Pause','duration':2}],
    }))))


PULSES=((.30,.70),(2.24,2.64),(2.80,3.20),(3.55,3.95),(4.55,4.95))


@pytest.mark.parametrize('delay, expected_cbpm_candidates', [(0,2),(1,2),(3,0)])
def test_actual_motor_and_delayed_visible_jrp(delay,expected_cbpm_candidates):
    s=run_jrp_closed_loop(fixture(), ScheduledMotorFixture(PULSES),
                          policy_label='scripted_fixture_NOT_STUDENT',visible_delay_frames=delay)
    assert s.actual_hits==5 and s.actual_misses==2
    assert s.physical_keydowns==5 and s.actual_display_hit_hooks==5
    assert s.unique_jrp_candidates == expected_cbpm_candidates
    assert s.false_jrp_candidates==0
    assert s.matched_jrp_candidates==expected_cbpm_candidates
    assert s.unique_jrp_forecasts==0  # baseline v1.4 already predicts these events
    assert s.actual_overload is False


def test_accepted_hit_is_missing_then_old_text_cannot_become_new_anchor():
    both=run_jrp_closed_loop(fixture(),ScheduledMotorFixture(PULSES),policy_label='scripted')
    skipped=run_jrp_closed_loop(fixture(),ScheduledMotorFixture(PULSES[1:]),policy_label='skip_first')
    assert skipped.actual_hits==both.actual_hits-1
    assert skipped.actual_display_hit_hooks==skipped.actual_hits
    assert skipped.actual_misses==both.actual_misses+1
    assert skipped.unique_jrp_candidates<both.unique_jrp_candidates
    assert skipped.false_jrp_candidates==0


def test_idle_motor_yields_no_accepted_hooks_or_cbpm_anchor():
    result=run_jrp_closed_loop(fixture(),IdlePolicy(),policy_label='idle')
    assert result.actual_hits==0
    assert result.actual_misses==7
    assert result.physical_keydowns==0
    assert result.actual_display_hit_hooks==0
    assert result.observed_text_changes==0
    assert result.unique_jrp_candidates==0


def test_policy_receives_public_immutable_values_not_segment_or_renderer():
    class Inspector:
        def __init__(self):self.n=0
        def reset(self):self.n=0
        def act(self,x):
            assert type(x) is CausalJrpPolicyInput
            assert len(x.hud)==263
            assert x.jrp_text is None or len(x.jrp_text)==3
            assert {f.name for f in fields(x)}=={'frame','hud','jrp_text','timing'}
            assert not hasattr(x,'segment') and not hasattr(x,'target_time_s')
            assert not hasattr(x,'floor_id') and not hasattr(x,'hit_count')
            self.n+=1
            return NKeyAction((-1.,)*8)
    inspector=Inspector()
    report=run_jrp_closed_loop(fixture(),inspector,policy_label='observer')
    assert inspector.n==report.frames


def test_bad_policy_action_is_rejected_before_motion():
    class BadPolicy:
        def reset(self):pass
        def act(self,x):return NKeyAction((float('nan'),)*8)
    with pytest.raises(ValueError,match='finite 8-key'):
        run_jrp_closed_loop(fixture(),BadPolicy(),policy_label='invalid')
