"""JRP Timing's effects on real CPU Motor scorer; no trained Student supplied."""
from __future__ import annotations
import json
from dataclasses import fields
import pytest
import torch
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.evaluation.jrp_closed_loop_v1 import (
    run_jrp_closed_loop, ScheduledMotorFixture, IdlePolicy,
    CausalJrpPolicyInput,
    FrozenPrestigeMotorPolicy,
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


def test_hidden_jrp_is_matched_baseline_with_same_physical_motor_outcome():
    visible=run_jrp_closed_loop(fixture(),ScheduledMotorFixture(PULSES),
                                policy_label='visible',jrp_visible=True)
    hidden=run_jrp_closed_loop(fixture(),ScheduledMotorFixture(PULSES),
                               policy_label='hidden',jrp_visible=False)
    assert (hidden.actual_hits,hidden.actual_misses,hidden.actual_overload)==(
        visible.actual_hits,visible.actual_misses,visible.actual_overload)
    assert hidden.jrp_visible is False and visible.jrp_visible is True
    assert hidden.unique_jrp_forecasts==0 and hidden.mean_jrp_confidence is None
    assert hidden.matched_combined==hidden.matched_baseline_elapsed
    assert visible.matched_combined>=hidden.matched_combined


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


class _TinyPrestige(torch.nn.Module):
    input_dim = 263
    key_count = 8

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step_with_timing(self, x, state):
        assert x.shape == (263,)
        mean = torch.linspace(-2., 2., 8, device=x.device)
        raw = torch.tensor((0., 0., 0., 1., 0., 0., 0., 0., 0., 0.), device=x.device)
        return mean, torch.ones(8, device=x.device), torch.tensor(0.), state + 1., raw


def test_frozen_prestige_adapter_preserves_263d_and_records_causal_timing():
    policy = FrozenPrestigeMotorPolicy(_TinyPrestige())
    policy.reset()
    update = type('Timing', (), {})()
    observation = CausalJrpPolicyInput(7, (0.,) * 263, ('120', '240', '4'), update)
    action = policy.act(observation)
    assert action.values == pytest.approx(tuple(torch.tanh(torch.linspace(-2., 2., 8))))
    assert len(policy.estimates) == 1
    estimate = policy.estimates[0]
    assert estimate.frame == 7
    assert estimate.predicted_due_s == pytest.approx(.170)
    assert estimate.confidence == pytest.approx(.5)
    assert estimate.chord_size == 2


def test_frozen_prestige_adapter_rejects_changed_legacy_contract():
    model = _TinyPrestige()
    model.input_dim = 264
    with pytest.raises(ValueError, match='263D / 8-key'):
        FrozenPrestigeMotorPolicy(model)
