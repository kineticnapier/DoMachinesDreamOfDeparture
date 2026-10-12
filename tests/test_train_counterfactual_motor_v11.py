"""CPU-only logic and causal-replay guards for Train-only v1.1 diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from dmdod.training.train_counterfactual_motor import MotorRollout
from dmdod.training.train_counterfactual_motor_v11 import (
    Probe, altered_rollout, balanced_subset, candidate_pool, coverage,
    followup_checks, gate_checks, key_event_sources, short_status,
)


def fake_rollout(*, used=(0,), hits=1, misses=0, early=3,
                 events=None, keydowns=3, overload=True,
                 duration=.4, terminated=False, frames=40):
    if events is None:
        events = [('left_2', .076), ('right_3', .102), ('right_4', .143)]
    return MotorRollout(
        observations=torch.zeros((frames,263)),
        raw_actions=torch.zeros((frames,8)),
        executed_actions=torch.zeros((frames,8)),
        used=tuple(used), missed=(),
        keydowns=tuple(('left_2', .050+i*.03) for i in range(keydowns)),
        early=tuple(events), hits=hits, misses=misses,
        too_early=early, overloaded=overload,
        duration_s=duration, terminated=terminated,
    )


def test_no_early_source_results_in_zero_cases():
    base=fake_rollout(events=[])
    assert candidate_pool(base,dt=.01,key_count=8)==[]
    assert balanced_subset([],budget=128,seed=4)==[]


def test_early_source_is_train_oracle_only_and_unknown_keys_fail_closed():
    assert len(key_event_sources(fake_rollout(),key_count=8))==3
    with pytest.raises(RuntimeError, match='Unknown physical key'):
        key_event_sources(fake_rollout(events=[('unknown',.05)]),key_count=8)


def test_unique_effective_cases_include_early_pair_and_all_eight():
    base=fake_rollout(frames=8)
    pool=candidate_pool(base,dt=.01,key_count=8)
    assert {case.kind for case in pool} == {'single','paired','all_8'}
    assert len({(c.keys,c.start,c.duration,c.delta) for c in pool})==len(pool)
    assert any(c.end>base.raw_actions.shape[0] for c in pool)
    assert all(c.start<base.raw_actions.shape[0] for c in pool)


def test_budget_balanced_over_amplitude_duration_key_groups():
    pool=candidate_pool(fake_rollout(),dt=.01,key_count=8)
    chosen=balanced_subset(pool,budget=128,seed=2026)
    assert len(chosen)==len(set(chosen))==128
    assert chosen==balanced_subset(pool,budget=128,seed=2026)
    assert {abs(c.delta) for c in chosen}=={.02,.05,.10,.20}
    assert {c.duration for c in chosen}=={1,2,4,8}
    assert {c.kind for c in chosen}=={'single','paired','all_8'}
    summary=coverage(pool,chosen)
    assert all(summary[k]['sampled'] for k in ('groups','delta','duration','lead'))


def test_short_gates_report_independent_failures():
    base=fake_rollout()
    teacher=fake_rollout(hits=3,used=(0,1,2),overload=False,events=[],early=0)
    alt=fake_rollout(used=(),hits=0,keydowns=0,early=3,overload=True)
    gates=gate_checks(base,alt,teacher)
    assert gates['no_overload'] is False
    assert gates['has_keydown'] is False
    assert gates['baseline_hit_ids_retained'] is False
    assert gates['too_early_lower'] is False
    assert gates['hits_not_lower'] is False
    assert short_status(gates,baseline_overload=True)=='rejected_short'


def test_safe_early_reduction_is_diagnostic_review_not_discarded():
    base=fake_rollout(overload=False,misses=0,early=3)
    teacher=fake_rollout(overload=False,hits=1,early=0)
    alt=fake_rollout(overload=False,hits=1,early=2)
    g=gate_checks(base,alt,teacher)
    assert all(g[k] for k in g if k!='safe_hit_gain')
    assert g['safe_hit_gain'] is False
    assert short_status(g,baseline_overload=False)=='safe_early_reduction_review_only'


def test_followup_rejects_deferred_too_early_and_overload():
    short=fake_rollout(early=1,overload=False)
    long=fake_rollout(early=4,overload=True,duration=.55)
    teacher_short=fake_rollout(early=0,overload=False)
    teacher_long=fake_rollout(early=0,overload=False,duration=.8)
    gates=followup_checks(short,long,teacher_short,teacher_long,horizon_s=.8)
    assert gates['reached_horizon'] is False
    assert gates['no_overload'] is False
    assert gates['late_early_within_expert_plus_one'] is False


def test_followup_accepts_real_continuation_with_hits():
    short=fake_rollout(early=1,used=(0,1),hits=2,overload=False)
    long=fake_rollout(early=2,used=(0,1,2),hits=3,overload=False,
                      duration=.8,keydowns=3)
    teacher_short=fake_rollout(early=0,hits=2,overload=False)
    teacher_long=fake_rollout(early=1,hits=3,overload=False,duration=.8)
    assert all(followup_checks(short,long,teacher_short,teacher_long,horizon_s=.8).values())


def test_followup_rejects_suppression_of_future_hit():
    short=fake_rollout(early=1,used=(0,),hits=1,overload=False)
    long=fake_rollout(early=1,used=(0,),hits=1,overload=False,duration=.8)
    teacher_short=fake_rollout(early=0,hits=1,overload=False)
    teacher_long=fake_rollout(early=1,hits=4,overload=False,duration=.8)
    gates=followup_checks(short,long,teacher_short,teacher_long,horizon_s=.8)
    assert gates['expert_relative_hits'] is False


def test_incorrect_budget_or_timestep_are_errors():
    with pytest.raises(ValueError):
        balanced_subset([],budget=0,seed=0)
    with pytest.raises(ValueError):
        candidate_pool(fake_rollout(),dt=0,key_count=8)


def test_changed_motor_observation_affects_next_student_action(monkeypatch):
    import dmdod.training.train_counterfactual_motor_v11 as module

    class Model:
        def eval(self):
            return self
        def initial_state(self,device):
            return torch.tensor([0.],device=device)
        def forward_step(self,x,state):
            return torch.cat((x[:1],torch.zeros(7,device=x.device))),None,None,state

    class Environment:
        def __init__(self):
            self.time=0.0
            self.current=torch.zeros(263)
            self._used=[False]
            self._missed=[False]
            self._trace_physical_keydowns=[]
            self._trace_too_early=[]
            self._hits=0;self._misses=0;self._too_early=0
            self._overload=SimpleNamespace(overloaded=False)
        def reset(self):
            return self.current
        def step(self,action):
            self.current=self.current.clone()
            self.current[0]=action.values[0]
            self.time+=.01
            return SimpleNamespace(observation=self.current,done=self.time>=.02)
        def privileged_episode_time_s(self):
            return self.time

    monkeypatch.setattr(module,'_environment',lambda *a:Environment())
    monkeypatch.setattr(module,'encode_n_key_hud_real_chart_observation',
                        lambda obs:tuple(float(v) for v in obs))
    prepared=SimpleNamespace(device=torch.device('cpu'),control_dt_s=.01)
    changed=altered_rollout(prepared,Model(),SimpleNamespace(),
                           Probe((0,),0,1,.2,1),horizon_s=.04)
    assert changed.raw_actions.shape==(2,8)
    assert changed.raw_actions[0,0]==0
    assert changed.executed_actions[0,0]==pytest.approx(.2)
    assert float(changed.raw_actions[1,0])==pytest.approx(float(torch.tanh(torch.tensor(.2))))
