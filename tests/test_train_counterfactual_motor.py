"""CPU-only semantic tests; CUDA/full dataset tests run on Windows."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from dmdod.training.train_counterfactual_motor import (
    LocalIntervention, MotorRollout, MotorCorrection,
    accept_counterfactual, assert_train_rollout_matches_dagger,
    auxiliary_loss, candidate_cases, patched_action,
)


def fake_rollout(*, used=(0,), hits=1, missed=(), misses=0, early=2,
                 keydowns=2, overload=False, duration=.4):
    return MotorRollout(
        observations=torch.zeros((40, 263)),
        raw_actions=torch.zeros((40, 8)),
        executed_actions=torch.zeros((40, 8)),
        used=tuple(used), missed=tuple(missed),
        keydowns=tuple(("left_2",i*.1) for i in range(keydowns)),
        early=tuple(("left_2", .080) for _ in range(early)),
        hits=hits, misses=misses, too_early=early,
        overloaded=overload, duration_s=duration,terminated=False,
    )


def test_patch_is_local_bounded_and_nonmutating():
    raw=(0.9,0.0,0.0,-0.9)
    case=LocalIntervention(key=0,start=2,duration=2,delta=.2)
    assert patched_action(raw,case,1)==raw
    assert patched_action(raw,case,2)==(1.,0.,0.,-.9)
    assert patched_action(raw,case,3)==(1.,0.,0.,-.9)
    assert patched_action(raw,case,4)==raw
    assert raw[0]==0.9


def test_train_candidates_bounded_and_cover_delta_durations():
    base=fake_rollout()
    cases=candidate_cases(base,dt=.01,key_count=8,max_cases=128)
    assert len(cases)<=128
    assert {abs(c.delta) for c in cases}=={.02,.05,.10,.20}
    assert {c.duration for c in cases}=={1,2,4,8}
    assert all(c.start>=0 and c.end<=40 for c in cases)
    assert len(cases)==len(set(cases))


def test_no_early_events_means_no_oracle_candidates():
    assert candidate_cases(fake_rollout(early=0),dt=.01,key_count=8)==[]


def test_reject_suppression_and_lost_hits():
    base=fake_rollout(used=(0,1),hits=2,overload=True,early=3)
    teacher=fake_rollout(hits=3,used=(0,1,2),early=0)
    assert accept_counterfactual(base,fake_rollout(keydowns=0,early=0),teacher)[0] is False
    assert accept_counterfactual(base,fake_rollout(used=(1,2),hits=2,early=0),teacher)[0] is False
    assert accept_counterfactual(base,fake_rollout(used=(0,1),hits=2,early=0),teacher)[0] is False


def test_accept_keeps_hits_and_decreases_early():
    base=fake_rollout(used=(0,),hits=1,overload=True,early=3)
    teacher=fake_rollout(used=(0,1,2),hits=3,early=0)
    alt=fake_rollout(used=(0,1,2),hits=3,early=1)
    yes, reason=accept_counterfactual(base,alt,teacher)
    assert yes and reason=="physical_improvement_train_only"


def test_teacher_reference_must_be_meaningful():
    base=fake_rollout(overload=True)
    teacher=fake_rollout(overload=True,early=0)
    alt=fake_rollout(early=0,hits=3,used=(0,1,2))
    assert accept_counterfactual(base,alt,teacher)[1]=="expert_reference_invalid"


def test_exact_train_dagger_state_check_fails_closed():
    base=fake_rollout()
    dagger=SimpleNamespace(frames=40, observations=base.observations.clone(),
                           student_actions=base.raw_actions.clone())
    assert_train_rollout_matches_dagger(base,dagger)
    dagger.observations[0,1]=1.
    with pytest.raises(RuntimeError, match="observations differ"):
        assert_train_rollout_matches_dagger(base,dagger)


class FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=torch.nn.Parameter(torch.zeros(8))

    def initial_state(self, device):
        return torch.zeros((1,),device=device)

    def forward_sequence(self, x, state):
        return self.weights.expand(x.shape[0], 8), None, state


def test_aux_loss_only_perturbed_key_and_frames():
    policy=FixedPolicy()
    obs=torch.zeros((10,263))
    target=torch.zeros((10,8))
    target[2:4,3]=.25
    correction=MotorCorrection(0,LocalIntervention(3,2,2,.25),obs,target,0,1,1)
    loss=auxiliary_loss(policy,correction)
    assert loss.detach().item()==pytest.approx(.0625)
    loss.backward()
    assert policy.weights.grad[3] == pytest.approx(-.5)
    assert torch.all(policy.weights.grad[:3]==0)
    assert torch.all(policy.weights.grad[4:]==0)


def test_aux_rejects_short_counterfactual_trajectory():
    policy=FixedPolicy()
    correction=MotorCorrection(0,LocalIntervention(3,9,2,.25),
                               torch.zeros((10,263)),torch.zeros((10,8)),0,1,1)
    with pytest.raises(RuntimeError,match="corrected frames were lost"):
        auxiliary_loss(policy,correction)
