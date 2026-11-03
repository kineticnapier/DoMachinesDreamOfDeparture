"""D32 fixed parameter-group intervention invariants and failure controls."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d32_step413_parameter_group as d32
from a002_exploratory_ppo import ActorCritic


def synthetic_d28():
    p = [0.3233119547367096, 0.2952822148799896,
         0.21348761022090912, 0.16791823506355286]
    q = [0.32353976368904114, 0.29383787512779236,
         0.21458901464939117, 0.16803324222564697]
    obs = [0.0] * 8
    return {
        "schema": "research-a-v2-l0-d28-step413-policy-reward-v1",
        "classification": "valid_step413_archived_policy_reward_audit",
        "source_run": 41515190995, "source_artifact": 13175252296,
        "arm": "a0", "seed": 320403, "targets": 16,
        "step413": {
            "before": {"step": 413, "time_s": 4.13, "observation": obs,
                       "action_probabilities": p, "sampled_action": 0,
                       "argmax_action": 0},
            "after_to_none": {"step": 413, "time_s": 4.13,
                              "observation": obs.copy(),
                              "action_probabilities": q,
                              "sampled_action": 2, "argmax_action": 0},
        },
        "step413_comparison": {"observation_exact": True,
                               "pre_motor_exact": True,
                               "pre_diagnostics_exact": True},
        "extra_at_4_327": {"before": None,
                           "after_to_none": {"immediate_reward": -0.05}},
    }


def two_models():
    torch.manual_seed(1)
    before = ActorCritic()
    after = copy.deepcopy(before)
    with torch.no_grad():
        after.shared[0].weight.add_(0.01)
        after.actor.bias.add_(0.05)
        after.critic.bias.add_(1.0)
    return before, after


def test_validate_d28_fixed_observation():
    obs, p, q = d32.validate_d28(synthetic_d28())
    assert len(obs) == 8
    assert math.isclose(p[2] + p[3], 0.381405845284462, abs_tol=1e-14)
    assert math.isclose(q[2] + q[3], 0.38262225687503815, abs_tol=1e-14)


@pytest.mark.parametrize("change", [
    lambda s: s.update({"seed": 1}),
    lambda s: s.update({"source_run": 0}),
    lambda s: s["step413_comparison"].update({"observation_exact": False}),
    lambda s: s["step413"]["after_to_none"].update({"observation": [1.0]*8}),
    lambda s: s["step413"]["before"].update({"action_probabilities": [0.1]*4}),
    lambda s: s["step413"]["before"].update({"sampled_action": 2}),
    lambda s: s["extra_at_4_327"]["after_to_none"].update({"immediate_reward": 0}),
])
def test_reject_corrupt_archived_d28(change):
    s = synthetic_d28()
    change(s)
    with pytest.raises(ValueError):
        d32.validate_d28(s)


def test_group_change_count_and_name_contract():
    before, after = two_models()
    changes = d32.group_changes(before, after)
    assert sum(x["elements"] for x in changes.values()) == 1509
    assert all(x["changed_elements"] > 0 for x in changes.values())
    with pytest.raises(ValueError):
        d32.group_for("new_layer.weight")


def test_group_swap_keeps_unselected_parameter_bits():
    before, after = two_models()
    mixed = d32.mixed_model(before, after, shared=False, actor=True, critic=False)
    bm, am, mm = (dict(m.named_parameters()) for m in (before, after, mixed))
    for name in bm:
        expected = am[name] if name.startswith("actor.") else bm[name]
        assert d32.d12.tensor_bits(mm[name]) == d32.d12.tensor_bits(expected)


def test_critic_negative_control_bit_exact():
    before, after = two_models()
    m11 = d32.mixed_model(before, after, shared=True, actor=True, critic=False)
    m111 = d32.mixed_model(before, after, shared=True, actor=True, critic=True)
    logits11, p11, _ = d32.infer(m11, [0.0] * 8)
    logits111, p111, _ = d32.infer(m111, [0.0] * 8)
    assert torch.equal(logits11, logits111)
    assert torch.equal(p11, p111)


def test_four_condition_probabilities_normalized():
    before, after = two_models()
    for shared, actor in ((False, False), (True, False),
                          (False, True), (True, True)):
        model = d32.mixed_model(before, after, shared=shared, actor=actor)
        _, _, output = d32.infer(model, [0.0] * 8)
        assert len(output["probabilities"]) == 4
        assert abs(sum(output["probabilities"]) - 1) < 1e-6


def test_shapley_reconstructs_total_difference_with_interaction():
    c = d32.contributions(0.38, 0.39, 0.385, 0.405)
    assert math.isclose(c["delta"], 0.025, abs_tol=1e-14)
    assert math.isclose(c["shared_symmetric_contribution"], 0.015,
                        abs_tol=1e-14)
    assert math.isclose(c["actor_symmetric_contribution"], 0.01,
                        abs_tol=1e-14)
    assert math.isclose(c["interaction"], 0.01, abs_tol=1e-14)
    assert c["reconstruction_abs_error"] <= 1e-12


def test_zero_shift_is_valid_not_successful_learning():
    c = d32.contributions(0.38, 0.38, 0.38, 0.38)
    assert c["delta"] == c["shared_symmetric_contribution"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_reject_nonfinite_or_out_of_range_probability(bad):
    with pytest.raises(ValueError):
        d32.contributions(0.38, bad, 0.39, 0.40)


def test_reject_wrong_observation_shape():
    before, _ = two_models()
    with pytest.raises(ValueError):
        d32.infer(before, [0.0] * 7)


def test_original_model_bits_unchanged_after_swap():
    before, after = two_models()
    old_b, old_a = d32.d18.weights_bits(before), d32.d18.weights_bits(after)
    d32.mixed_model(before, after, shared=True, actor=False, critic=True)
    assert d32.d18.weights_bits(before) == old_b
    assert d32.d18.weights_bits(after) == old_a
