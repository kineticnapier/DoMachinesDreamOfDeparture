from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_oracle_fingering_v02 as teacher
from dmdod.training import fingering_network_v01 as network


class FixedRanker(network.FingeringRanker):
    def forward(self, context, candidates):
        return torch.arange(len(candidates), 0, -1, dtype=torch.float32)


def test_alpha_zero_is_bit_exact_free_selection():
    model = FixedRanker()
    allocations = teacher.finger_options(1)
    eligible = [True] * len(allocations)
    context = torch.zeros(network.CONTEXT_DIM)
    free_index, free_scores = network.choose_candidate(model, context, allocations, eligible)
    styled_index, styled_scores, combined = network.choose_candidate_styled(
        model, context, allocations, eligible, alpha=0.0)
    assert styled_index == free_index
    assert styled_scores == free_scores
    assert all(value is None for value in combined)


def test_maximum_style_prefers_index_proxy_but_does_not_ban_auxiliary():
    model = FixedRanker()
    allocations = teacher.finger_options(1)
    context = torch.zeros(network.CONTEXT_DIM)
    eligible = [True] * len(allocations)
    selected, _, _ = network.choose_candidate_styled(
        model, context, allocations, eligible, alpha=1.0)
    assert allocations[selected][0].axis.endswith("_1")

    eligible = [finger[0].axis.endswith("_5") for finger in allocations]
    selected, _, _ = network.choose_candidate_styled(
        model, context, allocations, eligible, alpha=1.0)
    assert allocations[selected][0].axis.endswith("_5")


def test_style_cost_orders_proxy_classes_and_alpha_is_bounded():
    choices = {value[0].axis: value for value in teacher.finger_options(1)}
    assert network.human_proxy_style_cost(choices["left_1"]) < network.human_proxy_style_cost(choices["left_2"])
    assert network.human_proxy_style_cost(choices["left_4"]) < network.human_proxy_style_cost(choices["left_5"])
    model = FixedRanker()
    try:
        network.choose_candidate_styled(model, torch.zeros(network.CONTEXT_DIM),
                                        tuple(choices.values()), [True] * len(choices), alpha=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid alpha accepted")
