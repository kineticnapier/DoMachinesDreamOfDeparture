"""D30 fixed 4096-seed audit tests; no PPO, Motor or learning."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d30_common_rng_seed_sensitivity as d30


def probs(a=None, b=None):
    return (torch.tensor(a or [0.3, 0.3, 0.2, 0.2], dtype=torch.float32),
            torch.tensor(b or [0.3, 0.3, 0.2, 0.2], dtype=torch.float32))


def test_identical_distribution_never_disagrees():
    p, q = probs()
    result = d30.sensitivity(p, q, count=64)
    assert result["disagreement_count"] == 0
    assert result["total_variation_distance"] == 0
    assert result["total_draws"] == 128


def test_disjoint_one_hot_distributions_always_disagree():
    p, q = probs([1.0, 0, 0, 0], [0, 0, 1.0, 0])
    result = d30.sensitivity(p, q, count=64)
    assert result["disagreement_count"] == 64
    assert result["none_to_left"] == 64
    assert result["total_variation_distance"] == 1


def test_same_seed_is_deterministic_and_rng_is_coupled():
    p, q = probs([0.32, 0.29, 0.22, 0.17],
                 [0.33, 0.28, 0.22, 0.17])
    assert d30.sample_pair(p, q, 123) == d30.sample_pair(p, q, 123)
    assert d30.sensitivity(p, q, count=128) == d30.sensitivity(p, q, count=128)


def test_transition_marginals_and_changed_seed_log():
    p, q = probs([0.25] * 4, [0.1, 0.2, 0.3, 0.4])
    r = d30.sensitivity(p, q, count=64)
    matrix = r["transition_matrix_before_rows_after_columns"]
    assert sum(map(sum, matrix)) == 64
    assert [sum(row) for row in matrix] == r["before_action_counts"]
    assert [sum(matrix[i][j] for i in range(4)) for j in range(4)] == r["after_action_counts"]
    assert len(r["changed_seeds"]) == r["disagreement_count"]


@pytest.mark.parametrize("p,q", [
    ([0.5, 0.4, -0.1, 0.2], [0.25] * 4),
    ([0.2, 0.2, 0.2, 0.2], [0.25] * 4),
    ([0.25] * 4, [float("nan"), 0.25, 0.25, 0.25]),
])
def test_invalid_probabilities_are_rejected(p, q):
    with pytest.raises(ValueError):
        d30.sensitivity(torch.tensor(p, dtype=torch.float32),
                        torch.tensor(q, dtype=torch.float32), count=8)


def test_invalid_seed_count_rejected():
    p, q = probs()
    with pytest.raises(ValueError):
        d30.sensitivity(p, q, count=0)


def test_analyze_checks_d29_positive_control(monkeypatch):
    fake = {
        "pivot_rng_state_exact": True,
        "replay": {"before": {"matched_steps": 480},
                   "after_to_none": {"matched_steps": 480}},
        "pivot_cross_draws": {"before": {"before": 0, "after_to_none": 2}},
        "pivot_probability_float32": {
            "before": [0.3233119547, 0.2952822149, 0.2134876102, 0.1679182351],
            "after_to_none": [0.3235397637, 0.2938378751, 0.2145890146, 0.1680332422],
        },
    }
    monkeypatch.setattr(d30.d29, "analyze", lambda _source, _traces: fake)
    result = d30.analyze({}, {})
    assert result["classification"] == "valid_seed_sensitivity_audit"
    assert result["sampling"]["seed_count"] == 4096
    assert result["sampling"]["total_draws"] == 8192
    assert result["new_physical_keydowns"] == 0
    broken = copy.deepcopy(fake)
    broken["pivot_rng_state_exact"] = False
    monkeypatch.setattr(d30.d29, "analyze", lambda _source, _traces: broken)
    with pytest.raises(ValueError, match="pivot RNG mismatch"):
        d30.analyze({}, {})
