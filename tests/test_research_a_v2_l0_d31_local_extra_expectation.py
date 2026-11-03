"""D31 conditional A0 physical KeyDown expectation: source and failure controls."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d31_local_extra_expectation as d31


def sources():
    series = {name: {"keydown_count": n} for name, n in [
        ("before", 2), ("after_to_none", 3),
        ("before_step413_from_after", 3), ("after_step413_from_before", 2)]}
    d27 = {
        "schema": "research-a-v2-l0-d27-step413-single-command-v1",
        "classification": "valid_step413_single_command_intervention",
        "source_run": 41515190995, "source_artifact": 13175252296,
        "arm": "a0", "seed": 320403, "intervention_step": 413,
        "later_unchanged_step": 453, "negative_controls_exact": True,
        "pre_step413_states_exact": True,
        "step413_commands": {"before_action": 0, "after_to_none_action": 2},
        "series": series,
        "comparisons": {
            "before_step413_from_after": {
                "baseline": "before", "baseline_has_extra_keydown_at_4_327s": False,
                "has_extra_keydown_at_4_327s": True, "physical_keydown_count_delta": 1},
            "after_step413_from_before": {
                "baseline": "after_to_none",
                "baseline_has_extra_keydown_at_4_327s": True,
                "has_extra_keydown_at_4_327s": False, "physical_keydown_count_delta": -1},
        },
        "new_series_hit_extra_scoring": "unscored",
    }
    d30 = {
        "schema": "research-a-v2-l0-d30-common-rng-seed-sensitivity-v1",
        "classification": "valid_seed_sensitivity_audit",
        "source_run": 41515190995, "source_artifact": 13175252296,
        "arm": "a0", "source_seed": 320403, "step": 413,
        "d29_replay": {"before": 480, "after_to_none": 480,
                       "pivot_rng_state_exact": True,
                       "archived_step413_cross_draw": [0, 2]},
        "p_before": [0.3233119547367096, 0.2952822148799896,
                     0.21348761022090912, 0.16791823506355286],
        "p_after": [0.32353976368904114, 0.29383787512779236,
                    0.21458901464939117, 0.16803324222564697],
        "sampling": {"seed_count": 4096, "disagreement_count": 6},
    }
    return d27, d30


def test_a0_effective_action_mapping():
    assert d31.validate_a0_mapping() == [-1, -1, 1, 1]


def test_expected_probability_and_two_suffixes():
    d27, d30 = sources()
    r = d31.analyze(d27, d30)
    assert r["classification"] == "valid_local_a0_keydown_expectation"
    assert math.isclose(r["before_probability"], 0.381405845284462, abs_tol=1e-14)
    assert math.isclose(r["after_probability"], 0.38262225687503815, abs_tol=1e-14)
    assert math.isclose(r["difference_probability"], 0.0012164115905761719, abs_tol=1e-14)
    assert r["conditional_expectation_by_frozen_suffix"]["before_suffix"] == (
        r["conditional_expectation_by_frozen_suffix"]["after_suffix"])
    assert r["new_physical_keydowns"] == r["new_motor_replays"] == 0
    assert r["counterfactual_hit_extra_scoring"] == "unscored"


@pytest.mark.parametrize("field", ["negative_controls_exact", "pre_step413_states_exact"])
def test_reject_broken_d27_controls(field):
    d27, d30 = sources()
    d27[field] = False
    with pytest.raises(ValueError):
        d31.analyze(d27, d30)


def test_reject_wrong_direction_of_d27_intervention():
    d27, d30 = sources()
    d27["comparisons"]["before_step413_from_after"]["has_extra_keydown_at_4_327s"] = False
    with pytest.raises(ValueError, match="reversal"):
        d31.analyze(d27, d30)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update({"source_seed": 9}),
    lambda d: d["d29_replay"].update({"pivot_rng_state_exact": False}),
    lambda d: d.update({"p_before": [0.2, 0.2, 0.2, 0.2]}),
    lambda d: d.update({"p_after": [float("nan"), 0.2, 0.2, 0.2]}),
    lambda d: d["sampling"].update({"disagreement_count": 7}),
])
def test_reject_corrupt_d30(mutate):
    d27, d30 = sources()
    mutate(d30)
    with pytest.raises(ValueError):
        d31.analyze(d27, d30)


def test_reject_d27_scored_intervention():
    d27, d30 = sources()
    d27["new_series_hit_extra_scoring"] = "scored"
    with pytest.raises(ValueError, match="scoring"):
        d31.analyze(d27, d30)
