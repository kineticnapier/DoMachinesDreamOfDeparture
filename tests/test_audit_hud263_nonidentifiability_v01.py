from __future__ import annotations

import ast
import inspect

import pytest

from scripts import audit_hud263_nonidentifiability_v01 as audit


@pytest.fixture(scope="module")
def proof():
    return audit.run()


def test_pair_of_unseen_future_setspeed_values_has_same_entire_hud_prefix(proof):
    result = next(x for x in proof["results"] if x["case"] == "set_speed_value")
    assert result["status"] == "INDISTINGUISHABLE"
    assert result["compared_prefix_frames"] >= 15
    assert result["maximum_visible_hud263_feature_difference"] <= audit.SAME_HUD_EPS
    assert result["future_due_difference_ms_posthoc"] > 44.
    assert result["earliest_due_remaining_at_probe_ms"] >= 44.
    assert result["same_deterministic_v14_prediction"]


def test_unknown_countdown_count_can_shift_first_target_with_same_hud(proof):
    result = next(x for x in proof["results"] if x["case"] == "countdown_count")
    assert result["status"] == "INDISTINGUISHABLE"
    assert result["first_different_target_ordinal_posthoc"] == 0
    assert result["future_due_difference_ms_posthoc"] > 1000.
    assert result["compared_prefix_frames"] >= 30
    assert result["maximum_visible_hud263_feature_difference"] <= audit.SAME_HUD_EPS
    assert result["same_deterministic_v14_prediction"]


def test_hidden_counterfactual_truth_not_read_in_policy_hud_collection():
    tree = ast.parse(inspect.getsource(audit._collect_visible_prefix))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "targets" not in attributes
    assert "_used" not in attributes
    assert "_missed" not in attributes
    assert "chart_time_from_episode" not in attributes
    assert "target_index" not in attributes


def test_paired_chart_values_are_hidden_from_public_timing_encoder():
    a, b = audit.paired_charts("set_speed_value")
    assert a.targets[0].episode_time_s == b.targets[0].episode_time_s
    assert any(
        abs(x.episode_time_s - y.episode_time_s) > audit.MIN_DIFFERENT_DUE_S
        for x, y in zip(a.targets, b.targets)
    )
    c, d = audit.paired_charts("countdown_count")
    assert c.chart.countdown_ticks != d.chart.countdown_ticks
    assert len(c.targets) == len(d.targets)
    with pytest.raises(ValueError, match="unsupported"):
        audit.paired_charts("training")
