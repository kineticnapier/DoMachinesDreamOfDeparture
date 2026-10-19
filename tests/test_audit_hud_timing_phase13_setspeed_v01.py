from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from dmdod.adofai.chart import load_adofai
from scripts import audit_hud_timing_phase13_setspeed_v01 as phase13


@pytest.fixture(scope="module")
def audit_result():
    return phase13.run()


def test_fixed_scope_is_four_primary_events_plus_separate_reference():
    assert len(phase13.PRIMARY) == 4
    assert phase13.REFERENCE not in phase13.PRIMARY
    assert phase13.INTERVENTION_FACTOR == .125
    assert phase13.MOTOR_LEAD_S == .044


def test_counterfactual_changes_only_one_numeric_setspeed_field():
    name, event_index = phase13.PRIMARY[1]
    path = phase13.TRAIN_DIR / name
    chart = load_adofai(path)
    _, _, segment = phase13.phase12.capture(
        path, phase13.phase12.UNSEEN_END_S
    )
    floor_index = segment.targets[event_index].floor_index - 2
    alternate, intervention = phase13._intervene_numeric_setspeed(
        chart, floor_index=floor_index
    )
    assert intervention["only_one_action_changed"]
    assert intervention["visible_marker_preserved"]
    assert len(alternate.actions) == len(chart.actions)
    assert alternate.angles == chart.angles
    assert alternate.countdown_ticks == chart.countdown_ticks
    assert alternate.actions[intervention["action_sequence_index"]].event_type == "SetSpeed"


def test_policy_capture_has_no_truth_or_future_action_access():
    tree = ast.parse(inspect.getsource(phase13.capture_causal_hud))
    attributes = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({
        "targets", "actions", "episode_time_s", "target_time_s",
        "bpm_multiplier", "beats_per_minute", "floor_index",
    })


def test_all_four_histories_are_identical_but_due_truths_differ(audit_result):
    result, _ = audit_result
    assert result["verdict"] == "FUTURE_SETSPEED_PREDEADLINE_NONIDENTIFIABLE"
    assert result["primary_classification_counts"] == {
        "PREDEADLINE_NONIDENTIFIABLE": 4
    }
    for case in result["primary_cases"]:
        assert case["counterfactual_prefix"]["indistinguishable_through_deadline"]
        assert case["counterfactual_prefix"]["max_absolute_hud263_difference"] == 0.0
        assert case["counterfactual_prefix"]["countdown_histories_equal"]
        assert case["due_gap_ms_posthoc"] > 44.0
        assert not case["exact_due_truth_intersection_exists"]


def test_information_arrives_in_the_required_causal_order(audit_result):
    result, _ = audit_result
    for case in result["primary_cases"]:
        deadline = case["deadline_44ms_s_posthoc"]
        arrival = case["information_arrival"]
        assert arrival["future_setspeed_marker_first_visible_s"] <= deadline
        assert arrival["midspin_pair_first_safely_visible_s"] <= deadline
        assert arrival["postchange_numeric_bpm_first_distinguishable_s"] > deadline
        assert arrival["first_safe_due_forecast"]["observed_at_s"] > deadline
        assert arrival["numeric_bpm_arrival_minus_deadline_ms"] > 0.0
        assert arrival["due_forecast_arrival_minus_deadline_ms"] > 0.0


def test_reference_is_not_misclassified_as_setspeed(audit_result):
    result, traces = audit_result
    reference = result["reference_case"]
    assert reference["role"] == "reference_only_not_setspeed"
    assert reference["classification"] == "OBSERVABILITY_UNVERIFIED"
    assert reference["information_arrival"]["future_twirl_marker_first_visible_s"] < (
        reference["deadline_44ms_s_posthoc"]
    )
    assert len(traces["events"]) == 5


def test_phase12_safety_evidence_is_preserved(audit_result):
    result, _ = audit_result
    safety = result["safety_constraints"]
    assert safety == {
        "phase12_recovered_13_of_13": True,
        "phase12_motor_ready_9_of_13": True,
        "phase12_false_positive_zero": True,
        "phase12_success_regressions_zero": True,
        "predictor_change_made": False,
    }


def test_output_writer_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "phase13.json"
    phase13._write_new(output, {"format": "first"})
    with pytest.raises(FileExistsError):
        phase13._write_new(output, {"format": "second"})
