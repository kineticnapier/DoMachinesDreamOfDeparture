from __future__ import annotations

import pytest

from scripts import diagnose_hud_v14_predecessor_cascade_v02 as cascade


def _arm(*, miss: bool):
    return {
        "target_outcomes_posthoc": [
            {"ordinal": 0, "due_s": .10, "hit": True, "miss": False},
            {"ordinal": 1, "due_s": .16, "hit": not miss, "miss": miss},
        ],
        "first_seen_predictions": [
            {"local_id": 0, "due_local_s": .10, "first_seen_s": .03},
            {"local_id": 1, "due_local_s": .16, "first_seen_s": .09},
        ],
        "decisions": [
            {"local_id": 0, "choice": "left_1",
             "reason": "assigned", "step": 4, "extra_lead_s": .01},
            {"local_id": 1, "choice": "right_1",
             "reason": "assigned", "step": 10, "extra_lead_s": .02},
        ],
        "physical_switch_trace": [
            {"time_s": .159, "axis": "right_1", "edge": "down"},
        ],
    }


def _scorer(*, miss: bool):
    return {
        "score_replay_exact_match": True,
        "logical_down_target_routing": [
            {"time_s": .159, "key": "right_1",
             "consumed_target_ordinal": 0 if miss else 1,
             "assigned_error_ms": 59 if miss else -1,
             "margin": "very_late" if miss else "perfect"}
        ],
    }


def test_missed_target_lookback_finds_real_scorer_predecessor():
    record, scoring = _arm(miss=True), _scorer(miss=True)
    assert cascade.implicated_predecessors(scoring, 1, .16) == (0,)
    selected = cascade.target_view(record, scoring, 1)
    assert selected["forecast"]["first_seen_lead_ms"] == 70.
    assert selected["decision"]["choice"] == "right_1"
    assert selected["chosen_axis_nearest_physical_downs"][0][
        "closest_down"]["offset_ms"] == -1.
    assert selected["scorer_logical_down_routing_within_70ms"][0][
        "consumed_ordinal"] == 0


def test_cross_arm_comparison_never_uses_missed_target_as_parent():
    arms = {
        "hud_network": (_arm(miss=True), _scorer(miss=True)),
        "oracle_network": (_arm(miss=False), _scorer(miss=False)),
    }
    result = cascade.inspect_case(arms)
    assert len(result) == 1
    assert result[0]["missed_target_ordinal"] == 1
    assert result[0]["earlier_scorer_consumers"] == [0]
    assert result[0]["comparison"]["0"]["hud_network"]["hit"] is True
    assert result[0]["comparison"]["1"]["hud_network"]["miss"] is True
    assert result[0]["comparison"]["1"]["oracle_network"]["hit"] is True


def test_refuses_attribution_if_scorer_replay_did_not_match():
    arms = {
        "hud_network": (_arm(miss=True), {
            **_scorer(miss=True), "score_replay_exact_match": False
        })
    }
    with pytest.raises(ValueError, match="scorer replay mismatch"):
        cascade.inspect_case(arms)


def test_no_forecast_does_not_invent_physical_intent():
    record, scorer = _arm(miss=True), _scorer(miss=True)
    record["first_seen_predictions"] = record["first_seen_predictions"][:1]
    selected = cascade.target_view(record, scorer, 1)
    assert selected["forecast"] is None
    assert selected["decision"] is None
    assert selected["chosen_axis_nearest_physical_downs"] == []
