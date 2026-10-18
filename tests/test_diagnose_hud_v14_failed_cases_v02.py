from __future__ import annotations

from scripts import diagnose_hud_v14_failed_cases_v02 as diagnostic


def _arm(hit0: bool, *, network: bool = True):
    return {
        "stats": {
            "targets": 2, "hits": 1 + int(hit0), "misses": int(not hit0),
            "too_early": 0, "x_accuracy_percent": 100.0,
        },
        "forecast_new_token_count": 2,
        "target_outcomes_posthoc": [
            {"ordinal": 0, "due_s": .050, "hit": hit0, "miss": not hit0},
            {"ordinal": 1, "due_s": .100, "hit": True, "miss": False},
        ],
        "first_seen_predictions": [
            {"local_id": 10, "due_local_s": .051, "first_seen_s": .010},
            {"local_id": 11, "due_local_s": .100, "first_seen_s": .050},
        ],
        "decisions": [
            {"local_id": 10, "step": 1, "reason": "assigned",
             "choice": "right_5/tip" if network else "left_1",
             "extra_lead_s": 0.0},
            {"local_id": 11, "step": 5, "reason": "assigned",
             "choice": "right_1", "extra_lead_s": 0.0},
        ],
        "physical_switch_trace": [
            {"time_s": .075 if network else .050,
             "axis": "right_5" if network else "left_1",
             "edge": "down"},
            {"time_s": .105, "axis": "right_1", "edge": "down"},
            {"time_s": .120, "axis": "right_1", "edge": "up"},
        ],
    }


def test_target_differential_compares_same_ordinal_across_all_arms():
    oracle = _arm(True, network=False)
    hud_network = _arm(False)
    center = _arm(True, network=False)
    result = diagnostic.compare_case({
        "oracle_network": oracle,
        "hud_network": hud_network,
        "hud_center_first": center,
    })
    assert result["missed_ordinals_union"] == [0]
    detail = result["per_missed_ordinal"][0]["arms"]
    assert detail["oracle_network"]["hit"] is True
    assert detail["hud_network"]["miss"] is True
    assert detail["hud_center_first"]["hit"] is True
    assert detail["hud_network"]["decision"]["choice"] == "right_5/tip"
    assert detail["hud_network"]["physical_downs_within_100ms"][0]["axis"] == "right_5"
    assert detail["hud_network"]["physical_downs_within_100ms"][0]["error_ms"] == 25.0
    assert detail["hud_network"]["closest_forecast"]["first_seen_lead_ms"] == 41.0


def test_missing_timing_forecast_not_misattributed_to_motor():
    record = _arm(False)
    record["first_seen_predictions"] = record["first_seen_predictions"][1:]
    record["decisions"] = record["decisions"][1:]
    detail = diagnostic._per_target(record, 0)
    assert detail["category"] == "no_forecast_within_20ms"
    assert detail["decision"] is None
    assert detail["closest_forecast"] is None


def test_late_forecast_is_explicitly_reported():
    record = _arm(False)
    record["decisions"][0] = {
        "local_id": 10, "step": 3,
        "reason": "insufficient_observable_lead",
        "choice": None, "extra_lead_s": 0.0,
    }
    assert diagnostic._per_target(record, 0)["category"] == (
        "insufficient_observable_lead"
    )


def test_no_results_are_invented_for_failed_arms():
    result = diagnostic.compare_case({
        "oracle_network": {"arm": "oracle_network", "error": "TimeoutError"},
        "hud_network": _arm(False),
    })
    assert result["errors"] == {"oracle_network": "TimeoutError"}
    assert result["missed_ordinals_union"] == [0]
    assert result["per_missed_ordinal"][0]["arms"]["oracle_network"] == {
        "available": False,
        "reason": "arm errored or missing target trace",
    }


def test_fixed_case_scope_is_train_and_synthetic_only():
    assert diagnostic.CASES == (
        "ordinary-480bpm", "special-speed_double",
        "special-pause_1", "train-bool-5s",
    )
    assert diagnostic.ARMS == (
        "oracle_network", "hud_network", "hud_center_first"
    )
