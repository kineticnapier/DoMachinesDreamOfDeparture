from __future__ import annotations

import ast
import inspect

import pytest

from scripts import diagnose_hud_v14_observability_boundaries_v02 as audit


def test_sets_speed_hud_audit_is_causal_and_stays_in_frozen_horizon():
    segment = audit.load_case("special-speed_double", audit.DEFAULT_TRAIN_CHART)
    rows = audit.collect_visible_hud(segment)
    assert rows
    assert rows[0]["step"] == 0
    assert all(row["step"] == index for index, row in enumerate(rows))
    assert all(row["tile_bpm"] > 0 and row["real_bpm"] > 0 for row in rows)
    assert all(
        0 < due - row["local_clock_s"] <= .08000011
        for row in rows for due in row["predicted_due_s"]
    )
    assert any("SetSpeed" in f["markers"]
               for row in rows for f in row["visible_floors"])
    assert any(row["raw_status"] == "eligible_partial" for row in rows)
    report = audit.summarize_posthoc(
        rows, segment, "special-speed_double", selected=(5, 6)
    )
    assert [x["target_ordinal"] for x in report["selected_target_results"]] == [5, 6]
    for item in report["selected_target_results"]:
        assert item["raw_status_counts_in_110ms_before"]
        assert item["first_confirmed"] is not None
        assert item["first_confirmed"]["lead_ms"] <= 80.1
        assert abs(item["first_confirmed"]["due_error_ms"]) < .01


def test_unseen_future_target_is_not_backfilled_into_a_earlier_hud_frame():
    segment = audit.load_case("special-speed_double", audit.DEFAULT_TRAIN_CHART)
    stream = audit.collect_visible_hud(segment)
    target5 = float(segment.targets[5].episode_time_s)
    # Comparison uses future truth only after collecting the complete stream.
    earliest = audit._first_seen(stream, target5)
    assert earliest is not None
    assert earliest["clock_s"] <= target5
    assert earliest["lead_ms"] < 80.01
    assert all(
        not any(abs(token["due_local_s"] - target5) < .00001
                and token["confirmed_now"] for token in frame["pending"])
        for frame in stream if frame["step"] < earliest["frame"]
    )


def test_observability_audit_only_uses_target_truth_after_stream_collection():
    source = inspect.getsource(audit.collect_visible_hud)
    tree = ast.parse(source)
    attrs = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "targets" not in attrs
    assert "_used" not in attrs
    assert "_missed" not in attrs
    assert "target_ordinal" not in attrs
    assert "chart_time_from_episode" not in attrs


def test_missing_actual_hud_forecast_is_not_converted_to_oracle_prediction():
    summary = audit.summarize_posthoc(
        [], audit.load_case("special-speed_double", audit.DEFAULT_TRAIN_CHART),
        "special-speed_double", selected=(5,)
    )
    record = summary["selected_target_results"][0]
    assert record["first_confirmed"] is None
    assert not record["normal_motor_lead_sufficient"]


def test_only_predeclared_fixture_and_train_cases_allowed():
    assert audit.CASES == ("special-speed_double", "train-bool-5s")
    with pytest.raises(ValueError, match="unsupported"):
        audit.load_case("Validation20", audit.DEFAULT_TRAIN_CHART)
