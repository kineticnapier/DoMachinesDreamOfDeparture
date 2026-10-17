from __future__ import annotations

import pytest

from dmdod.evaluation import autoplay_sequence_v1 as evaluation


def test_ordered_matcher_is_unique_and_optimizes_error():
    pairs = evaluation._ordered_matches(
        [0.100, 0.103, 0.200], [0.101, 0.104, 0.300], 0.005
    )
    assert [(a, b) for a, b, _ in pairs] == [(0, 0), (1, 1)]
    assert sum(error for _, _, error in pairs) == pytest.approx(0.002)


def test_repeated_frame_forecasts_count_once_and_keep_first_lead():
    tracker = evaluation.CausalForecastReconciler()
    tracker.observe(0.00, (0.050,), (0.8,), eligible=True)
    tracker.observe(0.01, (0.0405,), (0.9,), eligible=True)
    tracker.observe(0.02, (0.0302,), (0.95,), eligible=True)
    tracker.observe(0.06, (), (), eligible=False)
    result = tracker.finish(cutoff_s=0.1)
    assert len(result) == 1
    assert result[0].confirmations == 3
    assert result[0].due_s == pytest.approx(0.0502)
    assert result[0].lead_ms == pytest.approx(50.2)


def test_dense_events_remain_distinct_during_reconciliation():
    tracker = evaluation.CausalForecastReconciler()
    tracker.observe(0.0, (0.002, 0.005, 0.008), (0.9, 0.9, 0.9), eligible=True)
    tracker.observe(0.010, (), (), eligible=False)
    result = tracker.finish(cutoff_s=0.02)
    assert [event.due_s for event in result] == pytest.approx((0.002, 0.005, 0.008))
    assert len({event.local_id for event in result}) == 3


def test_eligible_empty_forecast_cancels_contradicted_pending_event():
    tracker = evaluation.CausalForecastReconciler()
    tracker.observe(0.0, (0.050,), (1.0,), eligible=True)
    tracker.observe(0.01, (), (), eligible=True)
    assert tracker.finish(cutoff_s=0.1) == ()


def test_abstention_does_not_fabricate_or_cancel_evidence():
    tracker = evaluation.CausalForecastReconciler()
    tracker.observe(0.0, (0.030,), (0.7,), eligible=True)
    tracker.observe(0.01, (), (), eligible=False)
    tracker.observe(0.04, (), (), eligible=False)
    assert len(tracker.finish(cutoff_s=0.1)) == 1


def test_aggregate_preserves_chart_boundaries_and_sums_primary_counts():
    practical = {
        "matches": 1, "targets": 2, "predictions": 2, "false_predictions": 1,
        "tracking_rate": .5, "precision": .5, "f1": .5,
        "mae_ms": 2.0, "p95_ms": 2.0,
        "matched_errors_ms": [2.0], "matched_leads_ms": [30.0],
        "median_matched_lead_ms": 30.0,
        "lead_bins": {"10_20": 0, "20_40": 1, "40_80": 0},
        "cohorts": {"All": {"targets": 2, "matches": 1, "tracking_rate": .5}},
        "confidence_calibration": {"ece": .1, "bins": [
            {"lower": .8, "upper": 1.0, "count": 2,
             "mean_confidence": .9, "accuracy": .5},
        ]},
    }
    score = {
        "unforecastable_zero_time_targets": 1,
        "all_leads": {key: {**practical} for key in ("1", "5", "10", "20")},
        "practical_10ms_tolerance_10_80ms_lead": practical,
    }
    row = {"frame_metrics": {"frames": 10, "abstained_frames": 4,
                              "raw_80ms_count_mae": .3}, "score": score}
    result = evaluation.aggregate_method_rows([row, row])
    primary = result["practical_10ms_tolerance_10_80ms_lead"]
    assert primary["matches"] == 2 and primary["targets"] == 4
    assert primary["false_predictions"] == 2
    assert primary["median_matched_lead_ms"] == 30
    assert result["unforecastable_zero_time_targets"] == 2


def test_disjoint_frame_sequence_score_preserves_dense_cardinality():
    result = evaluation.score_disjoint_frame_sequences(
        [(0.0021, 0.0051, 0.0081), (), (0.004,)],
        [(0.002, 0.005, 0.008), (), (0.020,)],
        [True, True, True],
    )
    assert result["multi_target_frames"] == 1
    assert result["predicted_multi_frames"] == 1
    assert result["exact_count_frames"] == 3
    assert result["tolerances_ms"]["1"]["matches"] == 3
    assert result["tolerances_ms"]["1"]["exact_multi_event_frames"] == 1
    assert result["tolerances_ms"]["20"]["matches"] == 4
