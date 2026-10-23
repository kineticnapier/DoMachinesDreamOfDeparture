from __future__ import annotations

import sys
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_dmdogo_v08_train_generalization import collate


def manifest():
    return {
        "format": "dmdogo-v08-frozen-train-generalization-manifest-v1",
        "frozen_implementation_commit": "fixed",
        "cases": [
            {"selection_reason": "small_inter_onset_gaps",
             "chart": "a.adofai", "sha256": "a" * 64, "event_count": 40},
            {"selection_reason": "uneven_timing",
             "chart": "b.adofai", "sha256": "b" * 64, "event_count": 70},
        ],
    }


def result(sha, *, termination="chart_complete"):
    return {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "planning_mode": "gesture_v08",
        "chart_sha256": sha,
        "termination_reason": termination,
        "stats": {
            "targets": 40, "hits": 39, "misses": 1, "too_early": 0,
            "overloaded": False, "x_accuracy_percent": 98.,
            "timing_mae_ms": 3.1,
        },
        "physical": {"local_replans": 2, "deferred_events": 1},
    }


def test_missing_run_remains_pending_not_success():
    report = collate(manifest(), {"a" * 64: result("a" * 64)})
    assert not report["all_cases_completed"]
    assert report["completed_count"] == 1
    assert report["records"][1]["status"] == "pending"
    assert report["records"][1]["hits"] is None
    assert report["aggregate"]["total_misses_completed"] == 1


def test_distinct_complete_charts_are_aggregated_without_invented_accuracy():
    report = collate(manifest(), {
        "a" * 64: result("a" * 64),
        "b" * 64: result("b" * 64),
    })
    assert report["all_cases_completed"]
    assert report["aggregate"]["total_targets_completed"] == 80
    assert report["aggregate"]["total_hits_completed"] == 78
    assert report["aggregate"]["total_local_replans_completed"] == 4
    assert "average_x" not in report["aggregate"]


def test_wrong_chart_sha_or_policy_is_not_counted():
    bad = result("c" * 64)
    report = collate(manifest(), {"a" * 64: bad})
    assert report["records"][0]["status"] == "invalid_identity"
    assert report["completed_count"] == 0


def test_nonterminal_and_corrupt_metrics_are_not_success():
    report = collate(manifest(), {"a" * 64: result("a" * 64, termination="miss")})
    assert report["records"][0]["status"] == "did_not_complete"
    assert report["completed_count"] == 0
    broken = result("a" * 64)
    broken["stats"] = None
    report = collate(manifest(), {"a" * 64: broken})
    assert report["records"][0]["status"] == "invalid_metrics"


def test_reject_duplicate_or_unrecognized_selection():
    broken = manifest()
    broken["cases"][1]["sha256"] = broken["cases"][0]["sha256"]
    with pytest.raises(ValueError):
        collate(broken, {})
    broken = manifest()
    broken["format"] = "unknown"
    with pytest.raises(ValueError):
        collate(broken, {})
