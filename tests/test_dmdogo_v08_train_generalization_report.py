from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdogo_v08_provenance import (
    FROZEN_HEAD, WRAPPER_FORMAT, digest_json,
)
from report_dmdogo_v08_train_generalization import collate

CODE_DIGEST = "f" * 64
ARTIFACT_DIGEST = "d" * 64


def manifest():
    return {
        "format": "dmdogo-v08-frozen-train-generalization-manifest-v1",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_code_digest": CODE_DIGEST,
        "cases": [
            {"selection_reason": "small_inter_onset_gaps",
             "chart": "a.adofai", "sha256": "a" * 64, "event_count": 40},
            {"selection_reason": "uneven_timing",
             "chart": "b.adofai", "sha256": "b" * 64, "event_count": 40},
        ],
    }


def result(sha, *, termination="chart_complete", chart="a.adofai"):
    return {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "planning_mode": "gesture_v08",
        "chart": chart,
        "chart_sha256": sha,
        "termination_reason": termination,
        "total_events": 40,
        "_artifact_sha256": ARTIFACT_DIGEST,
        "stats": {
            "targets": 40, "hits": 39, "misses": 1, "too_early": 0,
            "overloaded": False, "x_accuracy_percent": 98.,
            "timing_mae_ms": 3.1,
        },
        "physical": {
            "local_replans": 2, "deferred_events": 1,
            "unassigned_events": 0,
        },
    }


def proof(sha):
    settings = {
        "planning_mode": "gesture_v08",
        "minimum_reuse_ms": 80.,
        "stop_after_s": None,
        "control_dt_s": .01,
    }
    environment = {"python_version": "same-test-runtime"}
    return {
        "format": WRAPPER_FORMAT,
        "state": "success",
        "runner_exit_code": 0,
        "chart_sha256": sha,
        "summary_sha256": ARTIFACT_DIGEST,
        "source": {
            "frozen_source_commit": FROZEN_HEAD,
            "working_runtime_sources_match": True,
            "code_digest": CODE_DIGEST,
            "actual_git_head": "commit-used-for-test",
        },
        "selection_manifest_frozen_code_digest": CODE_DIGEST,
        "settings": settings,
        "settings_digest": digest_json(settings),
        "environment": environment,
        "environment_digest": digest_json(environment),
    }


def datasets():
    return ({
        "a" * 64: result("a" * 64),
        "b" * 64: result("b" * 64, chart="b.adofai"),
    }, {
        "a" * 64: proof("a" * 64),
        "b" * 64: proof("b" * 64),
    })


def test_missing_run_remains_pending_not_success():
    results, proofs = datasets()
    del results["b" * 64]
    report = collate(manifest(), results, proofs)
    assert not report["all_cases_completed"]
    assert report["completed_count"] == 1
    assert report["records"][1]["status"] == "pending"
    assert report["aggregate"]["total_misses_completed"] == 1


def test_complete_consistent_charts_are_aggregated_without_fabricating_accuracy():
    results, proofs = datasets()
    report = collate(manifest(), results, proofs)
    assert report["all_cases_completed"]
    assert report["completed_count"] == 2
    assert report["aggregate"]["total_targets_completed"] == 80
    assert report["aggregate"]["total_hits_completed"] == 78
    assert report["aggregate"]["total_local_replans_completed"] == 4
    assert "average_x" not in report["aggregate"]


def test_wrong_chart_sha_or_policy_is_rejected():
    results, proofs = datasets()
    results["a" * 64]["chart_sha256"] = "c" * 64
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "invalid_identity"
    assert report["completed_count"] == 1


def test_terminated_early_and_corrupt_metrics_are_not_success():
    results, proofs = datasets()
    results["a" * 64]["termination_reason"] = "miss"
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "did_not_complete"
    results["a" * 64]["stats"] = None
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "invalid_metrics"


@pytest.mark.parametrize("field,invalid", [
    ("hits", -7), ("targets", 3), ("misses", 99),
    ("hits", True), ("too_early", -1), ("x_accuracy_percent", 999.),
    ("x_accuracy_percent", float("nan")),
    ("x_accuracy_percent", float("inf")),
    ("timing_mae_ms", -5.), ("timing_mae_ms", float("nan")),
    ("overloaded", 0),
])
def test_bad_scorer_values_are_never_counted(field, invalid):
    results, proofs = datasets()
    results["a" * 64]["stats"][field] = invalid
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "invalid_metrics"
    assert not report["all_cases_completed"]


@pytest.mark.parametrize("field,invalid", [
    ("local_replans", -1), ("deferred_events", float("inf")),
    ("unassigned_events", 42),
])
def test_bad_motor_values_are_never_counted(field, invalid):
    results, proofs = datasets()
    results["a" * 64]["physical"][field] = invalid
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "invalid_metrics"


def test_wrong_event_count_refused():
    results, proofs = datasets()
    results["a" * 64]["total_events"] = 39
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "invalid_metrics"


@pytest.mark.parametrize("edit", [
    "no_proof", "reuse_changed", "code_changed",
    "summary_changed", "source_not_verified", "wrong_commit",
    "manifest_changed", "exit_nonzero", "settings_digest_mismatch",
])
def test_provenance_fail_closed(edit):
    results, proofs = datasets()
    p = proofs["a" * 64]
    if edit == "no_proof":
        del proofs["a" * 64]
    elif edit == "reuse_changed":
        p["settings"]["minimum_reuse_ms"] = 1.
        p["settings_digest"] = digest_json(p["settings"])
    elif edit == "code_changed":
        p["source"]["code_digest"] = "1" * 64
    elif edit == "summary_changed":
        p["summary_sha256"] = "2" * 64
    elif edit == "source_not_verified":
        p["source"]["working_runtime_sources_match"] = False
    elif edit == "wrong_commit":
        p["source"]["frozen_source_commit"] = "different"
    elif edit == "manifest_changed":
        p["selection_manifest_frozen_code_digest"] = "different"
    elif edit == "exit_nonzero":
        p["runner_exit_code"] = 1
    elif edit == "settings_digest_mismatch":
        p["settings_digest"] = "no"
    report = collate(manifest(), results, proofs)
    assert report["records"][0]["status"] == "unverified_provenance"
    assert not report["all_cases_completed"]


def test_different_runtime_environment_rejected():
    results, proofs = datasets()
    proofs["b" * 64]["environment"]["python_version"] = "changed"
    proofs["b" * 64]["environment_digest"] = digest_json(
        proofs["b" * 64]["environment"])
    report = collate(manifest(), results, proofs)
    assert report["completed_count"] == 1
    assert report["records"][1]["status"] == "different_runtime_environment"


def test_missing_provenance_cannot_be_accepted():
    results, _ = datasets()
    report = collate(manifest(), results)
    assert report["completed_count"] == 0
    assert set(r["status"] for r in report["records"]) == {"unverified_provenance"}


def test_manifest_validation():
    results, proofs = datasets()
    broken = manifest()
    broken["cases"][1]["sha256"] = broken["cases"][0]["sha256"]
    with pytest.raises(ValueError):
        collate(broken, results, proofs)
    for key, value in (("format", "unknown"),
                       ("frozen_implementation_commit", "different"),
                       ("frozen_code_digest", "")):
        broken = manifest()
        broken[key] = value
        with pytest.raises(ValueError):
            collate(broken, results, proofs)
