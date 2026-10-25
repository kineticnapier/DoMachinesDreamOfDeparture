from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdogo_v08_provenance import FROZEN_HEAD, WRAPPER_FORMAT, digest_json
from report_dmdogo_v09_style_proxies import compile_report, REPORT_FORMAT

SHA = "a" * 64
SOURCE_DIGEST = "c" * 64
SUMMARY_DIGEST = "d" * 64
MANIFEST_DIGEST = "e" * 64


def fixture():
    case = {"sha256": SHA, "chart": "example.adofai",
            "selection_reason": "small_inter_onset_gaps", "event_count": 4}
    manifest = {
        "format": "dmdogo-v08-frozen-train-generalization-manifest-v1",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_code_digest": SOURCE_DIGEST,
        "cases": [case],
    }
    verified = {
        "format": "dmdogo-v08-train-generalization-report-v2",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_code_digest": SOURCE_DIGEST,
        "case_count": 1, "completed_count": 1, "all_cases_completed": True,
        "records": [{
            "chart_sha256": SHA, "status": "chart_complete",
            "hits": 4, "targets": 4, "misses": 0,
            "timing_mae_ms": 4., "local_replans": 0,
            "deferred_events": 0, "overloaded": False,
        }],
    }
    summary = {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "chart": "example.adofai",
        "chart_sha256": SHA, "planning_mode": "gesture_v08",
        "total_events": 4, "termination_reason": "chart_complete",
        "_artifact_sha256": SUMMARY_DIGEST,
        "stats": {
            "hits": 4, "targets": 4, "misses": 0, "too_early": 0,
            "timing_mae_ms": 4., "x_accuracy_percent": 100.,
            "overloaded": False,
        },
        "physical": {
            "local_replans": 0, "deferred_events": 0,
            "unassigned_events": 0,
        },
        "abstract_schedule": {"chunks": [{
            "start_event_index": 0, "end_event_index": 3,
            "event_count": 4, "gesture_family": "same_hand_roll",
            "axes": [["left_1"], ["left_2"], ["left_3"], ["left_4"]],
        }]},
    }
    times = [1000., 1050., 1100., 1150.]
    telemetry = {"decisions": [
        {"event_index": i, "due_ms": ms, "axes": [f"left_{i+1}"],
         "desired_axes": [f"left_{i+1}"],
         "deferral_frames": 0, "local_replan": False}
        for i, ms in enumerate(times)
    ]}
    replay = {
        "format": summary["format"], "chart": "example.adofai",
        "chart_sha256": SHA, "default_policy": "gesture_chunk_v08",
        "events": [
            {"event_index": i, "episode_s": ms/1000., "chord_size": 1}
            for i, ms in enumerate(times)],
        "policies": {"gesture_chunk_v08": {
            "keydowns": [
                {"time_ms": ms + 4, "axis": f"left_{i+1}"}
                for i, ms in enumerate(times)
            ],
        }},
    }
    settings = {"planning_mode": "gesture_v08", "minimum_reuse_ms": 80.,
                "stop_after_s": None}
    environment = {"python_version": "test"}
    proof = {
        "format": WRAPPER_FORMAT, "state": "success", "runner_exit_code": 0,
        "chart_sha256": SHA, "summary_sha256": SUMMARY_DIGEST,
        "source": {"frozen_source_commit": FROZEN_HEAD,
                   "code_digest": SOURCE_DIGEST,
                   "working_runtime_sources_match": True},
        "selection_manifest_frozen_code_digest": SOURCE_DIGEST,
        "selection_manifest_sha256": MANIFEST_DIGEST,
        "settings": settings, "settings_digest": digest_json(settings),
        "environment": environment, "environment_digest": digest_json(environment),
    }
    return manifest, verified, {SHA: (summary, telemetry, replay, proof)}


def test_complete_verified_train_data_yields_no_human_score():
    manifest, verified, artifacts = fixture()
    report = compile_report(
        manifest, verified, artifacts, manifest_digest=MANIFEST_DIGEST)
    assert report["format"] == REPORT_FORMAT
    assert report["case_count"] == 1
    assert report["aggregate"]["events"] == 4
    assert report["aggregate"]["contacts"] == 4
    assert report["aggregate"]["planned_axis_adopted_events"] == 4
    assert report["human_demonstrations_used"] is False
    assert report["human_likeness_verified"] is False
    assert report["aggregate"]["flag_counts"]["rapid_single_hand_sequence"] == 1


def test_incomplete_prior_report_refused():
    manifest, verified, artifacts = fixture()
    verified["all_cases_completed"] = False
    with pytest.raises(ValueError):
        compile_report(
            manifest, verified, artifacts,
            manifest_digest=MANIFEST_DIGEST)


def test_mutated_source_proof_refused():
    manifest, verified, artifacts = fixture()
    artifacts[SHA][3]["source"]["code_digest"] = "f" * 64
    with pytest.raises(ValueError, match="unverified_provenance"):
        compile_report(
            manifest, verified, artifacts,
            manifest_digest=MANIFEST_DIGEST)


def test_edited_summary_metrics_cannot_be_hidden():
    manifest, verified, artifacts = fixture()
    artifacts[SHA][0]["stats"]["hits"] = 3
    artifacts[SHA][0]["stats"]["misses"] = 1
    with pytest.raises(ValueError, match="saved physical metrics differ"):
        compile_report(
            manifest, verified, artifacts,
            manifest_digest=MANIFEST_DIGEST)


def test_wrong_manifest_digest_refused():
    manifest, verified, artifacts = fixture()
    with pytest.raises(ValueError, match="unverified_provenance"):
        compile_report(
            manifest, verified, artifacts,
            manifest_digest="0" * 64)
