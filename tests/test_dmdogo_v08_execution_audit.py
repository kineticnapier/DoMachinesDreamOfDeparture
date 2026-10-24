from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_dmdogo_v08_execution import audit, _match_unique_keydowns


def sample():
    summary = {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "chart": "example.adofai", "chart_sha256": "a" * 64,
        "termination_reason": "chart_complete", "total_events": 4,
        "stats": {"targets": 4, "hits": 4, "misses": 0},
        "abstract_schedule": {"chunks": [
            {"start_event_index": 0, "end_event_index": 1,
             "event_count": 2, "axes": [["left_1"], ["left_2"]]},
            {"start_event_index": 2, "end_event_index": 3,
             "event_count": 2, "axes": [["right_1"], ["right_2"]]},
        ]},
    }
    times = [1., 1.02, 1.21, 1.23]
    axes = ["left_1", "left_2", "right_1", "right_2"]
    telemetry = {"decisions": [
        {"event_index": i, "due_ms": times[i] * 1000,
         "axes": [axis], "local_replan": False, "deferral_frames": 0}
        for i, axis in enumerate(axes)
    ]}
    replay = {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "chart_sha256": "a" * 64,
        "default_policy": "gesture_chunk_v08",
        "events": [
            {"event_index": i, "episode_s": t}
            for i, t in enumerate(times)],
        "policies": {"gesture_chunk_v08": {
            "keydowns": [
                {"time_ms": t * 1000 + 3, "axis": axis}
                for t, axis in zip(times, axes)
            ],
        }},
    }
    return summary, telemetry, replay


def test_complete_frozen_baseline_adoption_and_keydown_integrity():
    report = audit(*sample())
    assert report["events"] == 4
    assert report["planned_axis_adoption_fraction"] == 1
    assert report["preserved_chunks"] == 2
    assert report["chosen_due_reuse"]["violation_count"] == 0
    assert report["physical_keydown_reuse"]["violation_count"] == 0
    assert report["unique_keydown_diagnostic"]["matched_contacts"] == 4
    assert report["unique_keydown_diagnostic"]["unused_keydowns"] == 0
    assert report["unique_keydown_diagnostic"]["unmatched_contact_count"] == 0


def test_local_replan_changes_gesture_without_changing_scorer():
    s, t, r = sample()
    t["decisions"][2]["axes"] = ["left_1"]
    t["decisions"][2]["local_replan"] = True
    r["policies"]["gesture_chunk_v08"]["keydowns"][2]["axis"] = "left_1"
    report = audit(s, t, r)
    assert report["planned_axis_adoption_count"] == 3
    assert report["preserved_chunks"] == 1
    assert report["declared_local_replans"] == 1
    assert report["scorer_reference_only"]["hits"] == 4


def test_keydown_cannot_be_reused_for_two_nearby_events():
    rows = [
        {"event_index": 0, "due_ms": 1000, "axes": ["left_1"]},
        {"event_index": 1, "due_ms": 1020, "axes": ["left_1"]},
    ]
    match = _match_unique_keydowns(rows, [{"time_ms": 1008, "axis": "left_1"}])
    assert match["matched_contacts"] == 1
    assert match["unmatched_contact_count"] == 1
    assert len({m["keydown_index"] for m in match["matches"]}) == 1


def test_ambiguous_nearby_keydowns_are_counted():
    rows = [{"event_index": 0, "due_ms": 1000, "axes": ["right_2"]}]
    keydowns = [
        {"time_ms": 1001, "axis": "right_2"},
        {"time_ms": 1010, "axis": "right_2"},
    ]
    match = _match_unique_keydowns(rows, keydowns)
    assert match["ambiguous_contact_count"] == 1
    assert match["unused_keydowns"] == 1


def test_bad_identity_or_incomplete_coverage_refused():
    s, t, r = sample()
    r["chart_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        audit(s, t, r)
    s, t, r = sample()
    t["decisions"].pop()
    with pytest.raises(ValueError):
        audit(s, t, r)
    s, t, r = sample()
    s["termination_reason"] = "diagnostic_prefix_complete"
    with pytest.raises(ValueError):
        audit(s, t, r)


def test_overlapping_chunk_ids_refused():
    s, t, r = sample()
    s["abstract_schedule"]["chunks"][1]["start_event_index"] = 1
    s["abstract_schedule"]["chunks"][1]["event_count"] = 3
    s["abstract_schedule"]["chunks"][1]["axes"] = [
        ["left_2"], ["right_1"], ["right_2"]]
    with pytest.raises(ValueError):
        audit(s, t, r)
