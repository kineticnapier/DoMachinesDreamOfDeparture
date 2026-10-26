from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dmdogo_v09_h01_conditions import analyze_conditions
from test_dmdogo_v09_style_proxies import sample


def test_maps_flags_to_chunks_and_builds_like_for_like_controls():
    axes = [
        "left_1", "left_3", "left_4", "left_2",
        "right_1", "right_2", "left_1", "left_2",
    ]
    summary, telemetry, replay = sample(axes=axes, gaps=[50.] * 7)
    report = analyze_conditions(summary, telemetry, replay)

    quick = report["quick_same_hand_depth_jump"]
    assert quick["detected_count"] == 2
    assert quick["control_count"] == 3
    assert quick["detected"][0]["ioi_ms"] == 50.
    assert quick["detected"][0]["local_bpm_equivalent"] == 1200.
    assert quick["detected"][0]["depth_distance"] == 2
    assert quick["detected"][0]["chunk_relation"] == "within_chunk"
    assert quick["detected"][1]["chunk_relation"] == "within_chunk"
    assert quick["detected_movement_tendencies"]["hand_counts"] == {
        "left": 2, "right": 0}

    rapid = report["rapid_single_hand_sequence"]
    assert rapid["detected_count"] == 1
    assert rapid["control_count"] == 2
    assert rapid["detected"][0]["event_count"] == 4
    assert rapid["detected"][0]["chunk_relation"] == "within_chunk"
    assert rapid["detected"][0]["movement_proxy"]["total_depth_distance"] == 5
    assert rapid["distributions"]["detected_ioi_ms"]["count"] == 3
    assert rapid["detected_movement_tendencies"]["hand_counts"] == {
        "left": 1, "right": 0}

    overlap = report["overlap"]
    assert overlap["quick_flags_inside_rapid_sequence"] == 2
    assert overlap["rapid_sequences_containing_quick_flag"] == 1


def test_rapid_sequence_can_cross_a_gesture_chunk_boundary():
    summary, telemetry, replay = sample(
        axes=["left_1", "left_2", "left_3", "left_4", "left_1"],
        gaps=[40.] * 4,
    )
    report = analyze_conditions(summary, telemetry, replay)
    run = report["rapid_single_hand_sequence"]["detected"][0]
    assert run["chunk_relation"] == "cross_chunk_boundary"
    assert run["chunk_boundary_crossings"] == 1
    assert len(run["chunks"]) == 2


def test_zero_ioi_is_kept_without_inventing_infinite_bpm():
    summary, telemetry, replay = sample(
        axes=["left_1", "left_4", "right_1"], gaps=[0., 50.])
    report = analyze_conditions(summary, telemetry, replay)
    jump = report["quick_same_hand_depth_jump"]["detected"][0]
    assert jump["ioi_ms"] == 0.
    assert jump["local_bpm_equivalent"] is None
    assert report["quick_same_hand_depth_jump"]["distributions"][
        "detected_local_bpm_equivalent"]["count"] == 0


def test_report_preserves_h0_thresholds_and_noncausal_interpretation():
    report = analyze_conditions(*sample())
    assert report["h0_thresholds"] == {
        "rapid_run_min_events": 4,
        "rapid_run_max_ioi_ms": 120.0,
        "quick_jump_min_depth_indices": 2,
    }
    assert report["thresholds_changed"] is False
    assert report["causal_claim"] is False
    assert report["human_likeness_verified"] is False
