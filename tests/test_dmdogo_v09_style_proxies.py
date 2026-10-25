from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dmdogo_v09_style_proxies import analyze, OUTPUT_FORMAT


def sample(axes=None, *, gaps=None):
    axes = axes or [
        "left_1", "left_2", "left_3", "left_4",
        "right_1", "right_2", "right_3", "right_4",
    ]
    gaps = gaps or [50.] * (len(axes) - 1)
    times = [1000.]
    for gap in gaps:
        times.append(times[-1] + gap)
    events = [
        {"event_index": i, "episode_s": ms / 1000., "chord_size": 1}
        for i, ms in enumerate(times)]
    decisions = [
        {"event_index": i, "due_ms": ms, "axes": [axis],
         "desired_axes": [axis], "local_replan": False,
         "deferral_frames": 0}
        for i, (axis, ms) in enumerate(zip(axes, times))
    ]
    chunks = [
        {"start_event_index": i, "end_event_index": min(i+3, len(axes)-1),
         "event_count": min(4, len(axes)-i),
         "gesture_family": "synthetic",
         "axes": [[x] for x in axes[i:i+4]]}
        for i in range(0, len(axes), 4)]
    summary = {
        "format": "dmdogo-gesture-chunk-physical-v0.8",
        "chart": "test.adofai", "chart_sha256": "a" * 64,
        "planning_mode": "gesture_v08", "termination_reason": "chart_complete",
        "total_events": len(axes),
        "abstract_schedule": {"chunks": chunks},
    }
    telemetry = {"decisions": decisions}
    replay = {
        "format": summary["format"], "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "default_policy": "gesture_chunk_v08",
        "events": events,
        "policies": {"gesture_chunk_v08": {"keydowns": [
            {"time_ms": ms + 4, "axis": axis}
            for ms, axis in zip(times, axes)
        ]}},
    }
    return summary, telemetry, replay


def test_reference_8key_style_metrics_without_human_labels():
    report = analyze(*sample())
    assert report["format"] == OUTPUT_FORMAT
    assert report["event_count"] == 8
    assert report["total_contacts"] == 8
    assert report["metrics"]["hand_contact_counts"] == {"left": 4, "right": 4}
    assert report["metrics"]["short_same_hand_pairs"] == 6
    assert report["metrics"]["short_hand_switch_pairs"] == 1
    assert report["metrics"]["planned_axis_adoption_fraction"] == 1.
    assert report["metrics"]["chunk_family_counts"] == {"synthetic": 2}
    assert report["human_demonstrations_used"] is False
    assert report["human_likeness_verified"] is False
    assert report["scores_are_human_preferences"] is False


def test_rapid_same_hand_four_note_roll_is_review_flag_only():
    report = analyze(*sample(axes=["left_1", "left_2", "left_3",
                                   "left_4"]))
    assert report["flag_counts"]["rapid_single_hand_sequence"] == 1
    flag = next(r for r in report["review_flags"]
                if r["kind"] == "rapid_single_hand_sequence")
    assert flag["detail"]["contacts"] == 4
    assert flag["start_event_index"] == 0
    assert "warning" not in flag


def test_fast_depth_jump_is_described_without_human_claim():
    report = analyze(*sample(axes=["left_1", "left_4", "right_1"]))
    assert report["flag_counts"]["quick_same_hand_depth_jump"] == 1
    assert not report["flag_policy"]["thresholds_calibrated_to_humans"]


def test_gap_breaks_fast_hand_run():
    report = analyze(*sample(
        axes=["left_1", "left_2", "left_3", "left_4"],
        gaps=[50., 600., 50.]))
    assert "rapid_single_hand_sequence" not in report["flag_counts"]
    assert report["metrics"]["short_same_hand_pairs"] == 2


def test_one_second_hand_window_detects_concentration():
    # 13 unique axes can be repeated in the *review input*, not a physical
    # reuse feasibility assertion. This is a descriptive analysis only.
    report = analyze(*sample(
        axes=["left_1", "left_2", "left_3", "left_4"] * 4,
        gaps=[60.] * 15))
    assert report["flag_counts"]["one_second_hand_concentration"] >= 1
    assert report["metrics"]["max_hand_contacts_in_rolling_second"]["left"] == 16


def test_replan_is_counted_and_not_mislabeled_as_planned_axis():
    s, t, r = sample()
    t["decisions"][3]["axes"] = ["right_1"]
    t["decisions"][3]["local_replan"] = True
    report = analyze(s, t, r)
    assert report["metrics"]["local_replan_count"] == 1
    assert report["metrics"]["planned_axis_adoption_fraction"] == 0.875


def test_empty_returned_flag_list_does_not_hide_counts():
    report = analyze(*sample(
        axes=["left_1", "left_2", "left_3", "left_4"]), max_flags=0)
    assert report["total_flags"] > 0
    assert report["returned_flags"] == 0
    assert not report["review_flags"]


@pytest.mark.parametrize("mutation", [
    "wrong_sha", "prefix", "missing_event", "wrong_due",
    "unknown_axis", "invalid_chunk", "duplicate_event",
    "incorrect_desired", "nonphysical_chord",
])
def test_invalid_v08_artifacts_fail_closed(mutation):
    s, t, r = sample()
    if mutation == "wrong_sha":
        r["chart_sha256"] = "b" * 64
    elif mutation == "prefix":
        s["termination_reason"] = "diagnostic_prefix_complete"
    elif mutation == "missing_event":
        t["decisions"].pop()
    elif mutation == "wrong_due":
        t["decisions"][1]["due_ms"] += 10
    elif mutation == "unknown_axis":
        t["decisions"][0]["axes"] = ["left_9"]
    elif mutation == "invalid_chunk":
        s["abstract_schedule"]["chunks"][0]["event_count"] = 3
    elif mutation == "duplicate_event":
        r["events"][1]["event_index"] = 0
    elif mutation == "incorrect_desired":
        t["decisions"][0]["desired_axes"] = ["right_1"]
    elif mutation == "nonphysical_chord":
        r["events"][0]["chord_size"] = 2
    with pytest.raises(ValueError):
        analyze(s, t, r)


def test_nonnegative_flag_limit():
    with pytest.raises(ValueError):
        analyze(*sample(), max_flags=-1)
