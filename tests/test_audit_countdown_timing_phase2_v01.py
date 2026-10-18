from __future__ import annotations

import ast
import inspect

from scripts import audit_countdown_timing_phase2_v01 as audit


def test_predeclared_synthetic_phase2_cases_pass() -> None:
    cases = []
    for countdown in (3, 5):
        for bpm in (120.0, 240.0, 480.0):
            segment = audit._synthetic_segment(
                bpm=bpm, countdown=countdown
            )
            cases.append(audit._case(
                segment, f"N{countdown}-{bpm}",
                max_s=min(segment.duration_s, 4.0),
            ))
    assert all(item["status"] == "PASS" for item in cases)
    assert all(item["unique_countdown_candidate_ids"] == 1 for item in cases)
    assert all(not item["false_candidate_ids_posthoc"] for item in cases)


def test_pitch_200_case_passes_without_fixed_go_delay() -> None:
    segment = audit._synthetic_segment(
        bpm=120.0, countdown=4, pitch=200.0
    )
    result = audit._case(
        segment, "pitch", max_s=min(segment.duration_s, 3.0)
    )
    assert result["status"] == "PASS"


def test_policy_collection_does_not_read_target_truth() -> None:
    tree = ast.parse(inspect.getsource(audit._collect_visible_stream))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "targets" not in attributes
    assert "target_time_s" not in attributes
    assert "episode_time_s" not in attributes


def test_posthoc_lead_lower_is_predicted_due_lower_not_truth() -> None:
    class Target:
        episode_time_s = 10.0

    class Segment:
        targets = (Target(),)

    rows = [{
        "step": 100,
        "observed_at_s": 1.0,
        "pending": [{
            "local_id": "countdown:0",
            "due_lower_s": 9.98,
            "due_upper_s": 10.02,
            "due_mid_s": 10.0,
            "source": "countdown_first_event",
        }],
    }]
    result = audit._posthoc_first_event(rows, Segment())
    assert result["first_matching_prediction"]["lead_lower_ms_posthoc"] == 8980.0
