from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_fingering_human_log_v011 import parse_human_log, render_html
from prepare_dmdogo_v09_human_input import (
    normalize_legacy, validate_normalized, FORMAT,
)


def test_existing_recorder_log_can_be_normalized_without_alignment_claim(tmp_path):
    log = tmp_path / "capture.txt"
    log.write_text(
        "# K04 = home key -> LI\n"
        "# K09 = home key -> RI\n"
        "100.000 K04:LI\n"
        "125.000 K09:RI\n",
        encoding="utf-8",
    )
    rows = parse_human_log(log)
    payload = normalize_legacy(rows, chart_sha256="A" * 64)
    assert payload["format"] == FORMAT
    assert payload["chart_sha256"] == "a" * 64
    assert payload["alignment"]["status"] == "unverified"
    assert not payload["alignment"]["event_correspondence_verified"]
    assert not payload["finger_verification"]["independent_finger_observation"]
    assert payload["records"][0]["axis_proxy"] == "left_1"
    assert payload["records"][1]["axis_proxy"] == "right_1"
    assert all(item["finger_label_evidence"] ==
               "declared_key_mapping_not_observed"
               for item in payload["records"])


@pytest.mark.parametrize("tamper", [
    "nonfinite", "negative", "unsorted", "wrong_finger",
    "changed_axis", "wrong_timebase", "false_verified_alignment",
    "empty_records", "wrong_sha", "different_valid_axis", "spoofed_finger_observation",
])
def test_unsafe_human_source_assumptions_rejected(tamper):
    rows = [
        {"time_ms": 100.0, "key": "K04",
         "axis": "left_1", "finger": "LI"},
        {"time_ms": 110.0, "key": "K09",
         "axis": "right_1", "finger": "RI"},
    ]
    payload = normalize_legacy(rows, chart_sha256="0" * 64)
    if tamper == "nonfinite":
        payload["records"][1]["time_ms"] = float("nan")
    elif tamper == "negative":
        payload["records"][0]["time_ms"] = -1.
    elif tamper == "unsorted":
        payload["records"][1]["time_ms"] = 10.
    elif tamper == "wrong_finger":
        payload["records"][0]["finger_label"] = "unknown"
    elif tamper == "changed_axis":
        payload["records"][0]["axis_proxy"] = "left_7"
    elif tamper == "wrong_timebase":
        payload["clock_reference"] = "oracle_episode_ms"
    elif tamper == "false_verified_alignment":
        payload["alignment"]["event_correspondence_verified"] = True
    elif tamper == "empty_records":
        payload["records"] = []
    elif tamper == "wrong_sha":
        payload["chart_sha256"] = "unknown"
    elif tamper == "different_valid_axis":
        payload["records"][0]["axis_proxy"] = "right_1"
    elif tamper == "spoofed_finger_observation":
        payload["finger_verification"]["independent_finger_observation"] = True
    with pytest.raises(ValueError):
        validate_normalized(payload)


def test_missing_or_bad_chart_identity_rejected():
    with pytest.raises(ValueError):
        normalize_legacy([{"time_ms": 1, "key": "K04",
                           "axis": "left_1", "finger": "LI"}],
                         chart_sha256="123")


def test_importer_cannot_claim_fingers_verified_by_recording():
    payload = normalize_legacy(
        [{"time_ms": 1.0, "key": "K04",
          "axis": "left_1", "finger": "LI"}],
        chart_sha256="f" * 64,
    )
    assert payload["finger_verification"]["method"] == "fixed_declaration_from_legacy_key_map"
    assert payload["records"][0]["finger_label_evidence"] == "declared_key_mapping_not_observed"


def test_human_fingering_html_formats_first_and_subsequent_intervals():
    report = {
        "chart": "synthetic",
        "alignment": {
            "status": "intervals_consistent",
            "estimated_human_minus_oracle_offset_ms": 0.0,
            "p95_absolute_residual_ms": 0.0,
            "p95_interval_difference_ms": 0.0,
        },
        "policies": {
            "free": {"axis_matches": 2, "hand_matches": 2},
            "human_proxy": {"axis_matches": 2, "hand_matches": 2},
        },
        "four_second_windows": [],
        "common_prefix_only": False,
        "human": {"median_repeat_ms": None},
        "limitations": [],
    }
    rows = [
        {
            "event_index": index,
            "human": {"time_ms": time_ms, "finger": "LI", "key": "K04"},
            "interval_ms": interval,
            "choices": {"free": ["left_1"], "human_proxy": ["left_1"]},
            "match": {"free": True, "human_proxy": True},
        }
        for index, time_ms, interval in ((0, 100.0, None), (1, 120.54, 20.54))
    ]
    html = render_html(report, rows)
    assert '<td>—</td>' in html
    assert '<td>20.5</td>' in html
    assert 'data-dense="0"' in html
    assert 'data-dense="1"' in html
