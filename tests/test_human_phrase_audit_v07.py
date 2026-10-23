from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from audit_human_phrase_v07 import analyze_replay


def test_offline_output_does_not_misattribute_v06_hits():
    rows = [
        {"event_index": i, "episode_s": i * .03125, "chord_size": 1,
         "decisions": {"pattern_first_v06": {"axes": ["left_1"]}}}
        for i in range(16)
    ]
    source = {
        "format": "dmdogo-pattern-first-physical-v0.6",
        "events": rows,
        "policies": {"pattern_first_v06": {"saved_metrics": {"hits": 16}}},
    }
    result = analyze_replay(source)
    assert result["summary"]["v07_hits"] is None
    assert result["summary"]["v07_physical_motor_evaluated"] is False
    assert result["v06_physical_scorer_metrics_only"]["hits"] == 16
    assert result["summary"]["different_from_v06_physical_allocation"] > 0


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError):
        analyze_replay({"format": "unrelated", "events": []})
