from __future__ import annotations

import sys
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_gesture_chunks_v08 import analyze


def test_v08_offline_audit_does_not_inherit_v06_physical_hits():
    source = {
        "format": "dmdogo-pattern-first-physical-v0.6",
        "events": [
            {"event_index": i, "episode_s": .03125 * i,
             "chord_size": 1,
             "decisions": {"pattern_first_v06": {"axes": ["left_1"]}}}
            for i in range(16)
        ],
        "policies": {
            "pattern_first_v06": {
                "saved_metrics": {"hits": 16, "x_accuracy_percent": 100.}
            }
        },
    }
    report = analyze(source)
    assert report["event_count"] == 16
    assert report["v08_physical_motor_evaluated"] is False
    assert report["v08_hits"] is None
    assert report["v08_x_accuracy_percent"] is None
    assert report["v06_saved_physical_stats_only"]["hits"] == 16
    assert len(report["chunks"]) > 1


def test_v08_audit_rejects_unrelated_source():
    with pytest.raises(ValueError):
        analyze({"format": "another-format", "events": []})
