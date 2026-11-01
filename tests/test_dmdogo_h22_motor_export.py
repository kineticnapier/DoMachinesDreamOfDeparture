from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_dmdogo_h22_motor_trace import validate_and_export

CHART = b'{"angleData":[90,90],"settings":{"bpm":120},"actions":[]}'
CHART_NAME = "level.adofai"


def fixture():
    n = 2
    summary = {
        "planning_mode": "gesture_v08",
        "chart": CHART_NAME,
        "chart_sha256": hashlib.sha256(CHART).hexdigest(),
        "total_events": n, "termination_reason": "chart_complete",
        "stats": {"targets": n, "hits": n, "misses": 0,
                  "overloaded": False, "physical_keydowns": n},
    }
    telemetry = {
        "decisions": [
            {"event_index": i, "chosen": "left_1" if i == 0 else "right_1",
             "due_ms": 1500. + i * 80,
             "keydowns": [{"axis": "left_1" if i == 0 else "right_1",
                           "episode_s": 1.508 + i * .080}]}
            for i in range(n)
        ],
        "physical": {"local_replans": 0, "unassigned_events": 0},
    }
    return summary, telemetry


def test_valid_two_event_export():
    summary, telemetry = fixture()
    row = validate_and_export(summary, telemetry, CHART, CHART_NAME)
    assert row["h22_strict_eligible"]
    assert row["chart_sha256"] == hashlib.sha256(CHART).hexdigest()
    assert row["human_likeness_verified"] is False


@pytest.mark.parametrize("mutate", [
    lambda s, t: s["stats"].update(hits=1, misses=1),
    lambda s, t: s.update(termination_reason="miss"),
    lambda s, t: t["physical"].update(local_replans=1),
    lambda s, t: t["decisions"][0]["keydowns"].clear(),
    lambda s, t: t["physical"].update(unassigned_events=1),
])
def test_failure_gates_never_claim_strict_eligible(mutate):
    summary, telemetry = fixture()
    mutate(summary, telemetry)
    assert not validate_and_export(summary, telemetry, CHART, CHART_NAME)["h22_strict_eligible"]


def test_chart_hash_mismatch_rejected():
    summary, telemetry = fixture()
    with pytest.raises(ValueError, match="SHA256"):
        validate_and_export(summary, telemetry, CHART + b"!", CHART_NAME)


def test_unordered_identity_rejected():
    summary, telemetry = fixture()
    telemetry["decisions"][1]["event_index"] = 0
    with pytest.raises(ValueError, match="event identities"):
        validate_and_export(summary, telemetry, CHART, CHART_NAME)
