from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_stock_hold_h0_v01 import inventory


def test_h0_inventory_records_hold_without_interpreting_duration(tmp_path):
    train = tmp_path / "Train"
    evidence = tmp_path / "evidence"
    train.mkdir()
    evidence.mkdir()
    chart = train / "fixture.adofai"
    chart.write_text(json.dumps({
        "angleData": [0, 90],
        "settings": {"bpm": 120, "pitch": 80},
        "actions": [
            {"floor": 1, "eventType": "Twirl"},
            {"floor": 1, "eventType": "Hold", "duration": "2"},
            {"floor": 2, "eventType": "Hold", "duration": -1,
             "active": False},
        ],
    }), encoding="utf-8")

    result = inventory(train, evidence)

    assert result["hold_chart_count"] == 1
    assert result["hold_action_count"] == 1
    hold = result["charts"][0]["holds"][0]
    assert hold["duration_raw"] == "2"
    assert hold["same_floor_actions"] == ["Twirl", "Hold"]
    assert result["evidence_version_caveat"]
