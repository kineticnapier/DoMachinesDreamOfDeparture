import json
from pathlib import Path

from scripts.audit_hold_h8_release_v01 import run


def test_h8_train25_abstains_without_live_stock_pixels() -> None:
    h5 = json.loads(Path(
        "reports/hold-h5-train-audit-20261020-r1.json"
    ).read_text(encoding="utf-8"))
    h7 = json.loads(Path(
        "reports/hold-h7-mesh-observability-20261020-r1.json"
    ).read_text(encoding="utf-8"))
    result = run(h5, h7)
    assert result["verdict"] == "HOLD_LIVE_VALIDATION_REQUIRED"
    assert all(result["checks"].values())
    assert result["fixed_train25"]["abstentions"] == 25
    assert result["fixed_train25"]["production_release_forecasts"] == 0
    assert not result["student_received_train_duration_truth"]
