from __future__ import annotations

from pathlib import Path

from scripts.audit_hold_h6_v01 import run


def test_hold_h6_quality_gate() -> None:
    result = run(
        Path("data/DMDOD-v140-tuf-p11-p20/Train"),
        Path("reports/hold-h5-train-audit-20261020-r1.json"),
        Path("reports/hold-h5-phase14-regression-20261020-r1.json"),
    )
    assert result["verdict"] == "HOLD_PARTIAL_INTEGRATION"
    assert all(result["checks"].values())
    assert result["determinism"]["second_identical"]
    assert result["component_status"]["hud_recognition"].endswith(
        "EXACT_DURATION_BLOCKED"
    )
