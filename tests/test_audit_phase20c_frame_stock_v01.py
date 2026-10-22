from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_phase20c_frame_stock_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase20c", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_frame_audit_preserves_target_and_category_parity() -> None:
    audit = _load()
    result = audit.run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
        Path("F:/adofai-code"),
    )

    assert result["verdicts"] == ["FRAME_ACCURATE_REFERENCE_VALIDATED"]
    assert result["no_fail"]["target_parity_count"] == 54
    assert result["no_fail"]["category_parity_count"] == 54
    assert result["no_fail"]["backlog"]["count"] == 50


def test_frame_audit_moves_automatic_transitions_off_keydown_timestamps() -> None:
    audit = _load()
    result = audit.run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
        Path("F:/adofai-code"),
    )

    assert result["deadlines"]["event36_no_fail"]["first_virtual_update_s"] == 3.25
    assert result["deadlines"]["event89_no_fail"]["first_virtual_update_s"] == 5.32
    assert result["deadlines"]["event36_fail"]["first_virtual_update_s"] == 3.32
    assert result["fail_snapshot"]["failure"]["observed_time_s"] == pytest.approx(3.32)
    assert result["repository_first_state_difference"]["fail"]["time_s"] == 3.25
