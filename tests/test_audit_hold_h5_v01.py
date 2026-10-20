from __future__ import annotations

from pathlib import Path

from scripts.audit_hold_h5_v01 import EXPECTED, run


TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")


def test_fixed_train_hold_timing_and_judgement_audit() -> None:
    result = run(TRAIN_DIR)
    assert result["verdict"] == "CODE_PARITY_PASS_RUNTIME_UNVERIFIED"
    assert all(result["checks"].values())
    assert result["summary"]["chart_count"] == len(EXPECTED) == 3
    assert result["summary"]["hold_count"] == 25
    assert result["summary"]["max_code_parity_error_s"] <= 1e-6
    assert result["timing_evidence"]["live_game_capture"] == "UNVERIFIED"
    assert result["judgement_evidence"]["live_game_capture"] == "UNVERIFIED"
