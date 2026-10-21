from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_phase20b_stock_floor_state_v01.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase20b_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_phase16_audit_reproduces_no_fail_backlog() -> None:
    audit = _load_module()

    result = audit.run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
        Path("F:/adofai-code"),
    )

    assert result["verdict"] == "STOCK_MODEL_BACKLOG_REPRODUCED"
    comparison = result["no_fail_comparison"]
    assert comparison["comparison_count"] == 54
    assert comparison["judged_target_parity_count"] == 54
    assert comparison["judgement_parity_count"] == 54
    assert comparison["intention_backlog"]["count"] == 50
    assert comparison["intention_backlog"]["first"]["policy_local_id"] == 38
    assert comparison["intention_backlog"]["last"]["policy_local_id"] == 89
    assert comparison["intention_backlog"]["resynchronized"]["policy_local_id"] == 90


def test_fixed_phase16_fail_snapshot_does_not_claim_continuing_parity() -> None:
    audit = _load_module()

    result = audit.run(
        ROOT / "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        ROOT / "data/DMDOD-v140-tuf-p11-p20/Train",
        Path("F:/adofai-code"),
    )

    snapshot = result["fail_mode_snapshot"]
    assert snapshot["first_input"]["target_ordinal"] == 36
    assert snapshot["first_input"]["judgement"] == "too_late"
    assert snapshot["failure"]["kind"] == "auto_miss_fail"
    assert snapshot["failure"]["observed_time_s"] == 3.3549999999997415
    assert snapshot["initialization"]["kind"] == "posthoc_evaluator_snapshot"
