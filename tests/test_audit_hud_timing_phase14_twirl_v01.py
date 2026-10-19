from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_hud_timing_phase14_twirl_v01 as phase14


def test_fixed_three_prove_visible_causal_recovery_and_motor_lead():
    manifest = json.loads(Path(
        "reports/hud-timing-phase11-selection-20261019-r1.json"
    ).read_text(encoding="utf-8"))
    result, traces = phase14.run(
        phase14.TRAIN_DIR, manifest, target_only=True
    )

    assert result["verdict"] == "TWIRL_CAUSAL_RECOVERY_PASS"
    assert result["fixed_3"]["new_clean_recoveries"] == 2
    assert result["fixed_3"]["candidate_clean"] == 3
    assert result["fixed_3"]["motor_ready_44ms"] == 3
    assert result["false_positive_ids"] == 0
    assert result["existing_success_regressions"] == 0
    assert len(traces["events"]) == 3
    assert all(
        event["twirl_marker_first_visible_s"]
        <= event["motor_deadline_s_posthoc"]
        for event in traces["events"]
    )


def test_phase14_audit_keeps_truth_out_of_policy_capture_source():
    assert phase14._source(twirl=False).twirl_recovery is False
    candidate = phase14._source(twirl=True)
    assert candidate.twirl_recovery is True
    assert candidate.midspin_recovery is True
