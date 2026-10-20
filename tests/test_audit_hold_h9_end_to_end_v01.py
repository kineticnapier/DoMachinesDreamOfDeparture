from __future__ import annotations

from scripts.audit_hold_h9_end_to_end_v01 import run_synthetic_contract_fixture


def test_h9_synthetic_contract_fixture_is_deterministic_and_complete():
    first = run_synthetic_contract_fixture()
    second = run_synthetic_contract_fixture()
    assert first == second
    assert first["evidence_scope"] == "synthetic_sensor_contract_only"
    assert first["judge_state"] == "completed"
    assert first["judge_failure"] is None
    assert first["hold_completion_credit"] == 1
    assert first["held_during_unrelated_keydown"]
    assert first["normal_input_resumed"]
    assert first["overlay_completed_ids"] == [7]
