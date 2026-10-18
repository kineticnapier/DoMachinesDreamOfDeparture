from __future__ import annotations

from scripts import audit_countdown_hud_v3_phase1 as audit


def test_phase1_audit_separates_legacy_ambiguity_from_v3_display() -> None:
    result = audit.run()
    assert result["status"] == "PASS"
    assert result["legacy_hud263"]["status"] == "INDISTINGUISHABLE"
    pair = result["countdown_3_vs_5"]
    assert pair["first_distinguishable_observation_a"]["digit"] == 2
    assert pair["first_distinguishable_observation_b"]["digit"] == 4
    assert pair["required_lead_met_in_fixture"]


def test_audit_contract_has_no_truth_fields_and_go_gap_is_not_constant() -> None:
    result = audit.run()
    assert result["capture_contract"]["forbidden_truth_fields_absent"]
    gaps = result["go_to_first_target_is_not_fixed"]
    assert gaps["status"] == "PASS"
    values = [x["go_to_first_target_ms_posthoc"] for x in gaps["fixtures"]]
    assert values[0] != values[1]
