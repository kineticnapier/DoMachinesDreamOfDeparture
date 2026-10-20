from scripts.audit_hold_h7_mesh_v01 import run


def test_h7_report_separates_mesh_buffer_proxy_and_live_evidence() -> None:
    result = run()
    assert result["verdict"] == "HOLD_LIVE_VALIDATION_REQUIRED"
    assert all(result["checks"].values())
    assert not result["causality"]["student_receives_hold_length"]
    assert all(result["unverified"].values())
