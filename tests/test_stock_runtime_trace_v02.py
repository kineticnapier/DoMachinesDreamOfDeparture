from pathlib import Path

import pytest

from dmdod.research.stock_runtime_trace_v02 import (
    audit_trace_sessions,
    load_runtime_jsonl_v02,
    parse_runtime_jsonl_v02,
)


FIXTURE = Path(__file__).parent / "fixtures" / "phase32-stock-runtime-multisession-synthetic.jsonl"


def test_multisession_and_nofail_are_separated() -> None:
    sessions = load_runtime_jsonl_v02(FIXTURE)
    assert [session.session_id for session in sessions] == ["nofail-synthetic", "fail-synthetic"]
    result = audit_trace_sessions(sessions)
    assert result["session_count"] == 2
    assert result["sessions"][0]["no_fail_values"] == [True]
    assert result["sessions"][1]["no_fail_values"] == [False]
    assert not result["live_runtime_present"]
    assert result["runtime_parity"] == "RUNTIME_PARITY_UNVERIFIED"


def test_extracts_auto_transition_keydown_mapping_and_signal_delay() -> None:
    result = audit_trace_sessions(load_runtime_jsonl_v02(FIXTURE))["sessions"][0]
    assert result["auto_transitions"][0]["frame"] == 325
    assert result["keydown_judgement_pairs"][0]["judgement_sequence"] == 9
    assert result["hit_text_signal_delays"][0]["delay_s"] == pytest.approx(0.0001)
    assert result["visible_cumulative_counter_delay"] == "UNKNOWN_NOT_HOOKED"


def test_detects_gap_duplicate_frame_and_timestamp_reversal_without_inventing_rows() -> None:
    lines = [
        '{"type":"stock_runtime_probe_header","session_id":"x","evidence_kind":"synthetic_fixture"}',
        '{"type":"event","session_id":"x","sequence":1,"frame":2,"realtime_s":2,"kind":"frame_begin","hook":"Simulated_PlayerControl_Update","no_fail":false}',
        '{"type":"event","session_id":"x","sequence":3,"frame":1,"realtime_s":1,"kind":"frame_end","hook":"Simulated_PlayerControl_Update","no_fail":false}',
        '{"type":"event","session_id":"x","sequence":3,"frame":1,"realtime_s":1,"kind":"frame_end","hook":"Simulated_PlayerControl_Update","no_fail":false}',
    ]
    session = parse_runtime_jsonl_v02(lines)[0]
    assert "SEQUENCE_GAP:2-2" in session.issues
    assert "DUPLICATE_SEQUENCE:3" in session.issues
    assert any(issue.startswith("FRAME_REVERSAL") for issue in session.issues)
    assert any(issue.startswith("TIME_REVERSAL") for issue in session.issues)
    assert len(session.events) == 3


def test_rejects_missing_fields_and_duplicate_session_ids() -> None:
    with pytest.raises(ValueError, match="missing hook"):
        parse_runtime_jsonl_v02([
            '{"type":"stock_runtime_probe_header","session_id":"x"}',
            '{"type":"event","session_id":"x","sequence":1,"frame":1,"realtime_s":1,"kind":"x","no_fail":false}',
        ])
    with pytest.raises(ValueError, match="session_id must be unique"):
        parse_runtime_jsonl_v02([
            '{"type":"stock_runtime_probe_header","session_id":"x"}',
            '{"type":"event","session_id":"x","sequence":1,"frame":1,"realtime_s":1,"kind":"x","hook":"x","no_fail":false}',
            '{"type":"stock_runtime_probe_header","session_id":"x"}',
            '{"type":"event","session_id":"x","sequence":1,"frame":1,"realtime_s":1,"kind":"x","hook":"x","no_fail":false}',
        ])


def test_internal_fields_never_become_student_output() -> None:
    result = audit_trace_sessions(load_runtime_jsonl_v02(FIXTURE))
    assert result["student_visible_fields"] == []
    assert result["evaluator_only"]
