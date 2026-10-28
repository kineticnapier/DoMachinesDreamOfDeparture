from pathlib import Path

import pytest

from dmdod.research.stock_runtime_trace_v02 import (
    audit_trace_sessions,
    load_runtime_jsonl_v02,
    parse_runtime_jsonl_v02,
    TraceEvent,
    TraceSession,
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


class _NoSliceEvents(tuple):
    """Reject event tuple slices so a quadratic lookup cannot slip back in."""

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("audit must not copy the remaining event tuple")
        return super().__getitem__(index)


def _synthetic_events(session_id: str, kinds: list[str]) -> _NoSliceEvents:
    return _NoSliceEvents(
        TraceEvent(
            session_id=session_id,
            sequence=index + 1,
            frame=index + 1,
            realtime_s=index * 0.01,
            kind=kind,
            no_fail=False,
            hook="synthetic",
            payload=(("judgement", kind), ("floor_evaluator_only", index)),
        )
        for index, kind in enumerate(kinds)
    )


def test_reverse_lookup_preserves_first_strictly_later_match_and_session_boundary() -> None:
    kinds = [
        "key_down", "auto_fail_miss", "judgement", "add_hit",
        "counter_update", "visible_judgement_signal", "key_down",
        "judgement", "add_hit", "visible_judgement_signal",
        "auto_fail_miss", "add_hit", "key_down",
    ]
    first = TraceSession("first", "synthetic_fixture", None, None, _synthetic_events("first", kinds), ())
    second = TraceSession(
        "second", "synthetic_fixture", None, None,
        _synthetic_events("second", ["judgement", "key_down"]), (),
    )
    result = audit_trace_sessions((first, second))
    first_result, second_result = result["sessions"]

    assert [pair["judgement_sequence"] for pair in first_result["keydown_judgement_pairs"]] == [2, 8, None]
    assert [pair["judgement"] for pair in first_result["keydown_judgement_pairs"]] == [
        "auto_fail_miss", "judgement", None,
    ]
    assert [item["next_judgement_sequence"] for item in first_result["auto_transitions"]] == [3, None]
    assert [item["visible_sequence"] for item in first_result["hit_text_signal_delays"]] == [5, 10, None]
    assert [item["signal_kind"] for item in first_result["hit_text_signal_delays"]] == [
        "counter_update", "visible_judgement_signal", None,
    ]
    assert [item["delay_s"] for item in first_result["hit_text_signal_delays"][:2]] == pytest.approx([0.01, 0.01])
    assert first_result["hit_text_signal_delays"][2]["delay_s"] is None
    assert second_result["keydown_judgement_pairs"][0]["judgement_sequence"] is None
    assert result["student_visible_fields"] == []
    assert result["evaluator_only"] is True
    assert result["runtime_parity"] == "RUNTIME_PARITY_UNVERIFIED"


def test_reverse_lookup_handles_long_synthetic_session_without_tuple_slicing() -> None:
    kinds = [
        "key_down", "auto_fail_miss", "judgement",
        "add_hit", "counter_update", "visible_judgement_signal",
    ] * 400
    session = TraceSession("long", "synthetic_fixture", None, None, _synthetic_events("long", kinds), ())
    result = audit_trace_sessions((session,))["sessions"][0]

    assert result["event_count"] == 2400
    assert len(result["keydown_judgement_pairs"]) == 400
    assert len(result["auto_transitions"]) == 400
    assert len(result["hit_text_signal_delays"]) == 400
    assert result["keydown_judgement_pairs"][-1]["judgement_sequence"] == 2396
    assert result["auto_transitions"][-1]["next_judgement_sequence"] == 2397
    assert result["hit_text_signal_delays"][-1]["visible_sequence"] == 2399
