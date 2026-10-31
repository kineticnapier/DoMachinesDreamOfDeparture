import json
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


@pytest.mark.parametrize("field,value", [
    ("no_fail", "false"),
    ("no_fail", "true"),
    ("no_fail", 0),
    ("no_fail", 1),
    ("sequence", "1"),
    ("sequence", 1.5),
    ("sequence", True),
    ("frame", "2"),
    ("frame", 2.5),
    ("frame", False),
    ("realtime_s", "1.0"),
    ("realtime_s", True),
])
def test_runtime_event_metadata_rejects_implicit_type_coercion(field: str, value: object) -> None:
    """Malformed JSON types must not silently change NoFail or frame identity."""
    header = {
        "type": "stock_runtime_probe_header", "session_id": "test",
        "evidence_kind": "synthetic_fixture",
    }
    event = {
        "type": "event", "session_id": "test", "sequence": 1,
        "frame": 2, "realtime_s": 1.0, "kind": "frame_begin",
        "no_fail": False, "hook": "Simulated_PlayerControl_Update",
    }
    event[field] = value
    with pytest.raises(ValueError, match="invalid sequence/frame/time/no_fail types"):
        parse_runtime_jsonl_v02([json.dumps(header), json.dumps(event)])


def test_runtime_event_accepts_integer_timestamp_and_boolean_nofail() -> None:
    """Integral JSON timestamps remain valid; no numeric representation is changed."""
    lines = [
        json.dumps({"type": "stock_runtime_probe_header", "session_id": "test"}),
        json.dumps({
            "type": "event", "session_id": "test", "sequence": 0,
            "frame": 0, "realtime_s": 0, "kind": "frame_begin",
            "no_fail": False, "hook": "Simulated_PlayerControl_Update",
        }),
    ]
    event = parse_runtime_jsonl_v02(lines)[0].events[0]
    assert event.no_fail is False
    assert event.sequence == event.frame == 0
    assert event.realtime_s == 0.0


@pytest.mark.parametrize("invalid_id", [1, True, ["test"], {"value": "test"}, ""])
def test_runtime_header_rejects_non_string_session_identity(invalid_id: object) -> None:
    header = {"type": "stock_runtime_probe_header", "session_id": invalid_id}
    with pytest.raises(ValueError, match="header session_id required"):
        parse_runtime_jsonl_v02([json.dumps(header)])


@pytest.mark.parametrize("field,invalid_value,error", [
    ("session_id", 1, "session mismatch"),
    ("session_id", True, "session mismatch"),
    ("kind", 1, "invalid kind/hook types"),
    ("kind", "", "invalid kind/hook types"),
    ("hook", False, "invalid kind/hook types"),
    ("hook", "", "invalid kind/hook types"),
])
def test_runtime_event_rejects_non_string_identity_and_hook_fields(
    field: str, invalid_value: object, error: str,
) -> None:
    header = {"type": "stock_runtime_probe_header", "session_id": "test"}
    event = {
        "type": "event", "session_id": "test", "sequence": 0,
        "frame": 0, "realtime_s": 0.0, "kind": "frame_begin",
        "no_fail": False, "hook": "Simulated_PlayerControl_Update",
    }
    event[field] = invalid_value
    with pytest.raises(ValueError, match=error):
        parse_runtime_jsonl_v02([json.dumps(header), json.dumps(event)])


def test_runtime_event_rejects_oversized_integer_timestamp_with_line_number() -> None:
    header = {"type": "stock_runtime_probe_header", "session_id": "test"}
    event = {
        "type": "event", "session_id": "test", "sequence": 0,
        "frame": 0, "realtime_s": 10 ** 400, "kind": "frame_begin",
        "no_fail": False, "hook": "Simulated_PlayerControl_Update",
    }
    with pytest.raises(ValueError, match=r"line 2: invalid sequence/frame/time"):
        parse_runtime_jsonl_v02([json.dumps(header), json.dumps(event)])


@pytest.mark.parametrize("raw,expected_line,key", [
    (
        '{"type":"stock_runtime_probe_header","session_id":"first","session_id":"second"}',
        1, "session_id",
    ),
    (
        '{"type":"event","session_id":"test","sequence":0,"sequence":1,'
        '"frame":0,"realtime_s":0,"kind":"frame_begin","no_fail":false,'
        '"hook":"Simulated_PlayerControl_Update"}',
        2, "sequence",
    ),
    (
        '{"type":"event","session_id":"test","sequence":0,'
        '"frame":0,"realtime_s":0,"kind":"frame_begin",'
        '"no_fail":false,"no_fail":true,"hook":"Simulated_PlayerControl_Update"}',
        2, "no_fail",
    ),
    (
        '{"type":"event","session_id":"test","sequence":0,'
        '"frame":0,"realtime_s":0,"kind":"frame_begin","no_fail":false,'
        '"hook":"Simulated_PlayerControl_Update",'
        '"payload":{"floor_evaluator_only":1,"floor_evaluator_only":2}}',
        2, "floor_evaluator_only",
    ),
])
def test_runtime_jsonl_rejects_duplicate_keys_at_any_depth(
    raw: str, expected_line: int, key: str,
) -> None:
    """Overwritten fields cannot be promoted into qualified runtime evidence."""
    header = '{"type":"stock_runtime_probe_header","session_id":"test"}'
    lines = [raw] if expected_line == 1 else [header, raw]
    with pytest.raises(ValueError, match=rf"line {expected_line}: duplicate JSON key: {key}"):
        parse_runtime_jsonl_v02(lines)


@pytest.mark.parametrize("raw,expected_line,reason", [
    ('{"type":"stock_runtime_probe_header","session_id":"test","extra":NaN}',
     1, "non-standard JSON constant: NaN"),
    ('{"payload":{"nested":[NaN]}}',
     2, "non-standard JSON constant: NaN"),
    ('{"payload":{"floor_evaluator_only":Infinity}}',
     2, "non-standard JSON constant: Infinity"),
    ('{"payload":{"floor_evaluator_only":-Infinity}}',
     2, "non-standard JSON constant: -Infinity"),
    ('{"payload":{"floor_evaluator_only":1e999}}',
     2, "non-finite JSON number: 1e999"),
    ('{"realtime_s":1e999}',
     2, "non-finite JSON number: 1e999"),
])
def test_runtime_jsonl_rejects_nonfinite_numbers_at_any_depth(
    raw: str, expected_line: int, reason: str,
) -> None:
    """Malformed evaluator-only fields must not survive JSON parsing."""
    header = '{"type":"stock_runtime_probe_header","session_id":"test"}'
    event = {
        "type": "event", "session_id": "test", "sequence": 0,
        "frame": 0, "realtime_s": 0.0, "kind": "frame_begin",
        "no_fail": False, "hook": "Simulated_PlayerControl_Update",
    }
    if expected_line == 1:
        lines = [raw]
    else:
        # Replace the event's specified field with its raw JSON token,
        # keeping the non-standard token intact rather than json.dumps it.
        if raw.startswith('{"payload"'):
            event_text = json.dumps(event)[:-1] + ',' + raw[1:]
        else:
            event_text = json.dumps(event).replace('"realtime_s": 0.0', '"realtime_s": 1e999')
        lines = [header, event_text]
    with pytest.raises(ValueError, match=f"line {expected_line}: {reason}"):
        parse_runtime_jsonl_v02(lines)


def test_runtime_jsonl_keeps_valid_finite_exponents_in_payload() -> None:
    header = '{"type":"stock_runtime_probe_header","session_id":"test"}'
    event = ('{"type":"event","session_id":"test","sequence":0,"frame":0,'
             '"realtime_s":1e-3,"kind":"frame_begin","no_fail":false,'
             '"hook":"Simulated_PlayerControl_Update",'
             '"payload":{"nested":[1e-4,-2.5e2]}}')
    result = parse_runtime_jsonl_v02([header, event])[0].events[0]
    assert result.realtime_s == pytest.approx(0.001)
    assert result.value("payload")["nested"] == pytest.approx([0.0001, -250.0])
