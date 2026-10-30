import json
from dataclasses import replace
from pathlib import Path

import pytest

from dmdod.research.stock_runtime_log_v01 import AUDITED_DLL_SHA256
from dmdod.research.stock_runtime_parity_v01 import compare_runtime_to_reference, extract_runtime_transitions
from dmdod.research.stock_runtime_trace_v02 import (
    REQUIRED_HOOKS,
    TraceEvent,
    TraceSession,
    load_runtime_jsonl_v02,
    parse_runtime_jsonl_v02,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "phase32-stock-runtime-multisession-synthetic.jsonl"
REFERENCE = ROOT / "reports" / "phase20c-frame-stock-state-20261022-r1.json"


def test_synthetic_capture_never_confirms_runtime_parity() -> None:
    session = load_runtime_jsonl_v02(FIXTURE)[0]
    result = compare_runtime_to_reference(session, json.loads(REFERENCE.read_text(encoding="utf-8")))
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
    assert not result["live_qualified"]
    assert not result["uses_virtual_10ms_as_runtime_time"]


def test_transition_extraction_requires_explicit_floor_map_for_target_identity() -> None:
    session = load_runtime_jsonl_v02(FIXTURE)[0]
    unknown = extract_runtime_transitions(session)
    mapped = extract_runtime_transitions(session, floor_to_target={37: 36})
    assert unknown[0].target_ordinal is None
    assert mapped[0].target_ordinal == 36


def test_live_explicit_category_mismatch_is_reported_without_claiming_success() -> None:
    lines = [
        '{"type":"stock_runtime_probe_header","session_id":"live","evidence_kind":"live_runtime","assembly_csharp_sha256":"0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD"}',
        '{"type":"event","session_id":"live","sequence":1,"frame":1,"realtime_s":1,"kind":"key_down","hook":"HitInputEvent","no_fail":true,"song_time_evaluator_only":3.261}',
        '{"type":"event","session_id":"live","sequence":2,"frame":1,"realtime_s":1.001,"kind":"judgement","hook":"AddHit","no_fail":true,"judgement":"Perfect","song_time_evaluator_only":3.261}',
        '{"type":"event","session_id":"live","sequence":3,"frame":1,"realtime_s":1.002,"kind":"judgement","hook":"SwitchChosen","hook_phase":"postfix","no_fail":true,"floor_before_evaluator_only":39,"floor_after_evaluator_only":40,"song_time_evaluator_only":3.261}',
    ]
    session = parse_runtime_jsonl_v02(lines)[0]
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    result = compare_runtime_to_reference(session, reference, floor_to_target={40: 37})
    assert result["explicit_mismatch"]
    # Required hook coverage is intentionally incomplete, so mismatch evidence is
    # retained without upgrading this tiny fixture to a live parity verdict.
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
    assert result["comparisons"][0]["fields"]["judgement"] == "MISMATCH"


def test_song_time_is_never_substituted_with_virtual_frame_grid() -> None:
    session = load_runtime_jsonl_v02(FIXTURE)[0]
    rows = extract_runtime_transitions(session)
    assert rows[0].song_time_s is None

@pytest.mark.parametrize("source,canonical", [
    ("TooEarly", "too_early"),
    ("too_early", "too_early"),
    ("VeryEarly", "very_early"),
    ("very_early", "very_early"),
    ("EarlyPerfect", "early_perfect"),
    ("early_perfect", "early_perfect"),
    ("LatePerfect", "late_perfect"),
    ("late_perfect", "late_perfect"),
    ("VeryLate", "very_late"),
    ("TooLate", "too_late"),
    ("FailMiss", "fail_miss"),
    ("Perfect", "perfect"),
])
def test_runtime_category_matches_stock_reference(source, canonical):
    def row(sequence, hook, *, kind="judgement", **extra):
        return json.dumps({
            "type": "event", "session_id": "synthetic",
            "sequence": sequence, "frame": 1,
            "realtime_s": 1.0 + sequence * 0.001,
            "kind": kind, "no_fail": True, "hook": hook, **extra,
        })

    session = parse_runtime_jsonl_v02([
        json.dumps({
            "type": "stock_runtime_probe_header",
            "session_id": "synthetic", "evidence_kind": "synthetic_fixture",
        }),
        row(0, "HitInputEvent", kind="key_down"),
        row(1, "AddHit", judgement=source),
        row(2, "SwitchChosen", hook_phase="postfix",
            floor_before_evaluator_only=1, floor_after_evaluator_only=2),
    ])[0]
    reference = {"no_fail": {
        "automatic_transitions": [],
        "event_37_90": [{
            "keydown_at_s": 3.0,
            "stock_target_ordinal": 0,
            "stock_judgement": canonical,
            "stock_current_floor_before": 1,
            "stock_current_floor_after": 2,
        }],
    }}
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
    )
    assert result["comparisons"][0]["fields"]["judgement"] == "MATCH"
    assert result["comparisons"][0]["fields"]["target_ordinal"] == "MATCH"
    assert not result["explicit_mismatch"]
    # A synthetic fixture cannot establish runtime parity.
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"


def _qualified_one_transition(
    *, song_time_s: float | None = 3.261,
) -> tuple[TraceSession, dict[str, object]]:
    """Synthetic construction of a *qualified-shaped* live session for tests.

    This is not an actual live capture and cannot validate external calibration.
    """
    def event(sequence: int, kind: str, hook: str, **payload: object) -> TraceEvent:
        return TraceEvent(
            session_id="qualified-test", sequence=sequence, frame=7,
            realtime_s=1.0 + sequence * 0.001, kind=kind,
            no_fail=True, hook=hook, payload=tuple(sorted(payload.items())),
        )

    timestamp = {} if song_time_s is None else {"song_time_evaluator_only": song_time_s}
    events = [
        event(0, "key_down", "HitInputEvent", **timestamp),
        event(1, "judgement", "AddHit", judgement="Perfect", **timestamp),
        event(
            2, "judgement", "SwitchChosen", hook_phase="postfix",
            floor_before_evaluator_only=1, floor_after_evaluator_only=2,
            **timestamp,
        ),
    ]
    for hook in sorted(REQUIRED_HOOKS - {"HitInputEvent", "AddHit", "SwitchChosen"}):
        events.append(event(len(events), "probe", hook))
    session = TraceSession(
        session_id="qualified-test", evidence_kind="live_runtime",
        dll_sha256=AUDITED_DLL_SHA256, game_build="test",
        events=tuple(events), issues=(),
    )
    reference = {"no_fail": {
        "automatic_transitions": [],
        "event_37_90": [{
            "keydown_at_s": 3.25,
            "stock_target_ordinal": 0,
            "stock_judgement": "perfect",
            "stock_current_floor_before": 1,
            "stock_current_floor_after": 2,
        }],
    }}
    return session, reference


def test_qualified_live_time_is_unknown_without_external_alignment() -> None:
    session, reference = _qualified_one_transition()
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
    )
    assert result["live_qualified"]
    assert result["comparisons"][0]["fields"]["song_time_s"] == "UNKNOWN"
    assert result["comparisons"][0]["song_time_delta_s"] == pytest.approx(0.011)
    assert not result["complete"]
    assert not result["explicit_mismatch"]
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"


def test_qualified_live_time_requires_residual_within_explicit_tolerance() -> None:
    session, reference = _qualified_one_transition()
    matched = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.011, song_time_tolerance_s=0.0001,
    )
    assert matched["comparisons"][0]["fields"]["song_time_s"] == "MATCH"
    assert matched["verdict"] == "RUNTIME_PARITY_CONFIRMED"
    mismatched = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.0, song_time_tolerance_s=0.001,
    )
    assert mismatched["comparisons"][0]["fields"]["song_time_s"] == "MISMATCH"
    assert mismatched["explicit_mismatch"]
    assert mismatched["verdict"] == "RUNTIME_PARITY_MISMATCH"


def test_qualified_live_missing_song_time_cannot_confirm() -> None:
    session, reference = _qualified_one_transition(song_time_s=None)
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.0, song_time_tolerance_s=0.01,
    )
    assert result["comparisons"][0]["fields"]["song_time_s"] == "UNKNOWN"
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"


@pytest.mark.parametrize("settings", [
    {"song_time_offset_s": 0.0},
    {"song_time_tolerance_s": 0.01},
    {"song_time_offset_s": float("nan"), "song_time_tolerance_s": 0.01},
    {"song_time_offset_s": 0.0, "song_time_tolerance_s": float("inf")},
    {"song_time_offset_s": 0.0, "song_time_tolerance_s": -0.01},
    {"song_time_offset_s": True, "song_time_tolerance_s": 0.01},
    {"song_time_offset_s": 0.0, "song_time_tolerance_s": True},
    {"song_time_offset_s": "0.011", "song_time_tolerance_s": 0.01},
    {"song_time_offset_s": 0.0, "song_time_tolerance_s": "0.01"},
    {"song_time_offset_s": 10**400, "song_time_tolerance_s": 0.01},
    {"song_time_offset_s": 0.0, "song_time_tolerance_s": 10**400},
])
def test_invalid_song_time_alignment_is_rejected(settings: dict[str, object]) -> None:
    session, reference = _qualified_one_transition()
    with pytest.raises(ValueError, match="song-time"):
        compare_runtime_to_reference(
            session, reference, floor_to_target={2: 0}, **settings,
        )


def test_finite_integer_alignment_parameters_remain_valid() -> None:
    """Numeric integer offsets/tolerances remain valid for calibrated callers."""
    session, reference = _qualified_one_transition()
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0, song_time_tolerance_s=1,
    )
    assert result["verdict"] == "RUNTIME_PARITY_CONFIRMED"


def test_zero_song_time_is_not_replaced_by_category_timestamp() -> None:
    session, _ = _qualified_one_transition()
    switch = session.events[2]
    payload = dict(switch.payload)
    payload["song_time_evaluator_only"] = 0.0
    updated_switch = replace(switch, payload=tuple(sorted(payload.items())))
    updated_session = replace(
        session,
        events=session.events[:2] + (updated_switch,) + session.events[3:],
    )
    transitions = extract_runtime_transitions(updated_session)
    assert len(transitions) == 1
    assert transitions[0].song_time_s == 0.0


@pytest.mark.parametrize("field,invalid_value", [
    ("floor_before_evaluator_only", "1"),
    ("floor_before_evaluator_only", 1.0),
    ("floor_before_evaluator_only", True),
    ("floor_after_evaluator_only", "2"),
    ("floor_after_evaluator_only", 2.0),
    ("floor_after_evaluator_only", True),
])
def test_runtime_floor_payload_rejects_implicit_type_coercion(
    field: str, invalid_value: object,
) -> None:
    """Malformed floor identities must not confirm synthetic qualified-shaped parity."""
    session, reference = _qualified_one_transition()
    switch = session.events[2]
    payload = dict(switch.payload)
    payload[field] = invalid_value
    altered = replace(switch, payload=tuple(sorted(payload.items())))
    session = replace(session, events=session.events[:2] + (altered,) + session.events[3:])
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.011, song_time_tolerance_s=0.0001,
    )
    field_name = "floor_before" if field.startswith("floor_before") else "floor_after"
    assert result["comparisons"][0]["fields"][field_name] == "UNKNOWN"
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
    assert not result["complete"]


@pytest.mark.parametrize("invalid_value", ["3.261", True, float("nan"), float("inf")])
def test_runtime_song_time_payload_rejects_implicit_type_coercion(
    invalid_value: object,
) -> None:
    """Invalid timestamps cannot be calibrated into a confirmed live comparison."""
    session, reference = _qualified_one_transition()
    events = []
    for event in session.events:
        payload = dict(event.payload)
        if "song_time_evaluator_only" in payload:
            payload["song_time_evaluator_only"] = invalid_value
        events.append(replace(event, payload=tuple(sorted(payload.items()))))
    result = compare_runtime_to_reference(
        replace(session, events=tuple(events)), reference, floor_to_target={2: 0},
        song_time_offset_s=0.011, song_time_tolerance_s=0.0001,
    )
    assert result["comparisons"][0]["fields"]["song_time_s"] == "UNKNOWN"
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
    assert not result["complete"]


def test_runtime_numeric_floor_and_song_time_remain_compatible() -> None:
    """Real JSON integer floors and numeric song times still match."""
    session, reference = _qualified_one_transition(song_time_s=3.261)
    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.011, song_time_tolerance_s=0.0001,
    )
    assert result["verdict"] == "RUNTIME_PARITY_CONFIRMED"


@pytest.mark.parametrize("switch_has_time", [True, False])
def test_previous_frame_category_cannot_confirm_later_switch(
    switch_has_time: bool,
) -> None:
    """A prior-frame AddHit cannot supply category or fallback time evidence.

    This qualified-shaped test is synthetic, not a real ADOFAI capture.
    """
    session, reference = _qualified_one_transition()
    events = list(session.events)
    events[0] = replace(events[0], frame=6)
    events[1] = replace(events[1], frame=6)
    if not switch_has_time:
        switch_payload = dict(events[2].payload)
        del switch_payload["song_time_evaluator_only"]
        events[2] = replace(events[2], payload=tuple(sorted(switch_payload.items())))
    session = replace(session, events=tuple(events))

    transition = extract_runtime_transitions(session)[0]
    assert transition.judgement is None
    assert events[1].sequence not in transition.source_sequences
    if not switch_has_time:
        assert transition.song_time_s is None

    result = compare_runtime_to_reference(
        session, reference, floor_to_target={2: 0},
        song_time_offset_s=0.011, song_time_tolerance_s=0.0001,
    )
    fields = result["comparisons"][0]["fields"]
    assert fields["judgement"] == "UNKNOWN"
    assert fields["song_time_s"] == ("MATCH" if switch_has_time else "UNKNOWN")
    assert not result["complete"]
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
