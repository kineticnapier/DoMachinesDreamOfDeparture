import json
from pathlib import Path

import pytest

from dmdod.research.stock_runtime_parity_v01 import compare_runtime_to_reference, extract_runtime_transitions
from dmdod.research.stock_runtime_trace_v02 import load_runtime_jsonl_v02, parse_runtime_jsonl_v02


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
