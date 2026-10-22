from pathlib import Path

import pytest

from dmdod.research.stock_runtime_log_v01 import (
    RuntimeEventKind,
    audit_runtime_capture,
    load_runtime_jsonl,
    parse_runtime_jsonl,
)


FIXTURE = Path(__file__).parent / "fixtures/phase25_stock_runtime_synthetic.jsonl"


def test_synthetic_runtime_fixture_parses_but_never_proves_parity() -> None:
    capture = load_runtime_jsonl(FIXTURE)
    result = audit_runtime_capture(capture)
    assert capture.events[1].kind is RuntimeEventKind.CHECK_POST_HOLD_FAIL
    assert result["audited_dll_match"]
    assert result["missing_required_kinds"] == []
    assert result["verdict"] == "RUNTIME_PARITY_UNVERIFIED"
    assert result["student_visible_fields"] == []


def test_runtime_parser_rejects_nonmonotonic_or_missing_header() -> None:
    with pytest.raises(ValueError):
        parse_runtime_jsonl(['{"type":"event"}'])
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    bad = [*lines, lines[-1].replace('"sequence":9', '"sequence":8')]
    with pytest.raises(ValueError):
        parse_runtime_jsonl(bad)


def test_evaluator_floor_fields_are_payload_not_student_output() -> None:
    capture = load_runtime_jsonl(FIXTURE)
    check = next(row for row in capture.events if row.kind is RuntimeEventKind.CHECK_POST_HOLD_FAIL)
    assert check.value("floor_evaluator_only") == 36
    assert audit_runtime_capture(capture)["evaluator_only"]
