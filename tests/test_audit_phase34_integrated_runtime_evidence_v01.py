from __future__ import annotations

import importlib.util
from hashlib import sha256
import subprocess
import sys
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_phase34_integrated_runtime_evidence_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase34", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def test_integrated_report_keeps_build_and_runtime_evidence_separate() -> None:
    result = _load().integrate(
        _report("phase30-dll-signature-audit-20261022-r1.json"),
        _report("phase31-stock-runtime-probe-build-20261022-r1.json"),
        _report("phase32-runtime-trace-ingestion-20261022-r1.json"),
        _report("phase33-stock-runtime-parity-20261022-r1.json"),
    )
    assert "DLL_SIGNATURES_VERIFIED" in result["verdicts"]
    assert "RUNTIME_TRACE_MOD_BUILT" in result["verdicts"]
    assert "RUNTIME_TRACE_MOD_UNVERIFIED" in result["verdicts"]
    assert "RUNTIME_PARITY_UNVERIFIED" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]
    assert "RUNTIME_TRACE_CAPTURED" not in result["verdicts"]


def test_synthetic_capture_cannot_confirm_visible_feedback_or_parity() -> None:
    result = _load().integrate(
        {"verdict": "DLL_SIGNATURES_VERIFIED"},
        {"build": {"success": True}, "runtime_loaded": False, "harmony_hooks_observed": False},
        {"live_runtime_present": False, "sessions": [{"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": "UNKNOWN_NOT_HOOKED"}]},
        {"verdict": "RUNTIME_PARITY_UNVERIFIED"},
    )
    assert "STOCK_VISIBLE_FEEDBACK_PARTIAL" in result["verdicts"]
    assert not result["gates"]["live_runtime_trace_captured"]
    assert result["student_visible_fields"] == []
    assert not result["production_changed"]


def test_full_gate_requires_runtime_load_hooks_counter_and_comparison() -> None:
    result = _complete_evidence([
        {"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01},
    ])
    assert "RUNTIME_TRACE_CAPTURED" in result["verdicts"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]



def _matching_reports(sessions: list[dict]) -> tuple[dict, dict, dict, dict]:
    captured = [
        {"session_id": f"session-{i}", "evidence_kind": "live_runtime",
         "no_fail_values": [bool(i % 2)], **row}
        for i, row in enumerate(sessions)
    ]
    compared = [
        {"session_id": row["session_id"], "no_fail": row["no_fail_values"][0],
         "live_qualified": True, "verdict": "RUNTIME_PARITY_CONFIRMED"}
        for row in captured
    ]
    return (
        {"verdict": "DLL_SIGNATURES_VERIFIED"},
        {"build": {"success": True}, "runtime_loaded": True, "harmony_hooks_observed": True},
        {"input": "traces/live.jsonl", "input_sha256": "a" * 64,
         "all_sessions_complete": True,
         "live_runtime_present": True, "sessions": captured},
        {"runtime": "traces/live.jsonl", "runtime_sha256": "a" * 64,
         "verdict": "RUNTIME_PARITY_CONFIRMED",
         "sessions": compared},
    )


def _complete_evidence(sessions: list[dict]) -> dict:
    return _load().integrate(*_matching_reports(sessions))


def test_missing_counter_field_cannot_confirm_visible_feedback() -> None:
    result = _complete_evidence([{"hook_coverage": ["ShowHitText"]}])
    assert "STOCK_VISIBLE_FEEDBACK_PARTIAL" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]
    assert not result["gates"]["visible_cumulative_counter_hook_present"]


@pytest.mark.parametrize(
    "delay",
    [None, "UNKNOWN_NOT_HOOKED", "NOT_MEASURED", float("nan"),
     float("inf"), float("-inf"), -0.01, True],
)
def test_unmeasured_or_invalid_counter_delay_cannot_confirm(delay: object) -> None:
    result = _complete_evidence([{
        "hook_coverage": ["ShowHitText"],
        "visible_cumulative_counter_delay": delay,
    }])
    assert "STOCK_VISIBLE_FEEDBACK_PARTIAL" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]
    assert not result["gates"]["visible_cumulative_counter_hook_present"]


def test_hit_text_and_counter_from_different_sessions_do_not_confirm() -> None:
    result = _complete_evidence([
        {"hook_coverage": ["ShowHitText"],
         "visible_cumulative_counter_delay": "UNKNOWN_NOT_HOOKED"},
        {"hook_coverage": [], "visible_cumulative_counter_delay": 0.01},
    ])
    assert result["gates"]["visible_hit_text_signal_present"]
    assert result["gates"]["visible_cumulative_counter_hook_present"]
    assert "STOCK_VISIBLE_FEEDBACK_PARTIAL" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def test_counter_without_hit_text_does_not_complete_runtime_gate() -> None:
    result = _complete_evidence([
        {"hook_coverage": [], "visible_cumulative_counter_delay": 0.01},
    ])
    assert "STOCK_VISIBLE_FEEDBACK_PARTIAL" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def test_zero_measured_delay_in_same_session_is_valid() -> None:
    result = _complete_evidence([
        {"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.0},
    ])
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]


def test_synthetic_visible_feedback_cannot_substitute_for_live_session() -> None:
    result = _complete_evidence([
        {"evidence_kind": "synthetic_fixture", "hook_coverage": ["ShowHitText"],
         "visible_cumulative_counter_delay": 0.01},
        {"evidence_kind": "live_runtime", "hook_coverage": ["ProcessKeyInputs"],
         "visible_cumulative_counter_delay": "UNKNOWN_NOT_HOOKED"},
    ])
    assert result["gates"]["live_runtime_trace_captured"]
    assert not result["gates"]["visible_hit_text_signal_present"]
    assert not result["gates"]["visible_cumulative_counter_hook_present"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" not in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def test_synthetic_only_sessions_cannot_be_live_even_with_true_flag() -> None:
    result = _complete_evidence([
        {"evidence_kind": "synthetic_fixture", "hook_coverage": ["ShowHitText"],
         "visible_cumulative_counter_delay": 0.01},
    ])
    assert not result["gates"]["live_runtime_trace_captured"]
    assert "RUNTIME_TRACE_CAPTURED" not in result["verdicts"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" not in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def test_truthy_string_live_flag_does_not_confirm_runtime() -> None:
    result = _load().integrate(
        {"verdict": "DLL_SIGNATURES_VERIFIED"},
        {"build": {"success": True}, "runtime_loaded": True, "harmony_hooks_observed": True},
        {"live_runtime_present": "false", "sessions": [
            {"evidence_kind": "live_runtime", "hook_coverage": ["ShowHitText"],
             "visible_cumulative_counter_delay": 0.01},
        ]},
        {"verdict": "RUNTIME_PARITY_CONFIRMED"},
    )
    assert not result["gates"]["live_runtime_trace_captured"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" not in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def test_hook_coverage_string_substring_is_not_exact_hook_evidence() -> None:
    result = _complete_evidence([
        {"hook_coverage": "not-ShowHitText", "visible_cumulative_counter_delay": 0.01},
    ])
    assert not result["gates"]["visible_hit_text_signal_present"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" not in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]


def _assert_parity_unverified(reports: tuple[dict, dict, dict, dict]) -> None:
    result = _load().integrate(*reports)
    assert "RUNTIME_PARITY_UNVERIFIED" in result["verdicts"]
    assert "RUNTIME_PARITY_CONFIRMED" not in result["verdicts"]
    assert "RUNTIME_PARITY_MISMATCH" not in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" in result["verdicts"]
    assert not result["gates"]["reference_comparison_complete"]


def test_parity_from_another_runtime_file_is_not_confirmed() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["runtime"] = "traces/other.jsonl"
    _assert_parity_unverified(reports)


def test_parity_session_id_must_match_phase32() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["sessions"][0]["session_id"] = "different-run"
    _assert_parity_unverified(reports)


def test_parity_nofail_mode_must_match_phase32() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["sessions"][0]["no_fail"] = True
    _assert_parity_unverified(reports)


def test_unqualified_phase33_session_cannot_confirm() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["sessions"][0]["live_qualified"] = False
    _assert_parity_unverified(reports)


def test_incomplete_phase32_capture_cannot_confirm_parity() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[2]["all_sessions_complete"] = False
    _assert_parity_unverified(reports)


def test_parity_summary_must_match_per_session_verdicts() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["verdict"] = "RUNTIME_PARITY_MISMATCH"
    _assert_parity_unverified(reports)


def test_mixed_live_and_synthetic_sessions_cannot_complete_parity() -> None:
    reports = _matching_reports([
        {"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01},
        {"evidence_kind": "synthetic_fixture", "hook_coverage": ["ShowHitText"],
         "visible_cumulative_counter_delay": 0.01},
    ])
    _assert_parity_unverified(reports)


def test_duplicate_phase33_session_id_cannot_confirm() -> None:
    reports = _matching_reports([
        {"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01},
        {"hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01},
    ])
    reports[3]["sessions"][1]["session_id"] = "session-0"
    _assert_parity_unverified(reports)


def test_qualified_live_mismatch_is_preserved() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["sessions"][0]["verdict"] = "RUNTIME_PARITY_MISMATCH"
    reports[3]["verdict"] = "RUNTIME_PARITY_MISMATCH"
    result = _load().integrate(*reports)
    assert "RUNTIME_PARITY_MISMATCH" in result["verdicts"]
    assert result["gates"]["reference_comparison_complete"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]


def test_windows_and_posix_separators_for_same_trace_are_accepted() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[3]["runtime"] = "traces\\live.jsonl"
    result = _load().integrate(*reports)
    assert "RUNTIME_PARITY_CONFIRMED" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]


@pytest.mark.parametrize(
    "capture_hash, parity_hash",
    [
        (None, "a" * 64),
        ("a" * 64, None),
        ("a" * 63, "a" * 64),
        ("not-a-sha256", "not-a-sha256"),
        ("a" * 64, "b" * 64),
        ("g" * 64, "g" * 64),
    ],
)
def test_parity_rejects_missing_malformed_or_different_trace_hash(
    capture_hash: object, parity_hash: object,
) -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[2]["input_sha256"] = capture_hash
    reports[3]["runtime_sha256"] = parity_hash
    _assert_parity_unverified(reports)


def test_matching_case_insensitive_sha256_is_accepted() -> None:
    reports = _matching_reports([{
        "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01,
    }])
    reports[2]["input_sha256"] = "abcdef" * 10 + "abcd"
    reports[3]["runtime_sha256"] = "ABCDEF" * 10 + "ABCD"
    result = _load().integrate(*reports)
    assert result["gates"]["reference_comparison_complete"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]


def test_phase32_and_phase33_hash_exact_same_trace_bytes(tmp_path: Path) -> None:
    fixture = ROOT / "tests" / "fixtures" / "phase32-stock-runtime-multisession-synthetic.jsonl"
    trace = tmp_path / "runtime.jsonl"
    trace.write_bytes(fixture.read_bytes())
    phase32_output = tmp_path / "phase32.json"
    phase33_output = tmp_path / "phase33.json"
    reference = ROOT / "reports" / "phase20c-frame-stock-state-20261022-r1.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_phase32_runtime_trace_ingestion_v01.py"),
         "--input", str(trace), "--output", str(phase32_output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_phase33_stock_runtime_parity_v01.py"),
         "--runtime", str(trace), "--reference", str(reference), "--output", str(phase33_output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    phase32 = json.loads(phase32_output.read_text(encoding="utf-8"))
    phase33 = json.loads(phase33_output.read_text(encoding="utf-8"))
    expected = sha256(trace.read_bytes()).hexdigest()
    assert phase32["input_sha256"] == expected
    assert phase33["runtime_sha256"] == expected
    assert phase32["input"] == phase33["runtime"] == str(trace)

    # Same path but different bytes must not qualify for parity.
    trace.write_bytes(trace.read_bytes() + b"\n")
    changed = sha256(trace.read_bytes()).hexdigest()
    assert changed != expected
    assert phase32["input_sha256"] != changed
