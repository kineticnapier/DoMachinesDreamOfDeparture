from __future__ import annotations

import importlib.util
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
    result = _load().integrate(
        {"verdict": "DLL_SIGNATURES_VERIFIED"},
        {"build": {"success": True}, "runtime_loaded": True, "harmony_hooks_observed": True},
        {"live_runtime_present": True, "sessions": [{"evidence_kind": "live_runtime", "hook_coverage": ["ShowHitText"], "visible_cumulative_counter_delay": 0.01}]},
        {"verdict": "RUNTIME_PARITY_CONFIRMED"},
    )
    assert "RUNTIME_TRACE_CAPTURED" in result["verdicts"]
    assert "STOCK_VISIBLE_FEEDBACK_CONFIRMED" in result["verdicts"]
    assert "INSUFFICIENT_EVIDENCE" not in result["verdicts"]



def _complete_evidence(sessions: list[dict]) -> dict:
    return _load().integrate(
        {"verdict": "DLL_SIGNATURES_VERIFIED"},
        {"build": {"success": True}, "runtime_loaded": True, "harmony_hooks_observed": True},
        {"live_runtime_present": True, "sessions": [
            {"evidence_kind": "live_runtime", **row} for row in sessions
        ]},
        {"verdict": "RUNTIME_PARITY_CONFIRMED"},
    )


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
