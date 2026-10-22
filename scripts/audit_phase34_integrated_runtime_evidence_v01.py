from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def integrate(phase30: dict, phase31: dict, phase32: dict, phase33: dict) -> dict[str, object]:
    verdicts: list[str] = []
    signature = phase30.get("verdict")
    verdicts.append(signature if signature in {"DLL_SIGNATURES_VERIFIED", "DLL_SIGNATURES_PARTIAL"} else "DLL_SIGNATURES_PARTIAL")
    build = phase31.get("build", {})
    if build.get("success"):
        verdicts.append("RUNTIME_TRACE_MOD_BUILT")
    verdicts.append("RUNTIME_TRACE_MOD_UNVERIFIED")

    live_captured = bool(phase32.get("live_runtime_present"))
    if live_captured:
        verdicts.append("RUNTIME_TRACE_CAPTURED")
    parity = phase33.get("verdict", "RUNTIME_PARITY_UNVERIFIED")
    if parity not in {"RUNTIME_PARITY_CONFIRMED", "RUNTIME_PARITY_MISMATCH"}:
        parity = "RUNTIME_PARITY_UNVERIFIED"
    verdicts.append(parity)

    sessions = phase32.get("sessions", [])
    has_hit_text = any("ShowHitText" in row.get("hook_coverage", []) for row in sessions)
    has_visible_counter = any(row.get("visible_cumulative_counter_delay") != "UNKNOWN_NOT_HOOKED" for row in sessions)
    if has_hit_text and has_visible_counter and live_captured:
        verdicts.append("STOCK_VISIBLE_FEEDBACK_CONFIRMED")
    else:
        verdicts.append("STOCK_VISIBLE_FEEDBACK_PARTIAL")

    complete = (
        signature == "DLL_SIGNATURES_VERIFIED"
        and build.get("success") is True
        and phase31.get("runtime_loaded") is True
        and phase31.get("harmony_hooks_observed") is True
        and live_captured
        and parity in {"RUNTIME_PARITY_CONFIRMED", "RUNTIME_PARITY_MISMATCH"}
        and has_visible_counter
    )
    if not complete:
        verdicts.append("INSUFFICIENT_EVIDENCE")
    return {
        "format": "dmdod-phase34-integrated-runtime-evidence-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": verdicts,
        "gates": {
            "decompiled_signatures_verified": signature == "DLL_SIGNATURES_VERIFIED",
            "probe_build_succeeded": build.get("success") is True,
            "probe_loaded_in_runtime": phase31.get("runtime_loaded") is True,
            "harmony_hooks_observed": phase31.get("harmony_hooks_observed") is True,
            "live_runtime_trace_captured": live_captured,
            "visible_hit_text_signal_present": has_hit_text,
            "visible_cumulative_counter_hook_present": has_visible_counter,
            "reference_comparison_complete": parity in {"RUNTIME_PARITY_CONFIRMED", "RUNTIME_PARITY_MISMATCH"},
        },
        "event_scope": [36, 37, 42, 47, 89],
        "runtime_parity_scope": "none until qualified live traces exist",
        "student_visible_fields": [],
        "production_changed": False,
        "next_required_evidence": [
            "load Phase 31 probe in the matching installed build",
            "capture separate NoFail and normal-Fail sessions around fixed event ranges",
            "verify sequence/hook coverage and frame perturbation",
            "identify and hook the actual cumulative judgement counter renderer",
            "supply an evaluator-only floor seqID to playable-target mapping",
            "compare the qualified trace with Phase 20C without imposing a 10ms runtime grid",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase30", type=Path, default=Path("reports/phase30-dll-signature-audit-20261022-r1.json"))
    parser.add_argument("--phase31", type=Path, default=Path("reports/phase31-stock-runtime-probe-build-20261022-r1.json"))
    parser.add_argument("--phase32", type=Path, default=Path("reports/phase32-runtime-trace-ingestion-20261022-r1.json"))
    parser.add_argument("--phase33", type=Path, default=Path("reports/phase33-stock-runtime-parity-20261022-r1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.phase30, args.phase31, args.phase32, args.phase33)]
    result = integrate(*inputs)
    result["inputs"] = [str(path) for path in (args.phase30, args.phase31, args.phase32, args.phase33)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
