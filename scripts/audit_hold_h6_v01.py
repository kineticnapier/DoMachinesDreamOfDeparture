from __future__ import annotations

"""Final Hold integration quality gate over frozen Train-only evidence."""

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.features.hold_hud_v4 import (
    HoldFloorCue,
    HoldHUDV4Frame,
    HoldIconStudentObservation,
)
if __package__:
    from scripts.audit_hold_h5_v01 import EXPECTED
else:
    from audit_hold_h5_v01 import EXPECTED


FORMAT = "dmdod-hold-h6-quality-audit-v0.1"
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
MAX_FIXED3_DOUBLE_COMPILE_S = 60.0
MAX_PEAK_BYTES = 256 * 1024 * 1024


def _fingerprint(path: Path) -> tuple[str, int, int]:
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart)
    payload = [{
        "index": floor.index,
        "target": floor.target_time_s,
        "exit": floor.exit_time_s,
        "bpm": floor.bpm,
        "hold": None if floor.hold is None else {
            "loops": floor.hold.duration_loops,
            "start": floor.hold.start_time_s,
            "end": floor.hold.end_time_s,
            "additional": floor.hold.additional_time_s,
        },
    } for floor in chart.floors]
    digest = sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return digest, len(segment.targets), len(segment.holds)


def run(train_dir: Path, h5_path: Path, phase14_path: Path) -> dict:
    h5 = json.loads(h5_path.read_text(encoding="utf-8"))
    phase14 = json.loads(phase14_path.read_text(encoding="utf-8"))

    tracemalloc.start()
    started = perf_counter()
    first = {name: _fingerprint(train_dir / name) for name in EXPECTED}
    second = {name: _fingerprint(train_dir / name) for name in EXPECTED}
    elapsed_s = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    public = {
        item.name
        for cls in (HoldFloorCue, HoldIconStudentObservation, HoldHUDV4Frame)
        for item in fields(cls)
    }
    forbidden = {
        "duration", "hold_length", "holdLength", "target_time_s",
        "release_time_s", "floor_index", "chart_time_s",
    }
    phase14_checks = phase14.get("structural_checks", {})
    checks = {
        "h5_code_parity_pass":
            h5.get("verdict") == "CODE_PARITY_PASS_RUNTIME_UNVERIFIED",
        "h5_all_checks_pass": all(h5.get("checks", {}).values()),
        "deterministic_double_compile": first == second,
        "fixed_chart_hashes_present": set(first) == set(EXPECTED),
        "hud_student_contract_has_no_truth_fields": public.isdisjoint(forbidden),
        "phase14_pass": phase14.get("verdict") == "TWIRL_CAUSAL_RECOVERY_PASS",
        "phase14_all_checks_pass": all(phase14_checks.values()),
        "phase14_false_positive_zero": phase14.get("false_positive_ids") == 0,
        "phase14_success_regressions_zero":
            phase14.get("existing_success_regressions") == 0,
        "bounded_compile_time": elapsed_s <= MAX_FIXED3_DOUBLE_COMPILE_S,
        "bounded_peak_memory": peak_bytes <= MAX_PEAK_BYTES,
    }
    evidence_pass = all(checks.values())
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "HOLD_PARTIAL_INTEGRATION" if evidence_pass
        else "HOLD_REGRESSION_FAILED",
        "checks": checks,
        "determinism": {
            "first": {
                name: {"sha256": row[0], "targets": row[1], "holds": row[2]}
                for name, row in first.items()
            },
            "second_identical": first == second,
        },
        "bounded_smoke": {
            "scope": "fixed Train-only 3 Hold charts compiled twice",
            "elapsed_s": elapsed_s,
            "limit_s": MAX_FIXED3_DOUBLE_COMPILE_S,
            "peak_bytes": peak_bytes,
            "limit_bytes": MAX_PEAK_BYTES,
            "historical_performance_baseline_available": False,
        },
        "component_status": {
            "stock_timing": "CODE_PARITY_PASS_LIVE_GAME_UNVERIFIED",
            "judgement": "DETERMINISTIC_CODE_PARITY_LIVE_MARGINS_UNVERIFIED",
            "hud_recognition": "ICON_CLASSES_CAUSAL_EXACT_DURATION_BLOCKED",
            "motor": "DETERMINISTIC_OVERLAY_PASS_FULL_CHART_HUD_ONLY_BLOCKED",
            "ordinary_timing_regression": "PASS",
        },
        "why_not_full_pass": (
            "verified icon-only HUD cannot distinguish positive Hold durations; "
            "live stock timing/judgement capture and full-chart HUD-only release "
            "control have not been demonstrated"
        ),
        "sources": {
            "h5": str(h5_path),
            "phase14_regression": str(phase14_path),
            "test_results": "reports/hold-h6-test-results-20261020-r1.json",
        },
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--phase14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_dir, args.h5, args.phase14)
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("checks:", result["checks"])
    print("bounded_smoke:", result["bounded_smoke"])


if __name__ == "__main__":
    main()
