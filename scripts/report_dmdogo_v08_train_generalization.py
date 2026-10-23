from __future__ import annotations

"""Collate independently executed frozen v0.8 Train-only physical results.

The manifest selected charts but did not execute Motor. This reporter never
runs or retrains a policy and refuses to interpret missing runs as successes.
"""

import argparse
import json
from pathlib import Path


def _result_row(case: dict, summary: dict | None) -> dict:
    row = {
        "selection_reason": case["selection_reason"],
        "chart": case["chart"],
        "chart_sha256": case["sha256"],
        "screened_event_count": case["event_count"],
        "status": "pending",
        "hits": None,
        "targets": None,
        "misses": None,
        "too_early": None,
        "x_accuracy_percent": None,
        "timing_mae_ms": None,
        "local_replans": None,
        "deferred_events": None,
        "overloaded": None,
    }
    if summary is None:
        return row
    if (summary.get("chart_sha256") != case["sha256"]
            or summary.get("planning_mode") != "gesture_v08"
            or summary.get("format") != "dmdogo-gesture-chunk-physical-v0.8"):
        row["status"] = "invalid_identity"
        return row
    stats, physical = summary.get("stats"), summary.get("physical")
    if not isinstance(stats, dict) or not isinstance(physical, dict):
        row["status"] = "invalid_metrics"
        return row
    row.update({
        "status": ("chart_complete" if summary.get("termination_reason") == "chart_complete"
                   else "did_not_complete"),
        "termination_reason": summary.get("termination_reason"),
        "hits": stats.get("hits"),
        "targets": stats.get("targets"),
        "misses": stats.get("misses"),
        "too_early": stats.get("too_early"),
        "x_accuracy_percent": stats.get("x_accuracy_percent"),
        "timing_mae_ms": stats.get("timing_mae_ms"),
        "local_replans": physical.get("local_replans"),
        "deferred_events": physical.get("deferred_events"),
        "overloaded": stats.get("overloaded"),
    })
    return row


def collate(manifest: dict, results: dict[str, dict]) -> dict:
    if manifest.get("format") != "dmdogo-v08-frozen-train-generalization-manifest-v1":
        raise ValueError("unexpected selection manifest format")
    cases = manifest["cases"]
    if len({case["sha256"] for case in cases}) != len(cases):
        raise ValueError("duplicate SHA256 in selection manifest")
    rows = [_result_row(case, results.get(case["sha256"])) for case in cases]
    completed = [row for row in rows if row["status"] == "chart_complete"]
    return {
        "format": "dmdogo-v08-train-generalization-report-v1",
        "frozen_implementation_commit": manifest["frozen_implementation_commit"],
        "case_count": len(rows),
        "completed_count": len(completed),
        "all_cases_completed": len(completed) == len(rows) and bool(rows),
        "records": rows,
        "aggregate": {
            "total_targets_completed": sum(row["targets"] or 0 for row in completed),
            "total_hits_completed": sum(row["hits"] or 0 for row in completed),
            "total_misses_completed": sum(row["misses"] or 0 for row in completed),
            "total_local_replans_completed": sum(
                row["local_replans"] or 0 for row in completed),
            "total_deferred_events_completed": sum(
                row["deferred_events"] or 0 for row in completed),
            "any_overload_completed": any(
                row["overloaded"] for row in completed),
        },
        "limitations": [
            "Missing, incomplete or mismatched charts do not count as successful.",
            "MAE and X are per-chart, not averaged across nonuniform event counts.",
            "A Train subset cannot demonstrate full distribution generalization.",
            "Hit/X does not establish human imitation; no human traces measured.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite comparison report")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summaries = {}
    for case in manifest.get("cases", []):
        source = args.results_root / case["sha256"][:12] / "summary.json"
        if source.is_file():
            try:
                summaries[case["sha256"]] = json.loads(
                    source.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                # Bad/corrupt artifacts are *not* represented as pending.
                summaries[case["sha256"]] = {
                    "chart_sha256": case["sha256"],
                    "planning_mode": "gesture_v08",
                    "format": "dmdogo-gesture-chunk-physical-v0.8",
                    "termination_reason": "corrupt_summary",
                }
    report = collate(manifest, summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(
        {"case_count": report["case_count"],
         "completed_count": report["completed_count"],
         "all_cases_completed": report["all_cases_completed"],
         "aggregate": report["aggregate"], "output": str(args.output)},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
