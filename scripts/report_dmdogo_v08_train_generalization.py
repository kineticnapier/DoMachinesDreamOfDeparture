from __future__ import annotations

"""Strict frozen-v0.8 Train physical evaluation reporter.

No physical Motor, retraining, or chart access here. Rejects implausible
metrics, missing execution provenance, changed source/settings/environment,
incomplete episodes and chart/summary identity mismatch.
"""

import argparse
import json
from math import isfinite
from pathlib import Path

from dmdogo_v08_provenance import (
    FROZEN_HEAD, MINIMUM_REUSE_MS, PLANNING_MODE,
    WRAPPER_FORMAT, digest_json, sha256_file,
)

RESULT_FORMAT = "dmdogo-gesture-chunk-physical-v0.8"
MANIFEST_FORMAT = "dmdogo-v08-frozen-train-generalization-manifest-v1"


def _natural(value, *, zero_ok: bool = True) -> bool:
    return (type(value) is int and value >= (0 if zero_ok else 1))


def _finite(value, *, min_value: float | None = None,
            max_value: float | None = None) -> bool:
    if type(value) not in (float, int) or not isfinite(value):
        return False
    return ((min_value is None or value >= min_value)
            and (max_value is None or value <= max_value))


def _valid_metrics(case: dict, summary: dict) -> bool:
    stats, physical = summary.get("stats"), summary.get("physical")
    if not isinstance(stats, dict) or not isinstance(physical, dict):
        return False
    for field in ("hits", "targets", "misses", "too_early"):
        if not _natural(stats.get(field), zero_ok=field != "targets"):
            return False
    if stats["hits"] + stats["misses"] != stats["targets"]:
        return False
    if not isinstance(stats.get("overloaded"), bool):
        return False
    if (not _finite(stats.get("x_accuracy_percent"), min_value=0, max_value=100)
            or not _finite(stats.get("timing_mae_ms"), min_value=0)):
        return False
    if ("signed_timing_error_ms" in stats
            and not _finite(stats["signed_timing_error_ms"])):
        return False
    if ("physical_keydowns" in stats
            and not _natural(stats["physical_keydowns"])):
        return False
    if not _natural(summary.get("total_events"), zero_ok=False):
        return False
    if summary["total_events"] != case.get("event_count"):
        return False
    for field in ("local_replans", "deferred_events", "unassigned_events"):
        if not _natural(physical.get(field)):
            return False
        if physical[field] > summary["total_events"]:
            return False
    for field in ("deferral_frames",):
        if field in physical and not _natural(physical[field]):
            return False
    if stats["targets"] < summary["total_events"]:
        return False
    return True


def _valid_provenance(proof: dict | None, case: dict, summary: dict,
                      *, frozen_code_digest: str, frozen_head: str,
                      manifest_digest: str | None) -> bool:
    if not isinstance(proof, dict) or not isinstance(proof.get("source"), dict):
        return False
    source = proof["source"]
    if (proof.get("format") != WRAPPER_FORMAT
            or proof.get("state") != "success"
            or proof.get("runner_exit_code") != 0
            or proof.get("chart_sha256") != case["sha256"]
            or source.get("frozen_source_commit") != frozen_head
            or source.get("code_digest") != frozen_code_digest
            or source.get("working_runtime_sources_match") is not True
            or proof.get("selection_manifest_frozen_code_digest") != frozen_code_digest):
        return False
    if manifest_digest is not None and proof.get("selection_manifest_sha256") != manifest_digest:
        return False
    # The summary bytes must be identical to those saved at actual run finish.
    if (not isinstance(proof.get("summary_sha256"), str)
            or proof.get("summary_sha256") != summary.get("_artifact_sha256")):
        return False
    settings = proof.get("settings")
    if not isinstance(settings, dict):
        return False
    if (settings.get("planning_mode") != PLANNING_MODE
            or type(settings.get("minimum_reuse_ms")) not in (float, int)
            or settings.get("minimum_reuse_ms") != MINIMUM_REUSE_MS
            or settings.get("stop_after_s", "missing") is not None
            or proof.get("settings_digest") != digest_json(settings)):
        return False
    environment = proof.get("environment")
    if (not isinstance(environment, dict)
            or proof.get("environment_digest") != digest_json(environment)):
        return False
    return True


def _result_row(case: dict, summary: dict | None,
                provenance: dict | None = None, *,
                frozen_code_digest: str | None = None,
                frozen_head: str = FROZEN_HEAD,
                manifest_digest: str | None = None) -> dict:
    row = {
        "selection_reason": case["selection_reason"],
        "chart": case["chart"],
        "chart_sha256": case["sha256"],
        "screened_event_count": case["event_count"],
        "status": "pending",
        "hits": None, "targets": None, "misses": None,
        "too_early": None, "x_accuracy_percent": None,
        "timing_mae_ms": None, "local_replans": None,
        "deferred_events": None, "unassigned_events": None,
        "overloaded": None,
    }
    if summary is None:
        return row
    if (summary.get("chart_sha256") != case["sha256"]
            or summary.get("chart") != case["chart"]
            or summary.get("planning_mode") != PLANNING_MODE
            or summary.get("format") != RESULT_FORMAT):
        row["status"] = "invalid_identity"
        return row
    if not _valid_metrics(case, summary):
        row["status"] = "invalid_metrics"
        return row
    if not frozen_code_digest or not _valid_provenance(
            provenance, case, summary,
            frozen_code_digest=frozen_code_digest,
            frozen_head=frozen_head, manifest_digest=manifest_digest):
        row["status"] = "unverified_provenance"
        return row
    stats, physical = summary["stats"], summary["physical"]
    row.update({
        "status": ("chart_complete" if summary.get("termination_reason") == "chart_complete"
                   else "did_not_complete"),
        "termination_reason": summary.get("termination_reason"),
        "hits": stats["hits"], "targets": stats["targets"],
        "misses": stats["misses"], "too_early": stats["too_early"],
        "x_accuracy_percent": stats["x_accuracy_percent"],
        "timing_mae_ms": stats["timing_mae_ms"],
        "local_replans": physical["local_replans"],
        "deferred_events": physical["deferred_events"],
        "unassigned_events": physical["unassigned_events"],
        "overloaded": stats["overloaded"],
        "environment_digest": provenance["environment_digest"],
        "settings_digest": provenance["settings_digest"],
        "actual_git_head": provenance["source"].get("actual_git_head"),
    })
    return row


def collate(manifest: dict, results: dict[str, dict],
            provenance: dict[str, dict] | None = None, *,
            manifest_digest: str | None = None) -> dict:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("unexpected selection manifest format")
    if manifest.get("frozen_implementation_commit") != FROZEN_HEAD:
        raise ValueError("selection manifest refers to an unrecognized source commit")
    code_digest = manifest.get("frozen_code_digest")
    if not isinstance(code_digest, str) or len(code_digest) != 64:
        raise ValueError("selection manifest lacks complete source-tree digest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("selection manifest contains no cases")
    if len({case["sha256"] for case in cases}) != len(cases):
        raise ValueError("duplicate SHA256 in selection manifest")
    proofs = provenance or {}
    rows = [
        _result_row(case, results.get(case["sha256"]), proofs.get(case["sha256"]),
                    frozen_code_digest=code_digest, manifest_digest=manifest_digest)
        for case in cases
    ]
    # One researcher cannot compare 'same frozen runtime' if the installed
    # Python packages or physical constants differ across case executions.
    reference = None
    for row in rows:
        if row["status"] not in ("chart_complete", "did_not_complete"):
            continue
        fingerprint = (row["environment_digest"], row["settings_digest"])
        if reference is None:
            reference = fingerprint
        elif reference != fingerprint:
            row["status"] = "different_runtime_environment"
    completed = [row for row in rows if row["status"] == "chart_complete"]
    return {
        "format": "dmdogo-v08-train-generalization-report-v2",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_code_digest": code_digest,
        "case_count": len(rows),
        "completed_count": len(completed),
        "all_cases_completed": len(completed) == len(rows),
        "records": rows,
        "aggregate": {
            "total_targets_completed": sum(row["targets"] for row in completed),
            "total_hits_completed": sum(row["hits"] for row in completed),
            "total_misses_completed": sum(row["misses"] for row in completed),
            "total_local_replans_completed": sum(row["local_replans"] for row in completed),
            "total_deferred_events_completed": sum(row["deferred_events"] for row in completed),
            "any_overload_completed": any(row["overloaded"] for row in completed),
        },
        "limitations": [
            "All accepted runs have verified frozen tracked Python/TOML code, not cryptographically attested execution.",
            "Historic bool machine configuration cannot be retroactively proven identical.",
            "Missing provenance/invalid metrics/mismatched runtime are NOT accepted.",
            "A completed chart can contain misses; all_cases_completed is not all_hits.",
            "No human recordings or HUD-only Student inference were evaluated.",
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
    summaries, proofs = {}, {}
    for case in manifest.get("cases", []):
        name = case["sha256"]
        root = args.results_root / name[:12]
        summary_file = root / "summary.json"
        proof_file = args.results_root / "_provenance" / f"{name[:12]}.json"
        if summary_file.is_file():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                summary["_artifact_sha256"] = sha256_file(summary_file)
                summaries[name] = summary
            except (ValueError, OSError):
                summaries[name] = {"format": "corrupt_summary"}
        if proof_file.is_file():
            try:
                proofs[name] = json.loads(proof_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                proofs[name] = {"format": "corrupt_provenance"}
    report = collate(
        manifest, summaries, proofs, manifest_digest=sha256_file(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "case_count": report["case_count"],
        "completed_count": report["completed_count"],
        "all_cases_completed": report["all_cases_completed"],
        "statuses": {s: sum(r["status"] == s for r in report["records"])
                     for s in sorted({r["status"] for r in report["records"]})},
        "aggregate": report["aggregate"], "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
