from __future__ import annotations

"""Summarize v0.9 review-only proxies over *verified* frozen v0.8 Train runs.

Single read-only command. Checks source evidence again rather than trusting
a report filename or prior PASS message. No Motor execution or human data.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_dmdogo_v09_style_proxies import analyze, OUTPUT_FORMAT
from dmdogo_v08_provenance import FROZEN_HEAD, sha256_file
from report_dmdogo_v08_train_generalization import _result_row

REPORT_FORMAT = "dmdogo-v09-train-style-proxy-report-v1"


def compile_report(manifest: dict, verified: dict, case_artifacts: dict, *,
                   manifest_digest: str,
                   max_flags_per_chart: int = 25) -> dict:
    if manifest.get("format") != "dmdogo-v08-frozen-train-generalization-manifest-v1":
        raise ValueError("unrecognized Train selection manifest")
    if (manifest.get("frozen_implementation_commit") != FROZEN_HEAD
            or verified.get("format") != "dmdogo-v08-train-generalization-report-v2"
            or verified.get("frozen_implementation_commit") != FROZEN_HEAD
            or verified.get("frozen_code_digest") != manifest.get("frozen_code_digest")):
        raise ValueError("frozen v0.8 identity mismatch")
    cases = manifest.get("cases", [])
    if (not cases
            or len({case["sha256"] for case in cases}) != len(cases)
            or verified.get("case_count") != len(cases)
            or verified.get("completed_count") != len(cases)
            or verified.get("all_cases_completed") is not True):
        raise ValueError("all preselected Train cases must be verified complete")
    records = verified.get("records")
    if (not isinstance(records, list) or len(records) != len(cases)
            or {row.get("chart_sha256") for row in records}
            != {case["sha256"] for case in cases}
            or any(row.get("status") != "chart_complete" for row in records)):
        raise ValueError("verified report has missing/incomplete cases")
    if type(max_flags_per_chart) is not int or max_flags_per_chart < 0:
        raise ValueError("invalid max_flags_per_chart")

    summaries = []
    aggregated_flags = Counter()
    total_contacts = 0
    hand_total = Counter()
    all_intact = 0
    total_events = 0

    for case in cases:
        if case["sha256"] not in case_artifacts:
            raise ValueError("missing a selected saved case")
        summary, telemetry, replay, proof = case_artifacts[case["sha256"]]
        row = _result_row(
            case, summary, proof,
            frozen_code_digest=manifest["frozen_code_digest"],
            frozen_head=FROZEN_HEAD, manifest_digest=manifest_digest,
        )
        if row["status"] != "chart_complete":
            raise ValueError(
                f"not a verified complete physical outcome: {case['sha256']}: "
                f"{row['status']}")
        prior = next(r for r in records if r["chart_sha256"] == case["sha256"])
        if any(prior.get(k) != row.get(k)
               for k in ("hits", "targets", "misses", "timing_mae_ms",
                         "local_replans", "deferred_events", "overloaded")):
            raise ValueError("saved physical metrics differ from verified report")

        features = analyze(summary, telemetry, replay,
                           max_flags=max_flags_per_chart)
        metric = features["metrics"]
        total_events += features["event_count"]
        total_contacts += features["total_contacts"]
        hand_total.update(metric["hand_contact_counts"])
        all_intact += round(
            metric["planned_axis_adoption_fraction"] *
            features["event_count"])
        aggregated_flags.update(features["flag_counts"])
        summaries.append({
            "chart": case["chart"],
            "chart_sha256": case["sha256"],
            "events": features["event_count"],
            "source_v08_status": "chart_complete",
            "v09_style_format": OUTPUT_FORMAT,
            "metrics": metric,
            "flag_counts": features["flag_counts"],
            "total_flags": features["total_flags"],
            "returned_review_flags": features["review_flags"],
            "flags_truncated": features["total_flags"] > features["returned_flags"],
        })
    return {
        "format": REPORT_FORMAT,
        "source_v08_commit": FROZEN_HEAD,
        "frozen_code_digest": manifest["frozen_code_digest"],
        "case_count": len(summaries),
        "cases": summaries,
        "aggregate": {
            "events": total_events,
            "contacts": total_contacts,
            "hand_contact_counts": {
                "left": hand_total["left"], "right": hand_total["right"],
            },
            "planned_axis_adopted_events": all_intact,
            "flag_counts": dict(sorted(aggregated_flags.items())),
        },
        "human_demonstrations_used": False,
        "human_likeness_verified": False,
        "execution_policy_unchanged": True,
        "interpretation": (
            "Review-only descriptive heuristics on three preselected Train "
            "charts; NO learned human preference, imitation score or "
            "representative population estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verified-report", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-flags-per-chart", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite v0.9 style report")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verified = json.loads(args.verified_report.read_text(encoding="utf-8"))
    artifacts = {}
    for case in manifest.get("cases", []):
        sha = case["sha256"]
        sub = args.results_root / sha[:12]
        def load(path):
            return json.loads(path.read_text(encoding="utf-8"))
        summary_path = sub / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        telemetry = load(sub / "telemetry-v08.json")
        replay = load(sub / "replay-v08.json")
        proof = load(args.results_root / "_provenance" / f"{sha[:12]}.json")
        artifacts[sha] = (summary, telemetry, replay, proof)
    report = compile_report(
        manifest, verified, artifacts,
        manifest_digest=sha256_file(args.manifest),
        max_flags_per_chart=args.max_flags_per_chart)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "case_count": report["case_count"],
        "events": report["aggregate"]["events"],
        "hand_contacts": report["aggregate"]["hand_contact_counts"],
        "flag_counts": report["aggregate"]["flag_counts"],
        "saved": str(args.output),
        "human_likeness_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
