from __future__ import annotations
"""Export actual v0.8 Motor telemetry into H2.2's strict, read-only input shape.

Raw human logs are never read or published. The runner's diagnostic association
between physical keydowns and events is NOT a canonical game judgment.
"""
import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "dmdogo-h22-motor-export-v1"


def validate_and_export(summary: dict, telemetry: dict, chart_bytes: bytes, chart_name: str) -> dict:
    if summary.get("planning_mode") != "gesture_v08":
        raise ValueError("expected gesture_v08 physical Motor run")
    if summary.get("chart") != chart_name:
        raise ValueError("chart name differs from runner report")
    expected_hash = hashlib.sha256(chart_bytes).hexdigest()
    if summary.get("chart_sha256") != expected_hash:
        raise ValueError("chart SHA256 differs from runner report")
    decisions = telemetry.get("decisions")
    physical = telemetry.get("physical")
    if not isinstance(decisions, list) or not isinstance(physical, dict):
        raise ValueError("missing physical decisions or diagnostics")
    if any(not isinstance(d, dict) or d.get("event_index") != i for i, d in enumerate(decisions)):
        raise ValueError("missing/unsorted event identities")
    n = summary.get("total_events")
    stats = summary.get("stats")
    if type(n) is not int or n < 1 or not isinstance(stats, dict):
        raise ValueError("invalid run count/stats")

    one_keydown_per_event = (
        len(decisions) == n and all(
            isinstance(d.get("keydowns"), list)
            and len(d["keydowns"]) == 1
            and isinstance(d["keydowns"][0], dict)
            and d["keydowns"][0].get("axis") == d.get("chosen")
            and d.get("chosen", "").startswith(("left_", "right_"))
            for d in decisions
        )
    )
    all_hit = (summary.get("termination_reason") == "chart_complete"
               and stats.get("targets") == n and stats.get("hits") == n
               and stats.get("misses") == 0 and stats.get("overloaded") is False
               and physical.get("unassigned_events") == 0)
    no_replan = physical.get("local_replans") == 0
    eligible = bool(one_keydown_per_event and all_hit and no_replan)
    return {
        "format": FORMAT,
        "stage": "executed",
        "candidate_id": "h22-gesture-v08-fresh-chart",
        "chart": chart_name,
        "chart_sha256": expected_hash,
        "chart_event_count": n,
        "termination_reason": summary.get("termination_reason"),
        "stats": stats,
        "physical_local_replan_count": physical.get("local_replans"),
        "physical": physical,
        "actual_keydown_count": stats.get("physical_keydowns"),
        "decisions": decisions,
        "h22_strict_eligible": eligible,
        "h22_gates": {
            "one_keydown_per_event": one_keydown_per_event,
            "all_hit": all_hit,
            "no_physical_replan": no_replan,
        },
        "source": "DMDOGO run_dmdogo_v06_pattern_first.py --planning-mode gesture_v08",
        "limitations": [
            "Event-keydown matching in runner telemetry is diagnostic, not an authoritative ADOFAI judgment.",
            "v0.8 offline planner and Oracle event time; not a HUD-only Student.",
            "This report contains no human keystroke data.",
            "A failed strict gate is preserved as failure, never silently converted into an all-hit claim.",
        ],
        "human_likeness_verified": False,
        "training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--chart", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing output")
    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8-sig"))
    report = validate_and_export(summary, telemetry, args.chart.read_bytes(), args.chart.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "chart": report["chart"],
        "chart_sha256": report["chart_sha256"],
        "eligible": report["h22_strict_eligible"],
        "gates": report["h22_gates"],
        "hits": report["stats"].get("hits"),
        "misses": report["stats"].get("misses"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
