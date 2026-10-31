from __future__ import annotations

"""DMDOGO v0.9 H0.10: apply frozen-width-24, quota-8 audit to ALL 22 H0.6 seams.

H0.6 improvement status is retained ONLY for stratified analysis, never
for choosing targets. Quota-8 is exactly the predeclared H0.8 rule.
All seven known-improvement outputs must reproduce H0.8 byte-for-byte as
parsed JSON objects; H0.8 quota-1 must still match saved H0.7.
No Motor, validation, training, timing, or frozen source mutation.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from dmdogo_h0_source_provenance import capture_audit_source_provenance
from statistics import mean, median

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_v09_h04_beam_decisions import _oracle_events
from audit_dmdogo_v09_h06_improved_suffix import FORMAT as H06_FORMAT
from audit_dmdogo_v09_h07_shadow_lane import FORMAT as H07_FORMAT
from audit_dmdogo_v09_h08_protected_quota import (
    FORMAT as H08_FORMAT, BEAM_WIDTH, _candidate_target, _h07_projection,
    trace_quota,
)
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h010-all-22-quota8-v1"
QUOTA = 8
EPS = 1e-7


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs(rows: list[dict], label: str) -> dict[tuple[int, int], dict]:
    mapping = {tuple(x["flag_pair"]): x for x in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate {label} event pair")
    return mapping


def _target_missing(quick_changes: dict, pair: tuple[int, int]) -> bool:
    """Whether the original targeted quick condition is absent at the same pair."""
    removed = {tuple(x["event_pair"]) for x in quick_changes["removed"]}
    added = {tuple(x["event_pair"]) for x in quick_changes["added"]}
    return pair in removed and pair not in added


def verify_improving_reference(entry: dict, result: dict, saved: dict) -> None:
    """Fail closed on any previously known quota-8 outcome discrepancy."""
    if not entry["improved_final_cost"]:
        raise ValueError("H0.8 reference can only cover the original seven")
    if (result != saved["variants"][str(QUOTA)]
            or saved["h06_forced_suffix_final_delta"]
            != entry["cost"]["final_delta"]):
        raise ValueError(
            "H0.10 previously known improving target does not reproduce "
            "saved H0.8 quota-8 outcome and H0.6 baseline")


def _measure(path: dict | None, pair: tuple[int, int]) -> dict | None:
    if path is None:
        return None
    quick = path["quick"]
    rapid = path["rapid"]
    return {
        "delta_cost_vs_v08": path["delta_cost_vs_frozen_v08"],
        "lower_cost": path["delta_cost_vs_frozen_v08"] < -EPS,
        "higher_cost": path["delta_cost_vs_frozen_v08"] > EPS,
        "roughly_equal_cost": abs(path["delta_cost_vs_frozen_v08"]) <= EPS,
        "is_protected": path["is_from_protected_lineage"],
        "target_quick_flag_removed": _target_missing(quick, pair),
        "new_quick_flag_count": quick["added_count"],
        "removed_quick_flag_count": quick["removed_count"],
        "new_quick_flag_locations": quick["added"],
        "removed_quick_flag_locations": quick["removed"],
        "new_rapid_flag_count": rapid["added_count"],
        "removed_rapid_flag_count": rapid["removed_count"],
        "new_rapid_flag_locations": rapid["added"],
        "removed_rapid_flag_locations": rapid["removed"],
        "changed_event_assignments": path["changed_event_assignment_count"],
        "minimum_axis_reuse_ms": path["minimum_axis_reuse_ms"],
        "abstract_reuse_under_80ms": path["axis_reuse_under_80ms"],
        "hand_axis_uses": path["hand_axis_uses"],
        "cost_terms": path["score_components"],
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None,
                "mean": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 9),
        "median": round(median(values), 9),
        "mean": round(mean(values), 9),
        "max": round(max(values), 9),
    }


def summarize(rows: list[dict]) -> dict:
    """Independent experimental runs; aggregate flags are NOT a merged chart."""
    tally = Counter()
    all_cost_deltas, all_assignment_changes = [], []
    tagged_deltas = []
    for row in rows:
        tally["target_count"] += 1
        raw = row["quota8_result"]
        tally["terminal_protected_survivors"] += raw[
            "protected_descendants_reached_terminal"]
        tally["rescue_windows"] += raw["slot_replacement_count"]
        tally["displaced_untagged_slots"] += raw[
            "total_displaced_nonprotected_slots"]
        tally["ever_underfilled_quota_windows"] += bool(
            raw["windows_with_tagged_count_below_quota"])
        for label, measured in (
            ("winner", row["winner_effects"]),
            ("best_protected", row["best_protected_effects"]),
        ):
            if measured is None:
                continue
            tally[f"{label}_lower_cost"] += measured["lower_cost"]
            tally[f"{label}_higher_cost"] += measured["higher_cost"]
            tally[f"{label}_roughly_equal_cost"] += measured["roughly_equal_cost"]
            tally[f"{label}_target_quick_removed"] += measured[
                "target_quick_flag_removed"]
            for field in ("new_quick_flag_count", "removed_quick_flag_count",
                          "new_rapid_flag_count", "removed_rapid_flag_count"):
                tally[f"{label}_{field}_across_independent_runs"] += measured[field]
            tally[f"{label}_introduces_quick_flag"] += measured[
                "new_quick_flag_count"] > 0
            tally[f"{label}_introduces_rapid_flag"] += measured[
                "new_rapid_flag_count"] > 0
            tally[f"{label}_abstract_reuse_violations"] += measured[
                "abstract_reuse_under_80ms"]
        tally["winner_is_protected"] += row["winner_effects"]["is_protected"]
        all_cost_deltas.append(row["winner_effects"]["delta_cost_vs_v08"])
        all_assignment_changes.append(
            row["winner_effects"]["changed_event_assignments"])
        if row["best_protected_effects"] is not None:
            tagged_deltas.append(row["best_protected_effects"][
                "delta_cost_vs_v08"])
    # Counter omits never-incremented values (especially important for the
    # 15 previously nonimproved cases). Keep the public schema fixed even
    # when a group has zero improved/protected/flag counts.
    mandatory = (
        "target_count", "terminal_protected_survivors",
        "rescue_windows", "displaced_untagged_slots",
        "ever_underfilled_quota_windows",
        "winner_lower_cost", "winner_higher_cost",
        "winner_roughly_equal_cost", "winner_is_protected",
        "winner_target_quick_removed",
        "best_protected_lower_cost", "best_protected_higher_cost",
        "best_protected_roughly_equal_cost",
        "best_protected_target_quick_removed",
        "winner_new_quick_flag_count_across_independent_runs",
        "winner_removed_quick_flag_count_across_independent_runs",
        "winner_new_rapid_flag_count_across_independent_runs",
        "winner_removed_rapid_flag_count_across_independent_runs",
        "best_protected_new_quick_flag_count_across_independent_runs",
        "best_protected_removed_quick_flag_count_across_independent_runs",
        "best_protected_new_rapid_flag_count_across_independent_runs",
        "best_protected_removed_rapid_flag_count_across_independent_runs",
        "winner_introduces_quick_flag", "winner_introduces_rapid_flag",
        "best_protected_introduces_quick_flag",
        "best_protected_introduces_rapid_flag",
        "winner_abstract_reuse_violations",
        "best_protected_abstract_reuse_violations",
    )
    return {
        **dict(sorted({key: tally[key] for key in
                       (*mandatory, *tally.keys())}.items())),
        "winner_cost_delta_distribution": _distribution(all_cost_deltas),
        "best_protected_cost_delta_distribution": _distribution(tagged_deltas),
        "winner_changed_events_distribution": _distribution(
            all_assignment_changes),
    }


def audit_case(chart: Path, summary: dict, telemetry: dict,
               replay: dict, h06_case: dict, h07_case: dict,
               h08_case: dict) -> dict:
    rows, frozen_chunks, _ = _validate(summary, telemetry, replay)
    events = _oracle_events(chart)
    if (summary["chart_sha256"] != h06_case["chart_sha256"]
            or summary["chart_sha256"] != h07_case["chart_sha256"]
            or summary["chart_sha256"] != h08_case["chart_sha256"]
            or any(x["event_count"] != len(events)
                   for x in (h06_case, h07_case, h08_case))
            or len(rows) != len(events)
            or any(e.event_index != row["event_index"]
                   or abs(e.due_s * 1000. - row["due_ms"]) > .005
                   for e, row in zip(events, rows))):
        raise ValueError("Train chart/Oracle/physical source identity mismatch")

    h07_by_pair = _pairs(h07_case["targets"], "H0.7")
    h08_by_pair = _pairs(h08_case["targets"], "H0.8")
    improved = [x for x in h06_case["evidence"] if x["improved_final_cost"]]
    if (len(improved) != h06_case["improved_count"]
            or len(h07_by_pair) != len(improved)
            or len(h08_by_pair) != len(improved)
            or set(h07_by_pair) != {tuple(x["flag_pair"]) for x in improved}
            or set(h08_by_pair) != set(h07_by_pair)):
        raise ValueError("H0.7/H0.8 reference cohort does not equal H0.6 improvements")
    for pair in h07_by_pair:
        if (_h07_projection(h08_by_pair[pair]["variants"]["1"])
                != _h07_projection(h07_by_pair[pair])):
            raise ValueError("H0.8 saved quota-1 control no longer matches H0.7")
    all_pairs = _pairs(h06_case["evidence"], "H0.6")
    if (len(all_pairs) != h06_case["targeted_count"]
            or set(h07_by_pair) - set(all_pairs)):
        raise ValueError("H0.6 target census changed")

    windows = GestureChunkPlannerV08()._windows(events)
    results = []
    for entry in h06_case["evidence"]:
        target = _candidate_target(entry, windows)
        pair = tuple(entry["flag_pair"])
        result = trace_quota(
            events, frozen_chunks, target,
            quota=QUOTA,
            expected_cost=summary["abstract_schedule"]["metrics"]["beam_cost"],
            expected_evaluated=summary["abstract_schedule"]["candidates_evaluated"],
            reference_final_cost=entry["cost"]["original_v08_final"],
        )
        if entry["improved_final_cost"]:
            verify_improving_reference(entry, result, h08_by_pair[pair])
        elif pair in h08_by_pair:
            raise ValueError("H0.6 nonimproved case already selected in H0.8")

        winner = _measure(result["beam_winner"], pair)
        best_protected = _measure(result["best_protected_terminal"], pair)
        if winner is None:
            raise ValueError("valid quota-8 search has no terminal winner")
        results.append({
            "flag_pair": entry["flag_pair"],
            "window_index": entry["window_index"],
            "h06_cohort": (
                "previously_improved_7" if entry["improved_final_cost"]
                else "previously_nonimproved_15"),
            "h06_forced_suffix_cost_delta": entry["cost"]["final_delta"],
            "admission_rank": result["admission_expanded_rank"],
            "winner_effects": winner,
            "best_protected_effects": best_protected,
            "quota8_result": result,
        })
    if len(results) != h06_case["targeted_count"]:
        raise ValueError("did not cover every original H0.6 target")

    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(events),
        "target_count": len(results),
        "cohort_count": {
            "previously_improved_7": sum(
                r["h06_cohort"] == "previously_improved_7" for r in results),
            "previously_nonimproved_15": sum(
                r["h06_cohort"] == "previously_nonimproved_15"
                for r in results),
        },
        "targets": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "verified-report", "h0-report", "h01-report",
                "h06-report", "h07-report", "h08-report", "results-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite saved H0.10 results")
    manifest, verified, h0, h01, h06, h07, h08 = [
        _read(path) for path in (
            args.manifest, args.verified_report, args.h0_report,
            args.h01_report, args.h06_report, args.h07_report, args.h08_report)]
    if (h06.get("format") != H06_FORMAT
            or h07.get("format") != H07_FORMAT
            or h08.get("format") != H08_FORMAT
            or h08.get("quotas") != [1, 2, 4, 8]):
        raise ValueError("source phase format or protected quota set mismatched")
    artifacts = {}
    for case in manifest["cases"]:
        digest = case["sha256"]
        root = args.results_root / digest[:12]
        summary_path = root / "summary.json"
        summary = _read(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[digest] = (
            summary, _read(root / "telemetry-v08.json"),
            _read(root / "replay-v08.json"),
            _read(args.results_root / "_provenance" / f"{digest[:12]}.json"))
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    if checked != h01:
        raise ValueError("frozen verified Train/H0.1 provenance differs")
    for phase in (h06, h07, h08):
        if (phase.get("source_v08_commit") != checked["source_v08_commit"]
                or phase.get("frozen_code_digest") != checked["frozen_code_digest"]
                or phase.get("event_count") != checked["aggregate"]["event_count"]
                or phase.get("case_count") != len(manifest["cases"])):
            raise ValueError("H0.6–H0.8 source identity mismatch")

    maps = [{case["chart_sha256"]: case for case in phase["cases"]}
            for phase in (h06, h07, h08)]
    expected = {case["sha256"] for case in manifest["cases"]}
    if any(len(mapping) != len(phase["cases"]) or set(mapping) != expected
           for mapping, phase in zip(maps, (h06, h07, h08))):
        raise ValueError("source report contains missing/duplicate/mismatched chart")

    root = Path(manifest["train_root"]).resolve()
    output = []
    for case in manifest["cases"]:
        digest = case["sha256"]
        chart = validate_chart(case, root)
        output.append(audit_case(
            chart, *artifacts[digest][:3], *(mapping[digest] for mapping in maps)))
    all_rows = [row for case in output for row in case["targets"]]
    groups = {
        "all_22": all_rows,
        "previously_improved_7": [
            row for row in all_rows if row["h06_cohort"] == "previously_improved_7"],
        "previously_nonimproved_15": [
            row for row in all_rows if row["h06_cohort"] == "previously_nonimproved_15"],
    }
    if (len(groups["all_22"]) != 22
            or len(groups["previously_improved_7"]) != 7
            or len(groups["previously_nonimproved_15"]) != 15):
        raise ValueError("Train 22-target / 7+15 grouping changed")
    aggregates = {name: summarize(rows) for name, rows in groups.items()}

    # Known seven cases: reproduce saved H0.8 aggregate exactly for quota 8.
    prior = h08["aggregate"]["8"]
    improved = aggregates["previously_improved_7"]
    for local, saved in (
        ("terminal_protected_survivors", "protected_terminal_survivors"),
        ("best_protected_lower_cost", "protected_lower_final_cost_than_v08"),
        ("winner_lower_cost", "beam_winner_lower_final_cost_than_v08"),
        ("winner_is_protected", "beam_winner_from_protected_lineage"),
        ("rescue_windows", "total_rescue_windows_across_independent_runs"),
        ("displaced_untagged_slots",
         "total_displaced_nonprotected_slots_across_independent_runs"),
    ):
        if improved[local] != prior[saved]:
            raise ValueError(f"H0.8 improving cohort aggregate differs: {local}")
    report = {
        "format": FORMAT,
        "audit_source_provenance": capture_audit_source_provenance(Path(__file__).resolve().parents[1]),
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "source_h06_sha256": sha256_file(args.h06_report),
        "source_h08_sha256": sha256_file(args.h08_report),
        "case_count": len(output),
        "event_count": sum(c["event_count"] for c in output),
        "targeted_flag_count": len(all_rows),
        "beam_width": BEAM_WIDTH,
        "protected_minimum_quota": QUOTA,
        "aggregate": aggregates,
        "cases": output,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "All 22 targets were selected for H0.1 quick flags; the 15-case cohort is not an independent holdout.",
            "H0.6 improvement groups were defined post hoc using forced-suffix cost, not human feedback.",
            "Each target is an independent counterfactual. Improvements, flags, slot replacements and changed assignments cannot be combined into one playable chart.",
            "Quota 8 is a modified beam pruning rule, not the original v0.8 planner; total beam width and original score stay 24/frozen.",
            "The global winner need not belong to the tagged lineage; report protected and winner outcomes separately.",
            "Quick and rapid flags are unvalidated style heuristics and may relocate rather than disappear.",
            "No actual Motor, Timing, Validation20, FINAL or human judgment was measured.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "targeted_flag_count": report["targeted_flag_count"],
        "beam_width": BEAM_WIDTH,
        "quota": QUOTA,
        "aggregate": {
            group: {
                key: summary[key] for key in (
                    "target_count", "terminal_protected_survivors",
                    "best_protected_lower_cost", "winner_lower_cost",
                    "winner_higher_cost", "winner_is_protected",
                    "winner_target_quick_removed",
                    "winner_new_quick_flag_count_across_independent_runs",
                    "winner_new_rapid_flag_count_across_independent_runs",
                    "rescue_windows", "displaced_untagged_slots",
                    "winner_cost_delta_distribution",
                    "winner_changed_events_distribution",
                )
            } for group, summary in aggregates.items()
        },
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
