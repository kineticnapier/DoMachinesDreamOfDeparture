from __future__ import annotations

"""H1.8: read-only diagnosis of H1.7 proxy ties, tradeoffs and sensitivity.

Join the exact H1.7 Pareto report to its H1.6 source and complete H1.1
Train-only event-axis assignments. Do not infer human preference from
abstract pressure counts; divergence windows are not causal explanations
of individual proxy changes. No simulation or model mutation.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import audit_dmdogo_h17_proxy_pareto as h17
from audit_dmdogo_h13_automated_fingering_quality import (
    H11_FORMAT, PRESSURE_KEYS, axis_tuple,
)
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-h18-proxy-identifiability-v1"
MAX_WINDOWS = 5


def check(value, message):
    if not value:
        raise ValueError(message)


def seven_axis_relation(a, b):
    """Pareto relation on any nonempty equal-sized integer vectors."""
    check(len(a) == len(b) and len(a) > 0, "invalid vector lengths")
    check(all(type(x) is int and x >= 0 for x in (*a, *b)),
          "invalid pressure counts")
    if a == b:
        return "equal_proxies"
    if all(x <= y for x, y in zip(a, b)):
        return "first_dominates"
    if all(y <= x for x, y in zip(a, b)):
        return "second_dominates"
    return "tradeoff"


def verify_h11(h11_report, h17_report, h16_report):
    """Recover complete indexed plans, fail closed on mismatched H1.1 input.

    H1.1 is SHA-pinned to H1.6; this does NOT replay H1.1 reconstruction.
    """
    check(isinstance(h11_report, dict) and h11_report.get("format") == H11_FORMAT
          and h11_report.get("chart_count") == 3
          and h11_report.get("target_count") == 22
          and h11_report.get("strict_lower_heuristic_cost_count") == 7
          and h11_report.get("complete_event_axis_assignments") is True
          and h11_report.get("human_preference_observed") is False
          and h11_report.get("alternate_motor_executed") is False
          and h11_report.get("motor_quality_verified") is False
          and h11_report.get("validation20_or_final_accessed") is False
          and h11_report.get("training_performed") is False
          and h11_report.get("frozen_planner_motor_timing_changed") is False,
          "H1.1 scope or provenance flags differ")
    charts = h11_report.get("charts")
    check(isinstance(charts, list) and len(charts) == 3,
          "H1.1 charts missing")
    h17_chart_map = {c["chart_sha256"]: c for c in h17_report["per_chart"]}
    h16_rows = {r["candidate_id"]: r for r in h16_report["pairs"]}
    plans = {}
    chart_times = {}
    found = set()
    for chart in charts:
        sha = chart.get("chart_sha256")
        check(sha in h17_chart_map, "unexpected H1.1 chart")
        expected = h17_chart_map[sha]
        events = chart.get("original_events_by_index")
        count = expected["original_source_events"]
        reconstructions = chart.get("reconstructions")
        check(chart.get("chart") == expected["chart"]
              and chart.get("event_count") == count
              and isinstance(events, list) and len(events) == count
              and isinstance(reconstructions, list)
              and len(reconstructions) == len(expected["schedules"]) - 1
              and chart.get("count") == len(reconstructions),
              "H1.1 chart/event/reconstruction census differs")
        original = []
        times = []
        ids = []
        prev_time = float("-inf")
        for row in events:
            check(isinstance(row, dict) and type(row.get("event_index")) is int,
                  "H1.1 event identity invalid")
            index = row["event_index"]
            due = h17.finite_number(row.get("due_s"), "H1.1 due_s")
            check((not ids or index > ids[-1]) and due >= prev_time,
                  "H1.1 event ordering differs")
            chord = row.get("chord_size")
            original.append(axis_tuple(row.get("original_axes"), chord, "H1.1 original"))
            ids.append(index)
            times.append(due)
            prev_time = due
        origin_id = expected["schedules"][0]["id"]
        check(origin_id == "original:" + sha[:12], "H1.7 original ID differs")
        plans[origin_id] = tuple(original)
        chart_times[sha] = (tuple(ids), tuple(times))
        scheduled = {r["id"] for r in expected["schedules"][1:]}
        check(len(scheduled) == len(reconstructions), "H1.7 alternative IDs differ")
        for rec in reconstructions:
            cid = rec.get("candidate_id")
            check(cid in scheduled and cid not in found,
                  "H1.1 unknown or duplicate alternative")
            found.add(cid)
            axes = rec.get("alternative_axes_by_event_order")
            check(isinstance(axes, list) and len(axes) == count,
                  f"{cid}: incomplete H1.1 assignment")
            alternative = tuple(
                axis_tuple(a, events[i]["chord_size"], f"{cid} axis")
                for i, a in enumerate(axes))
            changes = [
                {
                    "event_index": ids[i],
                    "due_s": times[i],
                    "original_axes": list(original[i]),
                    "alternative_axes": list(alternative[i]),
                }
                for i in range(count) if original[i] != alternative[i]
            ]
            saved_changes = rec.get("changed_events")
            check(saved_changes == changes
                  and rec.get("changed_event_count") == len(changes)
                  and h16_rows[cid]["changed_events"] == len(changes),
                  f"{cid}: event assignment differences do not match H1.1/H1.6")
            saved_cost = rec.get("cost")
            saved_delta = h16_rows[cid]["objective"]["alternative_minus_original"]
            check(isinstance(saved_cost, dict)
                  and abs(h17.finite_number(saved_cost.get("final_delta"), "cost")
                          - saved_delta) <= 2e-6,
                  f"{cid}: objective provenance differs")
            plans[cid] = alternative
        check({x for x in found if x.startswith(sha[:12])} == scheduled,
              "H1.1/H1.7 chart candidate coverage differs")
    check(set(plans) == {x["id"] for c in h17_report["per_chart"]
                        for x in c["schedules"]}
          and len(found) == 22 and len(plans) == 25,
          "H1.1/H1.7 full schedule coverage differs")
    return plans, chart_times


def divergence(first, second, ids, times):
    check(len(first) == len(second) == len(ids) == len(times),
          "unaligned event assignments")
    mismatch_indices = [i for i, (a, b) in enumerate(zip(first, second))
                        if a != b]
    # A run is continuous in the SOURCE timeline's event order, not wall time.
    runs = []
    if mismatch_indices:
        start = previous = mismatch_indices[0]
        for i in mismatch_indices[1:] + [None]:
            if i is not None and i == previous + 1:
                previous = i
                continue
            runs.append({
                "start_event_index": ids[start],
                "end_event_index": ids[previous],
                "start_due_s": times[start],
                "end_due_s": times[previous],
                "changed_events": previous - start + 1,
            })
            if i is not None:
                start = i
                previous = i
    ranked = sorted(runs, key=lambda r: (
        -r["changed_events"], r["start_event_index"]))
    return {
        "different_event_count": len(mismatch_indices),
        "different_event_fraction": round(len(mismatch_indices) / len(first), 6),
        "different_event_run_count": len(runs),
        "largest_different_event_runs": ranked[:MAX_WINDOWS],
        "same_event_axis_assignments": not mismatch_indices,
    }


def sensitivity(chart):
    schedules = chart["schedules"]
    original_nodes = {s["id"] for s in schedules
                      if s["on_proxy_pareto_frontier"]}
    original_pair_lookup = {
        (p["first_id"], p["second_id"]): p["proxy_relation"]
        for p in chart["pairwise_comparisons"]}
    changes = []
    for ignored in PRESSURE_KEYS:
        dominated = set()
        transitions = Counter()
        new_counts = Counter()
        for a, b in combinations(schedules, 2):
            keys = [k for k in PRESSURE_KEYS if k != ignored]
            first = tuple(a["proxy_counts"][k] for k in keys)
            second = tuple(b["proxy_counts"][k] for k in keys)
            new = seven_axis_relation(first, second)
            old = original_pair_lookup[(a["id"], b["id"])]
            new_counts[new] += 1
            if new != old:
                transitions[f"{old} -> {new}"] += 1
            if new == "first_dominates":
                dominated.add(b["id"])
            elif new == "second_dominates":
                dominated.add(a["id"])
        frontier = {s["id"] for s in schedules} - dominated
        changes.append({
            "removed_pressure_dimension": ignored,
            "frontier_schedule_count": len(frontier),
            "frontier_alternative_count": sum(
                x in frontier for x in frontier if not x.startswith("original:")),
            "lost_original_frontier_members": sorted(original_nodes - frontier),
            "new_frontier_members": sorted(frontier - original_nodes),
            "changed_comparison_count": sum(transitions.values()),
            "relation_transitions": dict(sorted(transitions.items())),
            "new_relation_counts": dict(sorted(new_counts.items())),
        })
    return changes


def analyze(h17_report, plans, chart_times):
    chart_results = []
    for chart in h17_report["per_chart"]:
        sha = chart["chart_sha256"]
        ids, times = chart_times[sha]
        info = {node["id"]: node for node in chart["schedules"]}
        ties = []
        tradeoffs = []
        conflicts = []
        for p in chart["pairwise_comparisons"]:
            a = p["first_id"]
            b = p["second_id"]
            va = info[a]["proxy_counts"]
            vb = info[b]["proxy_counts"]
            diff = divergence(plans[a], plans[b], ids, times)
            if p["proxy_relation"] == "equal_proxies":
                ties.append({
                    "first_id": a, "second_id": b,
                    **diff,
                })
            elif p["proxy_relation"] == "tradeoff":
                tradeoffs.append({
                    "first_id": a, "second_id": b,
                    "first_has_lower_count_on": [k for k in PRESSURE_KEYS
                                                 if va[k] < vb[k]],
                    "second_has_lower_count_on": [k for k in PRESSURE_KEYS
                                                  if vb[k] < va[k]],
                    **diff,
                })
            if p["cost_prefers_proxy_dominated"]:
                rel = p["proxy_relation"]
                check(rel in ("first_dominates", "second_dominates"),
                      "H1.7 cost conflict has no proxy dominator")
                winner, loser = (a, b) if rel == "first_dominates" else (b, a)
                conflicts.append({
                    "proxy_dominant_id": winner,
                    "cost_preferred_but_proxy_dominated_id": loser,
                    "strictly_lower_pressure_dimensions": [
                        k for k in PRESSURE_KEYS
                        if info[winner]["proxy_counts"][k]
                        < info[loser]["proxy_counts"][k]],
                    "cost_delta_disagreement": round(
                        info[winner]["six_term_cost_delta_vs_original"]
                        - info[loser]["six_term_cost_delta_vs_original"], 9),
                    **diff,
                })
        check(len(ties) == chart["summary"]["equal_proxy_pair_count"]
              and len(tradeoffs) == chart["summary"]["proxy_tradeoff_pair_count"]
              and len(conflicts)
                  == chart["summary"]["pairwise_cost_prefers_dominated_count"],
              "H1.8/H1.7 per-chart pair category drift")
        sensitivity_rows = sensitivity(chart)
        chart_results.append({
            "chart": chart["chart"],
            "chart_sha256": sha,
            "summary": {
                "equal_proxy_pair_count": len(ties),
                "ties_with_different_actual_assignments": sum(
                    not x["same_event_axis_assignments"] for x in ties),
                "ties_with_identical_actual_assignments": sum(
                    x["same_event_axis_assignments"] for x in ties),
                "tradeoff_pair_count": len(tradeoffs),
                "cost_prefers_dominated_pair_count": len(conflicts),
                "dimensions_whose_removal_changes_frontier": [
                    r["removed_pressure_dimension"] for r in sensitivity_rows
                    if r["lost_original_frontier_members"] or r["new_frontier_members"]],
                "dimensions_whose_removal_changes_pairwise_relation": [
                    r["removed_pressure_dimension"] for r in sensitivity_rows
                    if r["changed_comparison_count"] > 0],
            },
            "equal_proxy_pairs": ties,
            "tradeoff_pairs": tradeoffs,
            "cost_proxy_conflicts": conflicts,
            "leave_one_dimension_out": sensitivity_rows,
        })
    overall = {k: sum(c["summary"][k] for c in chart_results)
               for k in (
                   "equal_proxy_pair_count",
                   "ties_with_different_actual_assignments",
                   "ties_with_identical_actual_assignments",
                   "tradeoff_pair_count",
                   "cost_prefers_dominated_pair_count",
               )}
    check(overall["equal_proxy_pair_count"] == h17_report["overall"]["equal_proxy_pair_count"]
          and overall["tradeoff_pair_count"] == h17_report["overall"]["proxy_tradeoff_pair_count"]
          and overall["cost_prefers_dominated_pair_count"]
              == h17_report["overall"]["pairwise_cost_prefers_dominated_count"],
          "H1.8/H1.7 overall categories drifted")
    return overall, chart_results


def compile_analysis(h16_report, h17_report, h11_report, *,
                     h16_sha256, h17_sha256, h11_sha256):
    check(h17_report.get("format") == h17.FORMAT, "H1.7 format differs")
    check(h17_report.get("source_h16_report_sha256") == h16_sha256,
          "H1.7/H1.6 SHA256 mismatch")
    check(h17_report.get("source_underlying_sha256", {}).get("h11") == h11_sha256,
          "H1.7/H1.1 SHA256 mismatch")
    check(h17_report == h17.compile_analysis(h16_report, h16_sha256=h16_sha256),
          "H1.7 does not recompute from its exact H1.6 source")
    plans, times = verify_h11(h11_report, h17_report, h16_report)
    overall, charts = analyze(h17_report, plans, times)
    return {
        "format": FORMAT,
        "source_sha256": {
            "h11": h11_sha256, "h16": h16_sha256, "h17": h17_sha256,
        },
        "scope": {
            "train_charts": 3,
            "original_schedules": 3,
            "alternative_schedules": 22,
            "within_chart_pairs": 113,
            "new_motor_runs": 0,
            "training_performed": False,
            "validation20_or_final_accessed": False,
            "frozen_planner_motor_timing_changed": False,
        },
        "overall": overall,
        "per_chart": charts,
        "interpretation": {
            "human_likeness_verified": False,
            "human_preference_observed": False,
            "fatigue_or_comfort_measured": False,
            "hud_only_student_verified": False,
            "event_difference_regions_are_causes_of_proxy_changes": False,
            "leave_one_out_is_human_calibration": False,
            "statistical_significance_test_performed": False,
            "same_proxy_counts_do_not_imply_same_fingering": True,
            "recomputed_h17_from_saved_h16": True,
            "recomputed_h11_from_original_simulation": False,
            "limitations": [
                "H1.1/H1.6/H1.7 input files are SHA-pinned; raw H1.5 episodes are not rerun.",
                "Proxy equality does not prove fingerings are physically identical.",
                "Mismatch windows locate changed event-axis assignments, not the physical cause of pressure changes.",
                "No metric is a calibrated human strain, ease, or preference measurement.",
                "Three correlated Train charts; 113 pairs are not independent samples.",
                "Drop-one-dimension sensitivity is a diagnostic, not a selection rule.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h11-report", "h16-report", "h17-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing H1.8 analysis")
    h11_report = json.loads(args.h11_report.read_text(encoding="utf-8-sig"))
    h16_report = json.loads(args.h16_report.read_text(encoding="utf-8-sig"))
    h17_report = json.loads(args.h17_report.read_text(encoding="utf-8-sig"))
    result = compile_analysis(
        h16_report, h17_report, h11_report,
        h11_sha256=sha256_file(args.h11_report),
        h16_sha256=sha256_file(args.h16_report),
        h17_sha256=sha256_file(args.h17_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({
        "overall": result["overall"],
        "per_chart": [
            {"chart": c["chart"], "summary": c["summary"]}
            for c in result["per_chart"]],
        "saved": str(args.output),
        "human_likeness_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
