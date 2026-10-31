from __future__ import annotations

"""H1.9: read-only local assignment and six-term cost discrepancy audit.

Recheck H1.1, H1.6, H1.7, H1.8 as one pinned Train-only chain.
Distances are movement in *abstract axis-depth indices*, not human effort.
Per-event difference clustering locates changes, not causes of proxy scores.
"""

import argparse
from collections import Counter, deque
from itertools import combinations
import json
from pathlib import Path

import audit_dmdogo_h18_proxy_identifiability as h18
from audit_dmdogo_h1_quality_evidence import TERMS
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-h19-local-proxy-diagnostics-v1"
EPS = 1e-7
COST_TOL = 2e-6
QUICK_GAP_S = .120
WINDOW_S = 1.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    return h18.h17.finite_number(value, name)


def six_terms(values, name):
    require(isinstance(values, dict) and set(values) == set(TERMS),
            f"{name}: missing or unexpected six-term components")
    return {k: finite(values[k], f"{name}.{k}") for k in TERMS}


def cost_rows(h11_report, h16_report, h17_report):
    """Map each chart's original to a zero cost delta, and validate 22 alternatives."""
    h16 = {row["candidate_id"]: row for row in h16_report["pairs"]}
    result = {}
    count = 0
    for chart in h11_report["charts"]:
        sha = chart["chart_sha256"]
        original_id = f"original:{sha[:12]}"
        result[original_id] = {k: 0.0 for k in TERMS}
        for rec in chart["reconstructions"]:
            cid = rec["candidate_id"]
            require(cid in h16 and cid not in result, f"{cid}: duplicate/missing H1.6 cost")
            saved = rec["cost"]
            require(isinstance(saved, dict), f"{cid}: H1.1 cost absent")
            original = finite(saved.get("original_v08_final"), f"{cid} original cost")
            forced = finite(saved.get("forced_final"), f"{cid} forced cost")
            delta = finite(saved.get("final_delta"), f"{cid} final delta")
            terms = six_terms(saved.get("six_component_deltas"), f"{cid} H1.1 terms")
            h16terms = six_terms(h16[cid]["objective"].get("six_term_deltas"),
                                  f"{cid} H1.6 terms")
            require(abs((forced - original) - delta) <= COST_TOL
                    and abs(sum(terms.values()) - delta) <= COST_TOL
                    and abs(h16[cid]["objective"]["alternative_minus_original"] - delta)
                        <= COST_TOL
                    and all(abs(terms[k] - h16terms[k]) <= COST_TOL for k in TERMS),
                    f"{cid}: six-component historical cost sum or join differs")
            result[cid] = terms
            count += 1
    require(count == 22
            and set(result) == {s["id"] for c in h17_report["per_chart"]
                                for s in c["schedules"]},
            "H1.9 cost coverage differs")
    return result


def movement_counts(plan, times):
    """Unweighted index-depth travel on adjacent single-contact events.

    Reconstructed hand/axis names are virtual; do not equate this metric
    with physical displacement, finger biomechanics, or human comfort.
    """
    require(len(plan) == len(times), "movement timeline unaligned")
    travel = 0
    max_step = 0
    nonzero = 0
    same_hand_pairs = 0
    switch_pairs = 0
    for i in range(1, len(plan)):
        a, b = plan[i-1], plan[i]
        gap = times[i] - times[i-1]
        require(gap >= 0, "nonchronological event timeline")
        if len(a) != 1 or len(b) != 1 or gap > QUICK_GAP_S + 1e-9:
            continue
        left = a[0].split("_")
        right = b[0].split("_")
        if left[0] == right[0]:
            step = abs(int(left[1]) - int(right[1]))
            travel += step
            max_step = max(max_step, step)
            nonzero += step > 0
            same_hand_pairs += 1
        else:
            switch_pairs += 1
    return {
        "same_hand_quick_adjacent_pairs": same_hand_pairs,
        "sum_abstract_depth_steps": travel,
        "max_abstract_depth_step": max_step,
        "nonzero_depth_step_pairs": nonzero,
        "quick_opposite_hand_switch_pairs": switch_pairs,
    }


def difference_density(first, second, ids, times):
    """Sliding one-second mismatch density plus actual mismatch time extent."""
    require(len(first) == len(second) == len(ids) == len(times),
            "difference timeline unaligned")
    indices = [i for i, (a, b) in enumerate(zip(first, second)) if a != b]
    require(indices, "H1.8 proved non-identical candidates in this comparison")
    queue = deque()
    max_density = 0
    minimum_gap = None
    for i in indices:
        t = times[i]
        queue.append(t)
        while queue and t - queue[0] > WINDOW_S + 1e-9:
            queue.popleft()
        max_density = max(max_density, len(queue))
    for p, q in zip(indices, indices[1:]):
        dt = (times[q] - times[p]) * 1000
        require(dt >= 0, "negative changed-event separation")
        minimum_gap = dt if minimum_gap is None else min(minimum_gap, dt)
    return {
        "changed_event_count": len(indices),
        "first_changed_event_index": ids[indices[0]],
        "last_changed_event_index": ids[indices[-1]],
        "first_changed_due_s": times[indices[0]],
        "last_changed_due_s": times[indices[-1]],
        "changed_span_s": round(times[indices[-1]] - times[indices[0]], 6),
        "peak_changed_events_in_one_second": max_density,
        "minimum_changed_event_separation_ms": (
            None if minimum_gap is None else round(minimum_gap, 6)),
    }


def cost_conflict_breakdown(pair, cost_vectors, nodes):
    """Arithmetic contributions, not causes of the preference disagreement."""
    superior = pair["proxy_dominant_id"]
    preferred = pair["cost_preferred_but_proxy_dominated_id"]
    comparison = {
        k: round(cost_vectors[superior][k] - cost_vectors[preferred][k], 9)
        for k in TERMS
    }
    raw = {
        k: cost_vectors[superior][k] - cost_vectors[preferred][k]
        for k in TERMS
    }
    saved = finite(pair["cost_delta_disagreement"], "H1.8 disagreement")
    expected = (nodes[superior]["six_term_cost_delta_vs_original"] -
                nodes[preferred]["six_term_cost_delta_vs_original"])
    require(expected > EPS and saved > EPS
            and abs(expected - sum(raw.values())) <= COST_TOL
            and abs(expected - saved) <= COST_TOL,
            "H1.9 disagreement cost component reconstruction differs")
    return {
        "proxy_dominant_id": superior,
        "cost_preferred_id": preferred,
        "proxy_dominant_minus_cost_preferred_total": round(expected, 9),
        "component_deltas": comparison,
        "terms_favor_cost_preferred": [k for k in TERMS if raw[k] > EPS],
        "terms_favor_proxy_dominant": [k for k in TERMS if raw[k] < -EPS],
        "terms_neutral": [k for k in TERMS if abs(raw[k]) <= EPS],
        "proxy_dimensions_strictly_lower_in_dominant": (
            pair["strictly_lower_pressure_dimensions"]),
    }


def analyse(h18_report, h17_report, plans, chart_times, costs):
    output = []
    for h18chart, h17chart in zip(
            h18_report["per_chart"], h17_report["per_chart"]):
        sha = h17chart["chart_sha256"]
        require(sha == h18chart["chart_sha256"], "chart order mismatch")
        ids, times = chart_times[sha]
        nodes = {n["id"]: n for n in h17chart["schedules"]}
        original_ids = set(nodes)
        movement = {
            nid: movement_counts(plans[nid], times)
            for nid in original_ids
        }
        ties = []
        for pair in h18chart["equal_proxy_pairs"]:
            a, b = pair["first_id"], pair["second_id"]
            diff = difference_density(plans[a], plans[b], ids, times)
            require(pair["different_event_count"] == diff["changed_event_count"]
                    and pair["same_event_axis_assignments"] is False,
                    "H1.8 equal proxy divergence differs")
            aa = movement[a]
            bb = movement[b]
            unequal = [k for k in aa if aa[k] != bb[k]]
            ties.append({
                "first_id": a, "second_id": b,
                "different_local_movement_metrics": unequal,
                "movement_first": aa,
                "movement_second": bb,
                **diff,
            })
        conflicts = []
        for pair in h18chart["cost_proxy_conflicts"]:
            diff = cost_conflict_breakdown(pair, costs, nodes)
            ctx = difference_density(
                plans[pair["proxy_dominant_id"]],
                plans[pair["cost_preferred_but_proxy_dominated_id"]],
                ids, times)
            require(ctx["changed_event_count"] == pair["different_event_count"],
                    "H1.8 conflict divergence differs")
            conflicts.append({**diff, **ctx})
        tradeoffs = []
        for pair in h18chart["tradeoff_pairs"]:
            diff = difference_density(
                plans[pair["first_id"]], plans[pair["second_id"]], ids, times)
            require(diff["changed_event_count"] == pair["different_event_count"],
                    "H1.8 tradeoff divergence differs")
            tradeoffs.append({
                "first_id": pair["first_id"],
                "second_id": pair["second_id"],
                "first_lower_proxy_dimensions": pair["first_has_lower_count_on"],
                "second_lower_proxy_dimensions": pair["second_has_lower_count_on"],
                **diff,
            })
        require(len(ties) == h18chart["summary"]["equal_proxy_pair_count"]
                and len(conflicts) == h18chart["summary"]["cost_prefers_dominated_pair_count"]
                and len(tradeoffs) == h18chart["summary"]["tradeoff_pair_count"],
                "H1.8 category census changed")
        output.append({
            "chart": h18chart["chart"],
            "chart_sha256": sha,
            "summary": {
                "equal_proxy_pairs": len(ties),
                "equal_proxy_pairs_with_different_local_movement_metrics": sum(
                    bool(x["different_local_movement_metrics"]) for x in ties),
                "equal_proxy_pairs_still_equal_on_local_movement_metrics": sum(
                    not x["different_local_movement_metrics"] for x in ties),
                "cost_disagreements": len(conflicts),
                "tradeoff_pairs": len(tradeoffs),
                "cost_component_favors_preferred_counts": {
                    k: sum(k in c["terms_favor_cost_preferred"] for c in conflicts)
                    for k in TERMS
                },
                "cost_component_favors_proxy_dominant_counts": {
                    k: sum(k in c["terms_favor_proxy_dominant"] for c in conflicts)
                    for k in TERMS
                },
            },
            "ties": ties,
            "cost_proxy_conflicts": conflicts,
            "tradeoff_time_locality": tradeoffs,
        })
    totals = {key: sum(c["summary"][key] for c in output) for key in (
        "equal_proxy_pairs", "equal_proxy_pairs_with_different_local_movement_metrics",
        "equal_proxy_pairs_still_equal_on_local_movement_metrics",
        "cost_disagreements", "tradeoff_pairs",
    )}
    for name in ("cost_component_favors_preferred_counts",
                 "cost_component_favors_proxy_dominant_counts"):
        totals[name] = {
            k: sum(c["summary"][name][k] for c in output) for k in TERMS
        }
    require(totals["equal_proxy_pairs"]
                == sum(c["summary"]["equal_proxy_pair_count"]
                       for c in h18_report["per_chart"])
            and totals["cost_disagreements"]
                == h18_report["overall"]["cost_prefers_dominated_pair_count"]
            and totals["tradeoff_pairs"]
                == h18_report["overall"]["tradeoff_pair_count"],
            "H1.9 candidate pair census differs")
    return totals, output


def compile_analysis(h11_report, h16_report, h17_report, h18_report, *,
                     h11_sha256, h16_sha256, h17_sha256, h18_sha256):
    require(h18_report.get("format") == h18.FORMAT,
            "H1.8 report format differs")
    require(h18_report.get("source_sha256") == {
        "h11": h11_sha256, "h16": h16_sha256, "h17": h17_sha256,
    }, "H1.8 source digest mismatch")
    recomputed = h18.compile_analysis(
        h16_report, h17_report, h11_report,
        h11_sha256=h11_sha256,
        h16_sha256=h16_sha256,
        h17_sha256=h17_sha256,
    )
    require(h18_report == recomputed,
            "H1.8 report does not recompute from exact saved source")
    plans, times = h18.verify_h11(h11_report, h17_report, h16_report)
    costs = cost_rows(h11_report, h16_report, h17_report)
    overall, charts = analyse(h18_report, h17_report, plans, times, costs)
    return {
        "format": FORMAT,
        "source_sha256": {
            "h11": h11_sha256, "h16": h16_sha256,
            "h17": h17_sha256, "h18": h18_sha256,
        },
        "scope": {
            "distinct_train_charts": 3,
            "distinct_events": 10231,
            "alternative_schedules": 22,
            "equal_proxy_pairs": 31,
            "cost_conflict_pairs": 26,
            "tradeoff_pairs": 44,
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
            "movement_is_human_physiology": False,
            "cost_component_arithmetic_implies_causality": False,
            "difference_windows_explain_metric_changes_causally": False,
            "new_motor_traces_reviewed": False,
            "statistical_significance_test_performed": False,
            "limitations": [
                "Only 3 previously selected Train charts, with correlated alternatives.",
                "Derived counts and axis-depth steps are not calibrated human effort.",
                "Term differences are exact arithmetic accounting, not evidence of why a human chooses a fingering.",
                "Full H1.1/H1.6/H1.7/H1.8 saved reports are verified by SHA and recomputation, but original H1.5 traces are not rerun.",
                "Equality on added movement counts still does not imply identical event assignments.",
                "No training, Motor runs, holdout access, or intervention.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h11-report", "h16-report", "h17-report", "h18-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing H1.9 analysis")
    h11 = json.loads(args.h11_report.read_text(encoding="utf-8-sig"))
    h16 = json.loads(args.h16_report.read_text(encoding="utf-8-sig"))
    h17 = json.loads(args.h17_report.read_text(encoding="utf-8-sig"))
    h18report = json.loads(args.h18_report.read_text(encoding="utf-8-sig"))
    result = compile_analysis(
        h11, h16, h17, h18report,
        h11_sha256=sha256_file(args.h11_report),
        h16_sha256=sha256_file(args.h16_report),
        h17_sha256=sha256_file(args.h17_report),
        h18_sha256=sha256_file(args.h18_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "overall": result["overall"],
        "per_chart": [{"chart": c["chart"], "summary": c["summary"]}
                      for c in result["per_chart"]],
        "saved": str(args.output),
        "human_likeness_verified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
