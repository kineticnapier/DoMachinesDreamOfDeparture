from __future__ import annotations

"""H1.7: read-only Pareto audit of H1.6 abstract fingering proxies.

Compare complete ORIGINAL and alternative schedules only within the same
Train chart. A smaller proxy value means a smaller *abstract count*, not a
measured human preference, physical fatigue, or better gameplay. The frozen
six-term objective and Motor telemetry remain separate evidence channels.

No Motor runs, HUD inference, training, Validation20, FINAL or source edits.
The 22 alternatives are correlated counterfactuals on THREE charts.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import re

from audit_dmdogo_h13_automated_fingering_quality import PRESSURE_KEYS
from audit_dmdogo_h16_motor_proxy_relationships import FORMAT as H16_FORMAT
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-h17-within-chart-proxy-pareto-v1"
EXPECTED_EVENTS = (1540, 3637, 5054)
EXPECTED_ALTERNATIVES = (10, 10, 2)
COST_EPS = 1e-7
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: expected finite number")
    return value


def proxy_vector(metrics, label):
    require(isinstance(metrics, dict), f"{label}: invalid metrics")
    values = []
    for name in PRESSURE_KEYS:
        value = metrics.get(name)
        require(type(value) is int and value >= 0,
                f"{label}: {name} must be a nonnegative integer")
        values.append(value)
    return tuple(values)


def relation(first, second):
    """Direction refers solely to lower-is-better abstract pressure counts."""
    require(len(first) == len(second) == len(PRESSURE_KEYS),
            "proxy vector dimensionality differs")
    for x in (*first, *second):
        require(type(x) is int and x >= 0, "invalid proxy vector value")
    if first == second:
        return "equal_proxies"
    if all(a <= b for a, b in zip(first, second)):
        return "first_dominates"
    if all(b <= a for a, b in zip(first, second)):
        return "second_dominates"
    return "tradeoff"


def verify_source(report):
    """Fail closed on a drifted or incomplete H1.6 descriptive report."""
    require(isinstance(report, dict) and report.get("format") == H16_FORMAT,
            "not a supported H1.6 report")
    scope = report.get("scope")
    require(isinstance(scope, dict)
            and scope.get("distinct_train_charts") == 3
            and scope.get("distinct_original_events") == 10231
            and scope.get("independently_executed_alternative_episodes") == 22
            and scope.get("alternative_seams_per_chart") == [10, 10, 2]
            and scope.get("new_motor_runs") == 0
            and scope.get("source_v08_unchanged") is True,
            "H1.6 Train-only scope differs")
    require(report.get("human_likeness_verified") is False
            and report.get("human_preference_observed") is False
            and report.get("fatigue_or_comfort_measured") is False
            and report.get("hud_only_student_verified") is False
            and report.get("validation20_or_final_accessed") is False
            and report.get("frozen_planner_motor_timing_changed") is False,
            "H1.6 safety/evidence boundaries differ")
    overall = report.get("overall")
    require(isinstance(overall, dict)
            and overall.get("candidate_count") == 22
            and overall.get("all_hit_exact_plan_count") == 22
            and report.get("lower_cost_group", {}).get("candidate_count") == 7
            and report.get("not_lower_cost_group", {}).get("candidate_count") == 15,
            "H1.6 census count differs")
    charts = report.get("per_chart")
    pairs = report.get("pairs")
    require(isinstance(charts, list) and len(charts) == 3
            and isinstance(pairs, list) and len(pairs) == 22,
            "H1.6 missing charts/pairs")
    source_hashes = report.get("sources")
    require(isinstance(source_hashes, dict)
            and all(k in source_hashes and isinstance(source_hashes[k], str)
                    and SHA_RE.fullmatch(source_hashes[k])
                    for k in ("h11", "h13", "h15", "manifest", "verified")),
            "H1.6 source SHA256 provenance missing")

    grouped = {}
    seen = set()
    lower_count = 0
    for row in pairs:
        require(isinstance(row, dict), "invalid H1.6 pair")
        cid, sha = row.get("candidate_id"), row.get("chart_sha256")
        require(isinstance(sha, str) and SHA_RE.fullmatch(sha)
                and isinstance(cid, str)
                and re.fullmatch(re.escape(sha[:12]) + r"-\d+-\d+", cid)
                and cid not in seen,
                "missing or duplicate H1.6 candidate identity")
        seen.add(cid)
        require(row.get("physical_motor_success") is True
                and row.get("human_likeness_verified") is False,
                f"{cid}: Motor evidence boundary differs")
        physical = row.get("physical_motor")
        require(isinstance(physical, dict)
                and physical.get("misses") == 0
                and physical.get("local_replans") == 0
                and physical.get("hits") == row.get("chart_event_count")
                and physical.get("planned_axis_adoption_count") == row.get("chart_event_count")
                and physical.get("same_recorded_runtime_fingerprint") is True
                and physical.get("mae_direction_with_0p1ms_band") == "unchanged_within_band",
                f"{cid}: verified Motor/Timing census differs")
        objective = row.get("objective")
        require(isinstance(objective, dict), f"{cid}: missing objective")
        cost = finite_number(objective.get("alternative_minus_original"), "cost")
        lower = objective.get("lower_frozen_six_term_cost")
        require(type(lower) is bool and (cost < -COST_EPS) == lower,
                f"{cid}: objective gain flag differs")
        lower_count += lower
        abstract = row.get("abstract_fingering")
        require(isinstance(abstract, dict), f"{cid}: missing abstract fingering")
        original = abstract.get("original")
        alternative = abstract.get("alternative")
        ovec = proxy_vector(original, f"{cid} original")
        avec = proxy_vector(alternative, f"{cid} alternative")
        require(type(original.get("total_contacts")) is int
                and original["total_contacts"] > 0
                and alternative.get("total_contacts") == original["total_contacts"],
                f"{cid}: total contacts differ")
        deltas = abstract.get("pressure_proxy_deltas")
        require(isinstance(deltas, dict) and set(deltas) == set(PRESSURE_KEYS)
                and all(finite_number(deltas[k], k) == b - a
                        for k, a, b in zip(PRESSURE_KEYS, ovec, avec)),
                f"{cid}: proxy delta source differs")
        require(abstract.get("increased_proxy_count")
                == sum(b > a for a, b in zip(ovec, avec))
                and abstract.get("decreased_proxy_count")
                == sum(b < a for a, b in zip(ovec, avec)),
                f"{cid}: proxy sign counts differ")
        grouped.setdefault(sha, []).append(row)

    require(lower_count == 7 and len(grouped) == 3,
            "H1.6 cost/candidate census differs")
    for chart, events, count in zip(charts, EXPECTED_EVENTS, EXPECTED_ALTERNATIVES):
        require(isinstance(chart, dict), "invalid chart record")
        sha = chart.get("chart_sha256")
        rows = grouped.get(sha, [])
        summary = chart.get("summary")
        require(isinstance(sha, str) and SHA_RE.fullmatch(sha)
                and chart.get("original_source_events") == events
                and isinstance(summary, dict)
                and summary.get("candidate_count") == count
                and summary.get("all_hit_exact_plan_count") == count
                and len(rows) == count,
                "H1.6 per-chart census differs")
        first = rows[0]["abstract_fingering"]["original"]
        base = proxy_vector(first, "original baseline")
        for row in rows:
            require(row.get("chart") == chart.get("chart")
                    and row.get("chart_event_count") == events
                    and row["abstract_fingering"]["original"] == first
                    and proxy_vector(row["abstract_fingering"]["original"], "baseline") == base,
                    "different baseline or chart within one Train chart")
    require(set(grouped) == {c["chart_sha256"] for c in charts},
            "H1.6 unexpected chart identity")
    return charts, grouped


def analyze_chart(chart, rows):
    """Pareto comparison of one baseline plus same-chart alternatives."""
    sha = chart["chart_sha256"]
    baseline = rows[0]["abstract_fingering"]["original"]
    nodes = [{
        "id": f"original:{sha[:12]}",
        "kind": "original",
        "proxy": proxy_vector(baseline, "original"),
        "six_term_cost_delta": 0.0,
    }]
    for row in rows:
        nodes.append({
            "id": row["candidate_id"],
            "kind": "alternative",
            "proxy": proxy_vector(row["abstract_fingering"]["alternative"],
                                  row["candidate_id"]),
            "six_term_cost_delta": row["objective"]["alternative_minus_original"],
        })
    dominated_by = {node["id"]: [] for node in nodes}
    dominates = {node["id"]: [] for node in nodes}
    equals = {node["id"]: [] for node in nodes}
    comparisons = []
    for a, b in combinations(nodes, 2):
        verdict = relation(a["proxy"], b["proxy"])
        superior, inferior = (a, b) if verdict == "first_dominates" else (
            (b, a) if verdict == "second_dominates" else (None, None))
        if superior:
            dominates[superior["id"]].append(inferior["id"])
            dominated_by[inferior["id"]].append(superior["id"])
        if verdict == "equal_proxies":
            equals[a["id"]].append(b["id"])
            equals[b["id"]].append(a["id"])
        cost_conflict = bool(superior and
            inferior["six_term_cost_delta"] < superior["six_term_cost_delta"] - COST_EPS)
        comparisons.append({
            "first_id": a["id"],
            "second_id": b["id"],
            "proxy_relation": verdict,
            "cost_prefers_proxy_dominated": cost_conflict,
        })

    base_id = nodes[0]["id"]
    summaries = []
    for node in nodes:
        nid = node["id"]
        baseline_relation = ("self" if nid == base_id
                             else relation(node["proxy"], nodes[0]["proxy"]))
        summaries.append({
            "id": nid,
            "kind": node["kind"],
            "on_proxy_pareto_frontier": not dominated_by[nid],
            "dominated_by": sorted(dominated_by[nid]),
            "dominates": sorted(dominates[nid]),
            "equal_proxy_vectors_with": sorted(equals[nid]),
            "vs_original": baseline_relation,
            "six_term_cost_delta_vs_original": node["six_term_cost_delta"],
            "proxy_counts": dict(zip(PRESSURE_KEYS, node["proxy"])),
        })
    counts = Counter(x["proxy_relation"] for x in comparisons)
    alt = summaries[1:]
    strict_better_vs_orig = sum(x["vs_original"] == "first_dominates" for x in alt)
    strict_worse_vs_orig = sum(x["vs_original"] == "second_dominates" for x in alt)
    cost_gains = [x for x in alt if x["six_term_cost_delta_vs_original"] < -COST_EPS]
    summary = {
        "schedule_count": len(nodes),
        "comparison_count": len(comparisons),
        "proxy_dominance_pair_count": counts["first_dominates"] + counts["second_dominates"],
        "proxy_tradeoff_pair_count": counts["tradeoff"],
        "equal_proxy_pair_count": counts["equal_proxies"],
        "frontier_schedule_count": sum(x["on_proxy_pareto_frontier"] for x in summaries),
        "frontier_alternative_count": sum(x["on_proxy_pareto_frontier"] for x in alt),
        "original_on_frontier": summaries[0]["on_proxy_pareto_frontier"],
        "alternative_proxy_dominates_original_count": strict_better_vs_orig,
        "original_proxy_dominates_alternative_count": strict_worse_vs_orig,
        "alternative_equal_original_proxies_count": sum(
            x["vs_original"] == "equal_proxies" for x in alt),
        "alternative_tradeoff_with_original_count": sum(
            x["vs_original"] == "tradeoff" for x in alt),
        "cost_lower_alternatives": len(cost_gains),
        "cost_lower_but_original_proxy_dominates_count": sum(
            x["vs_original"] == "second_dominates" for x in cost_gains),
        "cost_lower_but_any_proxy_dominator_count": sum(
            bool(x["dominated_by"]) for x in cost_gains),
        "cost_lower_on_frontier_count": sum(
            x["on_proxy_pareto_frontier"] for x in cost_gains),
        "pairwise_cost_prefers_dominated_count": sum(
            x["cost_prefers_proxy_dominated"] for x in comparisons),
    }
    require(sum(counts.values()) == len(comparisons), "pair accounting failed")
    return {
        "chart": chart["chart"],
        "chart_sha256": sha,
        "original_source_events": chart["original_source_events"],
        "summary": summary,
        "schedules": summaries,
        "pairwise_comparisons": comparisons,
    }


def compile_analysis(h16_report, *, h16_sha256):
    charts, groups = verify_source(h16_report)
    require(isinstance(h16_sha256, str) and SHA_RE.fullmatch(h16_sha256),
            "invalid H1.6 report digest")
    output = [analyze_chart(chart, groups[chart["chart_sha256"]]) for chart in charts]
    summaries = [c["summary"] for c in output]
    summed = (
        "comparison_count", "proxy_dominance_pair_count",
        "proxy_tradeoff_pair_count", "equal_proxy_pair_count",
        "frontier_schedule_count", "frontier_alternative_count",
        "alternative_proxy_dominates_original_count",
        "original_proxy_dominates_alternative_count",
        "alternative_equal_original_proxies_count",
        "alternative_tradeoff_with_original_count",
        "cost_lower_alternatives",
        "cost_lower_but_original_proxy_dominates_count",
        "cost_lower_but_any_proxy_dominator_count",
        "cost_lower_on_frontier_count",
        "pairwise_cost_prefers_dominated_count",
    )
    overall = {key: sum(x[key] for x in summaries) for key in summed}
    overall["train_chart_count"] = 3
    overall["original_schedule_count"] = 3
    overall["alternative_schedule_count"] = 22
    overall["schedule_count"] = 25
    require(overall["comparison_count"] == 113
            and overall["cost_lower_alternatives"] == 7
            and overall["proxy_dominance_pair_count"]
              + overall["proxy_tradeoff_pair_count"]
              + overall["equal_proxy_pair_count"] == 113,
            "H1.7 chart/pair accounting failed")
    return {
        "format": FORMAT,
        "source_h16_report_sha256": h16_sha256,
        "source_h16_format": H16_FORMAT,
        "source_underlying_sha256": h16_report["sources"],
        "scope": {
            "distinct_train_charts": 3,
            "distinct_original_events": 10231,
            "original_schedules": 3,
            "alternative_schedules": 22,
            "same_chart_pairs": 113,
            "new_motor_runs": 0,
            "source_v08_unchanged": True,
            "validation20_or_final_accessed": False,
        },
        "objective": "Unweighted, exact eight-dimensional proxy Pareto comparison",
        "all_proxy_dimensions_lower_is_abstractly_preferred": list(PRESSURE_KEYS),
        "overall": overall,
        "per_chart": output,
        "interpretation": {
            "only_within_chart": True,
            "human_preference_observed": False,
            "human_likeness_verified": False,
            "fatigue_or_comfort_measured": False,
            "hud_only_student_verified": False,
            "statistical_significance_test_performed": False,
            "pareto_is_human_preference": False,
            "cost_proxy_disagreements_imply_bad_human_fingering": False,
            "h16_source_recomputed_from_raw_h15": False,
            "limitations": [
                "H1.7 validates the saved H1.6 structure, but does not rerun H1.6 raw H1.5 trace checks.",
                "Eight abstract proxy counts are not calibrated human effort or preference.",
                "Two different fingerings may have equal vectors; ties are not identical plans.",
                "The six-term objective and eight proxies share scheduling information.",
                "113 pairs are comparisons within three selected charts, not 113 independent samples.",
                "Motor Hit and Timing MAE are saturated/near-equal and are not ranking criteria.",
                "No human labels, strain measurements, or HUD-only Student trials are available.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h16-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing H1.7 analysis")
    h16 = json.loads(args.h16_report.read_text(encoding="utf-8-sig"))
    result = compile_analysis(h16, h16_sha256=sha256_file(args.h16_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({
        "overall": result["overall"],
        "per_chart": [{"chart": x["chart"], "summary": x["summary"]}
                      for x in result["per_chart"]],
        "saved": str(args.output),
        "human_likeness_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
