from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_dmdogo_h17_proxy_pareto as h17


def fixture_h16():
    charts = []
    pairs = []
    global_index = 0
    for chart_index, (count, events) in enumerate(zip(
            h17.EXPECTED_ALTERNATIVES, h17.EXPECTED_EVENTS)):
        sha = str(chart_index + 1) * 64
        chart_name = f"synthetic-{chart_index}"
        charts.append({
            "chart": chart_name,
            "chart_sha256": sha,
            "original_source_events": events,
            "summary": {
                "candidate_count": count,
                "all_hit_exact_plan_count": count,
            },
        })
        original = {name: 5 for name in h17.PRESSURE_KEYS}
        original["total_contacts"] = 50
        for i in range(count):
            cid = f"{sha[:12]}-{i}-{i+1}"
            case = global_index % 4
            alternative = deepcopy(original)
            if case == 0:
                for name in h17.PRESSURE_KEYS:
                    alternative[name] = 4
            elif case == 1:
                for name in h17.PRESSURE_KEYS:
                    alternative[name] = 6
            elif case == 2:
                alternative[h17.PRESSURE_KEYS[0]] = 4
                alternative[h17.PRESSURE_KEYS[1]] = 6
            # case 3: equal proxy vector even though the assignment differs
            delta = {k: alternative[k] - original[k] for k in h17.PRESSURE_KEYS}
            cost = -0.5 if global_index < 7 else 0.5
            pairs.append({
                "candidate_id": cid,
                "chart": chart_name,
                "chart_sha256": sha,
                "chart_event_count": events,
                "physical_motor_success": True,
                "human_likeness_verified": False,
                "physical_motor": {
                    "misses": 0,
                    "local_replans": 0,
                    "hits": events,
                    "planned_axis_adoption_count": events,
                    "same_recorded_runtime_fingerprint": True,
                    "mae_direction_with_0p1ms_band": "unchanged_within_band",
                },
                "objective": {
                    "alternative_minus_original": cost,
                    "lower_frozen_six_term_cost": cost < -h17.COST_EPS,
                },
                "abstract_fingering": {
                    "original": deepcopy(original),
                    "alternative": alternative,
                    "pressure_proxy_deltas": delta,
                    "increased_proxy_count": sum(v > 0 for v in delta.values()),
                    "decreased_proxy_count": sum(v < 0 for v in delta.values()),
                },
            })
            global_index += 1
    return {
        "format": h17.H16_FORMAT,
        "scope": {
            "distinct_train_charts": 3,
            "distinct_original_events": 10231,
            "independently_executed_alternative_episodes": 22,
            "alternative_seams_per_chart": [10, 10, 2],
            "new_motor_runs": 0,
            "source_v08_unchanged": True,
        },
        "human_likeness_verified": False,
        "human_preference_observed": False,
        "fatigue_or_comfort_measured": False,
        "hud_only_student_verified": False,
        "validation20_or_final_accessed": False,
        "frozen_planner_motor_timing_changed": False,
        "overall": {"candidate_count": 22, "all_hit_exact_plan_count": 22},
        "lower_cost_group": {"candidate_count": 7},
        "not_lower_cost_group": {"candidate_count": 15},
        "sources": {k: "a" * 64 for k in
                    ("h11", "h13", "h15", "manifest", "verified")},
        "per_chart": charts,
        "pairs": pairs,
    }


def test_exact_pareto_dominance_and_tradeoff():
    baseline = (5,) * len(h17.PRESSURE_KEYS)
    better = (4,) * len(h17.PRESSURE_KEYS)
    worse = (6,) * len(h17.PRESSURE_KEYS)
    tradeoff = (4, 6) + (5,) * (len(h17.PRESSURE_KEYS) - 2)
    assert h17.relation(better, baseline) == "first_dominates"
    assert h17.relation(worse, baseline) == "second_dominates"
    assert h17.relation(tradeoff, baseline) == "tradeoff"
    assert h17.relation(baseline, baseline) == "equal_proxies"


@pytest.mark.parametrize("vector", [
    (True,) * len(h17.PRESSURE_KEYS),
    (-1,) + (5,) * (len(h17.PRESSURE_KEYS) - 1),
    (float("nan"),) + (5,) * (len(h17.PRESSURE_KEYS) - 1),
    (5,) * (len(h17.PRESSURE_KEYS) - 1),
])
def test_reject_invalid_proxy_vectors(vector):
    with pytest.raises(ValueError):
        h17.relation(vector, (5,) * len(h17.PRESSURE_KEYS))


def test_three_chart_25_schedule_113_pair_audit():
    result = h17.compile_analysis(fixture_h16(), h16_sha256="b" * 64)
    assert result["format"] == h17.FORMAT
    assert result["overall"]["train_chart_count"] == 3
    assert result["overall"]["schedule_count"] == 25
    assert result["overall"]["comparison_count"] == 113
    assert result["overall"]["cost_lower_alternatives"] == 7
    assert [x["summary"]["comparison_count"] for x in result["per_chart"]] == [55, 55, 3]
    assert sum(len(c["schedules"]) for c in result["per_chart"]) == 25
    assert sum(len(c["pairwise_comparisons"]) for c in result["per_chart"]) == 113
    assert result["overall"]["cost_lower_but_original_proxy_dominates_count"] > 0
    assert result["overall"]["pairwise_cost_prefers_dominated_count"] > 0
    assert result["interpretation"]["pareto_is_human_preference"] is False
    assert result["scope"]["new_motor_runs"] == 0

    first = result["per_chart"][0]["schedules"]
    assert first[1]["vs_original"] == "first_dominates"
    assert first[2]["vs_original"] == "second_dominates"
    assert first[3]["vs_original"] == "tradeoff"
    assert first[4]["vs_original"] == "equal_proxies"
    assert first[0]["on_proxy_pareto_frontier"] is False
    assert first[1]["on_proxy_pareto_frontier"] is True
    assert first[2]["on_proxy_pareto_frontier"] is False


@pytest.mark.parametrize("corrupt,match", [
    (lambda r: r["pairs"].pop(), "missing charts/pairs"),
    (lambda r: r["pairs"][1].update(candidate_id=r["pairs"][0]["candidate_id"]),
     "candidate identity"),
    (lambda r: r["pairs"][1]["abstract_fingering"]["original"].update(
        {h17.PRESSURE_KEYS[0]: 6}), "proxy delta source differs"),
    (lambda r: r["pairs"][0]["abstract_fingering"]["pressure_proxy_deltas"].update(
        {h17.PRESSURE_KEYS[0]: -99}), "proxy delta source differs"),
    (lambda r: r["pairs"][0]["physical_motor"].update(
        {"misses": 1}), "Motor/Timing"),
    (lambda r: r["pairs"][0]["objective"].update(
        {"lower_frozen_six_term_cost": False}), "objective gain"),
    (lambda r: r["scope"].update({"distinct_train_charts": 22}), "scope"),
    (lambda r: r["pairs"][0]["abstract_fingering"]["alternative"].update(
        {h17.PRESSURE_KEYS[0]: True}), "nonnegative integer"),
])
def test_fail_closed_on_source_drift(corrupt, match):
    report = fixture_h16()
    corrupt(report)
    with pytest.raises(ValueError, match=match):
        h17.compile_analysis(report, h16_sha256="b" * 64)


def test_never_compare_across_charts():
    report = h17.compile_analysis(fixture_h16(), h16_sha256="b" * 64)
    for chart in report["per_chart"]:
        prefix = chart["chart_sha256"][:12]
        for pair in chart["pairwise_comparisons"]:
            assert prefix in pair["first_id"]
            assert prefix in pair["second_id"]
