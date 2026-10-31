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
import audit_dmdogo_h18_proxy_identifiability as h18
from test_dmdogo_h17_proxy_pareto import fixture_h16


def fixture_sources():
    """Synthetic full-size source event vectors; NOT a physiological sample."""
    h16 = fixture_h16()
    h17_report = h17.compile_analysis(h16, h16_sha256="a" * 64)
    charts = []
    for chart in h17_report["per_chart"]:
        sha, size = chart["chart_sha256"], chart["original_source_events"]
        original_events = [{
            "event_index": j,
            "due_s": j * 0.1,
            "chord_size": 1,
            "original_axes": ["left_1"],
        } for j in range(size)]
        reconstructions = []
        matching_rows = {x["candidate_id"]: x for x in h16["pairs"]
                         if x["chart_sha256"] == sha}
        for i, schedule in enumerate(chart["schedules"][1:]):
            cid = schedule["id"]
            candidate_axes = [["left_1"] for _ in range(size)]
            # Change a different event for each candidate, even when proxy
            # vectors happen to match.
            candidate_axes[10 + i] = ["right_1"]
            matching_rows[cid]["changed_events"] = 1
            reconstructions.append({
                "candidate_id": cid,
                "changed_event_count": 1,
                "changed_events": [{
                    "event_index": 10 + i,
                    "due_s": (10 + i) * 0.1,
                    "original_axes": ["left_1"],
                    "alternative_axes": ["right_1"],
                }],
                "alternative_axes_by_event_order": candidate_axes,
                "cost": {"final_delta":
                         schedule["six_term_cost_delta_vs_original"]},
            })
        charts.append({
            "chart": chart["chart"],
            "chart_sha256": sha,
            "event_count": size,
            "count": len(reconstructions),
            "original_events_by_index": original_events,
            "reconstructions": reconstructions,
        })
    h11 = {
        "format": h18.H11_FORMAT,
        "chart_count": 3,
        "target_count": 22,
        "strict_lower_heuristic_cost_count": 7,
        "complete_event_axis_assignments": True,
        "human_preference_observed": False,
        "alternate_motor_executed": False,
        "motor_quality_verified": False,
        "validation20_or_final_accessed": False,
        "training_performed": False,
        "frozen_planner_motor_timing_changed": False,
        "charts": charts,
    }
    return h16, h17_report, h11


def run(h16, h17_report, h11):
    return h18.compile_analysis(
        h16, h17_report, h11,
        h16_sha256="a" * 64,
        h17_sha256="b" * 64,
        h11_sha256="a" * 64,
    )


def test_h18_full_join_and_113_pairs():
    h16, h17_report, h11 = fixture_sources()
    output = run(h16, h17_report, h11)
    assert output["format"] == h18.FORMAT
    assert output["overall"]["equal_proxy_pair_count"] == (
        h17_report["overall"]["equal_proxy_pair_count"])
    assert output["overall"]["tradeoff_pair_count"] == (
        h17_report["overall"]["proxy_tradeoff_pair_count"])
    assert output["overall"]["cost_prefers_dominated_pair_count"] == (
        h17_report["overall"]["pairwise_cost_prefers_dominated_count"])
    assert output["overall"]["ties_with_identical_actual_assignments"] == 0
    assert output["overall"]["ties_with_different_actual_assignments"] > 0
    assert all(len(c["leave_one_dimension_out"]) == 8
               for c in output["per_chart"])
    assert all(t["different_event_count"] ==
               (1 if t["first_id"].startswith("original:") or
                     t["second_id"].startswith("original:") else 2)
               for c in output["per_chart"] for t in c["equal_proxy_pairs"])
    assert output["scope"]["new_motor_runs"] == 0
    assert output["interpretation"]["human_likeness_verified"] is False
    assert output["interpretation"]["event_difference_regions_are_causes_of_proxy_changes"] is False


def test_no_cross_chart_pairs_or_event_ids():
    h16, h17_report, h11 = fixture_sources()
    out = run(h16, h17_report, h11)
    assert sum(c["summary"]["equal_proxy_pair_count"] +
               c["summary"]["tradeoff_pair_count"] +
               len([p for p in h17_report["per_chart"][i]["pairwise_comparisons"]
                    if p["proxy_relation"] in ("first_dominates", "second_dominates")])
               for i, c in enumerate(out["per_chart"])) == 113
    for c in out["per_chart"]:
        for group in ("equal_proxy_pairs", "tradeoff_pairs"):
            for row in c[group]:
                assert c["chart_sha256"][:12] in row["first_id"]
                assert c["chart_sha256"][:12] in row["second_id"]
                # Synthetic alternatives change event (10 + i). If their
                # changed event indices are adjacent, the differences
                # form ONE contiguous run, not two separate runs.
                first, second = row["first_id"], row["second_id"]
                if first.startswith("original:") or second.startswith("original:"):
                    expected_events, expected_runs = 1, 1
                else:
                    first_i = int(first.split("-")[1])
                    second_i = int(second.split("-")[1])
                    expected_events = 2
                    expected_runs = 1 if abs(first_i - second_i) == 1 else 2
                assert row["different_event_count"] == expected_events
                assert row["different_event_run_count"] == expected_runs


def test_divergence_run_segmentation():
    ids = (10, 11, 12, 13, 14, 15)
    times = tuple(x / 10 for x in ids)
    a = (("left_1",),) * 6
    b = (("right_1",), ("right_1",), ("left_1",),
         ("right_1",), ("left_1",), ("right_1",))
    out = h18.divergence(a, b, ids, times)
    assert out["different_event_count"] == 4
    assert out["different_event_run_count"] == 3
    assert out["largest_different_event_runs"][0]["changed_events"] == 2
    assert out["largest_different_event_runs"][0]["start_event_index"] == 10
    assert h18.divergence(a, a, ids, times)["same_event_axis_assignments"] is True


def test_drop_one_axis_can_create_equalities_and_both_frontier_directions():
    # A 2D example expressed over eight dimensions.
    k = h18.PRESSURE_KEYS
    node = lambda id, vec, frontier: {
        "id": id,
        "proxy_counts": dict(zip(k, vec)),
        "on_proxy_pareto_frontier": frontier,
    }
    a = (0, 1) + (0,) * 6
    b = (1, 0) + (0,) * 6
    c = (1, 1) + (0,) * 6
    chart = {
        "schedules": [
            node("original:000000000000", a, True),
            node("alt1", b, True),
            node("alt2", c, False),
        ],
        "pairwise_comparisons": [
            {"first_id": "original:000000000000", "second_id": "alt1",
             "proxy_relation": "tradeoff"},
            {"first_id": "original:000000000000", "second_id": "alt2",
             "proxy_relation": "first_dominates"},
            {"first_id": "alt1", "second_id": "alt2",
             "proxy_relation": "first_dominates"},
        ],
    }
    res = h18.sensitivity(chart)
    drop_x = res[0]
    assert drop_x["lost_original_frontier_members"] == ["original:000000000000"]
    assert drop_x["frontier_schedule_count"] == 1
    assert drop_x["changed_comparison_count"] >= 1
    drop_y = res[1]
    assert drop_y["lost_original_frontier_members"] == ["alt1"]


@pytest.mark.parametrize("change,expected", [
    (lambda h16,h17,h11: h11["charts"][0]["reconstructions"][0].update(
        {"changed_event_count": 0}), "event assignment differences"),
    (lambda h16,h17,h11: h11["charts"][0]["reconstructions"][0][
        "alternative_axes_by_event_order"].pop(), "incomplete H1.1"),
    (lambda h16,h17,h11: h11["charts"][0]["original_events_by_index"][0].update(
        {"chord_size": 2}), "invalid frozen"),
    (lambda h16,h17,h11: h11["charts"][0]["reconstructions"][0]["cost"].update(
        {"final_delta": 900}), "objective provenance"),
    (lambda h16,h17,h11: h17["overall"].update(
        {"equal_proxy_pair_count": 1}), "does not recompute"),
    (lambda h16,h17,h11: h11.update(
        {"validation20_or_final_accessed": True}), "scope or provenance"),
])
def test_fail_closed_on_bad_inputs(change, expected):
    h16, h17_report, h11 = fixture_sources()
    change(h16, h17_report, h11)
    with pytest.raises(ValueError, match=expected):
        run(h16, h17_report, h11)


def test_reject_mismatched_h11_digest():
    h16, h17_report, h11 = fixture_sources()
    with pytest.raises(ValueError, match="H1.7/H1.1 SHA256 mismatch"):
        h18.compile_analysis(
            h16, h17_report, h11, h16_sha256="a" * 64,
            h17_sha256="b" * 64, h11_sha256="c" * 64)


def test_8_axes_nonnegative_and_no_silent_7d_switch():
    with pytest.raises(ValueError):
        h18.seven_axis_relation((True,) * 7, (0,) * 7)
    with pytest.raises(ValueError):
        h18.seven_axis_relation((1, 2), (1,))
    assert h18.seven_axis_relation((1, 2), (1, 2)) == "equal_proxies"
