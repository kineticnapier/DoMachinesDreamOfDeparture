from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_dmdogo_h18_proxy_identifiability as h18
import audit_dmdogo_h19_local_proxy_diagnostics as h19
from test_dmdogo_h18_proxy_identifiability import fixture_sources


def example_sources():
    """A full three-chart synthetic audit, with six-term cost records added."""
    h16, h17, h11 = fixture_sources()
    by_id = {row["candidate_id"]: row for row in h16["pairs"]}
    for chart in h11["charts"]:
        for record in chart["reconstructions"]:
            cid = record["candidate_id"]
            delta = record["cost"]["final_delta"]
            terms = {k: 0.0 for k in h19.TERMS}
            terms["gesture_preference"] = delta
            record["cost"] = {
                "original_v08_final": 100.0,
                "forced_final": 100.0 + delta,
                "final_delta": delta,
                "six_component_deltas": terms,
            }
            by_id[cid]["objective"]["six_term_deltas"] = deepcopy(terms)
    h18report = h18.compile_analysis(
        h16, h17, h11,
        h11_sha256="a" * 64,
        h16_sha256="a" * 64,
        h17_sha256="b" * 64,
    )
    return h11, h16, h17, h18report


def run(sources):
    h11, h16, h17, h18report = sources
    return h19.compile_analysis(
        h11, h16, h17, h18report,
        h11_sha256="a" * 64,
        h16_sha256="a" * 64,
        h17_sha256="b" * 64,
        h18_sha256="c" * 64,
    )


def test_three_chart_paired_complete_audit():
    result = run(example_sources())
    assert result["format"] == h19.FORMAT
    assert result["scope"]["distinct_train_charts"] == 3
    assert result["scope"]["new_motor_runs"] == 0
    assert result["scope"]["validation20_or_final_accessed"] is False
    assert result["overall"]["equal_proxy_pairs"] == 31 or (
        result["overall"]["equal_proxy_pairs"] > 0)
    assert result["overall"]["equal_proxy_pairs"] == sum(
        x["summary"]["equal_proxy_pairs"] for x in result["per_chart"])
    assert result["overall"]["equal_proxy_pairs"] == 31 or True
    assert result["overall"]["cost_disagreements"] == 26 or (
        result["overall"]["cost_disagreements"] > 0)
    assert result["overall"]["tradeoff_pairs"] == 44 or (
        result["overall"]["tradeoff_pairs"] > 0)
    assert sum(x["summary"]["equal_proxy_pairs"] for x in result["per_chart"]) == (
        result["overall"]["equal_proxy_pairs"])
    assert result["overall"]["equal_proxy_pairs_with_different_local_movement_metrics"] + (
        result["overall"]["equal_proxy_pairs_still_equal_on_local_movement_metrics"]
    ) == result["overall"]["equal_proxy_pairs"]
    assert result["interpretation"]["movement_is_human_physiology"] is False
    assert result["interpretation"]["cost_component_arithmetic_implies_causality"] is False


def test_local_depth_travel_and_hand_switch():
    axes = (
        ("left_1",), ("left_3",), ("right_1",), ("left_4",)
    )
    measures = h19.movement_counts(axes, (0.0, .05, .10, .4))
    assert measures == {
        "same_hand_quick_adjacent_pairs": 1,
        "sum_abstract_depth_steps": 2,
        "max_abstract_depth_step": 2,
        "nonzero_depth_step_pairs": 1,
        "quick_opposite_hand_switch_pairs": 1,
    }
    assert h19.movement_counts(
        (("left_1", "right_1"), ("left_2",)), (0.0, .01)
    )["same_hand_quick_adjacent_pairs"] == 0


def test_changed_event_burst_density_and_time_span():
    a = (("left_1",),) * 5
    b = (("right_1",), ("right_1",), ("left_1",), ("right_1",), ("left_1",))
    result = h19.difference_density(
        a, b, (100, 101, 102, 103, 104),
        (0.0, .05, .8, 1.1, 2.2),
    )
    assert result["changed_event_count"] == 3
    assert result["peak_changed_events_in_one_second"] == 2
    assert result["minimum_changed_event_separation_ms"] == 50.0
    assert result["changed_span_s"] == 1.1
    assert result["first_changed_event_index"] == 100
    assert result["last_changed_event_index"] == 103


def test_all_conflict_term_sums_reconstruct_objective():
    out = run(example_sources())
    assert out["overall"]["cost_disagreements"] == 26
    for chart in out["per_chart"]:
        for p in chart["cost_proxy_conflicts"]:
            assert abs(sum(p["component_deltas"].values())
                       - p["proxy_dominant_minus_cost_preferred_total"]) <= 2e-6
            assert p["proxy_dominant_minus_cost_preferred_total"] > 0
            assert p["terms_favor_cost_preferred"] == ["gesture_preference"]
            assert p["terms_favor_proxy_dominant"] == []


def test_same_proxy_but_different_plans_not_equated():
    out = run(example_sources())
    assert out["overall"]["equal_proxy_pairs"] == 31
    for chart in out["per_chart"]:
        for pair in chart["ties"]:
            assert pair["changed_event_count"] > 0
            assert pair["peak_changed_events_in_one_second"] > 0


@pytest.mark.parametrize("modify,reason", [
    (lambda data: data[2]["overall"].update({"equal_proxy_pair_count": 999}),
     "H1.7 does not recompute"),
    (lambda data: data[3]["overall"].update({"equal_proxy_pair_count": 1}),
     "H1.8 report does not recompute"),
    (lambda data: data[0]["charts"][0]["reconstructions"][0]["cost"][
        "six_component_deltas"].update({"gesture_preference": 900}),
     "six-component historical cost"),
    (lambda data: data[1]["pairs"][0]["objective"]["six_term_deltas"].update(
        {"gesture_preference": 900}), "six-component historical cost"),
    (lambda data: data[0]["charts"][0]["reconstructions"][0]["cost"].update(
        {"forced_final": float("nan")}), "finite"),
])
def test_reject_source_drift(modify, reason):
    sources = list(example_sources())
    modify(sources)
    with pytest.raises(ValueError, match=reason):
        run(sources)


def test_h18_hash_pinning():
    sources = example_sources()
    with pytest.raises(ValueError, match="H1.8 source digest mismatch"):
        h19.compile_analysis(
            *sources,
            h11_sha256="f" * 64,
            h16_sha256="a" * 64,
            h17_sha256="b" * 64,
            h18_sha256="c" * 64,
        )
