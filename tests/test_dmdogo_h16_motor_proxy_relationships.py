from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

import audit_dmdogo_h16_motor_proxy_relationships as h16


def fixture_pair(i=0, lower=True, chart_sha="e"*64):
    cid=f"{chart_sha[:12]}-{i}-{i+1}"
    original={
        "total_contacts":3,
        "hand_contacts":{"left":2,"right":1},
        "axis_contacts":{"left_1":2,"right_1":1},
        **{k:0 for k in h16.PRESSURE_KEYS},
    }
    alternative={
        "total_contacts":3,
        "hand_contacts":{"left":1,"right":2},
        "axis_contacts":{"left_1":1,"right_1":2},
        **{k:0 for k in h16.PRESSURE_KEYS},
    }
    original["quick_same_hand_depth_jump_pairs_120ms"]=1
    alternative["max_axis_contacts_in_half_second"]=1
    pressure={k:alternative[k]-original[k] for k in h16.PRESSURE_KEYS}
    proxy={
        "candidate_id":cid,"changed_event_count":2,
        "triage_tier":"P1_proxy_tradeoff",
        "heuristic_objective":{
            "alternative_minus_original":-.5 if lower else .5,
            "alternative_has_lower_frozen_heuristic_cost":lower,
            "six_term_deltas":{"term":-.5 if lower else .5},
        },
        "abstract_metrics":{"original":original,"alternative":alternative},
        "pressure_proxy_deltas":pressure,
        "increased_pressure_proxy_names":sorted(k for k,v in pressure.items() if v>0),
        "decreased_pressure_proxy_names":sorted(k for k,v in pressure.items() if v<0),
    }
    physical={
        "candidate_id":cid,"chart":"synthetic",
        "chart_sha256":chart_sha,"event_count":3,"changed_events":2,
        "status":"verified_episode","all_hit_and_exact_plan":True,
        "same_recorded_runtime_fingerprint_as_original":True,
        "selected_events":3,"planned_axis_adoption_count":3,
        "local_replans":0,"hits":3,"misses":0,"overloaded":False,
        "triage_tier":"P1_proxy_tradeoff",
        "heuristic_cost_delta":-.5 if lower else .5,
        "lower_heuristic_cost":lower,
        "original_timing_mae_ms":4.5,"timing_mae_ms":4.55,
        "timing_mae_delta_ms":.05,"original_deferred_events":1,
        "deferred_events":0,"original_hits":3,
    }
    return physical,proxy


def test_h15_saved_census_must_recompute_exactly():
    good={
        "format":h16.h15.FORMAT,
        "full_census_verified":True,
        "verified_episode_count":22,
        "all_hit_exact_plan_count":22,
        "pending_episode_count":0,
        "incomplete_episode_count":0,
        "matching_runtime_fingerprint_count":22,
        "reused_prior_h14_pilots":2,
        "human_likeness_verified":False,
        "validation20_or_final_accessed":False,
        "records":[{"status":"verified_episode"} for _ in range(22)],
    }
    h16.require_verified_h15(good,deepcopy(good))
    mismatch=deepcopy(good)
    mismatch["records"][0]["hits"]=2
    with pytest.raises(ValueError,match="unaudited"):
        h16.require_verified_h15(good,mismatch)
    fewer=deepcopy(good)
    fewer["verified_episode_count"]=21
    with pytest.raises(ValueError,match="incomplete"):
        h16.require_verified_h15(fewer,deepcopy(fewer))


def test_motor_diff_and_pressure_diff_do_not_imply_human_preference():
    motor,proxy=fixture_pair()
    row=h16.compare_pair(motor,proxy)
    assert row["objective"]["lower_frozen_six_term_cost"] is True
    assert row["physical_motor"]["alternative_minus_original_mae_ms"]==.05
    assert row["physical_motor"]["mae_direction_with_0p1ms_band"]=="unchanged_within_band"
    assert row["physical_motor"]["deferral_direction"]=="decreased"
    assert row["abstract_fingering"]["increased_proxy_count"]==1
    assert row["abstract_fingering"]["decreased_proxy_count"]==1
    assert row["abstract_fingering"]["hand_contacts_alternative_minus_original"]=={"left":-1,"right":1}
    assert row["abstract_fingering"]["axis_contacts_alternative_minus_original"]["left_1"]==-1
    assert row["human_likeness_verified"] is False


@pytest.mark.parametrize("mutate,message",[
    (lambda m,p: m.update({"all_hit_and_exact_plan":False}),"identity"),
    (lambda m,p: m.update({"planned_axis_adoption_count":2}),"identity"),
    (lambda m,p: m.update({"local_replans":1}),"identity"),
    (lambda m,p: m.update({"same_recorded_runtime_fingerprint_as_original":False}),"identity"),
    (lambda m,p: m.update({"timing_mae_delta_ms":8}),"MAE mismatch"),
    (lambda m,p: m.update({"original_deferred_events":None}),"reference missing"),
    (lambda m,p: p["pressure_proxy_deltas"].update({"axis_reuse_under_150ms":5}),"source delta differs"),
    (lambda m,p: p["abstract_metrics"]["alternative"]["axis_contacts"].update({"left_1":3}),"not conserved"),
    (lambda m,p: p["abstract_metrics"]["alternative"]["hand_contacts"].update({"right":3}),"hand contact conservation"),
    (lambda m,p: p["heuristic_objective"].update({"alternative_minus_original":7}),"cost or improvement"),
    (lambda m,p: p.update({"candidate_id":"wrong"}),"identity"),
])
def test_corrupt_pair_is_rejected(mutate,message):
    motor,proxy=fixture_pair()
    mutate(motor,proxy)
    with pytest.raises(ValueError,match=message):
        h16.compare_pair(motor,proxy)


def test_22_pairs_grouped_by_chart_and_cost_without_significance_claim():
    charts=[]
    original={}
    pairs=[]
    for chart_index,(count,size) in enumerate(((10,1540),(10,3637),(2,5054))):
        sha=str(chart_index+1)*64
        charts.append({
            "chart":"synthetic","chart_sha256":sha,
            "candidate_count":count,"event_count":size,
            "candidates":[],
        })
        original[sha]={
            "status":"chart_complete","hits":3,
            "timing_mae_ms":4.5,"local_replans":0,
            "deferred_events":1,
        }
        for i in range(count):
            lower=len(pairs)<7
            physical,proxy=fixture_pair(i,lower,sha)
            physical["chart"]="synthetic"
            charts[-1]["candidates"].append(proxy)
            pairs.append(physical)
    report=h16.compile_analysis(
        {},{"charts":charts},pairs,original,
        input_sha={"h15":"a"*64})
    assert report["scope"]["new_motor_runs"]==0
    assert report["overall"]["candidate_count"]==22
    assert report["lower_cost_group"]["candidate_count"]==7
    assert report["not_lower_cost_group"]["candidate_count"]==15
    assert report["overall"]["all_hit_exact_plan_count"]==22
    assert [x["summary"]["candidate_count"] for x in report["per_chart"]]==[10,10,2]
    assert report["overall"]["mae_direction_counts"]=={"unchanged_within_band":22}
    assert report["overall"]["deferral_direction_counts"]=={"decreased":22}
    assert report["human_likeness_verified"] is False
    assert report["interpretation"]["correlation_or_significance_test_performed"] is False


def test_missing_or_duplicate_22nd_join_rejected():
    c=[fixture_pair(i) for i in range(22)]
    proxies=[p for m,p in c]
    m=[a for a,p in c]
    with pytest.raises(ValueError,match="requires 22"):
        h16.compile_analysis({},{"charts":[{"candidates":proxies[:-1]}]},
                             m,{},input_sha={})


def test_direction_threshold_is_only_descriptive():
    assert h16.signed_bucket(-.101,.1)=="decreased"
    assert h16.signed_bucket(.1,.1)=="unchanged_within_band"
    assert h16.signed_bucket(.101,.1)=="increased"
    with pytest.raises(ValueError,match="finite"):
        h16.signed_bucket(float("nan"),.1)
