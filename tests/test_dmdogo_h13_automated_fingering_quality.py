from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

from audit_dmdogo_h13_automated_fingering_quality import (
    FORMAT, H11_FORMAT, PRESSURE_KEYS,
    metrics, timeline, flatten_chunks, compare_candidate, compile_audit, main,
)
from audit_dmdogo_h1_quality_evidence import H06_BLOB_SHA


def synthetic_chart(*, n=6, chart_id=1, targets=1, index_prefix="demo"):
    events=[
        {"event_index":i,"due_s":1.0+0.2*i,"chord_size":1,
         "original_axes":["left_1"] if i%2==0 else ["right_1"]}
        for i in range(n)]
    chunks=[{
        "start_event_index":i,"end_event_index":i,"event_count":1,
        "axes":[e["original_axes"][:]],
        "gesture_family":"single","rhythm":"single","reason":"test",
    } for i,e in enumerate(events)]
    cases=[]
    for c in range(targets):
        changed_index=1+c
        orig=events[changed_index]["original_axes"]
        swap=["right_2"] if orig==["right_1"] else ["left_2"]
        alt=[e["original_axes"][:] for e in events]
        alt[changed_index]=swap
        alt_chunks=deepcopy(chunks)
        alt_chunks[changed_index]["axes"]=[swap]
        term_names=(
            "gesture_preference","hand_exposure","axis_reuse_margin",
            "intra_gesture_hand_changes","intra_gesture_depth_distance",
            "chunk_join_transition")
        delta=-0.5 if chart_id==1 and c<7 else 0.2
        cases.append({
            "candidate_id":f"{index_prefix}-{chart_id}-{c}",
            "flag_pair":[changed_index-1,changed_index],
            "join_window_index":changed_index,
            "changed_chunk_indices":[changed_index],
            "changed_events":[{
                "event_index":changed_index,
                "due_s":events[changed_index]["due_s"],
                "original_axes":orig,
                "alternative_axes":swap,
            }],
            "changed_event_count":1,
            "alternative_axes_by_event_order":alt,
            "alternative_chunks":alt_chunks,
            "independent_intervention_only":True,
            "reached_terminal_abstract":True,
            "alternate_motor_result":"not_executed",
            "human_preference":"unobserved",
            "physical_reuse_80ms_proxy_violations":0,
            "cost":{
                "original_v08_final":100.0,
                "forced_final":100.0+delta,
                "final_delta":delta,
                "six_component_deltas":{
                    term:(delta if idx==0 else 0.0)
                    for idx,term in enumerate(term_names)},
            },
        })
    return {
        "chart":f"synthetic {chart_id}",
        "chart_sha256":f"{chart_id:064x}",
        "event_count":n,
        "count":targets,
        "original_events_by_index":events,
        "original_chunks":chunks,
        "reconstructions":cases,
    }


def fixture_source():
    # All charts have the exact historical *counts*, but deliberately
    # synthetic event/axis values. They are never human evidence.
    charts=[]
    target_counts=(10,10,2)
    for j,(n,k) in enumerate(zip((1540,3637,5054),target_counts),1):
        chart=synthetic_chart(n=n,chart_id=j,targets=k)
        for i,case in enumerate(chart["reconstructions"]):
            is_better=(j==1 and i<7)
            val=-0.5 if is_better else 0.2
            case["cost"]["forced_final"]=100.0+val
            case["cost"]["final_delta"]=val
            case["cost"]["six_component_deltas"]["gesture_preference"]=val
        charts.append(chart)
    return {
        "format":H11_FORMAT,
        "source_h06_git_blob":H06_BLOB_SHA,
        "source_v08_frozen_commit":
            "45f30bcaf968281400e2c2913a96dd95515222d2",
        "chart_count":3,
        "target_count":22,
        "strict_lower_heuristic_cost_count":7,
        "complete_event_axis_assignments":True,
        "alternate_motor_executed":False,
        "human_preference_observed":False,
        "motor_quality_verified":False,
        "validation20_or_final_accessed":False,
        "training_performed":False,
        "frozen_planner_motor_timing_changed":False,
        "cohort":{
            "total_events":10231,
            "independent_counterfactual_targets":22},
        "charts":charts,
    }


def test_metrics_detect_rapid_same_hand_and_short_reuse():
    rows=(
        (0,1000.0,("left_1",)),
        (1,1040.0,("left_4",)),
        (2,1080.0,("left_2",)),
        (3,1120.0,("left_3",)),
        (4,1170.0,("left_1",)),
    )
    m=metrics(rows)
    assert m["rapid_single_hand_runs_at_least_four"]==1
    assert m["longest_rapid_single_hand_run"]==5
    assert m["quick_same_hand_pairs_120ms"]==4
    assert m["quick_same_hand_depth_jump_pairs_120ms"]==3
    assert m["max_hand_contacts_in_one_second"]==5
    assert m["max_axis_contacts_in_half_second"]==2
    assert m["axis_reuse_under_80ms"]==0
    assert m["axis_reuse_under_100ms"]==0
    assert m["axis_reuse_under_150ms"]==0
    assert set(PRESSURE_KEYS).issubset(m)


def test_metrics_reuse_flags_and_invalid_axis_fail_closed():
    rows=((0,1000.,("right_2",)),(1,1070.,("right_2",)))
    m=metrics(rows)
    assert m["axis_reuse_under_80ms"]==1
    assert m["axis_reuse_under_100ms"]==1
    assert m["axis_reuse_under_150ms"]==1
    assert m["minimum_axis_reuse_ms"]==70.0
    chart=synthetic_chart()
    chart["original_events_by_index"][0]["original_axes"]=["right_5"]
    with pytest.raises(ValueError,match="invalid frozen"):
        timeline(chart["original_events_by_index"],field="original_axes")


def test_sample_complete_one_candidate_without_motor():
    chart=synthetic_chart()
    original=chart["original_events_by_index"]
    baseline=timeline(original,field="original_axes")
    m=metrics(baseline)
    x=compare_candidate(
        original,chart["original_chunks"],m,chart["reconstructions"][0])
    assert x["changed_event_count"]==1
    assert x["abstract_metrics"]["original"]==m
    assert x["human_review_required"] is False
    assert x["alternate_motor_feasibility"]=="not_tested"
    assert x["human_preference"]=="not_measured"
    assert x["heuristic_objective"][
        "alternative_has_lower_frozen_heuristic_cost"] is True


def test_failed_chunk_identity_and_tampered_changes_detected():
    chart=synthetic_chart()
    original=chart["original_events_by_index"]
    m=metrics(timeline(original,field="original_axes"))
    case=deepcopy(chart["reconstructions"][0])
    case["changed_events"][0]["event_index"]+=1
    with pytest.raises(ValueError,match="changed-event identity"):
        compare_candidate(original,chart["original_chunks"],m,case)
    case=deepcopy(chart["reconstructions"][0])
    case["alternative_chunks"][1]["axes"]=[["right_3"]]
    with pytest.raises(ValueError,match="chunk arrays differ"):
        compare_candidate(original,chart["original_chunks"],m,case)
    case=deepcopy(chart["reconstructions"][0])
    case["cost"]["final_delta"]=100.0
    with pytest.raises(ValueError,match="six-term cost delta"):
        compare_candidate(original,chart["original_chunks"],m,case)


def test_full_22_candidate_census_without_fabricated_quality():
    src=fixture_source()
    r=compile_audit(src,source_sha256="f"*64)
    assert r["format"]==FORMAT
    assert r["chart_count"]==3
    assert r["candidate_count"]==22
    assert r["source_event_count"]==10231
    assert r["original_objective_lower_alternatives"]==7
    assert r["human_reviews_observed"] is False
    assert r["human_reviews_required"] is False
    assert r["motor_executed"] is False
    assert r["human_likeness_verified"] is False
    assert sum(r["triage"][k] for k in ("P1","P2","P3"))==22
    assert len(r["triage"]["all_cases_ordered_by_diagnostic_priority"])==22
    assert len(r["triage"]["optional_first_five_case_ids"])==5
    assert all(x["conclusions_about_physical_comfort"] is False
        for c in r["charts"] for x in c["candidates"])


def test_refuses_fake_motor_or_human_claims():
    for key in ("alternate_motor_executed","human_preference_observed"):
        source=fixture_source()
        source[key]=True
        with pytest.raises(ValueError,match="provenance/claims"):
            compile_audit(source,source_sha256="f"*64)


def test_refuses_missing_candidate_and_cost_census_changes():
    source=fixture_source()
    source["charts"][0]["reconstructions"].pop()
    with pytest.raises(ValueError,match="chart/target census"):
        compile_audit(source,source_sha256="f"*64)
    source=fixture_source()
    source["charts"][0]["reconstructions"][0]["cost"]["final_delta"]=.2
    source["charts"][0]["reconstructions"][0]["cost"]["forced_final"]=100.2
    source["charts"][0]["reconstructions"][0]["cost"][
        "six_component_deltas"]["gesture_preference"]=.2
    with pytest.raises(ValueError,match="seven-improvement"):
        compile_audit(source,source_sha256="f"*64)


def test_report_never_overwrites(tmp_path,monkeypatch):
    source=tmp_path/"h11.json"
    source.write_text(json.dumps(fixture_source()),encoding="utf-8")
    output=tmp_path/"h13.json"
    monkeypatch.setattr(sys,"argv",["h13","--h11-report",str(source),
                                    "--output",str(output)])
    main()
    first=output.read_bytes()
    assert json.loads(first)["candidate_count"]==22
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes()==first
