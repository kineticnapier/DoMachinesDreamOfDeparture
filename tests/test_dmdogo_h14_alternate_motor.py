from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunk, GestureEvent, GesturePlan,
)
from run_dmdogo_h14_alternate_motor import (
    FORMAT, REUSE_S, Prepared, OneShotPlanner,
    _chunk_to_saved, _require_same_rows, _inject_plan,
    preflight_record, execute, main,
)
import run_dmdogo_h14_alternate_motor as h14


def fake_ready():
    events=(
        GestureEvent(0,1.0,1),
        GestureEvent(1,1.2,1),
        GestureEvent(2,1.4,1),
    )
    axes=(("left_1",),("right_1",),("left_1",))
    chunks=tuple(
        GestureChunk(i,i,1,"isolated","test","single",(a,))
        for i,a in enumerate(axes)
    )
    original=GesturePlan(
        True,axes,chunks,None,37,
        {"beam_cost":10.0,"chunk_count":3},
        {"source":"offline","physical_motor_evaluated":False,
         "human_likeness_verified":False})
    alternative=(
        ("left_1",),("right_2",),("left_1",))
    )
    cand={
        "candidate_id":"sample-one",
        "flag_pair":[0,1],
        "join_window_index":1,
        "changed_chunk_indices":[1],
        "changed_event_count":1,
        "changed_events":[{
            "event_index":1,"due_s":1.2,
            "original_axes":["right_1"],"alternative_axes":["right_2"],
        }],
        "alternative_chunks":[
            _chunk_to_saved(chunk) if i != 1 else {
                **_chunk_to_saved(chunk),"axes":[["right_2"]]}
            for i,chunk in enumerate(chunks)
        ],
        "alternative_axes_by_event_order":[list(x) for x in alternative],
        "independent_intervention_only":True,
        "reached_terminal_abstract":True,
        "alternate_motor_result":"not_executed",
        "cost":{
            "original_v08_final":10.0,
            "forced_final":9.5,
            "final_delta":-0.5,
        },
    }
    chart={
        "original_events_by_index":[{
            "event_index":e.event_index,"due_s":e.due_s,
            "chord_size":e.chord_size,
            "original_axes":list(axes[i])}
            for i,e in enumerate(events)],
        "original_chunks":[_chunk_to_saved(x) for x in chunks],
    }
    alt=_inject_plan(original,chart["original_chunks"],cand,events)
    original_stats={
        "hits":3,"targets":3,"misses":0,"too_early":0,
        "x_accuracy_percent":100.0,"timing_mae_ms":5.0,
        "local_replans":0,"deferred_events":0,"overloaded":False,
    }
    ready=Prepared(
        case={"chart":"sample.adofai","sha256":"1"*64},
        chart_path=Path("sample.adofai"),
        h11_sha="a"*64,h13_sha="b"*64,
        manifest_sha="c"*64,verified_sha="d"*64,
        runtime={
            "source":{"code_digest":"e"*64},
            "environment_digest":"z"*64,
        },
        historical={**original_stats,"environment_digest":"z"*64},
        source={},chart=chart,candidate=cand,segment=object(),
        events=events,original_plan=original,alternate_plan=alt,
    )
    return ready


def test_one_shot_alternate_not_original():
    ready=fake_ready()
    assert ready.alternate_plan.assignments[1] == ("right_2",)
    assert ready.original_plan.assignments[1] == ("right_1",)
    assert ready.alternate_plan.metrics["beam_cost"]==9.5
    assert ready.alternate_plan.metrics["ordinary_v08_plan_winner"] is False
    assert ready.alternate_plan.model_contract[
        "research_override_applied"] is True
    assert ready.alternate_plan.candidates_evaluated == 0


def test_source_original_plan_parity_rejects_bad_axis_and_time():
    ready=fake_ready()
    records=deepcopy(ready.chart["original_events_by_index"])
    _require_same_rows(ready.events,records,ready.original_plan)
    records[1]["original_axes"]=["left_4"]
    with pytest.raises(ValueError,match="differs from H1.1"):
        _require_same_rows(ready.events,records,ready.original_plan)
    records=deepcopy(ready.chart["original_events_by_index"])
    records[0]["due_s"] += 0.01
    with pytest.raises(ValueError,match="differs from H1.1"):
        _require_same_rows(ready.events,records,ready.original_plan)


def test_alternate_rejects_broken_reuse_and_corrupt_event_chord():
    ready=fake_ready()
    case=deepcopy(ready.candidate)
    case["alternative_chunks"][1]["axes"]=[["left_1"]]
    case["alternative_axes_by_event_order"][1]=["left_1"]
    case["changed_events"][0]["alternative_axes"]=["left_1"]
    # The left_1 at 1.2 and 1.4 still satisfies 80ms,
    # so force an actual <80ms gap in a second case.
    too_close=(
        GestureEvent(0,1.0,1),
        GestureEvent(1,1.04,1),
        GestureEvent(2,1.4,1),
    )
    case["changed_events"][0]["due_s"]=1.04
    with pytest.raises(ValueError,match="80ms"):
        _inject_plan(
            ready.original_plan,ready.chart["original_chunks"],
            case,too_close)
    case=deepcopy(ready.candidate)
    case["changed_event_count"]=2
    with pytest.raises(ValueError,match="changed-event"):
        _inject_plan(
            ready.original_plan,ready.chart["original_chunks"],
            case,ready.events)
    case=deepcopy(ready.candidate)
    case["alternative_chunks"][1]["start_event_index"]=10
    with pytest.raises(ValueError,match="segmentation"):
        _inject_plan(
            ready.original_plan,ready.chart["original_chunks"],
            case,ready.events)


def test_runner_symbol_is_swapped_once_and_original_restored(monkeypatch):
    ready=fake_ready()
    real_symbol=h14.frozen_runner.GestureChunkPlannerV08
    seen={}

    def controlled_motor(segment, **kwargs):
        assert kwargs["planning_mode"]=="gesture_v08"
        planner=h14.frozen_runner.GestureChunkPlannerV08(
            minimum_reuse_s=REUSE_S)
        alt=planner.plan(ready.events)
        assert alt.assignments==ready.alternate_plan.assignments
        with pytest.raises(ValueError,match="more than once"):
            planner.plan(ready.events)
        seen["used"]=True
        return {
            "stats":{
                "hits":2,"misses":1,"targets":3,
                "overloaded":False,
            },
            "physical":{
                "local_replans":1,
                "deferred_events":0,
                "deferral_frames":0,
                "unassigned_events":0,
            },
            "decisions":[{
                "event_index":i,
                "desired_axes":list(axes),
                "axes":list(axes) if i != 1 else ["right_3"],
                "local_replan":i==1,
                "policy":"gesture_chunk_v08",
            } for i,axes in enumerate(alt.assignments)],
            "keydowns":[{"axis":"left_1","time_ms":1000}],
            "compensation_telemetry":[],
            "termination_reason":"chart_complete",
        }

    monkeypatch.setattr(h14.frozen_runner,"run_pattern_arm",controlled_motor)
    result=execute(ready,wall_timeout_s=120.0)
    assert seen["used"]
    assert result["alternate_motor_executed"] is True
    assert result["run_is_frozen_v08_baseline"] is False
    assert result["planned_axis_adoption_count"] == 2
    assert result["selected_event_count"] == 3
    assert result["physical_local_replan_count"] == 1
    assert result["stats"]["misses"] == 1
    assert result["human_likeness_verified"] is False
    assert h14.frozen_runner.GestureChunkPlannerV08 is real_symbol


def test_runner_detects_wrong_actual_desired_axes(monkeypatch):
    ready=fake_ready()
    def misleading_motor(segment, **kwargs):
        h14.frozen_runner.GestureChunkPlannerV08().plan(ready.events)
        return {
            "decisions":[{
                "event_index":1,"desired_axes":["left_4"],
                "axes":["left_4"],
                "local_replan":False,"policy":"gesture_chunk_v08",
            }],
            "stats":{"hits":2,"misses":1,"targets":3},
            "physical":{"local_replans":0},
            "keydowns":[],"compensation_telemetry":[],
            "termination_reason":"chart_complete",
        }
    monkeypatch.setattr(h14.frozen_runner,"run_pattern_arm",misleading_motor)
    with pytest.raises(ValueError,match="did not reach unchanged Motor"):
        execute(ready,wall_timeout_s=120)


def test_preflight_distinguishes_runtime_mismatch_without_faking_result():
    ready=fake_ready()
    normal=preflight_record(ready)
    assert normal["alternate_motor_executed"] is False
    assert normal["same_runtime_fingerprint_as_original"] is True
    ready.runtime["environment_digest"]="different"
    drift=preflight_record(ready)
    assert drift["same_runtime_fingerprint_as_original"] is False
    assert drift["original_verified_metrics"]["hits"]==3


def test_main_create_once_abort_before_source_read(tmp_path, monkeypatch):
    output=tmp_path/"record.json"
    output.write_text("previous evidence",encoding="utf-8")
    monkeypatch.setattr(sys,"argv",[
        "h14","--manifest",str(tmp_path/"no-manifest"),
        "--verified-report",str(tmp_path/"no-verified"),
        "--h11-report",str(tmp_path/"no-h11"),
        "--h13-report",str(tmp_path/"no-h13"),
        "--results-root",str(tmp_path/"no-results"),
        "--candidate-id","sample-one","--output",str(output),
    ])
    with pytest.raises(SystemExit):
        main()
    assert output.read_text(encoding="utf-8")=="previous evidence"
