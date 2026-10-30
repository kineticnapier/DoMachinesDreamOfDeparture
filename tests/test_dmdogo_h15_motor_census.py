from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

import run_dmdogo_h15_motor_census as h15


def sample_case(tmp_path, *, cid="e142af91b178-181-182"):
    # Synthetic evidence is not real Motor output or human preference.
    out=tmp_path/"episode.json"
    status_path=tmp_path/"episode.status.json"
    hashes={k:ch*64 for k,ch in (
        ("h11","a"),("h13","b"),("manifest","c"),("verified","d"))}
    original={
        "hits":3,"targets":3,"misses":0,"too_early":0,
        "x_accuracy_percent":100.0,"timing_mae_ms":4.5,
        "local_replans":0,"deferred_events":0,"overloaded":False}
    evidence={cid:{
        "chart":"synthetic_chart","sha":"e"*64,
        "events":3,"changed":1,"cost_delta":-.5,
        "lower_cost":True,"triage":"P1_proxy_tradeoff",
        "original_metrics":original,
        "desired":{0:("left_1",),1:("right_2",),2:("left_2",)}}}
    result={
        "format":h15.H14_FORMAT,"stage":"executed",
        "candidate_id":cid,"chart":"synthetic_chart",
        "chart_sha256":"e"*64,"chart_event_count":3,
        "changed_event_count":1,
        "h11_source_sha256":hashes["h11"],
        "h13_source_sha256":hashes["h13"],
        "manifest_sha256":hashes["manifest"],
        "verified_report_sha256":hashes["verified"],
        "frozen_code_digest":"f"*64,
        "forced_heuristic_cost_delta":-.5,
        "run_is_frozen_v08_baseline":False,
        "alternate_motor_executed":True,
        "physical_motor_source":"unchanged_v08_runner_and_live_motor",
        "policy_source":"h11_reconstructed_single_seam_forced_suffix",
        "human_likeness_verified":False,
        "human_preference_measured":False,
        "validation20_or_final_accessed":False,
        "original_verified_metrics":original,
        "stats":{
            "targets":3,"hits":3,"misses":0,"too_early":0,
            "physical_keydowns":3,"overloaded":False,
            "x_accuracy_percent":100.0,"timing_mae_ms":4.7},
        "physical":{
            "local_replans":0,"deferred_events":0,
            "deferral_frames":0,"unassigned_events":0},
        "decisions":[{
            "event_index":i,"desired_axes":[axis],"axes":[axis],
            "local_replan":False,"policy":"gesture_chunk_v08"}
            for i,axis in enumerate(("left_1","right_2","left_2"))],
        "keydowns":[{"axis":"left_1"},{"axis":"right_2"},{"axis":"left_2"}],
        "compensation":[],"termination_reason":"chart_complete",
        "selected_event_count":3,"planned_axis_adoption_count":3,
        "physical_local_replan_count":0,
        "same_runtime_fingerprint_as_original":True,
    }
    status={
        "format":h15.H14_STATUS,"candidate_id":cid,
        "state":"success","exit_code":0,
        "physical_motor_requested":True,
        "output_file":str(out),
        "output_sha256":None,
    }
    return out,status_path,result,status,hashes,evidence


def stored(out,status_path,result,status):
    out.write_text(json.dumps(result),encoding="utf-8")
    status["output_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
    status_path.write_text(json.dumps(status),encoding="utf-8")


def validate_bundle(tmp_path,edit=None):
    out,sp,r,s,hashes,evidence=sample_case(tmp_path)
    if edit:edit(r,s)
    stored(out,sp,r,s)
    return h15.validate_result(
        r["candidate_id"],r,s,result_path=out,status_path=sp,
        pinned_digest=None,hashes=hashes,
        frozen_digest="f"*64,evidence=evidence)


def test_full_episode_all_hit_exact_plan(tmp_path):
    row=validate_bundle(tmp_path)
    assert row["status"]=="verified_episode"
    assert row["hits"]==3 and row["misses"]==0
    assert row["exact_plan_adopted"] is True
    assert row["all_hit_and_exact_plan"] is True
    assert row["timing_mae_delta_ms"]==.2
    assert row["human_preference_observed"] is False


def test_valid_physical_miss_is_counted_not_deleted(tmp_path):
    def change(result,status):
        result["stats"]["hits"]=2
        result["stats"]["misses"]=1
    row=validate_bundle(tmp_path,change)
    assert row["status"]=="verified_episode"
    assert row["all_hit"] is False
    assert row["all_hit_and_exact_plan"] is False


def test_local_replan_is_not_exact_plan(tmp_path):
    def change(result,status):
        result["decisions"][1]["axes"]=["right_3"]
        result["decisions"][1]["local_replan"]=True
        result["physical"]["local_replans"]=1
        result["physical_local_replan_count"]=1
        result["planned_axis_adoption_count"]=2
    row=validate_bundle(tmp_path,change)
    assert row["hits"]==3
    assert row["exact_plan_adopted"] is False
    assert row["all_hit_and_exact_plan"] is False


@pytest.mark.parametrize("edit,pattern",[
    (lambda r,s:r.update({"h11_source_sha256":"0"*64}),"provenance"),
    (lambda r,s:r["decisions"][1].update({"desired_axes":["left_4"]}),"desired-axis"),
    (lambda r,s:r["stats"].update({"hits":4}),"scorer"),
    (lambda r,s:r.update({"alternate_motor_executed":False}),"provenance"),
    (lambda r,s:r["physical"].update({"local_replans":2}),"selection counters"),
    (lambda r,s:r["decisions"][2].update({"event_index":1}),"desired-axis"),
])
def test_tampering_is_never_a_success(tmp_path,edit,pattern):
    with pytest.raises(ValueError,match=pattern):
        validate_bundle(tmp_path,edit)


def test_status_digest_mismatch_is_rejected(tmp_path):
    out,sp,r,s,hashes,evidence=sample_case(tmp_path)
    stored(out,sp,r,s)
    s["output_sha256"]="0"*64
    sp.write_text(json.dumps(s),encoding="utf-8")
    with pytest.raises(ValueError,match="status/result digest"):
        h15.validate_result(
            r["candidate_id"],r,s,result_path=out,status_path=sp,
            pinned_digest=None,hashes=hashes,
            frozen_digest="f"*64,evidence=evidence)


def test_historical_pilot_pin_mismatch_is_rejected(tmp_path):
    out,sp,r,s,hashes,evidence=sample_case(tmp_path)
    stored(out,sp,r,s)
    with pytest.raises(ValueError,match="pilot digest changed"):
        h15.validate_result(
            r["candidate_id"],r,s,result_path=out,status_path=sp,
            pinned_digest="0"*64,hashes=hashes,
            frozen_digest="f"*64,evidence=evidence)


def test_missing_and_failed_run_not_relaunched(tmp_path):
    cid="e142af91b178-181-182"
    out,sp,r,s,hashes,evidence=sample_case(tmp_path,cid=cid)
    empty=h15.inspect(
        cid,repo=tmp_path,result_root=tmp_path,
        status_root=tmp_path,hashes=hashes,frozen_digest="f"*64,
        evidence=evidence)
    assert empty["status"]=="pending"
    out=tmp_path/f"{cid}.json"
    out.write_text("stale output",encoding="utf-8")
    with pytest.raises(ValueError,match="incomplete result/status pair"):
        h15.inspect(cid,repo=tmp_path,result_root=tmp_path,
                    status_root=tmp_path,hashes=hashes,
                    frozen_digest="f"*64,evidence=evidence)


def test_partial_and_complete_22_report_have_distinct_flags():
    hashes={key:key*64 for key in ("h11","h13","manifest","verified")}
    evidence={}
    rows=[]
    for i in range(22):
        cid=f"f00abc001234-{i}-{i+1}"
        evidence[cid]={
            "sha":f"{1+i//10:064x}",
            "events":100,"changed":1,
            "cost_delta":-.5,"lower_cost":i<7,
        }
        rows.append({"candidate_id":cid,"status":"pending",
                     "reused_h14_pilot":False})
    a=h15.summary(rows,hashes=hashes,frozen_digest="f"*64,
                  evidence=evidence)
    assert a["full_census_verified"] is False
    assert a["pending_episode_count"]==22
    complete=[]
    for i,row in enumerate(rows):
        complete.append({
            **row,"status":"verified_episode",
            "reused_h14_pilot":i<2,
            "all_hit_and_exact_plan":i!=5,
            "same_recorded_runtime_fingerprint_as_original":True,
            "chart_sha256":evidence[row["candidate_id"]]["sha"]})
    b=h15.summary(complete,hashes=hashes,
                  frozen_digest="f"*64,evidence=evidence)
    assert b["full_census_verified"] is True
    assert b["verified_episode_count"]==22
    assert b["reused_prior_h14_pilots"]==2
    assert b["all_hit_exact_plan_count"]==21
    assert b["verified_but_not_all_hit_exact_count"]==1
    assert b["distinct_train_source_events"]==10231


def test_execute_one_never_overwrites_prior_artifact(tmp_path):
    cid="e142af91b178-181-182"
    out=tmp_path/"old.json"
    out.write_text("do not overwrite",encoding="utf-8")
    with pytest.raises(ValueError,match="refusing to overwrite"):
        h15.execute_one(
            None,cid,repo=tmp_path,out=out,
            status=tmp_path/"unused.json")
    assert out.read_text(encoding="utf-8")=="do not overwrite"


def test_run_mode_keeps_runner_one_candidate_and_explicit_execute(
        tmp_path,monkeypatch):
    repo=tmp_path
    runner=repo/"scripts"/"run_dmdogo_h14_alternate_motor.py"
    runner.parent.mkdir()
    runner.write_text("pass",encoding="utf-8")
    args=SimpleNamespace(
        manifest=tmp_path/"m.json",verified_report=tmp_path/"v.json",
        h11_report=tmp_path/"h11.json",h13_report=tmp_path/"h13.json",
        results_root=tmp_path/"source",wall_timeout_s=1200.)
    invoked={}
    def subprocess_spy(command,**kwargs):
        invoked["command"]=command
        invoked["kwargs"]=kwargs
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(h15.subprocess,"run",subprocess_spy)
    cid="e142af91b178-181-182"
    assert h15.execute_one(
        args,cid,repo=repo,out=tmp_path/"new"/"output.json",
        status=tmp_path/"logs"/"new.json")==0
    assert "--execute" in invoked["command"]
    assert invoked["command"].count("--candidate-id")==1
    assert cid in invoked["command"]
    assert "--stop-after-s" not in invoked["command"]
    assert invoked["kwargs"]["cwd"]==repo
    assert invoked["kwargs"]["timeout"]==1290.
