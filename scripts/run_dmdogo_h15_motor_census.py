from __future__ import annotations

"""H1.5: resume 22 independent Train-only alternative physical Motor episodes.

The two previously completed H1.4 pilots are reused only with pinned SHA256
and independently checked H1.4 status/telemetry. Remaining runs delegate
ONE case at a time to the unmodified H1.4 adapter and frozen Motor.
No blind manual grading, no v0.8 source edits, no Validation20/FINAL.
"""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from audit_dmdogo_h13_automated_fingering_quality import (
    FORMAT as H13_FORMAT, compile_audit,
)
from reconstruct_dmdogo_h11_fingering_pairs import FORMAT as H11_FORMAT
from run_dmdogo_h14_alternate_motor import (
    FORMAT as H14_FORMAT, STATUS_FORMAT as H14_STATUS,
)
from dmdogo_v08_provenance import FROZEN_HEAD, preflight, sha256_file

FORMAT = "dmdogo-h15-motor-22-census-v1"
PILOTS = {
    "e142af91b178-1089-1090": (
        "artifacts/dmdogo-h14/pilot-1089-r1.json",
        "logs/dmdogo-h14-pilot-1089-r1.status.json",
        "5cb3cb69b791d2da6fd630342575901a04cc37fa815f860f51da0542153e2e57"),
    "2ada1d0a06b5-1237-1238": (
        "artifacts/dmdogo-h14/pilot-1237-r1.json",
        "logs/dmdogo-h14-pilot-1237-r1.status.json",
        "0bf2d8bd91c83a5d79cc1be245455db993953c5409a445870ec6166f41c0c244"),
}
ID_PATTERN = re.compile(r"[a-f0-9]{12}-[0-9]+-[0-9]+\Z")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def nonnegative(value, *, integer=True):
    return (type(value) is int and value >= 0) if integer else (
        type(value) in (int, float) and math.isfinite(value) and value >= 0)


def load_context(args):
    h11, h13 = load(args.h11_report), load(args.h13_report)
    manifest, verified = load(args.manifest), load(args.verified_report)
    hashes = {
        "h11": sha256_file(args.h11_report),
        "h13": sha256_file(args.h13_report),
        "manifest": sha256_file(args.manifest),
        "verified": sha256_file(args.verified_report),
    }
    if (h11.get("format") != H11_FORMAT
            or h13.get("format") != H13_FORMAT
            or h11.get("source_manifest_sha256") != hashes["manifest"]
            or h13.get("source_h11_sha256") != hashes["h11"]
            or h11.get("source_v08_frozen_commit") != FROZEN_HEAD
            or h11.get("chart_count") != 3 or h11.get("target_count") != 22
            or h11.get("strict_lower_heuristic_cost_count") != 7
            or h11.get("complete_event_axis_assignments") is not True
            or h11.get("alternate_motor_executed") is not False
            or h11.get("human_preference_observed") is not False
            or h13.get("candidate_count") != 22
            or h13.get("original_objective_lower_alternatives") != 7):
        raise ValueError("H1.1/H1.3 input provenance differs")
    if compile_audit(h11, source_sha256=hashes["h11"]) != h13:
        raise ValueError("H1.3 report does not recompute from H1.1")
    if (verified.get("format") != "dmdogo-v08-train-generalization-report-v2"
            or verified.get("frozen_implementation_commit") != FROZEN_HEAD
            or manifest.get("frozen_implementation_commit") != FROZEN_HEAD
            or verified.get("frozen_code_digest") != manifest.get("frozen_code_digest")
            or verified.get("all_cases_completed") is not True
            or verified.get("completed_count") != 3):
        raise ValueError("original three Train baseline provenance differs")
    actual = preflight(Path(__file__).resolve().parents[1])
    if actual["source"]["code_digest"] != manifest["frozen_code_digest"]:
        raise ValueError("frozen source tree is modified")
    charts = h11["charts"]
    if (tuple(x["event_count"] for x in charts) != (1540, 3637, 5054)
            or tuple(len(x["reconstructions"]) for x in charts) != (10, 10, 2)):
        raise ValueError("original 3-Train, 22-candidate census changed")
    m = {x["sha256"]: x for x in manifest["cases"]}
    v = {x["chart_sha256"]: x for x in verified["records"]}
    q = {x["candidate_id"]: x for chart in h13["charts"]
         for x in chart["candidates"]}
    if (len(m) != 3 or len(v) != 3 or len(q) != 22
            or any(x["status"] != "chart_complete" for x in v.values())):
        raise ValueError("historical Train selection/cases differ")
    evidence = {}
    for chart in charts:
        sha = chart["chart_sha256"]
        if (sha not in m or sha not in v
                or m[sha]["chart"] != chart["chart"]
                or m[sha]["event_count"] != chart["event_count"]):
            raise ValueError("H1.1 chart differs from saved Train selection")
        original_ids = [r["event_index"] for r in chart["original_events_by_index"]]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("duplicated source event IDs")
        for cand in chart["reconstructions"]:
            cid = cand["candidate_id"]
            if (not ID_PATTERN.fullmatch(cid) or not cid.startswith(sha[:12]+"-")
                    or cid in evidence or cid not in q):
                raise ValueError("candidate identity duplicated or not from Train")
            score = float(cand["cost"]["final_delta"])
            if (cand["changed_event_count"] != q[cid]["changed_event_count"]
                    or abs(score-q[cid]["heuristic_objective"][
                        "alternative_minus_original"]) > 2e-6
                    or len(cand["alternative_axes_by_event_order"]) != len(original_ids)):
                raise ValueError("H1.1/H1.3 candidate differs")
            evidence[cid] = {
                "chart": chart["chart"], "sha": sha,
                "events": chart["event_count"],
                "changed": cand["changed_event_count"],
                "cost_delta": score, "lower_cost": score < -1e-7,
                "triage": q[cid]["triage_tier"],
                "original_metrics": v[sha],
                "desired": {i: tuple(a) for i,a in zip(
                    original_ids,cand["alternative_axes_by_event_order"],strict=True)},
            }
    order = tuple(x["candidate_id"] for x in h13["triage"][
        "all_cases_ordered_by_diagnostic_priority"])
    if (len(order) != 22 or len(set(order)) != 22
            or set(order) != set(evidence) or not set(PILOTS) <= set(evidence)):
        raise ValueError("H1.3 triage/22 candidates differ")
    return hashes, manifest["frozen_code_digest"], evidence, order


def candidate_paths(cid, *, repo, result_root, status_root):
    if cid in PILOTS:
        a,b,digest = PILOTS[cid]
        return repo/a, repo/b, digest
    return (result_root/f"{cid}.json",
            status_root/f"{cid}.status.json", None)


def validate_result(cid, result, status, *, result_path, status_path,
                    pinned_digest, hashes, frozen_digest, evidence):
    expected = evidence[cid]
    digest = sha256_file(result_path)
    if pinned_digest is not None and digest != pinned_digest:
        raise ValueError(f"{cid}: previously observed H1.4 pilot digest changed")
    if (status.get("format") != H14_STATUS
            or status.get("candidate_id") != cid
            or status.get("state") != "success" or status.get("exit_code") != 0
            or status.get("physical_motor_requested") is not True
            or status.get("output_sha256") != digest
            or not isinstance(status.get("output_file"),str)
            or Path(status["output_file"]).resolve() != result_path.resolve()):
        raise ValueError(f"{cid}: H1.4 status/result digest mismatch")
    if (result.get("format") != H14_FORMAT
            or result.get("stage") != "executed"
            or result.get("candidate_id") != cid
            or result.get("chart") != expected["chart"]
            or result.get("chart_sha256") != expected["sha"]
            or result.get("chart_event_count") != expected["events"]
            or result.get("changed_event_count") != expected["changed"]
            or result.get("h11_source_sha256") != hashes["h11"]
            or result.get("h13_source_sha256") != hashes["h13"]
            or result.get("manifest_sha256") != hashes["manifest"]
            or result.get("verified_report_sha256") != hashes["verified"]
            or result.get("frozen_code_digest") != frozen_digest
            or result.get("run_is_frozen_v08_baseline") is not False
            or result.get("alternate_motor_executed") is not True
            or result.get("physical_motor_source") != "unchanged_v08_runner_and_live_motor"
            or result.get("policy_source") != "h11_reconstructed_single_seam_forced_suffix"
            or result.get("human_likeness_verified") is not False
            or result.get("human_preference_measured") is not False
            or result.get("validation20_or_final_accessed") is not False):
        raise ValueError(f"{cid}: H1.4 execution provenance differs")
    if (type(result.get("forced_heuristic_cost_delta")) not in (int,float)
            or not math.isfinite(result["forced_heuristic_cost_delta"])
            or abs(result["forced_heuristic_cost_delta"]-expected["cost_delta"]) > 2e-6):
        raise ValueError(f"{cid}: H1.1 alternative objective differs")
    original = result.get("original_verified_metrics")
    for key in ("hits","targets","misses","too_early","x_accuracy_percent",
                "timing_mae_ms","local_replans","deferred_events","overloaded"):
        if not isinstance(original,dict) or original.get(key)!=expected[
                "original_metrics"][key]:
            raise ValueError(f"{cid}: verified original baseline was substituted")
    stats, physical = result.get("stats"), result.get("physical")
    decisions, keydowns = result.get("decisions"), result.get("keydowns")
    if (not isinstance(stats,dict) or not isinstance(physical,dict)
            or not isinstance(decisions,list) or not isinstance(keydowns,list)
            or not isinstance(result.get("compensation"),list)):
        raise ValueError(f"{cid}: missing complete physical telemetry")
    if any(not nonnegative(stats.get(k)) for k in (
            "targets","hits","misses","too_early","physical_keydowns")):
        raise ValueError(f"{cid}: malformed scorer totals")
    if (stats["targets"] != expected["events"]
            or stats["hits"]+stats["misses"] != stats["targets"]
            or stats["physical_keydowns"] != len(keydowns)
            or type(stats.get("overloaded")) is not bool
            or not nonnegative(stats.get("x_accuracy_percent"),integer=False)
            or stats["x_accuracy_percent"] > 100
            or not nonnegative(stats.get("timing_mae_ms"),integer=False)):
        raise ValueError(f"{cid}: physical scorer/keydown totals differ")
    if any(not nonnegative(physical.get(k)) for k in (
            "local_replans","deferred_events","deferral_frames","unassigned_events")):
        raise ValueError(f"{cid}: malformed physical counters")
    if (len(decisions) != result.get("selected_event_count")
            or len(decisions)+physical["unassigned_events"] != expected["events"]
            or result.get("physical_local_replan_count") != physical["local_replans"]
            or not nonnegative(result.get("planned_axis_adoption_count"))):
        raise ValueError(f"{cid}: physical selection counters differ")
    seen=set()
    adoption=0
    replans=0
    for row in decisions:
        i=row.get("event_index")
        if (type(i) is not int or i in seen or i not in expected["desired"]
                or tuple(row.get("desired_axes",())) != expected["desired"][i]
                or not isinstance(row.get("axes"),list)
                or row.get("policy")!="gesture_chunk_v08"
                or type(row.get("local_replan")) is not bool):
            raise ValueError(f"{cid}: actual desired-axis trace is not H1.1")
        seen.add(i)
        equal = row["axes"] == row["desired_axes"]
        adoption += equal
        replans += row["local_replan"]
        if equal == row["local_replan"]:
            raise ValueError(f"{cid}: local replan/axis adoption inconsistency")
    if (adoption != result["planned_axis_adoption_count"]
            or replans != physical["local_replans"]):
        raise ValueError(f"{cid}: adoption/replan aggregate differs")
    finish=result.get("termination_reason")
    if finish not in ("chart_complete","miss","overload"):
        raise ValueError(f"{cid}: unexpected termination mode")
    same=result.get("same_runtime_fingerprint_as_original")
    if type(same) is not bool:
        raise ValueError(f"{cid}: missing runtime fingerprint comparison")
    all_hit=(finish=="chart_complete" and stats["hits"]==expected["events"]
             and stats["misses"]==0 and stats["too_early"]==0
             and stats["overloaded"] is False)
    full_adoption=(len(decisions)==expected["events"]
                   and physical["unassigned_events"]==0
                   and adoption==len(decisions) and replans==0)
    return {
        "candidate_id":cid, "chart":expected["chart"],
        "chart_sha256":expected["sha"],
        "event_count":expected["events"],
        "changed_events":expected["changed"],
        "triage_tier":expected["triage"],
        "lower_heuristic_cost":expected["lower_cost"],
        "heuristic_cost_delta":expected["cost_delta"],
        "status":"verified_episode",
        "reused_h14_pilot":cid in PILOTS,
        "status_path":str(status_path),
        "status_sha256":sha256_file(status_path),
        "result_path":str(result_path),
        "result_sha256":digest,
        "same_recorded_runtime_fingerprint_as_original":same,
        "termination_reason":finish,
        "hits":stats["hits"],"misses":stats["misses"],
        "too_early":stats["too_early"],"overloaded":stats["overloaded"],
        "x_accuracy_percent":stats["x_accuracy_percent"],
        "timing_mae_ms":stats["timing_mae_ms"],
        "original_hits":original["hits"],
        "original_timing_mae_ms":original["timing_mae_ms"],
        "timing_mae_delta_ms":(
            round(stats["timing_mae_ms"]-original["timing_mae_ms"],6)
            if same else None),
        "selected_events":len(decisions),
        "planned_axis_adoption_count":adoption,
        "local_replans":replans,
        "deferred_events":physical["deferred_events"],
        "deferral_frames":physical["deferral_frames"],
        "episode_completed":finish=="chart_complete",
        "all_hit":all_hit,"exact_plan_adopted":full_adoption,
        "all_hit_and_exact_plan":all_hit and full_adoption,
        "human_preference_observed":False,
    }


def inspect(cid, *, repo, result_root, status_root,
            hashes, frozen_digest, evidence):
    path,status_path,pinned = candidate_paths(
        cid,repo=repo,result_root=result_root,status_root=status_root)
    if not path.exists() and not status_path.exists():
        return {"candidate_id":cid,"status":"pending",
                "reused_h14_pilot":cid in PILOTS}
    if not path.is_file() or not status_path.is_file():
        raise ValueError(f"{cid}: incomplete result/status pair; never auto-retry")
    status=load(status_path)
    if status.get("state")!="success":
        return {"candidate_id":cid,"status":"incomplete_run",
                "run_state":status.get("state"),
                "exit_code":status.get("exit_code"),
                "reused_h14_pilot":cid in PILOTS}
    return validate_result(
        cid,load(path),status,result_path=path,status_path=status_path,
        pinned_digest=pinned,hashes=hashes,frozen_digest=frozen_digest,
        evidence=evidence)


def summary(rows, *, hashes, frozen_digest, evidence):
    if len(rows)!=22:
        raise ValueError("not a 22-case audit")
    verified=[r for r in rows if r["status"]=="verified_episode"]
    pending=[r for r in rows if r["status"]=="pending"]
    incomplete=[r for r in rows if r["status"]=="incomplete_run"]
    passed=[r for r in verified if r["all_hit_and_exact_plan"]]
    by_chart=[]
    for sha in dict.fromkeys(evidence[k]["sha"] for k in evidence):
        total=sum(x["sha"]==sha for x in evidence.values())
        sub=[r for r in verified if r["chart_sha256"]==sha]
        by_chart.append({
            "chart_sha256":sha,"candidate_count":total,
            "verified_episode_count":len(sub),
            "all_hit_exact_plan_count":sum(x["all_hit_and_exact_plan"] for x in sub)})
    return {
        "format":FORMAT,"source_digests":hashes,
        "frozen_code_digest":frozen_digest,
        "distinct_train_charts":3,"distinct_train_source_events":10231,
        "independent_counterfactual_runs":22,
        "historically_lower_heuristic_cost_count":7,
        "reused_prior_h14_pilots":sum(x["reused_h14_pilot"] for x in verified),
        "verified_episode_count":len(verified),
        "pending_episode_count":len(pending),
        "incomplete_episode_count":len(incomplete),
        "full_census_verified":len(verified)==22,
        "all_hit_exact_plan_count":len(passed),
        "verified_but_not_all_hit_exact_count":len(verified)-len(passed),
        "matching_runtime_fingerprint_count":sum(
            x["same_recorded_runtime_fingerprint_as_original"]
            for x in verified),
        "charts":by_chart,"records":rows,
        "human_likeness_verified":False,
        "human_preference_observed":False,
        "hud_only_student_verified":False,
        "validation20_or_final_accessed":False,
        "frozen_planner_motor_timing_changed":False,
        "limitations":[
            "22 independent alternative episodes are NOT 22 independent charts.",
            "All data are selected Train charts and privileged exact Oracle timing.",
            "An all-hit episode with fallback/replanned axes is NOT proof of the proposed fingering.",
            "Simulated Motor feasibility does not establish human skill, preference or fatigue.",
            "Missing/failed/overloaded/missed episodes must not be silently excluded.",
            "An equal recorded runtime fingerprint does not ensure identical wall timing.",
            "Partial census progress is not a completed 22/22 Motor validation.",
        ],
    }


def execute_one(args, cid, *, repo, out, status):
    if out.exists() or status.exists():
        raise ValueError(f"{cid}: refusing to overwrite prior evidence")
    out.parent.mkdir(parents=True,exist_ok=True)
    status.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(repo/"scripts"/"run_dmdogo_h14_alternate_motor.py")]
    for option,path in (
        ("manifest",args.manifest),("verified-report",args.verified_report),
        ("h11-report",args.h11_report),("h13-report",args.h13_report),
        ("results-root",args.results_root)):
        command.extend(["--"+option,str(path.resolve())])
    command.extend([
        "--candidate-id",cid,"--output",str(out.resolve()),
        "--status-file",str(status.resolve()),
        "--wall-timeout-s",str(args.wall_timeout_s),"--execute"])
    # This launches the existing H1.4 isolated worker, not a new Motor.
    try:
        done=subprocess.run(command,cwd=repo,check=False,
                            timeout=args.wall_timeout_s+90.0)
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"{cid}: wrapper timeout; preserve partial files; no automatic retry") from err
    return done.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("manifest","verified-report","h11-report","h13-report",
                  "results-root"):
        p.add_argument("--"+field,required=True,type=Path)
    p.add_argument("--run-root",type=Path,
                   default=Path("artifacts/dmdogo-h15/motor-r1"))
    p.add_argument("--status-root",type=Path,
                   default=Path("logs/dmdogo-h15/motor-r1"))
    p.add_argument("--execute",action="store_true",
                   help="explicitly execute up to --max-new pending runs")
    p.add_argument("--max-new",type=int,default=20)
    p.add_argument("--wall-timeout-s",type=float,default=1200)
    p.add_argument("--complete-report",type=Path)
    args=p.parse_args()
    if not 1<=args.max_new<=20:
        p.error("--max-new must be 1..20")
    if not math.isfinite(args.wall_timeout_s) or not 30<=args.wall_timeout_s<=7200:
        p.error("--wall-timeout-s must be 30..7200")
    if args.complete_report is not None and args.complete_report.exists():
        p.error("refusing to overwrite a complete H1.5 report")
    repo=Path(__file__).resolve().parents[1]
    hashes,frozen_digest,evidence,order=load_context(args)
    def scan():
        return [inspect(cid,repo=repo,result_root=args.run_root,
                        status_root=args.status_root,hashes=hashes,
                        frozen_digest=frozen_digest,evidence=evidence)
                for cid in order]
    rows=scan()
    pending=[x["candidate_id"] for x in rows if x["status"]=="pending"]
    failed=[x for x in rows if x["status"]=="incomplete_run"]
    if set(PILOTS)&set(pending):
        p.error("both original H1.4 pilots are required; no fabricated reuse")
    if args.execute and failed:
        p.error("earlier failed/incomplete episodes require investigation; no automatic rerun")
    started=[]
    if args.execute:
        for cid in pending[:args.max_new]:
            out,status,_=candidate_paths(
                cid,repo=repo,result_root=args.run_root,
                status_root=args.status_root)
            print(json.dumps({
                "launch":cid,"index":len(started)+1,"max_new":args.max_new,
                "chart":evidence[cid]["chart"],
                "changed_events":evidence[cid]["changed"]},
                ensure_ascii=False),flush=True)
            exit_code=execute_one(args,cid,repo=repo,out=out,status=status)
            started.append({"candidate_id":cid,"exit_code":exit_code})
            if exit_code!=0:
                print(f"H1.5 stopped at failed case {cid} (exit={exit_code})",
                      flush=True)
                break
        rows=scan()
    report=summary(rows,hashes=hashes,frozen_digest=frozen_digest,
                   evidence=evidence)
    print(json.dumps({
        "total_candidates":22,
        "reused_prior_h14_pilots":report["reused_prior_h14_pilots"],
        "verified_episodes":report["verified_episode_count"],
        "pending_episodes":report["pending_episode_count"],
        "incomplete_episodes":report["incomplete_episode_count"],
        "all_hit_exact_plan_so_far":report["all_hit_exact_plan_count"],
        "full_census_verified":report["full_census_verified"],
        "launched_this_call":started,
        "remaining":[x["candidate_id"] for x in rows if x["status"]=="pending"],
        "human_likeness_verified":False,
    },ensure_ascii=False,indent=2),flush=True)
    if args.complete_report:
        if not report["full_census_verified"]:
            p.error("cannot write final report: fewer than 22 verified episodes")
        args.complete_report.parent.mkdir(parents=True,exist_ok=True)
        with args.complete_report.open("x",encoding="utf-8") as dest:
            json.dump(report,dest,ensure_ascii=False,indent=2)
            dest.write("\n")
        print("saved verified 22-case census: "+str(args.complete_report))
    if failed or any(item["exit_code"]!=0 for item in started):
        raise SystemExit(1)


if __name__=="__main__":
    main()
