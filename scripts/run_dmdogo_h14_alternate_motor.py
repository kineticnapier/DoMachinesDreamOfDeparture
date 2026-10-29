from __future__ import annotations

"""DMDOGO H1.4: strict preflight and isolated alternative physical Motor pilot.

The original v0.8 source/Runner/Motor/Timing is not edited. The isolated
child process *temporarily* replaces the imported Planner symbol in the
frozen runner with an attested ONE-SHOT full-chart H1.1 plan. The actual
Motor/eligibility/fallback and compensated step loop remain the unchanged
runner's implementation. Outputs use a NEW research format, never v0.8.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunk, GestureChunkPlannerV08, GestureEvent, GesturePlan,
)
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher
from audit_dmdogo_h13_automated_fingering_quality import (
    FORMAT as H13_FORMAT, compile_audit,
)
from reconstruct_dmdogo_h11_fingering_pairs import FORMAT as H11_FORMAT
from dmdogo_v08_provenance import FROZEN_HEAD, preflight, sha256_file
from report_dmdogo_v08_train_generalization import _result_row
from run_dmdogo_v08_verified_train_case import (
    case_from_manifest, validate_chart, validate_provenance_against_manifest,
)
import run_dmdogo_v06_pattern_first as frozen_runner

FORMAT = "dmdogo-h14-alternate-motor-pilot-v1"
STATUS_FORMAT = "dmdogo-h14-isolated-run-status-v1"
REUSE_S = .080
EPS = 1e-9


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj,handle,ensure_ascii=False,indent=2)
        handle.write("\n")


def _axis_rows(events, assignments):
    return [
        {
            "event_index": e.event_index,
            "due_s": e.due_s,
            "chord_size": e.chord_size,
            "axes": list(assignment),
        } for e, assignment in zip(events, assignments, strict=True)
    ]


def _require_same_rows(events, original: list[dict], original_plan: GesturePlan
                       ) -> None:
    if (not original_plan.feasible or len(events) != len(original)
            or len(original_plan.assignments) != len(events)):
        raise ValueError("original frozen plan/Oracle event coverage differs")
    for e, saved, axes in zip(
            events, original, original_plan.assignments, strict=True):
        if (e.event_index != saved["event_index"]
                or e.chord_size != saved["chord_size"]
                or abs(e.due_s - saved["due_s"]) > EPS
                or tuple(saved["original_axes"]) != axes):
            raise ValueError("original frozen plan differs from H1.1 event source")


def _chunk_to_saved(chunk: GestureChunk) -> dict:
    return {
        "start_event_index": chunk.start_event_index,
        "end_event_index": chunk.end_event_index,
        "event_count": chunk.event_count,
        "rhythm_kind": chunk.rhythm_kind,
        "start_reason": chunk.start_reason,
        "gesture_family": chunk.gesture_family,
        "axes": [list(row) for row in chunk.axes],
    }


def _inject_plan(original_plan: GesturePlan, original_chunks: list[dict],
                 candidate: dict, events) -> GesturePlan:
    if [_chunk_to_saved(x) for x in original_plan.chunks] != original_chunks:
        raise ValueError("original H1.1 chunk records differ from frozen v0.8 planner")
    alternative = candidate["alternative_chunks"]
    raw_assignments = candidate["alternative_axes_by_event_order"]
    if (len(alternative) != len(original_plan.chunks)
            or len(raw_assignments) != len(events)
            or candidate["reached_terminal_abstract"] is not True
            or candidate["independent_intervention_only"] is not True
            or candidate["alternate_motor_result"] != "not_executed"):
        raise ValueError("candidate is not a complete, independent abstract path")
    new_chunks = []
    new_assignments = []
    changes = []
    for original_chunk, row in zip(original_plan.chunks, alternative, strict=True):
        if any(row[field] != getattr(original_chunk, field)
               for field in ("start_event_index", "end_event_index",
                             "event_count", "rhythm_kind", "start_reason")):
            raise ValueError("alternate changed frozen chart segmentation")
        alt_axes = tuple(tuple(x) for x in row["axes"])
        if (len(alt_axes) != original_chunk.event_count
                or any(len(axes) < 1 or len(set(axes)) != len(axes)
                       for axes in alt_axes)
                or any(x not in frozen_runner.PatternMotorMapping.default_eight().physical_axes
                       for axes in alt_axes for x in axes)):
            raise ValueError("invalid H1.1 candidate motor axes")
        new_chunks.append(GestureChunk(
            original_chunk.start_event_index, original_chunk.end_event_index,
            original_chunk.event_count, original_chunk.rhythm_kind,
            original_chunk.start_reason, row["gesture_family"], alt_axes))
        new_assignments.extend(alt_axes)
    old_assignments = original_plan.assignments
    if (tuple(new_assignments) != tuple(tuple(x) for x in raw_assignments)
            or len(new_assignments) != len(old_assignments)):
        raise ValueError("new chunk/assignment vectors disagree")
    for e, old, new in zip(events, old_assignments, new_assignments, strict=True):
        if len(new) != e.chord_size:
            raise ValueError("alternate chord assignment mismatch")
        if old != new:
            changes.append({
                "event_index": e.event_index, "due_s": e.due_s,
                "original_axes": list(old), "alternative_axes": list(new)})
    if (changes != candidate["changed_events"]
            or len(changes) != candidate["changed_event_count"]
            or len(changes) == 0):
        raise ValueError("candidate changed-event list differs from H1.1")
    different_chunks = [
        idx for idx, (a,b) in enumerate(zip(
            original_plan.chunks, new_chunks, strict=True))
        if a.axes != b.axes or a.gesture_family != b.gesture_family]
    if different_chunks != candidate["changed_chunk_indices"]:
        raise ValueError("candidate changed chunk indices differ")
    last_use = {}
    for e, axes in zip(events, new_assignments, strict=True):
        for axis in axes:
            prior = last_use.get(axis)
            if prior is not None and e.due_s - prior < REUSE_S-1e-9:
                raise ValueError("H1.1 path violates frozen 80ms abstract reuse")
            last_use[axis] = e.due_s
    delta = candidate["cost"]["final_delta"]
    alt_cost = candidate["cost"]["forced_final"]
    if (type(delta) not in (int, float)
            or type(alt_cost) not in (int, float)
            or not (math.isfinite(delta) and math.isfinite(alt_cost))):
        raise ValueError("nonfinite counterfactual cost")
    if abs(original_plan.metrics["beam_cost"]-candidate[
            "cost"]["original_v08_final"]) > 2e-6:
        raise ValueError("frozen objective differs from source H1.1 cost")
    # The metrics in the runner's abstract plan must NOT masquerade as
    # a normally selected v0.8 beam winner. This path was injected.
    return replace(
        original_plan, assignments=tuple(new_assignments),
        chunks=tuple(new_chunks), candidates_evaluated=0,
        metrics={
            "research_path_kind": "h1.4_experimental_h0.6_forced_suffix",
            "beam_cost": alt_cost,
            "beam_cost_evidence": "h1.1_pinned_reconstructed_objective",
            "ordinary_v08_plan_winner": False,
            "original_frozen_beam_cost": original_plan.metrics["beam_cost"],
            "alternate_minus_frozen_objective_cost": delta,
            "changed_events": len(changes),
            "original_v08_candidates_evaluated_for_preflight":
                original_plan.candidates_evaluated,
        },
        model_contract={
            **original_plan.model_contract,
            "research_override_applied": True,
            "source": "verified_h11_full_chart_oracle_assignments",
            "method": "single_reconstructed_seam_and_forced_suffix",
            "physical_motor_evaluated": False,
            "human_likeness_verified": False,
        })


@dataclass(frozen=True)
class Prepared:
    case: dict
    chart_path: Path
    h11_sha: str
    h13_sha: str
    manifest_sha: str
    verified_sha: str
    runtime: dict
    historical: dict
    source: dict
    chart: dict
    candidate: dict
    segment: object
    events: tuple[GestureEvent, ...]
    original_plan: GesturePlan
    alternate_plan: GesturePlan


def prepare(args) -> Prepared:
    repo = Path(__file__).resolve().parents[1]
    manifest = read(args.manifest)
    h11 = read(args.h11_report)
    h13 = read(args.h13_report)
    verified = read(args.verified_report)
    if (h11.get("format") != H11_FORMAT or
            h13.get("format") != H13_FORMAT or
            h13.get("source_h11_sha256") != sha256_file(args.h11_report)):
        raise ValueError("H1.1/H1.3 source or digest mismatch")
    if (h11.get("target_count") != 22 or h13.get("candidate_count") != 22
            or h11.get("strict_lower_heuristic_cost_count") != 7
            or h13.get("original_objective_lower_alternatives") != 7):
        raise ValueError("counterfactual census differs")
    recomputed_h13 = compile_audit(h11, source_sha256=sha256_file(args.h11_report))
    if recomputed_h13 != h13:
        raise ValueError("recomputed H1.3 report differs")
    if (verified.get("format") != "dmdogo-v08-train-generalization-report-v2"
            or verified.get("frozen_implementation_commit") != FROZEN_HEAD
            or verified.get("frozen_code_digest") != manifest.get("frozen_code_digest")
            or verified.get("all_cases_completed") is not True
            or verified.get("completed_count") != 3):
        raise ValueError("original three-Train verification report mismatch")
    if (h11.get("source_sha256", {}).get("manifest")
            != sha256_file(args.manifest)
            or h11.get("source_sha256", {}).get("h06") is None
            or h11.get("source_manifest_sha256")
            != sha256_file(args.manifest)):
        raise ValueError("H1.1 original Train manifest provenance changed")
    cases = h11.get("charts")
    if (not isinstance(cases, list) or len(cases) != 3):
        raise ValueError("H1.1 source charts missing")
    matches = [
        (chart, candidate) for chart in cases
        for candidate in chart["reconstructions"]
        if candidate["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("candidate ID must identify exactly one H1.1 path")
    chart, candidate = matches[0]
    by_id = [
        row for source_chart in h13["charts"] for row in source_chart["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(by_id) != 1 or by_id[0]["changed_event_count"] != candidate[
            "changed_event_count"]:
        raise ValueError("chosen H1.3 audited candidate mismatch")
    selection = case_from_manifest(manifest, chart["chart_sha256"])
    if (chart["event_count"] != selection["event_count"]
            or chart["chart"] != selection["chart"]):
        raise ValueError("H1.1 chart identity disagrees with Train manifest")
    chart_path = validate_chart(selection, Path(manifest["train_root"]).resolve())
    runtime = preflight(repo)
    validate_provenance_against_manifest(runtime, manifest)
    verified_rows = [
        row for row in verified["records"]
        if row["chart_sha256"] == selection["sha256"]]
    if len(verified_rows) != 1 or verified_rows[0]["status"] != "chart_complete":
        raise ValueError("no verified original completed Train baseline")
    original_result = args.results_root / selection["sha256"][:12] / "summary.json"
    proof_path = (args.results_root / "_provenance" /
                  f"{selection['sha256'][:12]}.json")
    if not original_result.is_file() or not proof_path.is_file():
        raise ValueError("original physical Train summary or provenance missing")
    original_summary = read(original_result)
    original_summary["_artifact_sha256"] = sha256_file(original_result)
    historical = _result_row(
        selection, original_summary, read(proof_path),
        frozen_code_digest=manifest["frozen_code_digest"],
        frozen_head=FROZEN_HEAD,
        manifest_digest=sha256_file(args.manifest))
    if historical != verified_rows[0]:
        # Prior saved report must be precisely reproducible for this case.
        raise ValueError("original verified Train outcome no longer matches saved report")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    timing = AutoplayTimingTeacher.from_segment(segment).events
    events = tuple(GestureEvent(
        x.event_index, x.episode_time_s, x.chord_size) for x in timing)
    if len(events) != chart["event_count"]:
        raise ValueError("recompiled event count changed")
    original_plan = GestureChunkPlannerV08(minimum_reuse_s=REUSE_S).plan(events)
    _require_same_rows(events, chart["original_events_by_index"], original_plan)
    alternate_plan = _inject_plan(
        original_plan, chart["original_chunks"], candidate, events)
    return Prepared(
        selection, chart_path, sha256_file(args.h11_report),
        sha256_file(args.h13_report), sha256_file(args.manifest),
        sha256_file(args.verified_report), runtime, historical,
        h11, chart, candidate, segment, events, original_plan,
        alternate_plan)


def preflight_record(ready: Prepared) -> dict:
    match = (
        ready.runtime["environment_digest"] ==
        ready.historical["environment_digest"])
    return {
        "format": FORMAT,
        "stage": "preflight",
        "candidate_id": ready.candidate["candidate_id"],
        "chart": ready.case["chart"],
        "chart_sha256": ready.case["sha256"],
        "chart_event_count": len(ready.events),
        "h11_source_sha256": ready.h11_sha,
        "h13_source_sha256": ready.h13_sha,
        "manifest_sha256": ready.manifest_sha,
        "verified_report_sha256": ready.verified_sha,
        "frozen_code_digest": ready.runtime["source"]["code_digest"],
        "run_environment_digest": ready.runtime["environment_digest"],
        "original_environment_digest": ready.historical["environment_digest"],
        "same_runtime_fingerprint_as_original": match,
        "original_verified_metrics": {
            k: ready.historical[k] for k in (
                "hits", "targets", "misses", "too_early",
                "x_accuracy_percent", "timing_mae_ms", "local_replans",
                "deferred_events", "overloaded")},
        "forced_heuristic_cost_delta": ready.candidate["cost"]["final_delta"],
        "changed_event_count": ready.candidate["changed_event_count"],
        "abstraction": "frozen_v08_motor_with_one_h11_forced_seam_schedule",
        "proposed_plan_is_original_v08_winner": False,
        "alternate_motor_executed": False,
        "human_likeness_verified": False,
    }


class OneShotPlanner:
    """Runner's isolated runtime symbol override. No source-file mutation."""

    def __init__(self, ready: Prepared):
        self.ready = ready
        self.calls = 0

    def factory(self, *, minimum_reuse_s: float = .080):
        if abs(minimum_reuse_s - REUSE_S) > 1e-12:
            raise ValueError("physical H1.4 must retain frozen 80ms reuse")
        outer = self

        class Proxy:
            def plan(self, events):
                outer.calls += 1
                if outer.calls != 1:
                    raise ValueError("H1.4 planner was called more than once")
                source = tuple(events)
                if (len(source) != len(ready.events)
                        or any((a.event_index != b.event_index
                                or a.chord_size != b.chord_size
                                or abs(a.due_s-b.due_s) > EPS)
                               for a,b in zip(source,ready.events,strict=True))):
                    raise ValueError("Motor oracle events differ from H1.4 attested source")
                return ready.alternate_plan

        return Proxy()


def execute(ready: Prepared, *, wall_timeout_s: float) -> dict:
    """Physical Motor execution only after full provenance preflight."""
    slot = OneShotPlanner(ready)
    # One and only one adapter symbol in the runner is replaced in memory.
    # v0.8 Planner, selection loop, live Motor, compensation and physics
    # source files remain untouched; result never claims normal v0.8.
    with patch.object(
        frozen_runner, "GestureChunkPlannerV08", slot.factory):
        result = frozen_runner.run_pattern_arm(
            ready.segment, minimum_reuse_s=REUSE_S,
            stop_after_s=None, wall_timeout_s=wall_timeout_s,
            planning_mode="gesture_v08")
    if slot.calls != 1:
        raise ValueError("H1.4 experimental plan was not used exactly once")
    expected = {
        event.event_index: tuple(axes)
        for event, axes in zip(
            ready.events, ready.alternate_plan.assignments, strict=True)}
    for row in result["decisions"]:
        if (tuple(row["desired_axes"]) != expected[row["event_index"]]
                or row["policy"] != "gesture_chunk_v08"):
            raise ValueError("experimental desired axes did not reach unchanged Motor")
    stats = result["stats"]
    physical = result["physical"]
    executed = len(result["decisions"])
    if (stats["hits"] + stats["misses"] != stats["targets"]
            or executed > len(ready.events)
            or physical["local_replans"] !=
            sum(bool(d["local_replan"]) for d in result["decisions"])):
        raise ValueError("physical episode metrics/decision census inconsistent")
    adoption = sum(
        row["axes"] == row["desired_axes"] for row in result["decisions"])
    preflight = preflight_record(ready)
    return {
        **preflight,
        "stage": "executed",
        "alternate_motor_executed": True,
        "termination_reason": result["termination_reason"],
        "stats": stats,
        "physical": physical,
        "selected_event_count": executed,
        "planned_axis_adoption_count": adoption,
        "physical_local_replan_count": physical["local_replans"],
        "actual_keydown_count": len(result["keydowns"]),
        "decisions": result["decisions"],
        "keydowns": result["keydowns"],
        "compensation": result["compensation_telemetry"],
        "run_is_frozen_v08_baseline": False,
        "physical_motor_source": "unchanged_v08_runner_and_live_motor",
        "policy_source": "h11_reconstructed_single_seam_forced_suffix",
        "comparison_to_original_is_controlled_runtime": (
            preflight["same_runtime_fingerprint_as_original"]),
        "human_likeness_verified": False,
        "human_preference_measured": False,
        "validation20_or_final_accessed": False,
        "limitations": [
            "Research-only isolated planner-symbol substitution, never a normal frozen v0.8 planner output.",
            "H1.1 gives Oracle timing/full assignment; this is not HUD-only planning.",
            "A fallback local replan changes the executed axes; inspect adoption and deferrals, not just Hits.",
            "Different runtime fingerprint prevents controlled before-after comparison; baseline only descriptive.",
            "Physical success for simulated Motor does not prove a human can play the chosen fingering.",
        ],
    }


def _run_worker(args) -> None:
    ready = prepare(args)
    if not 30 <= args.wall_timeout_s <= 7200:
        raise ValueError("outer wall timeout must be 30..7200 seconds")
    report = execute(ready, wall_timeout_s=args.wall_timeout_s)
    _write_once(args.output, report)
    print(json.dumps({
        "case": report["candidate_id"],
        "termination": report["termination_reason"],
        "physical": report["physical"],
        "stats": report["stats"],
        "planned_axis_adoption_count": report["planned_axis_adoption_count"],
        "selected_event_count": report["selected_event_count"],
        "same_runtime_fingerprint_as_original": report[
            "same_runtime_fingerprint_as_original"],
        "saved": str(args.output),
    },ensure_ascii=False,indent=2),flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "verified-report", "h11-report", "h13-report",
                 "results-root", "candidate-id", "output"):
        p.add_argument(
            "--"+name, required=True,
            type=Path if name not in ("candidate-id",) else str)
    p.add_argument("--execute", action="store_true",
                   help="explicitly start ONE new alternate Motor run")
    p.add_argument("--status-file", type=Path)
    p.add_argument("--wall-timeout-s",type=float,default=1200)
    p.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite earlier H1.4 result")
    if args._worker:
        if not args.execute or args.status_file is None:
            p.error("worker must have explicit execution/status parameters")
        _run_worker(args)
        return
    if args.execute:
        if args.status_file is None or args.status_file.exists():
            p.error("execution requires unused --status-file")
        if (not math.isfinite(args.wall_timeout_s)
                or not 30 <= args.wall_timeout_s <= 7200):
            p.error("invalid 30..7200 outer wall timeout")
        # The same script is executed in an isolated child, which cannot
        # alter the parent's frozen runner module. The parent bounds
        # compilation, candidate validation and the Motor episode.
        cmd=[sys.executable, str(Path(__file__).resolve()), "--_worker",
             "--execute", "--status-file", str(args.status_file),
             "--wall-timeout-s", str(args.wall_timeout_s)]
        for name in ("manifest", "verified-report", "h11-report",
                     "h13-report", "results-root", "candidate-id", "output"):
            value=getattr(args,name.replace("-","_"))
            cmd.extend(["--"+name,str(value)])
        started=datetime.now(timezone.utc).isoformat()
        rc=1
        state="worker_failed"
        try:
            completed=subprocess.run(
                cmd, cwd=Path(__file__).resolve().parents[1],
                check=False, timeout=args.wall_timeout_s)
            rc=completed.returncode
            state="success" if rc==0 else "worker_failed"
        except subprocess.TimeoutExpired:
            rc=124
            state="outer_timeout"
        status={
            "format": STATUS_FORMAT,
            "candidate_id":args.candidate_id,
            "started_at_utc":started,
            "finished_at_utc":datetime.now(timezone.utc).isoformat(),
            "state":state,
            "exit_code":rc,
            "output_file":str(args.output),
            "output_sha256": (
                sha256_file(args.output) if args.output.is_file() else None),
            "physical_motor_requested":True,
        }
        _write_once(args.status_file,status)
        if rc!=0:
            raise SystemExit(rc)
        print(json.dumps(status,indent=2,ensure_ascii=False))
        return
    if args.status_file is not None:
        p.error("preflight does not use status file; omit --status-file")
    ready=prepare(args)
    report=preflight_record(ready)
    _write_once(args.output,report)
    print(json.dumps({
        "stage":"preflight",
        "case":report["candidate_id"],
        "events":report["chart_event_count"],
        "change_count":report["changed_event_count"],
        "same_runtime_fingerprint_as_original":
            report["same_runtime_fingerprint_as_original"],
        "saved":str(args.output),
        "motor_executed":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
