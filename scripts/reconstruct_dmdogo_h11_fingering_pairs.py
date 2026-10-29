from __future__ import annotations

"""DMDOGO H1.1: reconstruct every H0.6 abstract original/forced fingering.

Only verified preselected frozen v0.8 Train artifacts are allowed.
All 22 independent H0.6 forced-seam alternatives must reproduce the
original per-case H0.6 audit exactly before any new output is written.
No modified planner, Motor, human demonstrations, Validation20, FINAL,
training or physical execution. All outputs are create-once.
"""

import argparse
import json
from pathlib import Path

from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_h1_quality_evidence import (
    H06_BLOB_SHA, compile_register, git_blob_sha,
)
from audit_dmdogo_v09_h04_beam_decisions import (
    FORMAT as H04_FORMAT, _oracle_events,
)
from audit_dmdogo_v09_h05_nonquick_suffix import (
    FORMAT as H05_FORMAT, trace_unrestricted, _cost_terms,
)
from audit_dmdogo_v09_h06_improved_suffix import (
    FORMAT as H06_FORMAT,
    _assignment_metrics, _event_axes, _quick_flags, _rapid_flags,
    _flag_delta, _component_delta, _force_suffix_detailed,
    audit_case as audit_case_h06,
)
from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from dmdogo_v08_provenance import FROZEN_HEAD, sha256_file
from report_dmdogo_v09_style_proxies import REPORT_FORMAT as H0_FORMAT
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-h11-paired-fingering-reconstruction-v1"
EPS = 1e-7
TOL = 2e-6


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def reconstruction(events, original_chunks: list[dict],
                   forced_chunks: list[dict],
                   saved_h06: dict, *, candidate_id: str) -> dict:
    """Pure detailed representation plus strict H0.6 replay parity.

    One source chart's original axes are stored once, separately from
    this candidate's fully aligned alternative event-axis vector.
    """
    if len(original_chunks) != len(forced_chunks):
        raise ValueError("original/forced chunk count changed")
    by_old = _event_axes(original_chunks)
    by_new = _event_axes(forced_chunks)
    ordered_ids = [e.event_index for e in events]
    if (len(set(ordered_ids)) != len(ordered_ids)
            or set(by_old) != set(ordered_ids)
            or set(by_new) != set(ordered_ids)):
        raise ValueError("original/forced event coverage mismatch")
    changed_chunks = [
        i for i, (a, b) in enumerate(zip(
            original_chunks, forced_chunks, strict=True))
        if a["axes"] != b["axes"]
        or a["gesture_family"] != b["gesture_family"]]
    changes = [
        {
            "event_index": event.event_index,
            "due_s": event.due_s,
            "original_axes": list(by_old[event.event_index]),
            "alternative_axes": list(by_new[event.event_index]),
        }
        for event in events
        if by_old[event.event_index] != by_new[event.event_index]
    ]
    effects = saved_h06["assignment_effects"]
    if (len(changed_chunks) != effects["changed_chunk_count"]
            or changed_chunks != effects["changed_chunk_indices"]
            or len(changes) != effects["changed_event_assignment_count"]
            or (changed_chunks[0] if changed_chunks else None)
            != effects["first_changed_chunk"]
            or (changed_chunks[-1] if changed_chunks else None)
            != effects["last_changed_chunk"]):
        raise ValueError("H0.6 changed chunk/event coverage disagrees")

    # Reconstruct the exact independently scored assignment metrics,
    # including abstract reuse safety and the observed flag identities.
    old_metrics = _assignment_metrics(events, original_chunks)
    new_metrics = _assignment_metrics(events, forced_chunks)
    if (old_metrics != saved_h06["original_assignment_metrics"]
            or new_metrics != saved_h06["forced_assignment_metrics"]
            or new_metrics["abstract_reuse_under_80ms_count"] != 0):
        raise ValueError("H0.6 physical-axis metrics/reuse gate differ")
    flags = saved_h06["flag_effects"]
    new_quick = _quick_flags(events, forced_chunks)
    old_quick = _quick_flags(events, original_chunks)
    old_rapid = _rapid_flags(events, original_chunks)
    new_rapid = _rapid_flags(events, forced_chunks)
    if (_flag_delta(old_quick, new_quick) != flags["quick"]
            or _flag_delta(old_rapid, new_rapid) != flags["rapid"]
            or flags["target_quick_flag_removed"] is not (
                tuple(saved_h06["flag_pair"]) not in new_quick)):
        raise ValueError("H0.6 flag change identities differ")

    # Check that all chunk-to-event rows retain the original exact
    # event start/end identities; only fingering family/axes may differ.
    for idx, (old, new) in enumerate(zip(
            original_chunks, forced_chunks, strict=True)):
        for key in ("start_event_index", "end_event_index", "event_count",
                    "rhythm_kind", "start_reason"):
            if old[key] != new[key]:
                raise ValueError(
                    f"alternative changed frozen segmentation at chunk {idx}: {key}")

    end_rejoin = effects["permanent_rejoin_chunk"]
    if end_rejoin is not None:
        if not (end_rejoin > saved_h06["window_index"]
                and end_rejoin <= len(original_chunks)
                and all(i not in changed_chunks for i in range(
                    end_rejoin, len(original_chunks)))):
            raise ValueError("saved permanent rejoin is not a true suffix rejoin")

    return {
        "candidate_id": candidate_id,
        "flag_pair": saved_h06["flag_pair"],
        "join_window_index": saved_h06["window_index"],
        "changed_chunk_indices": changed_chunks,
        "changed_event_count": len(changes),
        "changed_events": changes,
        "original_and_alternative_event_order": "chart.original_events_by_index",
        "alternative_axes_by_event_order": [
            list(by_new[event_id]) for event_id in ordered_ids
        ],
        "alternative_chunks": forced_chunks,
        "quick_flag_delta": flags["quick"],
        "rapid_flag_delta": flags["rapid"],
        "physical_reuse_80ms_proxy_violations": (
            new_metrics["abstract_reuse_under_80ms_count"]),
        "human_preference": "unobserved",
        "alternate_motor_result": "not_executed",
        "independent_intervention_only": True,
    }


def reconstruct_chart(chart: Path, summary: dict,
                      telemetry: dict, replay: dict,
                      h01_case: dict, h04_case: dict,
                      h05_case: dict, h06_case: dict,
                      register_by_id: dict) -> dict:
    """Full source-parity pass, followed by complete 22-way path recovery."""
    ordered, original_chunks, _ = _validate(summary, telemetry, replay)
    events = _oracle_events(chart)
    if (len(events) != len(ordered)
            or [x.event_index for x in events] != [
                row["event_index"] for row in ordered]
            or any(abs(ev.due_s * 1000 - row["due_ms"]) > .005
                   for ev, row in zip(events, ordered))):
        raise ValueError("compiled chart event identities/times differ from replay")
    if (summary["chart_sha256"] != h06_case["chart_sha256"]
            or summary["chart_sha256"] != h05_case["chart_sha256"]
            or summary["chart_sha256"] != h04_case["chart_sha256"]):
        raise ValueError("chart SHA256 disagreement across H0.4/H0.5/H0.6")
    recomputed = audit_case_h06(
        chart, summary, telemetry, replay,
        h01_case, h04_case, h05_case)
    if recomputed != h06_case:
        raise ValueError("saved H0.6 full chart audit differs from recomputation")

    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    if len(windows) != len(original_chunks):
        raise ValueError("frozen original window segmentation mismatch")
    ordered_original = _event_axes(original_chunks)
    original_records = [
        {
            "event_index": e.event_index,
            "due_s": e.due_s,
            "chord_size": e.chord_size,
            "original_axes": list(ordered_original[e.event_index]),
        }
        for e in events
    ]
    lineage, seeds, frozen_winner = trace_unrestricted(
        events, original_chunks, h04_case["flag_evidence"],
        summary["abstract_schedule"]["metrics"]["beam_cost"],
        summary["abstract_schedule"]["candidates_evaluated"])
    if len(lineage) != len(h06_case["evidence"]):
        raise ValueError("tagged target count disagrees with H0.6")
    cost = round(frozen_winner.score, 9)
    reconstructed = []
    for index, (item, source) in enumerate(zip(
            lineage, h06_case["evidence"], strict=True)):
        join = item["window_index"]
        pair = item["flag_pair"]
        if (pair != source["flag_pair"]
                or join != source["window_index"]
                or item["nonquick_axes"] != source["nonquick_axes"]
                or item["nonquick_gesture_family"] !=
                source["nonquick_gesture_family"]
                or item["nonquick_join_expanded_rank"] !=
                source["beam_lineage"]["join_expanded_rank"]
                or item["lineage_loss_reason"] !=
                source["beam_lineage"]["loss_reason"]):
            raise ValueError("H0.6 target/lineage source identity changed")

        id_ = f"{summary['chart_sha256'][:12]}-{pair[0]}-{pair[1]}"
        if id_ not in register_by_id:
            raise ValueError("H1.0 evidence registry is missing candidate")
        register = register_by_id[id_]
        if (register["event_pair"] != pair
                or register["window_index"] != join):
            raise ValueError("H1.0 target identity mismatch")

        candidate = _candidate_gestures(windows[join])[
            item["nonquick_candidate_index"]]
        if (candidate.family != item["nonquick_gesture_family"]
                or [list(x) for x in candidate.axes] != item["nonquick_axes"]):
            raise ValueError("source candidate generator choice changed")
        path = _force_suffix_detailed(
            seeds[index], windows, join, candidate,
            beam_width=planner.beam_width)
        if not path["reached_terminal"]:
            raise ValueError("H0.6 alternative no longer reaches terminal")
        chunks = original_chunks[:join] + path["chunks"]
        if len(chunks) != len(original_chunks):
            raise ValueError("H0.6 path did not cover full chart")
        original_terms = _cost_terms(frozen_winner)
        if (path["final_cost"] != source["cost"]["forced_final"]
                or cost != source["cost"]["original_v08_final"]
                or path["final_cost_terms"] !=
                source["cost"]["forced_components"]
                or _component_delta(path["final_cost_terms"], original_terms)
                != source["cost"]["six_component_deltas"]
                or abs(path["final_cost"] - cost -
                       source["cost"]["final_delta"]) > TOL):
            raise ValueError("forced suffix exact score/components mismatch")
        output = reconstruction(
            events, original_chunks, chunks, source, candidate_id=id_)
        if (output["changed_event_count"] !=
                register["assignment_difference"]["changed_event_count"]
                or register["abstract_score"]["lower_frozen_heuristic_cost"]
                is not source["improved_final_cost"]):
            raise ValueError("H1.0 candidate quality-ledger identity mismatch")
        output["cost"] = source["cost"]
        output["source_h06_forced_candidate_indices"] = path["candidate_indices"]
        output["reached_terminal_abstract"] = True
        output["source_h06_suffix_candidates_evaluated"] = (
            path["suffix_candidates_evaluated"])
        output["beam_source_loss"] = source["beam_lineage"]
        reconstructed.append(output)

    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(events),
        "chunk_count": len(original_chunks),
        "source_original_frozen_cost": cost,
        "original_events_by_index": original_records,
        "original_chunks": original_chunks,
        "reconstructions": reconstructed,
        "count": len(reconstructed),
    }


def compile_reconstruction(manifest: dict, verified: dict, h0: dict,
                           h01: dict, h04: dict, h05: dict, h06: dict,
                           h1: dict, case_artifacts: dict,
                           chart_paths: dict[str, Path], *,
                           manifest_digest: str,
                           source_digests: dict[str, str]) -> dict:
    """Everything verified before returning. Caller writes only on success."""
    if (h06.get("format") != H06_FORMAT
            or h06.get("source_v08_commit") != FROZEN_HEAD
            or h06.get("planner_motor_timing_runner_changed") is not False
            or h06.get("physical_motor_executed") is not False
            or h04.get("format") != H04_FORMAT
            or h05.get("format") != H05_FORMAT):
        raise ValueError("saved H0.4–H0.6 provenance mismatch")
    reconstructed_h1 = compile_register(h06, input_blob_sha=H06_BLOB_SHA)
    if reconstructed_h1 != h1:
        raise ValueError("H1.0 quality registry differs from its saved source")
    h01_replay = compile_train_report(
        manifest, verified, h0, case_artifacts,
        manifest_digest=manifest_digest)
    if h01_replay != h01:
        raise ValueError("original H0.1 Train report disagrees with verified data")
    manifest_cases = manifest["cases"]
    ids = [x["sha256"] for x in manifest_cases]
    if (len(ids) != 3 or len(set(ids)) != 3
            or len(h06["cases"]) != len(ids)
            or h06["event_count"] != 10231
            or not set(ids) == set(chart_paths)
            or not set(ids) == set(case_artifacts)):
        raise ValueError("original three-chart Train cohort differs")
    case_by_h01 = {x["chart_sha256"]: x for x in h01["cases"]}
    case_by_h04 = {x["chart_sha256"]: x for x in h04["cases"]}
    case_by_h05 = {x["chart_sha256"]: x for x in h05["cases"]}
    case_by_h06 = {x["chart_sha256"]: x for x in h06["cases"]}
    if any(set(d) != set(ids) for d in (
            case_by_h01, case_by_h04, case_by_h05, case_by_h06)):
        raise ValueError("saved H0.1/H0.4/H0.5/H0.6 case census differs")
    register_by_id = {
        x["candidate_id"]: x for x in h1["candidates"]}
    if len(register_by_id) != 22:
        raise ValueError("duplicate/missing H1.0 candidate ID")
    charts = []
    for case in manifest_cases:
        sha = case["sha256"]
        summary, telemetry, replay, _ = case_artifacts[sha]
        charts.append(reconstruct_chart(
            chart_paths[sha], summary, telemetry, replay,
            case_by_h01[sha], case_by_h04[sha],
            case_by_h05[sha], case_by_h06[sha], register_by_id))

    records = [x for chart in charts for x in chart["reconstructions"]]
    if (len(records) != 22
            or len({r["candidate_id"] for r in records}) != 22
            or sum(r["cost"]["final_delta"] < -EPS for r in records) != 7
            or sum(chart["event_count"] for chart in charts) != 10231
            or {r["candidate_id"] for r in records} != set(register_by_id)):
        raise ValueError("H1.1 original 22-target/seven-improvement census differs")
    return {
        "format": FORMAT,
        "source_v08_frozen_commit": FROZEN_HEAD,
        "source_h06_git_blob": H06_BLOB_SHA,
        "source_sha256": source_digests,
        "source_manifest_sha256": manifest_digest,
        "cohort": h1["cohort"],
        "chart_count": len(charts),
        "target_count": len(records),
        "strict_lower_heuristic_cost_count": sum(
            r["cost"]["final_delta"] < -EPS for r in records),
        "complete_event_axis_assignments": True,
        "path_scope": (
            "Every counterfactual independently replaces just one frozen "
            "join seam and replans its OWN suffix. The 22 alternatives "
            "must never be merged into one trajectory."
        ),
        "charts": charts,
        "human_preference_observed": False,
        "alternate_motor_executed": False,
        "motor_quality_verified": False,
        "validation20_or_final_accessed": False,
        "training_performed": False,
        "frozen_planner_motor_timing_changed": False,
        "limitations": [
            "All 22 are previously selected Train seams, not an independent holdout.",
            "Score improvement is only the untrained frozen six-term heuristic.",
            "80ms abstract same-axis reuse is a grammar constraint, not a human comfort guarantee.",
            "Original physical Train success is not evidence of alternate physical Motor success.",
            "No blinded A/B judgments or independently observed fingering labels.",
            "Full event-axis vectors can be used for later masked visualization, but are not yet a human-likeness result.",
            "Reconstruction requires the original three verified Train charts and historical report files; no synthetic substitution permitted.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "verified-report", "h0-report", "h01-report",
        "h04-report", "h05-report", "h06-report", "h1-report",
        "results-root", "output",
    ):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H1.1 results")
    sources = {
        "manifest": args.manifest,
        "verified": args.verified_report,
        "h0": args.h0_report,
        "h01": args.h01_report,
        "h04": args.h04_report,
        "h05": args.h05_report,
        "h06": args.h06_report,
        "h1": args.h1_report,
    }
    if git_blob_sha(args.h06_report) != H06_BLOB_SHA:
        raise ValueError("H0.6 report differs from frozen H1.0 source blob")
    loaded = {key: load(path) for key, path in sources.items()}
    manifest = loaded["manifest"]
    artifacts = {}
    chart_paths = {}
    train_root = Path(manifest["train_root"]).resolve()
    if train_root.name.casefold() != "train":
        raise ValueError("only previously selected Train charts allowed")
    for case in manifest.get("cases", []):
        sha = case["sha256"]
        chart_paths[sha] = validate_chart(case, train_root)
        base = args.results_root / sha[:12]
        summary_path = base / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary,
            load(base / "telemetry-v08.json"),
            load(base / "replay-v08.json"),
            load(args.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    source_digests = {
        key: sha256_file(path) for key, path in sources.items()}
    result = compile_reconstruction(
        loaded["manifest"], loaded["verified"], loaded["h0"],
        loaded["h01"], loaded["h04"], loaded["h05"], loaded["h06"],
        loaded["h1"], artifacts, chart_paths,
        manifest_digest=source_digests["manifest"],
        source_digests=source_digests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "chart_count": result["chart_count"],
        "total_events": result["cohort"]["total_events"],
        "counterfactuals": result["target_count"],
        "lower_frozen_heuristic_cost": result[
            "strict_lower_heuristic_cost_count"],
        "changed_event_counts": [
            {
                "chart_sha256": ch["chart_sha256"],
                "cases": [
                    {
                        "candidate_id": case["candidate_id"],
                        "changed_events": case["changed_event_count"],
                        "changed_chunks": len(case["changed_chunk_indices"]),
                    }
                    for case in ch["reconstructions"]
                ],
            }
            for ch in result["charts"]
        ],
        "saved": str(args.output),
        "human_preference_observed": False,
        "alternate_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
