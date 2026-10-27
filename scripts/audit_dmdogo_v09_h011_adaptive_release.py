from __future__ import annotations

"""DMDOGO H0.11: diagnose 22 quota-8 runs and test early release.

Exploratory, NOT held-out: release quota after five join-inclusive chunks
if no protected descendant has ranked first in any expanded population.
The criterion is based on H0.9's known seven successes, not independently
validated. Release means ordinary stable top-24 pruning, NOT deleting the tag.
All v0.8 code/scoring/beam-width/80ms-reuse rules remain frozen.
No physical Motor, validation, FINAL, or training.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_v09_h04_beam_decisions import _oracle_events
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state, _cost_terms
from audit_dmdogo_v09_h06_improved_suffix import FORMAT as H06_FORMAT
from audit_dmdogo_v09_h08_protected_quota import (
    BEAM_WIDTH, TARGET_TAG, _candidate_target, _describe,
    choose_quota,
)
from audit_dmdogo_v09_h07_shadow_lane import (
    BeamPath, Node, Proposal, _chunk_record_from,
)
from audit_dmdogo_v09_h09_quota_mechanism import (
    trace_window_mechanism, compare_exact_h08,
)
from audit_dmdogo_v09_h010_all_quick_seams import FORMAT as H010_FORMAT
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h011-adaptive-quota-release-v1"
QUOTA = 8
OBSERVE_CHUNKS = 5  # join, join+1, ..., join+4 inclusive
EPS = 1e-7


def stop_after_grace(*, chunks_seen: int, ever_protected_rank1: bool,
                     already_released: bool = False) -> bool:
    """Predeclared observed-information policy. True only once."""
    if not isinstance(chunks_seen, int) or chunks_seen < 0:
        raise ValueError("invalid number of observed chunks")
    return (not already_released and chunks_seen == OBSERVE_CHUNKS
            and not ever_protected_rank1)


def _difference(components_a: dict, components_b: dict) -> dict:
    if set(components_a) != set(components_b):
        raise ValueError("different frozen cost term sets")
    return {key: round(components_a[key] - components_b[key], 9)
            for key in components_a}


def _stats(rows) -> dict:
    rows = list(rows)
    if not rows:
        return {"count": 0, "min": None, "median": None,
                "mean": None, "max": None}
    return {
        "count": len(rows), "min": round(min(rows), 9),
        "median": round(median(rows), 9),
        "mean": round(mean(rows), 9),
        "max": round(max(rows), 9),
    }


def quota8_diagnostic(source_row: dict, window_trace: dict) -> dict:
    """Cumulative-score decomposition and *observed* early rank metrics.

    The full trace comes from an independently re-executed exact H0.10
    quota-8 population. No future signal is used by the release policy.
    """
    compare_exact_h08(window_trace, source_row)
    windows = window_trace["windows"]
    if not windows:
        raise ValueError("no quota-8 windows")
    protected = source_row["best_protected_terminal"]
    winner = source_row["beam_winner"]
    if protected is None or winner is None:
        raise ValueError("saved 22-case quota8 run lacked terminal hypotheses")
    if protected["axis_reuse_under_80ms"] or winner["axis_reuse_under_80ms"]:
        raise ValueError("abstract reuse rule violated")
    early = windows[:OBSERVE_CHUNKS]
    gaps = [
        item["best_tagged_minus_best_untagged"] for item in windows
        if item["best_tagged_minus_best_untagged"] is not None
    ]
    early_gaps = [
        item["best_tagged_minus_best_untagged"] for item in early
        if item["best_tagged_minus_best_untagged"] is not None
    ]
    # Checkpoints for readable cost-growth comparisons; the complete
    # per-window gap series is also retained to avoid selection artifacts.
    checkpoints = []
    indices = sorted({0, 1, 2, 3, 4, len(windows)//4,
                      len(windows)//2, 3*len(windows)//4, len(windows)-1})
    for i in indices:
        if i >= len(windows):
            continue
        row = windows[i]
        checkpoints.append({
            "offset_chunks": i,
            "window_index": row["window_index"],
            "best_tagged_minus_untagged": row[
                "best_tagged_minus_best_untagged"],
            "natural_tagged_count": row["natural_tagged_count"],
            "best_tagged_expanded_rank": row["best_tagged_expanded_rank"],
            "rescue_slots": row["replaced_slots"],
        })
    return {
        "first_rank1_offset": next((
            i for i, item in enumerate(windows)
            if item["best_tagged_expanded_rank"] == 1), None),
        "ever_rank1_by_grace": any(
            x["best_tagged_expanded_rank"] == 1 for x in early),
        "grace_chunk_count": len(early),
        "grace_first_window": early[0]["window_index"],
        "grace_last_window": early[-1]["window_index"],
        "grace_tagged_minus_untagged": early_gaps,
        "full_tagged_minus_untagged": gaps,
        "gap_distribution": _stats(gaps),
        "first_and_late_gap_checkpoints": checkpoints,
        "minimum_gap": min(gaps) if gaps else None,
        "maximum_gap": max(gaps) if gaps else None,
        "final_best_protected_minus_winner": round(
            protected["final_cost"] - winner["final_cost"], 9),
        "final_term_differences_protected_minus_winner": _difference(
            protected["score_components"], winner["score_components"]),
        "full_quota8_rescue_windows": source_row["slot_replacement_count"],
        "full_quota8_displaced_slots": source_row[
            "total_displaced_nonprotected_slots"],
        "first_protected_rank1_window": next((
            x["window_index"] for x in windows
            if x["best_tagged_expanded_rank"] == 1), None),
        "natural_quota_windows": sum(
            x["natural_tagged_count"] >= QUOTA for x in windows),
    }


def trace_adaptive(events, frozen_chunks: list[dict], target: dict,
                   *, expected_cost: float,
                   expected_evaluated: int,
                   reference_final_cost: float) -> dict:
    """Independent full-chart replay with online release after grace.

    Exactly quota8 through the grace window. If none of those *past*
    expanded populations had a protected #1, subsequent windows use
    vanilla stable top24. A surviving tagged hypothesis is not deleted.
    """
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    canonical = planner.plan(events)
    from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
    if (not canonical.feasible or len(windows) != len(frozen_chunks)
            or [_chunk_record(x) for x in canonical.chunks] != frozen_chunks
            or canonical.metrics["beam_cost"] != expected_cost
            or canonical.candidates_evaluated != expected_evaluated
            or round(reference_final_cost, 6) != expected_cost):
        raise ValueError("frozen v0.8 planning source mismatch")
    join = target["window_index"]
    if (not 0 < join < len(windows)
            or windows[join].events[0].event_index != target["flag_pair"][1]
            or windows[join-1].events[-1].event_index != target["flag_pair"][0]):
        raise ValueError("target seam moved")
    states = [BeamPath(_seed_state(events[0].due_s), None)]
    tag_seen_rank1 = False
    stopped = False
    stopped_at = None
    rescue_windows = []
    displaced = 0
    observed_count = 0
    admission_rank = None
    eligible_expansions_total = 0
    for wi, window in enumerate(windows):
        prev_due = windows[wi-1].events[-1].due_s if wi else None
        family = frozen_chunks[wi]["gesture_family"]
        axes = tuple(tuple(x) for x in frozen_chunks[wi]["axes"])
        expanded = []
        for parent_rank, parent in enumerate(states, 1):
            for ci, candidate in enumerate(_candidate_gestures(window)):
                eligible_expansions_total += 1
                nxt = _next(parent.state, window, candidate, prev_due,
                            frozen_candidate=(
                                candidate.axes == axes and candidate.family == family))
                if nxt is None:
                    continue
                if (wi == join and parent.state.frozen_prefix
                        and ci == target["nonquick_candidate_index"]):
                    nxt.tags = frozenset({TARGET_TAG})
                expanded.append(Proposal(
                    nxt, parent.node, _chunk_record_from(window, candidate),
                    parent_rank, ci))
        if not expanded:
            raise ValueError(f"abstract beam expansion empty at {wi}")
        expanded.sort(key=lambda x: x.state.score)
        if wi < join:
            keep = expanded[:BEAM_WIDTH]
            if sum(x.state.frozen_prefix for x in keep) != 1:
                raise ValueError("original selected path pruned before join")
        else:
            if wi == join:
                tagged = [(rank, item) for rank, item in enumerate(expanded, 1)
                          if item.protected]
                if len(tagged) != 1:
                    raise ValueError("cannot uniquely tag H0.6 target")
                rank, item = tagged[0]
                if (rank != target["nonquick_join_expanded_rank"]
                        or item.candidate_index != target["nonquick_candidate_index"]
                        or item.chunk["axes"] != target["nonquick_axes"]):
                    raise ValueError("H0.6 target rank/identity mismatch")
                admission_rank = rank
            if not stopped:
                keep, selection = choose_quota(expanded, quota=QUOTA)
                if selection["displaced_nonprotected_count"]:
                    rescue_windows.append(wi)
                    displaced += selection["displaced_nonprotected_count"]
                tag_seen_rank1 |= bool(expanded[0].protected)
                observed_count += 1
                if stop_after_grace(
                        chunks_seen=observed_count,
                        ever_protected_rank1=tag_seen_rank1):
                    stopped = True
                    stopped_at = wi
            else:
                keep = expanded[:BEAM_WIDTH]
        states = [BeamPath(row.state, Node(row.parent, row.chunk))
                  for row in keep]

    winner = states[0]
    tagged_terminal = [(rank, p) for rank, p in enumerate(states, 1)
                       if TARGET_TAG in p.state.tags]
    protected = tagged_terminal[0][1] if tagged_terminal else None
    return {
        "flag_pair": target["flag_pair"],
        "window_index": join,
        "width": BEAM_WIDTH,
        "quota_until_release": QUOTA,
        "observation_chunks_before_decision": OBSERVE_CHUNKS,
        "ever_protected_rank1_before_decision": tag_seen_rank1,
        "quota_released": stopped,
        "release_decision_window": stopped_at,
        "first_post_release_window": stopped_at + 1 if stopped else None,
        "protected_terminal_survived": protected is not None,
        "best_protected_terminal_rank": (
            tagged_terminal[0][0] if tagged_terminal else None),
        "admission_expanded_rank": admission_rank,
        "rescue_windows": rescue_windows,
        "rescue_window_count": len(rescue_windows),
        "displaced_untagged_slots": displaced,
        "total_candidates_evaluated": eligible_expansions_total,
        "beam_winner": _describe(
            events, frozen_chunks, winner, reference_final_cost),
        "best_protected_terminal": _describe(
            events, frozen_chunks, protected, reference_final_cost),
        "physical_motor_executed": False,
    }


def _validate_reports(checked: dict, manifest: dict, h06: dict,
                      h08: dict, h010: dict, h06_path: Path,
                      h08_path: Path) -> list[dict]:
    if (h06.get("format") != H06_FORMAT
            or h010.get("format") != H010_FORMAT
            or h010.get("beam_width") != BEAM_WIDTH
            or h010.get("protected_minimum_quota") != QUOTA
            or h010.get("source_h06_sha256") != sha256_file(h06_path)
            or h010.get("source_h08_sha256") != sha256_file(h08_path)):
        raise ValueError("phase H0.10/6/8 metadata mismatch")
    for report in (h06, h08, h010):
        if (report.get("source_v08_commit") != checked["source_v08_commit"]
                or report.get("frozen_code_digest") != checked["frozen_code_digest"]
                or report.get("event_count") != checked["aggregate"]["event_count"]
                or report.get("case_count") != len(manifest["cases"])):
            raise ValueError("frozen Train provenance mismatch")
    required = {case["sha256"] for case in manifest["cases"]}
    if len(required) != len(manifest["cases"]):
        raise ValueError("duplicate Train chart SHA")
    for phase in (h06, h08, h010):
        cases = phase["cases"]
        by_sha = {case["chart_sha256"]: case for case in cases}
        if len(by_sha) != len(cases) or set(by_sha) != required:
            raise ValueError("missing/duplicate/mismatched Train charts")
    if (h010.get("targeted_flag_count") != 22
            or h010["aggregate"]["all_22"]["target_count"] != 22
            or h010["aggregate"]["previously_improved_7"]["target_count"] != 7
            or h010["aggregate"]["previously_nonimproved_15"]["target_count"] != 15):
        raise ValueError("target cohorts no longer match the saved H0.10 census")
    return [{c["chart_sha256"]: c for c in report["cases"]}
            for report in (h06, h08, h010)]


def _aggregate(rows: list[dict]) -> dict:
    c = Counter()
    by_cost = []
    term_totals = Counter()
    for row in rows:
        previous = row["quota8_reference"]
        adaptive = row["adaptive"]
        c["target_count"] += 1
        c["released"] += adaptive["quota_released"]
        c["kept_quota8"] += not adaptive["quota_released"]
        c["adaptive_winner_improved_vs_v08"] += (
            adaptive["beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS)
        c["quota8_winner_improved_vs_v08"] += (
            previous["beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS)
        c["adaptive_winner_changed_vs_quota8"] += (
            adaptive["beam_winner"] != previous["beam_winner"])
        c["adaptive_winner_worse_than_quota8"] += (
            adaptive["beam_winner"]["final_cost"]
            > previous["beam_winner"]["final_cost"] + EPS)
        c["adaptive_winner_better_than_quota8"] += (
            adaptive["beam_winner"]["final_cost"]
            < previous["beam_winner"]["final_cost"] - EPS)
        c["adaptive_winner_target_quick_removed"] += any(
            tuple(item["event_pair"]) == tuple(row["flag_pair"])
            for item in adaptive["beam_winner"]["quick"]["removed"])
        c["adaptive_winner_new_quick"] += adaptive[
            "beam_winner"]["quick"]["added_count"]
        c["adaptive_winner_new_rapid"] += adaptive[
            "beam_winner"]["rapid"]["added_count"]
        c["quota8_rescue_windows"] += previous["slot_replacement_count"]
        c["adaptive_rescue_windows"] += adaptive["rescue_window_count"]
        c["quota8_displaced_slots"] += previous[
            "total_displaced_nonprotected_slots"]
        c["adaptive_displaced_slots"] += adaptive["displaced_untagged_slots"]
        by_cost.append(adaptive["beam_winner"]["final_cost"]
                       - previous["beam_winner"]["final_cost"])
        for term, delta in row["diagnostic"][
                "final_term_differences_protected_minus_winner"].items():
            term_totals[term] += delta
    keys = (
        "target_count", "released", "kept_quota8",
        "adaptive_winner_improved_vs_v08",
        "quota8_winner_improved_vs_v08",
        "adaptive_winner_changed_vs_quota8",
        "adaptive_winner_worse_than_quota8",
        "adaptive_winner_better_than_quota8",
        "adaptive_winner_target_quick_removed",
        "adaptive_winner_new_quick", "adaptive_winner_new_rapid",
        "quota8_rescue_windows", "adaptive_rescue_windows",
        "quota8_displaced_slots", "adaptive_displaced_slots",
    )
    output = {k: c[k] for k in keys}
    output["saved_rescue_windows"] = (c["quota8_rescue_windows"]
                                      - c["adaptive_rescue_windows"])
    output["saved_displaced_slots"] = (c["quota8_displaced_slots"]
                                       - c["adaptive_displaced_slots"])
    output["adaptive_minus_quota8_final_cost"] = _stats(by_cost)
    output["sum_best_protected_minus_winner_components"] = {
        term: round(delta, 9) for term, delta in term_totals.items()
    }
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "verified-report", "h0-report",
                  "h01-report", "h06-report", "h08-report",
                  "h010-report", "results-root", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H0.11 output")
    manifest, verified, h0, h01, h06, h08, h010 = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.manifest, args.verified_report,
                     args.h0_report, args.h01_report, args.h06_report,
                     args.h08_report, args.h010_report)]
    artifacts = {}
    for item in manifest["cases"]:
        sha = item["sha256"]
        root = args.results_root / sha[:12]
        summary_file = root / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        summary["_artifact_sha256"] = sha256_file(summary_file)
        def read(name):
            return json.loads(name.read_text(encoding="utf-8"))
        artifacts[sha] = (
            summary, read(root / "telemetry-v08.json"),
            read(root / "replay-v08.json"),
            read(args.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    if checked != h01:
        raise ValueError("saved H0.1 differs from verified v0.8 Train run")
    m06, m08, m010 = _validate_reports(
        checked, manifest, h06, h08, h010, args.h06_report, args.h08_report)
    train_root = Path(manifest["train_root"]).resolve()
    cases = []
    for manifest_case in manifest["cases"]:
        sha = manifest_case["sha256"]
        chart = validate_chart(manifest_case, train_root)
        summary, telemetry, replay, _ = artifacts[sha]
        ordered, chunks, _ = _validate(summary, telemetry, replay)
        events = _oracle_events(chart)
        if (len(ordered) != len(events)
                or any(e.event_index != d["event_index"]
                       or abs(e.due_s*1000 - d["due_ms"]) > .005
                       for e, d in zip(events, ordered))):
            raise ValueError("unrounded Oracle events differ from saved decisions")
        c06, c08, c010 = m06[sha], m08[sha], m010[sha]
        if any(case["event_count"] != len(events)
               for case in (c06, c08, c010)):
            raise ValueError("Train event count differs by phase")
        h06_pairs = {
            tuple(e["flag_pair"]): e for e in c06["evidence"]}
        h010_pairs = {
            tuple(e["flag_pair"]): e for e in c010["targets"]}
        if (len(h06_pairs) != len(c06["evidence"])
                or len(h010_pairs) != len(c010["targets"])
                or set(h06_pairs) != set(h010_pairs)):
            raise ValueError("22-target report pair identities differ")
        windows = GestureChunkPlannerV08()._windows(events)
        rows = []
        for entry in c06["evidence"]:
            pair = tuple(entry["flag_pair"])
            existing = h010_pairs[pair]
            if existing["h06_cohort"] != (
                "previously_improved_7" if entry["improved_final_cost"]
                else "previously_nonimproved_15"):
                raise ValueError("H0.6/H0.10 cohort assignment mismatch")
            target = _candidate_target(entry, windows)
            saved = existing["quota8_result"]
            diagnostic = trace_window_mechanism(events, chunks, target)
            diagnosis = quota8_diagnostic(saved, diagnostic)
            adaptive = trace_adaptive(
                events, chunks, target,
                expected_cost=summary["abstract_schedule"]["metrics"]["beam_cost"],
                expected_evaluated=summary["abstract_schedule"]["candidates_evaluated"],
                reference_final_cost=entry["cost"]["original_v08_final"])
            if adaptive["admission_expanded_rank"] != saved["admission_expanded_rank"]:
                raise ValueError("adaptive trace tagged different candidate")
            if adaptive["ever_protected_rank1_before_decision"] != diagnosis[
                    "ever_rank1_by_grace"] and (
                    diagnosis["grace_chunk_count"] >= OBSERVE_CHUNKS):
                raise ValueError("online decision differs from saved quota8 observation")
            if not adaptive["quota_released"]:
                # With no release the entire adaptive replay MUST be exactly
                # H0.10's quota-8 search, not merely the final cost.
                for key, prior_key in (
                    ("beam_winner", "beam_winner"),
                    ("best_protected_terminal", "best_protected_terminal"),
                    ("rescue_windows", "slot_replacement_windows"),
                    ("displaced_untagged_slots",
                     "total_displaced_nonprotected_slots"),
                ):
                    if adaptive[key] != saved[prior_key]:
                        raise ValueError("unreleased adaptive replay diverges from H0.10")
            rows.append({
                "flag_pair": entry["flag_pair"],
                "h06_cohort": existing["h06_cohort"],
                "target_window": target["window_index"],
                "diagnostic": diagnosis,
                "quota8_reference": saved,
                "adaptive": adaptive,
            })
        if len(rows) != c010["target_count"]:
            raise ValueError("case target count mismatched")
        cases.append({
            "chart": summary["chart"],
            "chart_sha256": sha,
            "event_count": len(events),
            "target_count": len(rows),
            "targets": rows,
        })
    all_rows = [row for case in cases for row in case["targets"]]
    groups = {
        "all_22": all_rows,
        "previously_improved_7": [
            r for r in all_rows if r["h06_cohort"] == "previously_improved_7"],
        "previously_nonimproved_15": [
            r for r in all_rows if r["h06_cohort"] == "previously_nonimproved_15"],
    }
    if tuple(len(groups[x]) for x in groups) != (22, 7, 15):
        raise ValueError("H0.10 22-target split changed")
    aggregate = {name: _aggregate(records)
                 for name, records in groups.items()}
    full = aggregate["all_22"]
    if (full["quota8_rescue_windows"]
            != h010["aggregate"]["all_22"]["rescue_windows"]
            or full["quota8_displaced_slots"]
            != h010["aggregate"]["all_22"]["displaced_untagged_slots"]
            or full["quota8_winner_improved_vs_v08"]
            != h010["aggregate"]["all_22"]["winner_lower_cost"]):
        raise ValueError("saved H0.10 aggregates mismatched")
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "h010_source_sha256": sha256_file(args.h010_report),
        "case_count": len(cases),
        "event_count": sum(case["event_count"] for case in cases),
        "target_count": len(all_rows),
        "quota": QUOTA,
        "beam_width": BEAM_WIDTH,
        "observe_chunks": OBSERVE_CHUNKS,
        "release_rule": (
            "After the fifth join-inclusive chunk, if a protected "
            "candidate has never ranked #1, cease quota reservation "
            "starting with the sixth; keep natural score-sorted top 24 "
            "without forcibly removing tagged candidates."),
        "aggregate": aggregate,
        "cases": cases,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Release trigger was chosen after observing H0.9's seven improving cases; NOT an out-of-sample rule.",
            "The 15 nonimproving cases are also drawn from the same preselected flagged Train charts, not a held-out cohort.",
            "No definitive safety guarantee follows from five observations or positive cumulative score gaps; later reversals are possible.",
            "Each target is an independent counterfactual; aggregate savings do not measure runtime or a combined chart.",
            "The modified beam selection changes competition among candidates, including untagged lineages.",
            "Six objective components are untrained heuristics and do not prove human-like movement or Motor viability.",
            "Physical Motor, Validation20, FINAL and training were not run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "target_count": report["target_count"],
        "aggregate": aggregate,
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
