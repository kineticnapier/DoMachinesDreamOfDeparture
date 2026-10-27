from __future__ import annotations

"""H0.9: trace why width-24, quota-8 protected search works on seven H0.6 cases.

Read-only recomputation against exact saved H0.8 per-target quota-8 rows.
H0.8 quota-1 must still match H0.7. No Planner/Motor/Timing changes.
All observations describe heuristic search populations, NOT human skill.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_v09_h04_beam_decisions import _oracle_events
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h06_improved_suffix import FORMAT as H06_FORMAT
from audit_dmdogo_v09_h07_shadow_lane import FORMAT as H07_FORMAT
from audit_dmdogo_v09_h08_protected_quota import (
    FORMAT as H08_FORMAT, QUOTAS, BEAM_WIDTH, TARGET_TAG,
    _candidate_target, _h07_projection, choose_quota, trace_quota,
)
from audit_dmdogo_v09_h07_shadow_lane import Proposal, _chunk_record_from
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h09-quota-mechanism-v1"
TARGET_QUOTA = 8
EPS = 1e-7


def _first(values: list[dict], predicate):
    return next((row["window_index"] for row in values if predicate(row)), None)


def _permanent_from(values: list[dict], predicate):
    """First step after which predicate is continuously true until terminal."""
    if not values:
        return None
    first_stable = None
    for row in reversed(values):
        if predicate(row):
            first_stable = row["window_index"]
        else:
            break
    return first_stable


def summarize_windows(records: list[dict], quota: int = TARGET_QUOTA) -> dict:
    if not records:
        raise ValueError("no post-join windows")
    expected_indices = range(records[0]["window_index"],
                             records[0]["window_index"] + len(records))
    if [r["window_index"] for r in records] != list(expected_indices):
        raise ValueError("window trace is not contiguous")
    for row in records:
        if not 0 <= row["natural_tagged_count"] <= BEAM_WIDTH:
            raise ValueError("invalid natural tagged count")
        if not 0 <= row["retained_tagged_count"] <= BEAM_WIDTH:
            raise ValueError("invalid retained tagged count")
        if row["replaced_slots"] > row["retained_tagged_count"]:
            raise ValueError("impossible protected replacement")
    return {
        "windows_after_and_including_join": len(records),
        "rescue_windows": sum(r["replaced_slots"] > 0 for r in records),
        "displaced_untagged_slots": sum(r["replaced_slots"] for r in records),
        "quota_underfilled_windows": sum(
            r["retained_tagged_count"] < quota for r in records),
        "natural_at_least_quota_windows": sum(
            r["natural_tagged_count"] >= quota for r in records),
        "natural_zero_tagged_windows": sum(
            r["natural_tagged_count"] == 0 for r in records),
        "natural_at_least_one_tagged_windows": sum(
            r["natural_tagged_count"] >= 1 for r in records),
        "retained_tagged_peak": max(r["retained_tagged_count"] for r in records),
        "natural_tagged_peak": max(r["natural_tagged_count"] for r in records),
        "first_natural_quota_window": _first(
            records, lambda r: r["natural_tagged_count"] >= quota),
        "first_permanently_natural_quota_window": _permanent_from(
            records, lambda r: r["natural_tagged_count"] >= quota),
        "first_tagged_rank_one_window": _first(
            records, lambda r: r["best_tagged_expanded_rank"] == 1),
        "first_tagged_outscores_untagged_window": _first(
            records, lambda r: r["best_tagged_minus_best_untagged"] is not None
            and r["best_tagged_minus_best_untagged"] < -EPS),
        "first_no_untagged_feasible_window": _first(
            records, lambda r: r["feasible_untagged_count"] == 0),
        "first_natural_top24_contains_tagged_window": _first(
            records, lambda r: r["natural_tagged_count"] >= 1),
    }


def trace_window_mechanism(events, frozen_chunks: list[dict], target: dict,
                           *, quota: int = TARGET_QUOTA) -> dict:
    """Second independent trace, same H0.8 state/order, with pre-cut counts.

    This is not a new intervention. Returned diagnostics are checked against
    the exact stored/recomputed H0.8 quota-8 result before reporting.
    """
    if quota not in QUOTAS:
        raise ValueError("undeclared quota")
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    join = target["window_index"]
    if not 0 < join < len(windows) or len(windows) != len(frozen_chunks):
        raise ValueError("window segmentation invalid")
    if (windows[join].events[0].event_index != target["flag_pair"][1]
            or windows[join - 1].events[-1].event_index != target["flag_pair"][0]):
        raise ValueError("target seam differs from source")
    states = [_seed_state(events[0].due_s)]
    collected = []
    admission_rank = None
    for wi, window in enumerate(windows):
        candidates = _candidate_gestures(window)
        previous_due = windows[wi - 1].events[-1].due_s if wi else None
        frozen_axes = tuple(tuple(x) for x in frozen_chunks[wi]["axes"])
        frozen_family = frozen_chunks[wi]["gesture_family"]
        expansions = []
        for parent_rank, state in enumerate(states, 1):
            for candidate_index, candidate in enumerate(candidates):
                nxt = _next(
                    state, window, candidate, previous_due,
                    frozen_candidate=(
                        candidate.axes == frozen_axes
                        and candidate.family == frozen_family),
                )
                if nxt is None:
                    continue
                if (wi == join and state.frozen_prefix
                        and candidate_index == target["nonquick_candidate_index"]):
                    nxt.tags = frozenset({TARGET_TAG})
                expansions.append(Proposal(
                    nxt, None, _chunk_record_from(window, candidate),
                    parent_rank, candidate_index))
        if not expansions:
            raise ValueError(f"all abstract candidates failed at {wi}")
        expansions.sort(key=lambda p: p.state.score)
        if wi < join:
            keep = expansions[:BEAM_WIDTH]
            if sum(p.state.frozen_prefix for p in keep) != 1:
                raise ValueError("frozen v0.8 selected predecessor diverged")
        else:
            if wi == join:
                chosen = [(i, p) for i, p in enumerate(expansions, 1)
                          if p.protected]
                if len(chosen) != 1:
                    raise ValueError("tagged alternative missing/duplicated")
                admission_rank, chosen_p = chosen[0]
                if (admission_rank != target["nonquick_join_expanded_rank"]
                        or chosen_p.candidate_index != target["nonquick_candidate_index"]
                        or chosen_p.chunk["axes"] != target["nonquick_axes"]):
                    raise ValueError("original H0.6 alternative mismatch")
            natural = expansions[:BEAM_WIDTH]
            best_tagged_rank = next(
                (i for i, p in enumerate(expansions, 1) if p.protected), None)
            best_untagged = next(
                (p for p in expansions if not p.protected), None)
            best_tagged_score = (
                expansions[best_tagged_rank - 1].state.score
                if best_tagged_rank is not None else None)
            best_untagged_score = (
                best_untagged.state.score if best_untagged is not None else None)
            keep, status = choose_quota(expansions, quota=quota)
            live_rank = next((i for i, p in enumerate(keep, 1)
                              if p.protected), None)
            collected.append({
                "window_index": wi,
                "event_range": [window.events[0].event_index,
                                window.events[-1].event_index],
                "feasible_expansions": len(expansions),
                "feasible_tagged_count": status["feasible_tagged_expansions"],
                "feasible_untagged_count": (
                    len(expansions) - status["feasible_tagged_expansions"]),
                "natural_tagged_count": status["natural_tagged_kept"],
                "retained_tagged_count": status["tagged_kept"],
                "replaced_slots": status["displaced_nonprotected_count"],
                "best_tagged_expanded_rank": best_tagged_rank,
                "best_tagged_retained_rank": live_rank,
                "best_untagged_score": (
                    round(best_untagged_score, 9)
                    if best_untagged_score is not None else None),
                "best_tagged_score": (
                    round(best_tagged_score, 9)
                    if best_tagged_score is not None else None),
                "best_tagged_minus_best_untagged": (
                    round(best_tagged_score - best_untagged_score, 9)
                    if best_tagged_score is not None
                    and best_untagged_score is not None else None),
                "natural_cutoff_cost": round(natural[-1].state.score, 9),
                "was_naturally_in_top24": (
                    best_tagged_rank is not None
                    and best_tagged_rank <= BEAM_WIDTH),
                "protected_is_current_winner": bool(
                    keep and keep[0].protected),
            })
        states = [p.state for p in keep]

    tagged = [p for p in states if p.tags]
    if not tagged:
        raise ValueError("H0.8 quota trace lost every tagged hypothesis")
    if admission_rank is None:
        raise ValueError("no join recorded")
    return {
        "flag_pair": target["flag_pair"],
        "quota": quota,
        "admission_rank": admission_rank,
        "terminal_best_cost": round(states[0].score, 9),
        "terminal_best_is_tagged": bool(states[0].tags),
        "terminal_best_tagged_cost": round(tagged[0].score, 9),
        "terminal_best_tagged_rank": next(
            i for i, state in enumerate(states, 1) if state.tags),
        "summary": summarize_windows(collected, quota),
        "windows": collected,
    }


def compare_exact_h08(trace: dict, saved_variant: dict) -> None:
    """Fail closed if a separately replayed trace diverges from the H0.8 path."""
    summary = trace["summary"]
    if (trace["admission_rank"] != saved_variant["admission_expanded_rank"]
            or trace["terminal_best_cost"]
            != saved_variant["beam_winner"]["final_cost"]
            or trace["terminal_best_is_tagged"]
            != saved_variant["beam_winner"]["is_from_protected_lineage"]
            or trace["terminal_best_tagged_cost"]
            != saved_variant["best_protected_terminal"]["final_cost"]
            or trace["terminal_best_tagged_rank"]
            != saved_variant["best_protected_terminal_rank"]
            or summary["rescue_windows"] != saved_variant["slot_replacement_count"]
            or summary["displaced_untagged_slots"]
            != saved_variant["total_displaced_nonprotected_slots"]
            or summary["quota_underfilled_windows"]
            != saved_variant["windows_with_tagged_count_below_quota"]
            or summary["retained_tagged_peak"]
            != saved_variant["peak_retained_tagged_hypotheses"]
            or [r["window_index"] for r in trace["windows"]
                if r["replaced_slots"] > 0]
            != saved_variant["slot_replacement_windows"]):
        raise ValueError("window diagnostics diverge from exact H0.8 result")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "verified-report", "h0-report", "h01-report",
                 "h06-report", "h07-report", "h08-report", "results-root",
                 "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H0.9 result")
    manifest, verified, h0, h01, h06, h07, h08 = [
        _load(x) for x in (
            args.manifest, args.verified_report, args.h0_report,
            args.h01_report, args.h06_report, args.h07_report,
            args.h08_report)]
    if (h06.get("format") != H06_FORMAT
            or h07.get("format") != H07_FORMAT
            or h08.get("format") != H08_FORMAT
            or h08.get("quotas") != list(QUOTAS)):
        raise ValueError("unexpected saved phase/report format")
    artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        root = args.results_root / sha[:12]
        summary_path = root / "summary.json"
        summary = _load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary, _load(root / "telemetry-v08.json"),
            _load(root / "replay-v08.json"),
            _load(args.results_root / "_provenance" / f"{sha[:12]}.json"))
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    if checked != h01:
        raise ValueError("H0.1 baseline provenance/observations mismatch")
    for phase in (h06, h07, h08):
        if (phase.get("source_v08_commit") != checked["source_v08_commit"]
                or phase.get("frozen_code_digest") != checked["frozen_code_digest"]
                or phase.get("event_count") != checked["aggregate"]["event_count"]
                or phase.get("case_count") != len(manifest["cases"])):
            raise ValueError("source v0.8 identity mismatch")
    required = {case["sha256"] for case in manifest["cases"]}
    maps = [{case["chart_sha256"]: case for case in phase["cases"]}
            for phase in (h06, h07, h08)]
    if any(len(m) != len(phase["cases"]) or set(m) != required
           for m, phase in zip(maps, (h06, h07, h08))):
        raise ValueError("H0.6–H0.8 case identity mismatch")
    cases_out = []
    train_root = Path(manifest["train_root"]).resolve()
    for case in manifest["cases"]:
        sha = case["sha256"]
        chart = validate_chart(case, train_root)
        summary, telemetry, replay, _ = artifacts[sha]
        rows, chunks, _ = _validate(summary, telemetry, replay)
        events = _oracle_events(chart)
        if (len(events) != len(rows)
                or any(e.event_index != r["event_index"]
                       or abs(e.due_s * 1000 - r["due_ms"]) > .005
                       for e, r in zip(events, rows))):
            raise ValueError("original Oracle events did not recompile")
        h06case, h07case, h08case = [m[sha] for m in maps]
        if (h06case["event_count"] != len(events)
                or h07case["event_count"] != len(events)
                or h08case["event_count"] != len(events)):
            raise ValueError("case event counts changed")
        w = GestureChunkPlannerV08()._windows(events)
        h07by = {tuple(row["flag_pair"]): row for row in h07case["targets"]}
        h08by = {tuple(row["flag_pair"]): row for row in h08case["targets"]}
        if (len(h07by) != len(h07case["targets"])
                or len(h08by) != len(h08case["targets"])):
            raise ValueError("duplicate source targets")
        case_out = []
        for target in h06case["evidence"]:
            if not target["improved_final_cost"]:
                continue
            key = tuple(target["flag_pair"])
            saved07, saved08 = h07by.get(key), h08by.get(key)
            if saved07 is None or saved08 is None:
                raise ValueError("missing H0.7/H0.8 target in improved cohort")
            # H0.8 is a self-contained, strict control of H0.7 quota=1.
            if (_h07_projection(saved08["variants"]["1"])
                    != _h07_projection(saved07)):
                raise ValueError("saved H0.8 quota1 no longer matches H0.7")
            original = _candidate_target(target, w)
            if saved08["h06_forced_suffix_final_delta"] != target["cost"]["final_delta"]:
                raise ValueError("saved H0.8 forced suffix baseline mismatch")
            # Independently recompute quota 8 (exact saved path, cost,
            # flags and axis-reuse checks) before accepting window diagnostics.
            source08 = trace_quota(
                events, chunks, original, quota=TARGET_QUOTA,
                expected_cost=summary["abstract_schedule"]["metrics"]["beam_cost"],
                expected_evaluated=summary["abstract_schedule"]["candidates_evaluated"],
                reference_final_cost=target["cost"]["original_v08_final"])
            if source08 != saved08["variants"][str(TARGET_QUOTA)]:
                raise ValueError("saved H0.8 quota8 is not reproduced")
            diag = trace_window_mechanism(events, chunks, original)
            compare_exact_h08(diag, source08)
            winner = source08["beam_winner"]
            tagged = source08["best_protected_terminal"]
            if (winner is None or tagged is None
                    or not tagged["is_from_protected_lineage"]):
                raise ValueError("quota8 selected survivor is missing")
            case_out.append({
                "flag_pair": target["flag_pair"],
                "window_index": target["window_index"],
                "h06_forced_delta": target["cost"]["final_delta"],
                "h08_quota8_winner_delta": winner["delta_cost_vs_frozen_v08"],
                "h08_quota8_tagged_delta": tagged["delta_cost_vs_frozen_v08"],
                "h08_quota8_winner_tagged": winner["is_from_protected_lineage"],
                "new_quick_locations": winner["quick"]["added"],
                "removed_quick_locations": winner["quick"]["removed"],
                "new_rapid_locations": winner["rapid"]["added"],
                "removed_rapid_locations": winner["rapid"]["removed"],
                "changed_event_assignment_count": winner["changed_event_assignment_count"],
                "minimum_abstract_reuse_ms": winner["minimum_axis_reuse_ms"],
                "diagnostics": diag,
            })
        if len(case_out) != h06case["improved_count"]:
            raise ValueError("H0.6 seven-case selection changed")
        if len(case_out) != len(h08case["targets"]):
            raise ValueError("H0.8 target coverage changed")
        cases_out.append({
            "chart": summary["chart"],
            "chart_sha256": sha,
            "event_count": len(events),
            "target_count": len(case_out),
            "targets": case_out,
        })
    targets = [t for case in cases_out for t in case["targets"]]
    if len(targets) != 7:
        raise ValueError("expected seven predeclared improving targets")
    totals = Counter()
    for target in targets:
        s = target["diagnostics"]["summary"]
        totals["rescue_windows"] += s["rescue_windows"]
        totals["displaced_slots"] += s["displaced_untagged_slots"]
        totals["natural_at_least_quota_windows"] += s["natural_at_least_quota_windows"]
        totals["quota_underfilled_windows"] += s["quota_underfilled_windows"]
        totals["winner_from_protected_lineage"] += target["h08_quota8_winner_tagged"]
        totals["better_final_cost"] += target["h08_quota8_winner_delta"] < -EPS
        totals["new_quick_flags_across_independent_runs"] += len(
            target["new_quick_locations"])
        totals["removed_quick_flags_across_independent_runs"] += len(
            target["removed_quick_locations"])
        totals["new_rapid_flags_across_independent_runs"] += len(
            target["new_rapid_locations"])
        totals["removed_rapid_flags_across_independent_runs"] += len(
            target["removed_rapid_locations"])
    expected08 = h08["aggregate"]["8"]
    if (totals["rescue_windows"]
            != expected08["total_rescue_windows_across_independent_runs"]
            or totals["displaced_slots"]
            != expected08["total_displaced_nonprotected_slots_across_independent_runs"]
            or totals["winner_from_protected_lineage"]
            != expected08["beam_winner_from_protected_lineage"]
            or totals["better_final_cost"]
            != expected08["beam_winner_lower_final_cost_than_v08"]):
        raise ValueError("H0.9 aggregate differs from saved H0.8")
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "source_h08_sha256": sha256_file(args.h08_report),
        "event_count": sum(c["event_count"] for c in cases_out),
        "case_count": len(cases_out),
        "quota": TARGET_QUOTA,
        "beam_width": BEAM_WIDTH,
        "targeted_cases": len(targets),
        "aggregate": dict(totals),
        "cases": cases_out,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Seven targets were selected after H0.6 improvements; results are conditional.",
            "Window counts are across separate per-target runs and must not be interpreted as simultaneous chart changes.",
            "Natural tagged retention is measured inside the intervention-altered beam population, not the original unrestricted v0.8 beam.",
            "First cost-rank reversal is descriptive, not the causal origin of later outcomes.",
            "New quick/rapid flags are unvalidated style-review proxies, not proof of ergonomic harm.",
            "Only the untrained heuristic and abstract 80ms reuse rule are tested; physical Motor, Validation20 and FINAL remain untouched.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "targeted_cases": report["targeted_cases"],
        "aggregate": report["aggregate"],
        "first_natural_quota_windows": [
            t["diagnostics"]["summary"]["first_natural_quota_window"]
            for t in targets],
        "first_tagged_rank_one_windows": [
            t["diagnostics"]["summary"]["first_tagged_rank_one_window"]
            for t in targets],
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
