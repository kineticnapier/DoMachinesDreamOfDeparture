from __future__ import annotations

"""DMDOGO H0.8: compare protected-lineage quotas 1, 2, 4 and 8 inside width 24.

Counterfactual offline abstract search ONLY. Frozen v0.8 source, scoring,
candidate grammar, 80ms reuse, physical Motor and hyperparameters unchanged.
Quota 1 must reproduce the saved H0.7 report before anything is written.
"""

import argparse
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record, _oracle_events
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state, _cost_terms
from audit_dmdogo_v09_h06_improved_suffix import (
    FORMAT as H06_FORMAT, _assignment_metrics, _event_axes,
    _quick_flags, _rapid_flags,
)
from audit_dmdogo_v09_h07_shadow_lane import (
    FORMAT as H07_FORMAT, BEAM_WIDTH, TARGET_TAG,
    Node, BeamPath, Proposal, _chunk_record_from,
    _path_chunks, _event_count_chunks, _delta_flags,
)
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h08-protected-quota-sweep-v1"
QUOTAS = (1, 2, 4, 8)
EPS = 1e-7


def choose_quota(proposals: list[Proposal], *, quota: int,
                 width: int = BEAM_WIDTH) -> tuple[list[Proposal], dict]:
    """Retain min(quota, eligible tagged count) in a width-limited beam.

    First select ordinary top-W, then replace worst-scored *untagged*
    survivors until the desired tagged quota is met. All sorting is
    stable score-only (frozen heuristic, no term/weight adjustments).
    No tagged paths are fabricated if fewer than quota are feasible.
    """
    if not isinstance(quota, int) or not 1 <= quota <= width:
        raise ValueError("quota must be between 1 and width")
    if not isinstance(width, int) or width < 2 or not proposals:
        raise ValueError("invalid beam width/proposals")
    ordered = sorted(proposals, key=lambda p: p.state.score)
    regular = ordered[:width]
    tagged_total = sum(p.protected for p in ordered)
    natural = sum(p.protected for p in regular)
    desired = min(quota, tagged_total, width)
    needed = desired - natural
    best_rank = next((rank for rank, row in enumerate(ordered, 1)
                      if row.protected), None)
    if needed <= 0:
        return regular, {
            "feasible_tagged_expansions": tagged_total,
            "natural_tagged_kept": natural,
            "tagged_kept": natural,
            "requested_quota": quota,
            "requested_quota_satisfied": natural >= quota,
            "desired_tagged_count": desired,
            "displaced_nonprotected_count": 0,
            "best_tagged_expanded_rank": best_rank,
        }
    extras = [row for row in ordered[width:] if row.protected][:needed]
    evict = [row for row in reversed(regular) if not row.protected][:needed]
    if len(extras) != needed or len(evict) != needed:
        raise ValueError("inconsistent feasible tagged quota")
    evicted_ids = {id(row) for row in evict}
    survivors = [row for row in regular if id(row) not in evicted_ids] + extras
    ranks = {id(row): rank for rank, row in enumerate(ordered)}
    survivors.sort(key=lambda row: ranks[id(row)])
    if (len(survivors) != len(regular) or len(survivors) > width
            or sum(row.protected for row in survivors) < desired):
        raise ValueError("quota selection invariant violated")
    return survivors, {
        "feasible_tagged_expansions": tagged_total,
        "natural_tagged_kept": natural,
        "tagged_kept": sum(row.protected for row in survivors),
        "requested_quota": quota,
        "requested_quota_satisfied": sum(row.protected for row in survivors) >= quota,
        "desired_tagged_count": desired,
        "displaced_nonprotected_count": needed,
        "best_tagged_expanded_rank": best_rank,
    }


def _candidate_target(h06_entry, windows) -> dict:
    lineage = h06_entry["beam_lineage"]
    win_id = h06_entry["window_index"]
    pair = h06_entry["flag_pair"]
    if not 0 < win_id < len(windows):
        raise ValueError("invalid H0.6 targeted window")
    if (windows[win_id].events[0].event_index != pair[1]
            or windows[win_id - 1].events[-1].event_index != pair[0]):
        raise ValueError("H0.6 flag pair is not a real frozen seam")
    eligible = [(i, candidate) for i, candidate
                in enumerate(_candidate_gestures(windows[win_id]))
                if [list(a) for a in candidate.axes] == h06_entry["nonquick_axes"]
                and candidate.family == h06_entry["nonquick_gesture_family"]]
    if len(eligible) != 1:
        raise ValueError("H0.6 targeted nonquick candidate not unique")
    return {
        "flag_pair": pair,
        "window_index": win_id,
        "nonquick_candidate_index": eligible[0][0],
        "nonquick_axes": h06_entry["nonquick_axes"],
        "nonquick_join_expanded_rank": lineage["join_expanded_rank"],
    }


def _describe(events, frozen_chunks: list[dict], path: BeamPath | None,
              precise_frozen_cost: float) -> dict | None:
    if path is None:
        return None
    chunks = _path_chunks(path.node)
    if (len(chunks) != len(frozen_chunks)
            or _event_count_chunks(chunks) != len(events)):
        raise ValueError("counterfactual path does not cover frozen chart")
    original_axes = _event_axes(frozen_chunks)
    other_axes = _event_axes(chunks)
    count = sum(original_axes[e.event_index] != other_axes[e.event_index]
                for e in events)
    before_q, after_q = _quick_flags(events, frozen_chunks), _quick_flags(events, chunks)
    before_r, after_r = _rapid_flags(events, frozen_chunks), _rapid_flags(events, chunks)
    metric = _assignment_metrics(events, chunks)
    if metric["abstract_reuse_under_80ms_count"]:
        raise ValueError("abstract reuse violation introduced")
    return {
        "final_cost": round(path.state.score, 9),
        "delta_cost_vs_frozen_v08": round(path.state.score - precise_frozen_cost, 9),
        "is_from_protected_lineage": TARGET_TAG in path.state.tags,
        "changed_event_assignment_count": count,
        "quick": _delta_flags(before_q, after_q),
        "rapid": _delta_flags(before_r, after_r),
        "axis_reuse_under_80ms": metric["abstract_reuse_under_80ms_count"],
        "minimum_axis_reuse_ms": metric["minimum_axis_reuse_ms"],
        "hand_axis_uses": metric["hand_axis_uses"],
        "score_components": _cost_terms(path.state),
    }


def trace_quota(events, frozen_chunks: list[dict], target: dict,
                *, quota: int, expected_cost: float,
                expected_evaluated: int,
                reference_final_cost: float) -> dict:
    """Exact frozen prefix; one tagged target; fixed-width quota after join."""
    if quota not in QUOTAS:
        raise ValueError("unexpected non-predeclared quota")
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    canonical = planner.plan(events)
    if (not canonical.feasible
            or [_chunk_record(chunk) for chunk in canonical.chunks] != frozen_chunks
            or canonical.candidates_evaluated != expected_evaluated
            or canonical.metrics["beam_cost"] != expected_cost
            or round(reference_final_cost, 6) != expected_cost):
        raise ValueError("frozen v0.8 source/beam does not reproduce")

    join = target["window_index"]
    if (not 0 < join < len(windows)
            or windows[join].events[0].event_index != target["flag_pair"][1]
            or windows[join-1].events[-1].event_index != target["flag_pair"][0]):
        raise ValueError("target is not an actual chunk seam")
    alternative_id = target["nonquick_candidate_index"]
    states = [BeamPath(_seed_state(events[0].due_s), None)]
    evaluated_through_join = 0
    admission_rank = None
    rescued_windows = []
    replacement_count = 0
    rescued_slot_counts = []
    underfilled_windows = 0
    protected_dead_window = None
    peak_protected_count = 0
    for win_id, window in enumerate(windows):
        candidates = _candidate_gestures(window)
        previous_due = windows[win_id - 1].events[-1].due_s if win_id else None
        axes = tuple(tuple(x) for x in frozen_chunks[win_id]["axes"])
        family = frozen_chunks[win_id]["gesture_family"]
        expansions: list[Proposal] = []
        for parent_rank, parent in enumerate(states, 1):
            for ci, candidate in enumerate(candidates):
                if win_id <= join:
                    evaluated_through_join += 1
                next_state = _next(
                    parent.state, window, candidate, previous_due,
                    frozen_candidate=(
                        candidate.axes == axes and candidate.family == family),
                )
                if next_state is None:
                    continue
                if (win_id == join and parent.state.frozen_prefix
                        and ci == alternative_id):
                    next_state.tags = frozenset({TARGET_TAG})
                expansions.append(Proposal(
                    next_state, parent.node,
                    _chunk_record_from(window, candidate),
                    parent_rank, ci))
        if not expansions:
            raise ValueError(f"all expansions failed at window {win_id}")
        expansions.sort(key=lambda item: item.state.score)
        if win_id < join:
            keep = expansions[:BEAM_WIDTH]
            if sum(p.state.frozen_prefix for p in keep) != 1:
                raise ValueError("original selected prefix lost before join")
        else:
            if win_id == join:
                tagged = [(rank, p) for rank, p in enumerate(expansions, 1)
                          if p.protected]
                if len(tagged) != 1:
                    raise ValueError("target does not have unique tagged alternative")
                rank, proposal = tagged[0]
                if (rank != target["nonquick_join_expanded_rank"]
                        or proposal.candidate_index != alternative_id
                        or proposal.chunk["axes"] != target["nonquick_axes"]):
                    raise ValueError("original nonquick target/rank changed")
                admission_rank = rank
            keep, selection = choose_quota(expansions, quota=quota)
            retained = sum(p.protected for p in keep)
            peak_protected_count = max(peak_protected_count, retained)
            if retained < quota:
                underfilled_windows += 1
            replaced = selection["displaced_nonprotected_count"]
            if replaced:
                rescued_windows.append(win_id)
                replacement_count += replaced
                rescued_slot_counts.append(replaced)
            if not selection["feasible_tagged_expansions"] and protected_dead_window is None:
                protected_dead_window = win_id
        states = [BeamPath(p.state, Node(p.parent, p.chunk)) for p in keep]
    tagged_terminal = [(i, p) for i, p in enumerate(states, 1)
                       if TARGET_TAG in p.state.tags]
    if protected_dead_window is None and not tagged_terminal:
        raise ValueError("protected branch disappeared without recorded dead end")
    best_tagged = tagged_terminal[0][1] if tagged_terminal else None
    winner = states[0]
    return {
        "flag_pair": target["flag_pair"],
        "window_index": join,
        "admission_expanded_rank": admission_rank,
        "admitted_within_natural_top24": admission_rank <= BEAM_WIDTH,
        "total_candidates_evaluated_before_and_at_join": evaluated_through_join,
        "protected_descendants_reached_terminal": best_tagged is not None,
        "protected_dead_window": protected_dead_window,
        "slot_replacement_count": len(rescued_windows),
        "slot_replacement_windows": rescued_windows,
        "best_protected_terminal_rank": (
            tagged_terminal[0][0] if tagged_terminal else None),
        "beam_winner": _describe(
            events, frozen_chunks, winner, reference_final_cost),
        "best_protected_terminal": _describe(
            events, frozen_chunks, best_tagged, reference_final_cost),
        "width": BEAM_WIDTH,
        "physical_motor_executed": False,
        "quota": quota,
        "total_displaced_nonprotected_slots": replacement_count,
        "maximum_displaced_in_one_window": max(rescued_slot_counts, default=0),
        "peak_retained_tagged_hypotheses": peak_protected_count,
        "windows_with_tagged_count_below_quota": underfilled_windows,
        "note": "Tagged hypotheses share one original candidate at admission; quota only adds diversity in later descendants.",
    }


def _h07_projection(row: dict) -> dict:
    fields = (
        "flag_pair", "window_index", "admission_expanded_rank",
        "admitted_within_natural_top24",
        "total_candidates_evaluated_before_and_at_join",
        "protected_descendants_reached_terminal", "protected_dead_window",
        "slot_replacement_count", "slot_replacement_windows",
        "best_protected_terminal_rank", "beam_winner",
        "best_protected_terminal", "width", "physical_motor_executed",
    )
    return {name: row[name] for name in fields}


def audit_case(chart: Path, summary: dict, telemetry: dict,
               replay: dict, h06_case: dict,
               h07_case: dict) -> dict:
    rows, frozen_chunks, _ = _validate(summary, telemetry, replay)
    events = _oracle_events(chart)
    if (len(events) != len(rows)
            or any(e.event_index != row["event_index"]
                   or abs(e.due_s * 1000 - row["due_ms"]) > .005
                   for e, row in zip(events, rows))):
        raise ValueError("Oracle events mismatch saved v0.8 decisions")
    if (summary["chart_sha256"] != h06_case["chart_sha256"]
            or h06_case["chart_sha256"] != h07_case["chart_sha256"]
            or len(events) != h06_case["event_count"]
            or len(events) != h07_case["event_count"]):
        raise ValueError("source Train case identity differs across phases")
    h07_by_pair = {
        tuple(x["flag_pair"]): x for x in h07_case["targets"]
    }
    if len(h07_by_pair) != len(h07_case["targets"]):
        raise ValueError("duplicate H0.7 target")
    outputs = []
    for entry in h06_case["evidence"]:
        if not entry["improved_final_cost"]:
            continue
        target = _candidate_target(entry, GestureChunkPlannerV08()._windows(events))
        pair = tuple(entry["flag_pair"])
        reference = h07_by_pair.get(pair)
        if reference is None:
            raise ValueError("missing H0.7 control for preselected H0.6 target")
        variants = {}
        for quota in QUOTAS:
            result = trace_quota(
                events, frozen_chunks, target, quota=quota,
                expected_cost=summary["abstract_schedule"]["metrics"]["beam_cost"],
                expected_evaluated=summary["abstract_schedule"]["candidates_evaluated"],
                reference_final_cost=entry["cost"]["original_v08_final"])
            if quota == 1:
                if _h07_projection(result) != _h07_projection(reference):
                    raise ValueError(
                        "H0.8 single-slot control disagrees with saved H0.7; "
                        "stop without writing findings")
            variants[str(quota)] = result
        outputs.append({
            "flag_pair": entry["flag_pair"],
            "h06_forced_suffix_final_delta": entry["cost"]["final_delta"],
            "variants": variants,
        })
    if len(outputs) != h06_case["improved_count"]:
        raise ValueError("H0.6 improved target cohort changed")
    if len(outputs) != len(h07_case["targets"]):
        raise ValueError("H0.7 selected target cohort changed")
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(events),
        "target_count": len(outputs),
        "targets": outputs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "verified-report", "h0-report", "h01-report",
                "h06-report", "h07-report", "results-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite prior H0.8 result")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    manifest, verified, h0, h01, h06, h07 = [
        load(p) for p in (args.manifest, args.verified_report, args.h0_report,
                         args.h01_report, args.h06_report, args.h07_report)]
    artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        root = args.results_root / sha[:12]
        summary_path = root / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary, load(root / "telemetry-v08.json"),
            load(root / "replay-v08.json"),
            load(args.results_root / "_provenance" / f"{sha[:12]}.json"))
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    if checked != h01:
        raise ValueError("H0.1 did not reproduce source Train artifacts")
    if (h06.get("format") != H06_FORMAT
            or h07.get("format") != H07_FORMAT
            or h06.get("source_v08_commit") != checked["source_v08_commit"]
            or h07.get("source_v08_commit") != checked["source_v08_commit"]
            or h06.get("frozen_code_digest") != checked["frozen_code_digest"]
            or h07.get("frozen_code_digest") != checked["frozen_code_digest"]
            or h06.get("event_count") != checked["aggregate"]["event_count"]
            or h07.get("event_count") != checked["aggregate"]["event_count"]
            or h06.get("case_count") != len(manifest["cases"])
            or h07.get("case_count") != len(manifest["cases"])):
        raise ValueError("H0.6/H0.7 report provenance mismatch")
    by06 = {c["chart_sha256"]: c for c in h06["cases"]}
    by07 = {c["chart_sha256"]: c for c in h07["cases"]}
    required = {c["sha256"] for c in manifest["cases"]}
    if (set(by06) != required or set(by07) != required
            or len(by06) != len(h06["cases"])
            or len(by07) != len(h07["cases"])):
        raise ValueError("H0.6/H0.7 source case selection changed")
    root = Path(manifest["train_root"]).resolve()
    records = []
    for case in manifest["cases"]:
        sha = case["sha256"]
        chart = validate_chart(case, root)
        records.append(audit_case(
            chart, *artifacts[sha][:3], by06[sha], by07[sha]))
    rows = [x for c in records for x in c["targets"]]
    if len(rows) != h06["aggregate"]["improved_final_cost_count"]:
        raise ValueError("H0.8 case selection count changed")
    summary_by_quota = {}
    for quota in QUOTAS:
        variants = [x["variants"][str(quota)] for x in rows]
        summary_by_quota[str(quota)] = {
            "quota": quota, "beam_width": BEAM_WIDTH,
            "target_count": len(variants),
            "protected_terminal_survivors": sum(
                x["protected_descendants_reached_terminal"] for x in variants),
            "protected_lower_final_cost_than_v08": sum(
                x["best_protected_terminal"] is not None
                and x["best_protected_terminal"]["delta_cost_vs_frozen_v08"] < -EPS
                for x in variants),
            "beam_winner_lower_final_cost_than_v08": sum(
                x["beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS
                for x in variants),
            "beam_winner_from_protected_lineage": sum(
                x["beam_winner"]["is_from_protected_lineage"] for x in variants),
            "total_rescue_windows_across_independent_runs": sum(
                x["slot_replacement_count"] for x in variants),
            "total_displaced_nonprotected_slots_across_independent_runs": sum(
                x["total_displaced_nonprotected_slots"] for x in variants),
        }
    baseline = summary_by_quota["1"]
    check = h07["aggregate"]
    for name, source in (
        ("target_count", "targeted_improved_h06_cases"),
        ("protected_terminal_survivors", "protected_reached_terminal"),
        ("protected_lower_final_cost_than_v08", "protected_better_than_original"),
        ("beam_winner_lower_final_cost_than_v08", "beam_winner_better_than_original"),
        ("beam_winner_from_protected_lineage", "beam_winner_from_protected_lineage"),
        ("total_rescue_windows_across_independent_runs",
         "total_rescue_slot_replacements_across_separate_runs"),
    ):
        if baseline[name] != check[source]:
            raise ValueError(f"H0.7 aggregate mismatch for {name}")
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "event_count": sum(c["event_count"] for c in records),
        "case_count": len(records),
        "quotas": list(QUOTAS),
        "aggregate": summary_by_quota,
        "cases": records,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "hyperparameters_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Quota protection changes admission/pruning, not the original frozen beam.",
            "One tagged ancestor branches into distinct descendants only after the targeted seam.",
            "H0.6 selected seven improving cases post hoc; no out-of-sample inference.",
            "Results count seven independent experiments, not a combined chart improvement.",
            "A lower heuristic score is not physical Motor success or human-likeness.",
            "No validation/FINAL, physical Motor or learned data evaluated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "aggregate": report["aggregate"],
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
