from __future__ import annotations

"""H0.6: explain the 22 H0.5 forced non-quick suffixes, especially 7 gains.

This is a read-only, abstract-planner audit. It reconstructs the frozen v0.8
beam, retains the forced suffix path that H0.5 did not serialize, and compares
that path with the exact saved v0.8 assignment. It does not execute Motor,
change planner weights, or infer human preference.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from statistics import mean, median

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunk,
    GestureChunkPlannerV08,
    GestureEvent,
    _axis_parts,
    _candidate_gestures,
)
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from audit_dmdogo_v09_h04_beam_decisions import (
    FORMAT as H04_FORMAT,
    TERMS,
    _chunk_record,
    _oracle_events,
    _quick_at_join,
)
from audit_dmdogo_v09_h05_nonquick_suffix import (
    FORMAT as H05_FORMAT,
    Branch,
    _cost_terms,
    _next,
    audit_case as audit_case_h05,
    trace_unrestricted,
)
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart


FORMAT = "dmdogo-v09-h06-improved-suffix-attribution-v1"
EPSILON = 1e-7


@dataclass(slots=True)
class _Path:
    state: Branch
    chunks: tuple[dict, ...]
    candidate_indices: tuple[int, ...]


def _annotation(window, candidate) -> dict:
    return _chunk_record(GestureChunk(
        window.events[0].event_index,
        window.events[-1].event_index,
        len(window.events),
        window.rhythm,
        window.reason,
        candidate.family,
        candidate.axes,
    ))


def _force_suffix_detailed(seed: Branch, windows, start_window: int,
                           start_candidate, *, beam_width: int = 24) -> dict:
    """Repeat H0.5's frozen forced search while retaining its winning path."""
    states = [_Path(seed, (_annotation(windows[start_window], start_candidate),),
                    (next(i for i, item in enumerate(
                        _candidate_gestures(windows[start_window]))
                          if item == start_candidate),))]
    counted = 0
    per_window = []
    for win_id in range(start_window + 1, len(windows)):
        window = windows[win_id]
        previous_due = windows[win_id - 1].events[-1].due_s
        candidates = _candidate_gestures(window)
        expanded = []
        for path in states:
            for candidate_index, candidate in enumerate(candidates):
                counted += 1
                branch = _next(path.state, window, candidate, previous_due,
                               frozen_candidate=False)
                if branch is not None:
                    expanded.append(_Path(
                        branch,
                        path.chunks + (_annotation(window, candidate),),
                        path.candidate_indices + (candidate_index,),
                    ))
        if not expanded:
            return {
                "reached_terminal": False,
                "failure_reason": "no_reuse_safe_beam_continuation",
                "first_failed_chunk_index": win_id,
                "suffix_candidates_evaluated": counted,
                "final_cost": None,
                "final_cost_terms": None,
                "chunks": [],
                "candidate_indices": [],
                "search_windows": per_window,
            }
        expanded.sort(key=lambda item: item.state.score)
        states = expanded[:beam_width]
        per_window.append({
            "window_index": win_id,
            "feasible_expansions": len(expanded),
            "kept": len(states),
            "best_cost": round(states[0].state.score, 9),
            "cutoff_cost": round(states[-1].state.score, 9),
        })
    best = states[0]
    return {
        "reached_terminal": True,
        "failure_reason": None,
        "first_failed_chunk_index": None,
        "suffix_candidates_evaluated": counted,
        "final_cost": round(best.state.score, 9),
        "final_cost_terms": _cost_terms(best.state),
        "chunks": list(best.chunks),
        "candidate_indices": list(best.candidate_indices),
        "search_windows": per_window,
    }


def _find_frozen_candidate(window, chunk: dict):
    axes = tuple(tuple(row) for row in chunk["axes"])
    matches = [candidate for candidate in _candidate_gestures(window)
               if candidate.axes == axes
               and candidate.family == chunk["gesture_family"]]
    if len(matches) != 1:
        raise ValueError("saved v0.8 chunk is not a unique frozen candidate")
    return matches[0]


def _fixed_original_suffix(seed: Branch, windows, frozen_chunks: list[dict],
                           start_window: int) -> dict:
    """Force one seam, then keep every later v0.8 chunk unchanged."""
    state = seed
    for win_id in range(start_window + 1, len(windows)):
        candidate = _find_frozen_candidate(windows[win_id],
                                           frozen_chunks[win_id])
        state = _next(state, windows[win_id], candidate,
                      windows[win_id - 1].events[-1].due_s,
                      frozen_candidate=True)
        if state is None:
            return {
                "reached_terminal": False,
                "first_failed_chunk_index": win_id,
                "failure_reason": "original_suffix_abstract_reuse_infeasible",
                "final_cost": None,
                "final_cost_terms": None,
            }
    return {
        "reached_terminal": True,
        "first_failed_chunk_index": None,
        "failure_reason": None,
        "final_cost": round(state.score, 9),
        "final_cost_terms": _cost_terms(state),
    }


def _event_axes(chunks: list[dict]) -> dict[int, tuple[str, ...]]:
    result = {}
    for chunk in chunks:
        expected = chunk["end_event_index"] - chunk["start_event_index"] + 1
        if len(chunk["axes"]) != expected:
            raise ValueError("chunk event/axis cardinality mismatch")
        for offset, axes in enumerate(chunk["axes"]):
            event_index = chunk["start_event_index"] + offset
            if event_index in result:
                raise ValueError("duplicate chunk event assignment")
            result[event_index] = tuple(axes)
    return result


def _quick_flags(events: tuple[GestureEvent, ...], chunks: list[dict]) -> dict:
    by_event = _event_axes(chunks)
    result = {}
    for left, right in zip(events, events[1:]):
        a, b = by_event[left.event_index], by_event[right.event_index]
        gap = right.due_s - left.due_s
        if len(a) == len(b) == 1 and _quick_at_join(a[0], b[0], gap):
            result[(left.event_index, right.event_index)] = {
                "event_pair": [left.event_index, right.event_index],
                "ioi_ms": round(gap * 1000., 6),
                "from_axis": a[0],
                "to_axis": b[0],
                "depth_distance": abs(_axis_parts(a[0])[1]
                                      - _axis_parts(b[0])[1]),
            }
    return result


def _rapid_flags(events: tuple[GestureEvent, ...], chunks: list[dict]) -> dict:
    by_event = _event_axes(chunks)
    runs = []
    run = [events[0]]
    for event in events[1:]:
        previous = run[-1]
        left, right = by_event[previous.event_index], by_event[event.event_index]
        gap = event.due_s - previous.due_s
        same_hand = (
            len(left) == len(right) == 1
            and _axis_parts(left[0])[0] == _axis_parts(right[0])[0]
        )
        if same_hand and 0 <= gap <= .120:
            run.append(event)
        else:
            if len(run) >= 4:
                runs.append(run)
            run = [event]
    if len(run) >= 4:
        runs.append(run)
    return {
        (run[0].event_index, run[-1].event_index): {
            "event_range": [run[0].event_index, run[-1].event_index],
            "event_count": len(run),
            "axes": [list(by_event[event.event_index]) for event in run],
            "ioi_ms": [round((b.due_s - a.due_s) * 1000., 6)
                       for a, b in zip(run, run[1:])],
        }
        for run in runs
    }


def _distribution(values) -> dict:
    rows = sorted(float(value) for value in values)
    if not rows:
        return {"count": 0, "min": None, "median": None,
                "mean": None, "max": None}
    return {
        "count": len(rows),
        "min": round(rows[0], 9),
        "median": round(median(rows), 9),
        "mean": round(mean(rows), 9),
        "max": round(rows[-1], 9),
    }


def _assignment_metrics(events: tuple[GestureEvent, ...],
                        chunks: list[dict]) -> dict:
    by_event = _event_axes(chunks)
    due = {event.event_index: event.due_s for event in events}
    axis_counts = Counter()
    hand_counts = Counter()
    prior = {}
    reuse = []
    pure_hands = []
    for event in events:
        axes = by_event[event.event_index]
        hands = {_axis_parts(axis)[0] for axis in axes}
        pure_hands.append(next(iter(hands)) if len(hands) == 1 else "mixed")
        for axis in axes:
            axis_counts[axis] += 1
            hand_counts[_axis_parts(axis)[0]] += 1
            if axis in prior:
                reuse.append((due[event.event_index] - prior[axis]) * 1000.)
            prior[axis] = due[event.event_index]
    longest = 0
    current = 0
    last = None
    for hand in pure_hands:
        if hand != "mixed" and hand == last:
            current += 1
        elif hand != "mixed":
            last, current = hand, 1
        else:
            last, current = None, 0
        longest = max(longest, current)
    hand_switches = sum(
        a != b for a, b in zip(pure_hands, pure_hands[1:])
        if a != "mixed" and b != "mixed")
    return {
        "axis_use_counts": dict(sorted(axis_counts.items())),
        "hand_axis_uses": {hand: hand_counts[hand]
                           for hand in ("left", "right")},
        "hand_axis_use_imbalance": abs(hand_counts["left"]
                                       - hand_counts["right"]),
        "pure_hand_event_switches": hand_switches,
        "longest_single_hand_event_run": longest,
        "axis_reuse_interval_ms": _distribution(reuse),
        "minimum_axis_reuse_ms": round(min(reuse), 6) if reuse else None,
        "axis_reuse_under_150ms_count": sum(value < 150. - 1e-9
                                             for value in reuse),
        "axis_reuse_under_100ms_count": sum(value < 100. - 1e-9
                                             for value in reuse),
        "abstract_reuse_under_80ms_count": sum(value < 80. - 1e-9
                                                for value in reuse),
    }


def _numeric_delta(alternate: dict, original: dict, names: tuple[str, ...]) -> dict:
    return {name: round(alternate[name] - original[name], 9)
            for name in names}


def _flag_delta(original: dict, alternate: dict) -> dict:
    old, new = set(original), set(alternate)
    return {
        "original_count": len(old),
        "alternate_count": len(new),
        "added_count": len(new - old),
        "removed_count": len(old - new),
        "added": [alternate[pair] for pair in sorted(new - old)],
        "removed": [original[pair] for pair in sorted(old - new)],
    }


def _component_delta(alternate: dict, original: dict) -> dict:
    return {name: round(alternate[name] - original[name], 9)
            for name in TERMS}


def _same_number(left, right) -> bool:
    return isclose(float(left), float(right), rel_tol=0., abs_tol=1e-6)


def audit_case(chart: Path, summary: dict, telemetry: dict, replay: dict,
               h01_case: dict, h04_case: dict, h05_case: dict) -> dict:
    rows, frozen_chunks, _ = _validate(summary, telemetry, replay)
    if summary["chart_sha256"] != h05_case["chart_sha256"]:
        raise ValueError("H0.5 chart identity mismatch")
    recomputed_h05 = audit_case_h05(
        chart, summary, telemetry, replay, h01_case, h04_case)
    if recomputed_h05 != h05_case:
        raise ValueError("saved H0.5 case differs from exact recomputation")

    events = _oracle_events(chart)
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    h04_evidence = h04_case["flag_evidence"]
    detail, seeds, frozen_winner = trace_unrestricted(
        events, frozen_chunks, h04_evidence,
        summary["abstract_schedule"]["metrics"]["beam_cost"],
        summary["abstract_schedule"]["candidates_evaluated"],
    )
    original_terms = _cost_terms(frozen_winner)
    original_metrics = _assignment_metrics(events, frozen_chunks)
    original_quick = _quick_flags(events, frozen_chunks)
    original_rapid = _rapid_flags(events, frozen_chunks)
    expected_quick = {
        (item["start_event_index"], item["end_event_index"])
        for item in h01_case["quick_same_hand_depth_jump"]["detected"]
    }
    if set(original_quick) != expected_quick:
        raise ValueError("exact v0.8 quick flags differ from H0.1")

    evidence_out = []
    for target_id, (lineage, saved) in enumerate(zip(
            detail, h05_case["flag_evidence"], strict=True)):
        if {**lineage, "forced_nonquick_suffix": saved["forced_nonquick_suffix"]} != saved:
            raise ValueError("H0.5 lineage ordering changed")
        win_id = lineage["window_index"]
        candidates = _candidate_gestures(windows[win_id])
        candidate = candidates[lineage["nonquick_candidate_index"]]
        forced = _force_suffix_detailed(
            seeds[target_id], windows, win_id, candidate,
            beam_width=planner.beam_width)
        old_forced = saved["forced_nonquick_suffix"]
        for name in ("reached_terminal", "failure_reason",
                     "first_failed_chunk_index", "suffix_candidates_evaluated",
                     "final_cost", "final_cost_terms"):
            if forced[name] != old_forced[name]:
                raise ValueError(f"detailed forced suffix disagrees on {name}")
        if not forced["reached_terminal"]:
            raise ValueError("H0.5 says every forced suffix reaches terminal")
        alternate_chunks = frozen_chunks[:win_id] + forced["chunks"]
        if len(alternate_chunks) != len(frozen_chunks):
            raise ValueError("forced suffix chunk cardinality changed")

        alternate_metrics = _assignment_metrics(events, alternate_chunks)
        alternate_quick = _quick_flags(events, alternate_chunks)
        alternate_rapid = _rapid_flags(events, alternate_chunks)
        fixed = _fixed_original_suffix(
            seeds[target_id], windows, frozen_chunks, win_id)
        if fixed["reached_terminal"]:
            fixed["final_cost_delta_from_original_v08"] = round(
                fixed["final_cost"] - frozen_winner.score, 9)
            fixed["final_cost_component_deltas_from_original_v08"] = (
                _component_delta(fixed["final_cost_terms"], original_terms))
        else:
            fixed["final_cost_delta_from_original_v08"] = None
            fixed["final_cost_component_deltas_from_original_v08"] = None

        changed = [index for index, (old, new) in enumerate(
            zip(frozen_chunks, alternate_chunks, strict=True))
            if old["axes"] != new["axes"]
            or old["gesture_family"] != new["gesture_family"]]
        changed_events = sum(
            old_axes != new_axes
            for old, new in zip(frozen_chunks, alternate_chunks, strict=True)
            for old_axes, new_axes in zip(old["axes"], new["axes"], strict=True)
        )
        permanent_rejoin = None
        for index in range(win_id + 1, len(frozen_chunks)):
            if all(i not in changed for i in range(index, len(frozen_chunks))):
                permanent_rejoin = index
                break

        quick_delta = _flag_delta(original_quick, alternate_quick)
        rapid_delta = _flag_delta(original_rapid, alternate_rapid)
        pair = tuple(lineage["flag_pair"])
        added_quick_pairs = [tuple(item["event_pair"])
                             for item in quick_delta["added"]]
        nearest_added = min(
            (min(abs(a - pair[0]), abs(b - pair[1]))
             for a, b in added_quick_pairs), default=None)
        loss_window = lineage["first_lineage_lost_window"]
        loss_chunk = frozen_chunks[loss_window] if loss_window is not None else None
        forced_delta = old_forced["final_cost_delta_from_original_v08"]
        evidence_out.append({
            "flag_pair": list(pair),
            "window_index": win_id,
            "improved_final_cost": forced_delta < -EPSILON,
            "beam_lineage": {
                "join_expanded_rank": lineage["nonquick_join_expanded_rank"],
                "join_survived_width_24": lineage[
                    "nonquick_join_survives_original_beam"],
                "first_lost_window_index": loss_window,
                "chunks_survived_after_forced_join": (
                    loss_window - win_id if loss_window is not None else None),
                "loss_reason": lineage["lineage_loss_reason"],
                "loss_chunk_event_range": (
                    [loss_chunk["start_event_index"], loss_chunk["end_event_index"]]
                    if loss_chunk else None),
            },
            "cost": {
                "original_v08_final": round(frozen_winner.score, 9),
                "forced_final": forced["final_cost"],
                "final_delta": forced_delta,
                "improvement_amount": round(-forced_delta, 9),
                "immediate_delta_at_join": lineage[
                    "nonquick_immediate_cost_delta"],
                "original_components": original_terms,
                "forced_components": forced["final_cost_terms"],
                "six_component_deltas": _component_delta(
                    forced["final_cost_terms"], original_terms),
            },
            "single_seam_fixed_original_suffix_control": fixed,
            "downstream_replanning_gain_vs_fixed_suffix": (
                round(forced["final_cost"] - fixed["final_cost"], 9)
                if fixed["reached_terminal"] else None),
            "flag_effects": {
                "target_quick_flag_removed": pair not in alternate_quick,
                "quick": quick_delta,
                "rapid": rapid_delta,
                "nearest_new_quick_pair_event_offset": nearest_added,
                "flag_movement_is_identity_claim": False,
            },
            "assignment_effects": {
                "changed_chunk_count": len(changed),
                "changed_chunk_indices": changed,
                "changed_event_assignment_count": changed_events,
                "first_changed_chunk": changed[0] if changed else None,
                "last_changed_chunk": changed[-1] if changed else None,
                "permanent_rejoin_chunk": permanent_rejoin,
                "next_1_chunk_changed": sum(i in changed
                                             for i in range(win_id, win_id + 1)),
                "next_4_chunks_changed": sum(i in changed
                                              for i in range(win_id, win_id + 4)),
                "next_8_chunks_changed": sum(i in changed
                                              for i in range(win_id, win_id + 8)),
            },
            "original_assignment_metrics": original_metrics,
            "forced_assignment_metrics": alternate_metrics,
            "assignment_metric_deltas": _numeric_delta(
                alternate_metrics, original_metrics,
                ("hand_axis_use_imbalance", "pure_hand_event_switches",
                 "longest_single_hand_event_run",
                 "axis_reuse_under_150ms_count",
                 "axis_reuse_under_100ms_count",
                 "abstract_reuse_under_80ms_count")),
            "hand_axis_use_deltas": {
                hand: alternate_metrics["hand_axis_uses"][hand]
                - original_metrics["hand_axis_uses"][hand]
                for hand in ("left", "right")
            },
            "minimum_axis_reuse_delta_ms": round(
                alternate_metrics["minimum_axis_reuse_ms"]
                - original_metrics["minimum_axis_reuse_ms"], 9),
            "nonquick_axes": lineage["nonquick_axes"],
            "nonquick_gesture_family": lineage["nonquick_gesture_family"],
        })

    improved = [item for item in evidence_out if item["improved_final_cost"]]
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(rows),
        "targeted_count": len(evidence_out),
        "improved_count": len(improved),
        "evidence": evidence_out,
    }


def _cohort_summary(rows: list[dict]) -> dict:
    fixed_deltas = [
        row["single_seam_fixed_original_suffix_control"][
            "final_cost_delta_from_original_v08"]
        for row in rows
        if row["single_seam_fixed_original_suffix_control"][
            "final_cost_delta_from_original_v08"] is not None
    ]
    components = {
        name: {
            "sum": round(sum(row["cost"]["six_component_deltas"][name]
                             for row in rows), 9),
            "mean": round(mean(row["cost"]["six_component_deltas"][name]
                               for row in rows), 9) if rows else None,
        }
        for name in TERMS
    }
    return {
        "count": len(rows),
        "immediate_cost_delta": _distribution(
            row["cost"]["immediate_delta_at_join"] for row in rows),
        "final_cost_delta": _distribution(
            row["cost"]["final_delta"] for row in rows),
        "join_expanded_rank": _distribution(
            row["beam_lineage"]["join_expanded_rank"] for row in rows),
        "chunks_survived_in_original_beam": _distribution(
            row["beam_lineage"]["chunks_survived_after_forced_join"]
            for row in rows),
        "loss_reasons": dict(sorted(Counter(
            row["beam_lineage"]["loss_reason"] for row in rows).items())),
        "fixed_original_suffix_final_delta": _distribution(fixed_deltas),
        "downstream_replanning_gain": _distribution(
            row["downstream_replanning_gain_vs_fixed_suffix"]
            for row in rows
            if row["downstream_replanning_gain_vs_fixed_suffix"] is not None),
        "six_component_deltas": components,
        "changed_chunks": _distribution(
            row["assignment_effects"]["changed_chunk_count"] for row in rows),
        "changed_event_assignments": _distribution(
            row["assignment_effects"]["changed_event_assignment_count"]
            for row in rows),
        "new_quick_flags_across_independent_runs": sum(
            row["flag_effects"]["quick"]["added_count"] for row in rows),
        "removed_quick_flags_across_independent_runs": sum(
            row["flag_effects"]["quick"]["removed_count"] for row in rows),
        "new_rapid_flags_across_independent_runs": sum(
            row["flag_effects"]["rapid"]["added_count"] for row in rows),
        "removed_rapid_flags_across_independent_runs": sum(
            row["flag_effects"]["rapid"]["removed_count"] for row in rows),
        "minimum_axis_reuse_delta_ms": _distribution(
            row["minimum_axis_reuse_delta_ms"] for row in rows),
        "under_150ms_reuse_count_delta": _distribution(
            row["assignment_metric_deltas"]["axis_reuse_under_150ms_count"]
            for row in rows),
        "pure_hand_switch_delta": _distribution(
            row["assignment_metric_deltas"]["pure_hand_event_switches"]
            for row in rows),
        "left_axis_use_delta": _distribution(
            row["hand_axis_use_deltas"]["left"] for row in rows),
        "right_axis_use_delta": _distribution(
            row["hand_axis_use_deltas"]["right"] for row in rows),
    }


def _aggregate(cases: list[dict]) -> dict:
    all_rows = [row for case in cases for row in case["evidence"]]
    improved = [row for row in all_rows if row["improved_final_cost"]]
    component_sums = {
        name: round(sum(row["cost"]["six_component_deltas"][name]
                        for row in improved), 9)
        for name in TERMS
    }
    return {
        "targeted_count": len(all_rows),
        "improved_final_cost_count": len(improved),
        "improved_final_cost_distribution": _distribution(
            row["cost"]["improvement_amount"] for row in improved),
        "improved_beam_loss_reasons": dict(sorted(Counter(
            row["beam_lineage"]["loss_reason"] for row in improved).items())),
        "improved_component_delta_sums": component_sums,
        "improved_target_quick_removed_count": sum(
            row["flag_effects"]["target_quick_flag_removed"] for row in improved),
        "improved_new_quick_flag_count_across_independent_runs": sum(
            row["flag_effects"]["quick"]["added_count"] for row in improved),
        "improved_removed_quick_flag_count_across_independent_runs": sum(
            row["flag_effects"]["quick"]["removed_count"] for row in improved),
        "improved_new_rapid_flag_count_across_independent_runs": sum(
            row["flag_effects"]["rapid"]["added_count"] for row in improved),
        "improved_removed_rapid_flag_count_across_independent_runs": sum(
            row["flag_effects"]["rapid"]["removed_count"] for row in improved),
        "improved_fixed_original_suffix_terminal_count": sum(
            row["single_seam_fixed_original_suffix_control"]["reached_terminal"]
            for row in improved),
        "improved_fixed_original_suffix_also_lower_cost_count": sum(
            row["single_seam_fixed_original_suffix_control"][
                "final_cost_delta_from_original_v08"] is not None
            and row["single_seam_fixed_original_suffix_control"][
                "final_cost_delta_from_original_v08"] < -EPSILON
            for row in improved),
        "improved_changed_chunk_count_distribution": _distribution(
            row["assignment_effects"]["changed_chunk_count"] for row in improved),
        "improved_changed_event_count_distribution": _distribution(
            row["assignment_effects"]["changed_event_assignment_count"]
            for row in improved),
        "improved_cohort": _cohort_summary(improved),
        "non_improved_cohort": _cohort_summary(
            [row for row in all_rows if not row["improved_final_cost"]]),
        "physical_motor_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verified-report", required=True, type=Path)
    parser.add_argument("--h0-report", required=True, type=Path)
    parser.add_argument("--h01-report", required=True, type=Path)
    parser.add_argument("--h04-report", required=True, type=Path)
    parser.add_argument("--h05-report", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing H0.6 result")

    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = load(args.manifest)
    verified = load(args.verified_report)
    h0 = load(args.h0_report)
    h01 = load(args.h01_report)
    h04 = load(args.h04_report)
    h05 = load(args.h05_report)
    artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        prefix = args.results_root / sha[:12]
        summary_path = prefix / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary,
            load(prefix / "telemetry-v08.json"),
            load(prefix / "replay-v08.json"),
            load(args.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    if checked != h01:
        raise ValueError("saved H0.1 report differs from verified Train artifacts")
    if (h04.get("format") != H04_FORMAT
            or h05.get("format") != H05_FORMAT
            or h05.get("source_v08_commit") != checked["source_v08_commit"]
            or h05.get("frozen_code_digest") != checked["frozen_code_digest"]
            or h05.get("event_count") != checked["aggregate"]["event_count"]):
        raise ValueError("H0.4/H0.5 source identity mismatch")
    h04_by_sha = {case["chart_sha256"]: case for case in h04["cases"]}
    h05_by_sha = {case["chart_sha256"]: case for case in h05["cases"]}
    expected_sha = {case["sha256"] for case in manifest["cases"]}
    if set(h04_by_sha) != expected_sha or set(h05_by_sha) != expected_sha:
        raise ValueError("H0.4/H0.5 case selection changed")

    train_root = Path(manifest["train_root"]).resolve()
    cases = []
    for case in manifest["cases"]:
        sha = case["sha256"]
        chart = validate_chart(case, train_root)
        summary, telemetry, replay, _ = artifacts[sha]
        h01_case = next(item for item in checked["cases"]
                        if item["chart_sha256"] == sha)
        cases.append(audit_case(
            chart, summary, telemetry, replay, h01_case,
            h04_by_sha[sha], h05_by_sha[sha]))
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "case_count": len(cases),
        "event_count": sum(case["event_count"] for case in cases),
        "aggregate": _aggregate(cases),
        "cases": cases,
        "planner_motor_timing_runner_changed": False,
        "hyperparameters_changed": False,
        "physical_motor_executed": False,
        "causal_effect_established": False,
        "human_likeness_verified": False,
        "limitations": [
            "Each forced seam is an independent counterfactual and the 22 results cannot be combined into one chart.",
            "The forced search contains only descendants of one selected non-quick seed, unlike the original global beam.",
            "The fixed-original-suffix control isolates downstream replanning but remains an abstract heuristic counterfactual.",
            "Added and removed flag locations do not establish that a gesture identity moved.",
            "Six objective terms are frozen untrained heuristics, not measurements of comfort, fatigue, or human preference.",
            "Abstract 80ms reuse feasibility does not establish physical Motor feasibility or Hits.",
        ],
    }
    if report["aggregate"]["targeted_count"] != 22:
        raise ValueError("H0.6 must cover all 22 H0.5 targets")
    if report["aggregate"]["improved_final_cost_count"] != 7:
        raise ValueError("H0.6 must reproduce the seven H0.5 improvements")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "events": report["event_count"],
        "targets": report["aggregate"]["targeted_count"],
        "improved": report["aggregate"]["improved_final_cost_count"],
        "aggregate": report["aggregate"],
        "saved": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
