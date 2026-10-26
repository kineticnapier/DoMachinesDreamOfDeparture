from __future__ import annotations

"""H0.4 read-only reconstruction of the frozen v0.8 Gesture Chunk beam.

No altered planner, candidate policy, hyperparameters, Motor execution,
learning, or human demonstrations. Report explanations only if exact
chosen chunks, beam objective and candidate count agree with frozen v0.8.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from math import exp, isclose
from pathlib import Path
import json

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher
from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent, GestureChunk, _candidate_gestures,
    _axis_parts,
)
from analyze_dmdogo_v09_style_proxies import _validate
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h04-frozen-beam-choice-audit-v1"
TERMS = (
    "gesture_preference", "hand_exposure", "axis_reuse_margin",
    "intra_gesture_hand_changes", "intra_gesture_depth_distance",
    "chunk_join_transition",
)


@dataclass(slots=True)
class Node:
    parent: Node | None
    window_index: int
    candidate_index: int
    beam_rank: int
    expanded_rank: int
    score: float
    terms: dict
    gesture: GestureChunk


@dataclass(slots=True)
class State:
    score: float
    previous: dict
    last_axis: str | None
    exposure: tuple[float, float]
    exposure_t: float
    node: Node | None


def _quick_at_join(left: str | None, right: str | None, gap_s: float) -> bool:
    if not left or not right or not 0 <= gap_s <= .120:
        return False
    lh, ld = _axis_parts(left)
    rh, rd = _axis_parts(right)
    return lh == rh and abs(ld - rd) >= 2


def _expand(state: State, window, candidate, previous_due: float | None):
    """Numerical operation order intentionally mirrors frozen v0.8 plan()."""
    last_use = state.previous.copy()
    extra = candidate.preference
    terms = dict.fromkeys(TERMS, 0.)
    terms["gesture_preference"] = float(candidate.preference)
    exposure_now = list(state.exposure)
    last_time = state.exposure_t
    for event, axes in zip(window.events, candidate.axes):
        decay = exp(-max(0., event.due_s - last_time) / .75)
        exposure_now = [value * decay for value in exposure_now]
        last_time = event.due_s
        for axis in axes:
            hand_index = 0 if axis.startswith("left") else 1
            inc = .45 * exposure_now[hand_index]
            extra += inc
            terms["hand_exposure"] += inc
            exposure_now[hand_index] += 1.
            prior = last_use.get(axis)
            if prior is not None:
                duration = event.due_s - prior
                if duration <= 1e-12 or duration + 1e-12 < .080:
                    return None
                inc = max(0., .150 - duration) * 2.
                extra += inc
                terms["axis_reuse_margin"] += inc
            last_use[axis] = event.due_s
    flat = [axis for axes in candidate.axes for axis in axes]
    hand_changes = sum(_axis_parts(a)[0] != _axis_parts(b)[0]
                       for a, b in zip(flat, flat[1:]))
    inc = .35 * hand_changes
    extra += inc
    terms["intra_gesture_hand_changes"] += inc
    for a, b in zip(flat, flat[1:]):
        ha, da = _axis_parts(a)
        hb, db = _axis_parts(b)
        if ha == hb:
            inc = .10 * max(0, abs(db - da) - 1)
            extra += inc
            terms["intra_gesture_depth_distance"] += inc
    if state.last_axis is not None and flat and previous_due is not None:
        gap = window.events[0].due_s - previous_due
        if gap <= .050:
            ha, da = _axis_parts(state.last_axis)
            hb, db = _axis_parts(flat[0])
            inc = .12 if ha != hb else .05 * max(0, abs(db - da) - 1)
            extra += inc
            terms["chunk_join_transition"] += inc
    annotation = GestureChunk(
        window.events[0].event_index, window.events[-1].event_index,
        len(window.events), window.rhythm, window.reason,
        candidate.family, candidate.axes,
    )
    return (state.score + extra, last_use, flat[-1],
            tuple(exposure_now), last_time, terms, annotation, extra)


def _chunk_record(chunk: GestureChunk) -> dict:
    return {
        "start_event_index": chunk.start_event_index,
        "end_event_index": chunk.end_event_index,
        "event_count": chunk.event_count,
        "rhythm_kind": chunk.rhythm_kind,
        "start_reason": chunk.start_reason,
        "gesture_family": chunk.gesture_family,
        "axes": [list(axes) for axes in chunk.axes],
    }


def audit_beam(events: tuple[GestureEvent, ...], frozen_chunks: list[dict],
               expected_cost: float, expected_evaluated: int,
               flagged_pairs: list[tuple[int, int]], *,
               check_planner: bool = True) -> dict:
    """Replay the original beam, retaining candidate snapshots at flagged seams.

    No pruning decisions are inferred from a locally optimized proxy.
    If the instrumented beam fails to reproduce the frozen plan, fail closed.
    """
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    if len(windows) != len(frozen_chunks):
        raise ValueError("frozen Planner chunk segmentation not reproduced")
    if not events or len({e.event_index for e in events}) != len(events):
        raise ValueError("invalid Oracle event set")

    seam_ids = {b for a, b in flagged_pairs}
    indices = {window.events[0].event_index: i
               for i, window in enumerate(windows)}
    targeted = {}
    for prior, current in flagged_pairs:
        if current not in indices or current <= prior:
            raise ValueError("H0.1 quick flag is not a chunk start")
        i = indices[current]
        if not i or windows[i-1].events[-1].event_index != prior:
            raise ValueError("H0.1 pair does not cross adjacent chunks")
        targeted[i] = (prior, current)
    if len(targeted) != len(flagged_pairs):
        raise ValueError("duplicate H0.1 target window")

    states = [State(0., {}, None, (0., 0.), events[0].due_s, None)]
    evaluated = 0
    snapshots = {}
    window_stats = {}
    for index, window in enumerate(windows):
        previous_due = (windows[index - 1].events[-1].due_s if index else None)
        gap = (window.events[0].due_s - previous_due
               if previous_due is not None else float("inf"))
        candidates = _candidate_gestures(window)
        expanded = []
        rejected = 0
        # Sorted states at every step are preserved in frozen traversal order.
        for parent_rank, state in enumerate(states, 1):
            for candidate_index, candidate in enumerate(candidates):
                evaluated += 1
                result = _expand(state, window, candidate, previous_due)
                if result is None:
                    rejected += 1
                    continue
                (total, last_use, last_axis, exposure, last_time,
                 terms, annotation, incremental) = result
                quick = _quick_at_join(
                    state.last_axis, candidate.axes[0][0], gap)
                expanded.append((
                    total, last_use, last_axis, exposure, last_time,
                    state.node, parent_rank, candidate_index, terms,
                    annotation, incremental, quick,
                ))
        if not expanded:
            raise ValueError(f"replayed frozen beam infeasible at chunk {index}")
        # Exactly mirrors v0.8 stable score-only sorting, including ties.
        expanded.sort(key=lambda row: row[0])
        if index in targeted:
            snapshots[index] = {
                "previous_due_s": previous_due,
                "gap_ms": gap * 1000.,
                "expanded_feasible": len(expanded),
                "reuse_rejected": rejected,
                "candidate_count": len(candidates),
                "rows": [
                    {
                        "expanded_rank": rank,
                        "parent_rank": row[6],
                        "candidate_index": row[7],
                        "score": row[0],
                        "incremental_cost": row[10],
                        "terms": row[8],
                        "quick": row[11],
                        "first_axes": list(row[9].axes[0]),
                        "gesture_family": row[9].gesture_family,
                        "survived_cut": rank <= planner.beam_width,
                    }
                    for rank, row in enumerate(expanded, 1)
                ],
            }
        new_states = []
        for beam_rank, row in enumerate(expanded[:planner.beam_width], 1):
            (total, last_use, last_axis, exposure, last_time,
             parent, parent_rank, candidate_index, terms, annotation,
             incremental, quick) = row
            node = Node(parent, index, candidate_index,
                        beam_rank, beam_rank, total, terms, annotation)
            new_states.append(State(total, last_use, last_axis,
                                    exposure, last_time, node))
        states = new_states

    # Lowest cumulative score at final window is the actual frozen choice.
    final = states[0]
    nodes = []
    node = final.node
    while node is not None:
        nodes.append(node)
        node = node.parent
    nodes.reverse()
    if len(nodes) != len(frozen_chunks):
        raise ValueError("beam path length differs from source chunks")
    actual_chunks = [_chunk_record(node.gesture) for node in nodes]
    if actual_chunks != frozen_chunks:
        for i, (actual, expected) in enumerate(zip(actual_chunks, frozen_chunks)):
            if actual != expected:
                raise ValueError(f"frozen beam diverged at chunk {i}")
        raise ValueError("frozen beam source/output mismatch")
    if evaluated != expected_evaluated:
        raise ValueError(f"beam candidate count differs: {evaluated} != {expected_evaluated}")
    if not isclose(round(final.score, 6), expected_cost, abs_tol=1e-6, rel_tol=0):
        raise ValueError("frozen beam objective differs from saved planner")
    if check_planner:
        # Independently call the frozen implementation. This is a read-only
        # offline plan(), not a physical Motor episode or a new training run.
        canonical = planner.plan(events)
        if (not canonical.feasible
                or tuple(_chunk_record(c) for c in canonical.chunks)
                != tuple(actual_chunks)
                or canonical.candidates_evaluated != evaluated
                or not isclose(canonical.metrics["beam_cost"],
                               round(final.score, 6), abs_tol=1e-6, rel_tol=0)):
            raise ValueError("instrumented beam differs from canonical frozen Planner")

    evidence = []
    for index, (prior_id, current_id) in sorted(targeted.items()):
        chosen = nodes[index]
        if chosen.parent is None:
            raise ValueError("target cannot occur in first chunk")
        snap = snapshots[index]
        eligible = [r for r in snap["rows"]
                    if r["parent_rank"] == chosen.parent.beam_rank]
        chosen_entry = next((
            r for r in eligible
            if r["candidate_index"] == chosen.candidate_index
        ), None)
        if (chosen_entry is None or not chosen_entry["quick"]
                or chosen_entry["expanded_rank"] != chosen.expanded_rank):
            raise ValueError("frozen targeted quick jump not reproduced in beam")
        not_quick = [r for r in eligible if not r["quick"]]
        cheapest_not_quick = min(not_quick, key=lambda r: r["score"]) if not_quick else None
        # Counterfactual candidates share exactly the original predecessor
        # state, but do NOT share the eventual future beam path.
        evidence.append({
            "flag_pair": [prior_id, current_id],
            "window_index": index,
            "previous_chunk": _chunk_record(nodes[index-1].gesture),
            "chosen_chunk": _chunk_record(chosen.gesture),
            "join_gap_ms": round(snap["gap_ms"], 6),
            "beam_width": planner.beam_width,
            "feasible_expansions_all_parents": snap["expanded_feasible"],
            "reuse_rejected_all_parents": snap["reuse_rejected"],
            "prior_beam_rank": chosen.parent.beam_rank,
            "chosen_expanded_rank": chosen_entry["expanded_rank"],
            "chosen_beam_rank": chosen.beam_rank,
            "chosen_cost_terms": {
                k: round(v, 9) for k, v in chosen.terms.items()
            },
            "chosen_incremental_cost": round(chosen_entry["incremental_cost"], 9),
            "chosen_cumulative_cost": round(chosen_entry["score"], 9),
            "same_parent_feasible_options": len(eligible),
            "same_parent_nonquick_options": len(not_quick),
            "same_parent_nonquick_survived": sum(
                r["survived_cut"] for r in not_quick),
            "same_parent_nonquick_cheaper_than_chosen": sum(
                r["score"] < chosen_entry["score"] for r in not_quick),
            "best_same_parent_nonquick": (
                {
                    "gesture_family": cheapest_not_quick["gesture_family"],
                    "first_axes": cheapest_not_quick["first_axes"],
                    "expanded_rank": cheapest_not_quick["expanded_rank"],
                    "survived_cut": cheapest_not_quick["survived_cut"],
                    "incremental_cost": round(
                        cheapest_not_quick["incremental_cost"], 9),
                    "cumulative_cost_delta_from_chosen": round(
                        cheapest_not_quick["score"] - chosen_entry["score"], 9),
                    "cost_terms": {
                        k: round(v, 9)
                        for k, v in cheapest_not_quick["terms"].items()
                    },
                } if cheapest_not_quick else None),
        })
    totals = Counter()
    for item in evidence:
        totals["targeted_quick_flags"] += 1
        totals["nonquick_feasible_under_same_predecessor"] += (
            item["same_parent_nonquick_options"] > 0)
        totals["nonquick_in_top_beam_under_same_predecessor"] += (
            item["same_parent_nonquick_survived"] > 0)
        totals["nonquick_cheaper_locally_under_same_predecessor"] += (
            item["same_parent_nonquick_cheaper_than_chosen"] > 0)
        totals["quick_path_was_not_best_expansion"] += (
            item["chosen_expanded_rank"] > 1)
    return {
        "chunk_count": len(windows),
        "events": len(events),
        "candidates_evaluated": evaluated,
        "frozen_final_beam_cost": round(final.score, 6),
        "exact_frozen_chunk_path_matched": True,
        "exact_frozen_candidate_count_matched": True,
        "exact_frozen_beam_cost_matched": True,
        "canonical_frozen_planner_independently_matched": check_planner,
        "aggregate": dict(totals),
        "flag_evidence": evidence,
    }


def _oracle_events(chart: Path) -> tuple[GestureEvent, ...]:
    compiled = compile_adofai(load_adofai(chart))
    segment = build_playable_segment(compiled)
    return tuple(GestureEvent(e.event_index, e.episode_time_s, e.chord_size)
                 for e in AutoplayTimingTeacher.from_segment(segment).events)


def audit_case(chart_path: Path, summary: dict, telemetry: dict, replay: dict,
               h01_case: dict) -> dict:
    ordered, chunks, _ = _validate(summary, telemetry, replay)
    events = _oracle_events(chart_path)
    if (len(events) != len(ordered)
            or [e.event_index for e in events] !=
            [r["event_index"] for r in ordered]):
        raise ValueError("recompiled Oracle event IDs differ from frozen replay")
    if any(abs(e.due_s * 1000. - row["due_ms"]) > .005
           for e, row in zip(events, ordered)):
        raise ValueError("recompiled Oracle onset times differ from frozen replay")
    source = summary["abstract_schedule"]
    expected = source["metrics"]
    flags = h01_case["quick_same_hand_depth_jump"]["detected"]
    targets = [(f["start_event_index"], f["end_event_index"]) for f in flags]
    body = audit_beam(events, chunks, expected["beam_cost"],
                      source["candidates_evaluated"], targets)
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        **body,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--verified-report", type=Path, required=True)
    p.add_argument("--h0-report", type=Path, required=True)
    p.add_argument("--h01-report", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("refusing to overwrite H0.4 audit")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = load(a.manifest)
    verified = load(a.verified_report)
    h0 = load(a.h0_report)
    h01 = load(a.h01_report)
    case_artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        root = a.results_root / sha[:12]
        summary_path = root / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        case_artifacts[sha] = (
            summary, load(root / "telemetry-v08.json"),
            load(root / "replay-v08.json"),
            load(a.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    checked = compile_train_report(
        manifest, verified, h0, case_artifacts,
        manifest_digest=sha256_file(a.manifest))
    if checked != h01:
        raise ValueError("saved H0.1 does not match verified v0.8 sources")
    train_root = Path(manifest["train_root"]).resolve()
    cases = []
    for case in manifest["cases"]:
        chart = validate_chart(case, train_root)
        summary, telemetry, replay, _ = case_artifacts[case["sha256"]]
        h01_case = next(item for item in checked["cases"]
                        if item["chart_sha256"] == case["sha256"])
        cases.append(audit_case(chart, summary, telemetry, replay, h01_case))
    agg = Counter()
    for case in cases:
        agg.update(case["aggregate"])
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "case_count": len(cases),
        "event_count": sum(c["events"] for c in cases),
        "aggregate": dict(agg),
        "cases": cases,
        "physical_motor_executed": False,
        "human_likeness_verified": False,
        "causal_explanation_verified": False,
        "limitations": [
            "Replay reproduces the frozen algorithm's selected beam path; this is a mechanistic audit, not causal effect identification.",
            "Cheaper immediate alternatives may lose after future beam steps or future constraints.",
            "Choice costs reflect heuristic priors, not observed human preference or calibrated physiology.",
            "The review flags select 22 cases; they do not estimate population-wide harm or frequency.",
            "No alternative is tested against physical Motor or a human performer.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "events": report["event_count"],
        "case_count": report["case_count"],
        "aggregate": report["aggregate"],
        "saved": str(a.output),
        "physical_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
