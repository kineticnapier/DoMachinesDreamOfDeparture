from __future__ import annotations

"""H0.7: a diagnostic one-slot preservation policy for seven H0.6 gains.

The original frozen v0.8 planner and its width=24 remain UNCHANGED.
This independent counterfactual always keeps 24 or fewer hypotheses.
At one selected seam it admits the same best non-quick candidate as
H0.5; later it preserves at least one descendant by substituting for
the worst regular beam hypothesis only when necessary. Not production.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record, _oracle_events
from audit_dmdogo_v09_h05_nonquick_suffix import (
    Branch, _cost_terms, _next, _seed_state,
)
from audit_dmdogo_v09_h06_improved_suffix import (
    FORMAT as H06_FORMAT, _assignment_metrics, _quick_flags, _rapid_flags,
    _event_axes,
)
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_style_proxies import _validate
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h07-shadow-lane-v1"
BEAM_WIDTH = 24
TARGET_TAG = 0


@dataclass(slots=True)
class Node:
    parent: Node | None
    chunk: dict


@dataclass(slots=True)
class Path:
    state: Branch
    node: Node | None


@dataclass(slots=True)
class Proposal:
    state: Branch
    parent: Node | None
    chunk: dict
    parent_rank: int
    candidate_index: int

    @property
    def protected(self) -> bool:
        return TARGET_TAG in self.state.tags


def choose_beam(proposals: list[Proposal], *, width: int = BEAM_WIDTH
                ) -> tuple[list[Proposal], dict]:
    """Stable sort by unmodified v0.8 cost; preserve one tagged lineage.

    Keeping a path already in top-24 is not an intervention. If no
    tagged extension is feasible, report this, do not invent one.
    """
    if width < 2 or not proposals:
        raise ValueError("invalid beam selection")
    ordered = sorted(proposals, key=lambda x: x.state.score)
    regular = ordered[:width]
    if any(row.protected for row in regular):
        return regular, {
            "protected_feasible": True,
            "protected_natural_survival": True,
            "slot_replacement": False,
            "best_protected_expanded_rank": next(
                i for i, row in enumerate(ordered, 1) if row.protected),
            "displaced_rank": None,
        }
    best_tagged = next(
        ((rank, row) for rank, row in enumerate(ordered, 1)
         if row.protected), None)
    if best_tagged is None:
        return regular, {
            "protected_feasible": False,
            "protected_natural_survival": False,
            "slot_replacement": False,
            "best_protected_expanded_rank": None,
            "displaced_rank": None,
        }
    rank, protected = best_tagged
    displaced = len(regular) if len(regular) == width else None
    survivors = regular[:width-1] + [protected]
    # Sort by original expanded rank, preserving tie order.
    ordered_index = {id(row): i for i, row in enumerate(ordered)}
    survivors.sort(key=lambda row: ordered_index[id(row)])
    return survivors, {
        "protected_feasible": True,
        "protected_natural_survival": False,
        "slot_replacement": True,
        "best_protected_expanded_rank": rank,
        "displaced_rank": displaced,
    }


def _path_chunks(node: Node | None) -> list[dict]:
    output = []
    while node is not None:
        output.append(node.chunk)
        node = node.parent
    return list(reversed(output))


def _event_count_chunks(chunks: list[dict]) -> int:
    return sum(chunk["event_count"] for chunk in chunks)


def _delta_flags(original: dict, candidate: dict) -> dict:
    old, new = set(original), set(candidate)
    return {
        "original_count": len(old),
        "candidate_count": len(new),
        "added": [candidate[k] for k in sorted(new - old)],
        "removed": [original[k] for k in sorted(old - new)],
        "added_count": len(new - old),
        "removed_count": len(old - new),
    }


def trace_one_slot(events, frozen_chunks: list[dict],
                   target: dict, *, expected_cost: float,
                   expected_evaluated: int) -> dict:
    """Original beam up to the target; independent protected selection thereafter."""
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    if not windows or len(windows) != len(frozen_chunks):
        raise ValueError("frozen segmentation mismatch")
    canonical = planner.plan(events)
    expected_chunks = [_chunk_record(c) for c in canonical.chunks]
    if (not canonical.feasible or expected_chunks != frozen_chunks
            or canonical.candidates_evaluated != expected_evaluated
            or canonical.metrics["beam_cost"] != expected_cost):
        raise ValueError("frozen v0.8 Planner/source mismatch")

    target_window = target["window_index"]
    join_pair = target["flag_pair"]
    target_index = target["nonquick_candidate_index"]
    if (not 0 < target_window < len(windows)
            or windows[target_window].events[0].event_index != join_pair[1]
            or windows[target_window-1].events[-1].event_index != join_pair[0]):
        raise ValueError("H0.6 target does not refer to a frozen chunk seam")
    if target["nonquick_join_expanded_rank"] < 1:
        raise ValueError("invalid H0.5 nonquick expanded rank")

    states = [Path(_seed_state(events[0].due_s), None)]
    total_candidates_before_join = 0
    protected_first_rank = None
    rescue_windows = []
    protected_dead_window = None
    protected_started = False
    for win_id, window in enumerate(windows):
        candidates = _candidate_gestures(window)
        previous_due = windows[win_id-1].events[-1].due_s if win_id else None
        frozen_axes = tuple(tuple(x) for x in frozen_chunks[win_id]["axes"])
        frozen_family = frozen_chunks[win_id]["gesture_family"]
        expansions = []
        for rank, parent in enumerate(states, 1):
            for candidate_index, candidate in enumerate(candidates):
                if win_id <= target_window:
                    total_candidates_before_join += 1
                next_state = _next(
                    parent.state, window, candidate, previous_due,
                    frozen_candidate=(
                        candidate.axes == frozen_axes
                        and candidate.family == frozen_family),
                )
                if next_state is None:
                    continue
                if (win_id == target_window
                        and parent.state.frozen_prefix
                        and candidate_index == target_index):
                    next_state.tags = frozenset({TARGET_TAG})
                expansions.append(Proposal(
                    next_state, parent.node, _chunk_record_from(window, candidate),
                    rank, candidate_index,
                ))
        if not expansions:
            raise ValueError(f"all candidate paths failed at window {win_id}")
        # v0.8 stable cost-only sort; do not change scoring.
        expansions.sort(key=lambda p: p.state.score)
        if win_id < target_window:
            keep = expansions[:BEAM_WIDTH]
            if sum(p.state.frozen_prefix for p in keep) != 1:
                raise ValueError("frozen selected prefix pruned before target")
        else:
            if win_id == target_window:
                tagged = [(i, p) for i, p in enumerate(expansions, 1)
                          if p.protected]
                if len(tagged) != 1:
                    raise ValueError("target seam alternative is not unique")
                chosen_rank, chosen = tagged[0]
                if (chosen_rank != target["nonquick_join_expanded_rank"]
                        or chosen.candidate_index != target_index
                        or list(chosen.chunk["axes"]) != target["nonquick_axes"]):
                    raise ValueError("H0.5 nonquick candidate rank/identity mismatch")
                protected_first_rank = chosen_rank
                protected_started = True
            keep, status = choose_beam(expansions)
            if status["slot_replacement"]:
                rescue_windows.append(win_id)
            if not status["protected_feasible"] and protected_dead_window is None:
                protected_dead_window = win_id
        states = [Path(p.state, Node(p.parent, p.chunk)) for p in keep]
    if not protected_started:
        raise ValueError("targeted alternative was never selected")

    winner = states[0]
    tagged = [(i, p) for i, p in enumerate(states, 1)
              if TARGET_TAG in p.state.tags]
    chosen_protected = tagged[0][1] if tagged else None
    if protected_dead_window is None and not chosen_protected:
        raise ValueError("tag disappeared without accounting for infeasibility")

    def characterize(state: Path | None):
        if state is None:
            return None
        chunks = _path_chunks(state.node)
        if len(chunks) != len(windows) or _event_count_chunks(chunks) != len(events):
            raise ValueError("counterfactual does not cover entire chart")
        original_axes = _event_axes(frozen_chunks)
        alternate_axes = _event_axes(chunks)
        changed = sum(
            alternate_axes[e.event_index] != original_axes[e.event_index]
            for e in events)
        quick0, quick1 = _quick_flags(events, frozen_chunks), _quick_flags(events, chunks)
        rapid0, rapid1 = _rapid_flags(events, frozen_chunks), _rapid_flags(events, chunks)
        metric = _assignment_metrics(events, chunks)
        if metric["abstract_reuse_under_80ms_count"] != 0:
            raise ValueError("shadow result violates original abstract reuse")
        return {
            "final_cost": round(state.state.score, 9),
            "delta_cost_vs_frozen_v08": round(
                state.state.score - expected_cost, 9),
            "is_from_protected_lineage": TARGET_TAG in state.state.tags,
            "changed_event_assignment_count": changed,
            "quick": _delta_flags(quick0, quick1),
            "rapid": _delta_flags(rapid0, rapid1),
            "axis_reuse_under_80ms": metric["abstract_reuse_under_80ms_count"],
            "minimum_axis_reuse_ms": metric["minimum_axis_reuse_ms"],
            "hand_axis_uses": metric["hand_axis_uses"],
            "score_components": _cost_terms(state.state),
        }

    return {
        "flag_pair": join_pair,
        "window_index": target_window,
        "admission_expanded_rank": protected_first_rank,
        "admitted_within_natural_top24": protected_first_rank <= BEAM_WIDTH,
        "total_candidates_evaluated_before_and_at_join": total_candidates_before_join,
        "protected_descendants_reached_terminal": chosen_protected is not None,
        "protected_dead_window": protected_dead_window,
        "slot_replacement_count": len(rescue_windows),
        "slot_replacement_windows": rescue_windows,
        "best_protected_terminal_rank": tagged[0][0] if tagged else None,
        "beam_winner": characterize(winner),
        "best_protected_terminal": characterize(chosen_protected),
        "width": BEAM_WIDTH,
        "physical_motor_executed": False,
    }


def _chunk_record_from(window, candidate):
    return {
        "start_event_index": window.events[0].event_index,
        "end_event_index": window.events[-1].event_index,
        "event_count": len(window.events),
        "rhythm_kind": window.rhythm,
        "start_reason": window.reason,
        "gesture_family": candidate.family,
        "axes": [list(x) for x in candidate.axes],
    }


def audit_case(chart, summary: dict, telemetry: dict,
               replay: dict, h06_case: dict) -> dict:
    rows, frozen_chunks, _ = _validate(summary, telemetry, replay)
    events = _oracle_events(chart)
    if (len(events) != len(rows)
            or any(e.event_index != r["event_index"]
                   or abs(1000*e.due_s-r["due_ms"]) > .005
                   for e, r in zip(events, rows))):
        raise ValueError("Oracle event IDs/times differ from frozen decisions")
    if (h06_case["chart_sha256"] != summary["chart_sha256"]
            or h06_case["event_count"] != len(events)):
        raise ValueError("H0.6 case source mismatch")
    seen = set()
    outputs = []
    for entry in h06_case["evidence"]:
        if not entry["improved_final_cost"]:
            continue
        pair = tuple(entry["flag_pair"])
        if pair in seen:
            raise ValueError("duplicate improved H0.6 target")
        seen.add(pair)
        target = entry["beam_lineage"]
        windows = GestureChunkPlannerV08()._windows(events)
        flagged_window = windows[entry["window_index"]]
        matching = [
            i for i, candidate in enumerate(_candidate_gestures(flagged_window))
            if [list(x) for x in candidate.axes] == entry["nonquick_axes"]
            and candidate.family == entry["nonquick_gesture_family"]
        ]
        if len(matching) != 1:
            raise ValueError("H0.6 nonquick candidate cannot be uniquely reconstructed")
        record = trace_one_slot(
            events, frozen_chunks,
            {**target, "flag_pair": entry["flag_pair"],
             "window_index": entry["window_index"],
             "nonquick_candidate_index": matching[0],
             "nonquick_join_expanded_rank": target["join_expanded_rank"],
             "nonquick_axes": entry["nonquick_axes"]},
            expected_cost=summary["abstract_schedule"]["metrics"]["beam_cost"],
            expected_evaluated=summary["abstract_schedule"]["candidates_evaluated"],
        )
        record["h06_forced_final_delta"] = entry["cost"]["final_delta"]
        record["h06_forced_changed_event_count"] = entry[
            "assignment_effects"]["changed_event_assignment_count"]
        outputs.append(record)
    if len(outputs) != h06_case["improved_count"]:
        raise ValueError("H0.6 improved cohort differs from source report")
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(events),
        "improved_h06_targets": len(outputs),
        "targets": outputs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--verified-report", type=Path, required=True)
    p.add_argument("--h0-report", type=Path, required=True)
    p.add_argument("--h01-report", type=Path, required=True)
    p.add_argument("--h06-report", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("refusing to overwrite previous H0.7 result")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    manifest, verified, h0, h01, h06 = map(
        load, (a.manifest, a.verified_report, a.h0_report,
               a.h01_report, a.h06_report))
    artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        root = a.results_root / sha[:12]
        summary_path = root / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary, load(root / "telemetry-v08.json"),
            load(root / "replay-v08.json"),
            load(a.results_root / "_provenance" / f"{sha[:12]}.json"))
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(a.manifest))
    if checked != h01:
        raise ValueError("H0.1 does not reproduce verified Train source")
    if (h06.get("format") != H06_FORMAT
            or h06.get("source_v08_commit") != checked["source_v08_commit"]
            or h06.get("frozen_code_digest") != checked["frozen_code_digest"]
            or h06.get("event_count") != checked["aggregate"]["event_count"]
            or h06.get("case_count") != len(manifest["cases"])):
        raise ValueError("H0.6 report provenance mismatch")
    by_sha = {c["chart_sha256"]: c for c in h06["cases"]}
    if len(by_sha) != len(manifest["cases"]):
        raise ValueError("missing or duplicated H0.6 chart identity")
    results = []
    train_root = Path(manifest["train_root"]).resolve()
    for case in manifest["cases"]:
        sha = case["sha256"]
        if sha not in by_sha:
            raise ValueError("H0.6 selection differs from frozen Train")
        chart = validate_chart(case, train_root)
        results.append(audit_case(
            chart, *artifacts[sha][:3], by_sha[sha]))
    rows = [t for c in results for t in c["targets"]]
    if len(rows) != h06["aggregate"]["improved_final_cost_count"]:
        raise ValueError("H0.7 improved cohort does not match H0.6")
    aggregate = {
        "targeted_improved_h06_cases": len(rows),
        "protected_reached_terminal": sum(
            row["protected_descendants_reached_terminal"] for row in rows),
        "protected_better_than_original": sum(
            row["best_protected_terminal"] is not None
            and row["best_protected_terminal"]["delta_cost_vs_frozen_v08"] < -1e-7
            for row in rows),
        "beam_winner_better_than_original": sum(
            row["beam_winner"]["delta_cost_vs_frozen_v08"] < -1e-7
            for row in rows),
        "beam_winner_from_protected_lineage": sum(
            row["beam_winner"]["is_from_protected_lineage"] for row in rows),
        "total_rescue_slot_replacements_across_separate_runs": sum(
            row["slot_replacement_count"] for row in rows),
    }
    report = {
        "format": FORMAT,
        "case_count": len(results),
        "event_count": sum(c["event_count"] for c in results),
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "aggregate": aggregate,
        "cases": results,
        "experimental_search_rule": (
            "one tagged lineage receives one of the unchanged 24 slots "
            "if otherwise absent, replacing the worst currently retained candidate"),
        "v08_frozen_source_changed": False,
        "physical_motor_executed": False,
        "human_likeness_verified": False,
        "causal_effect_established": False,
        "limitations": [
            "The shadow selection rule is a counterfactual policy, NOT unmodified frozen v0.8.",
            "One protected slot changes competition among other candidates and may eject v0.8's path.",
            "The 7 cases were selected after seeing H0.6 cost gains (selection bias).",
            "Individual interventions are independent and must not be combined as one chart.",
            "The six-term cost is not a measure of human preference or physical feasibility.",
            "No new Motor test, training, Validation20 or FINAL.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "aggregate": aggregate,
        "saved": str(a.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
