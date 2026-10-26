from __future__ import annotations

"""H0.5: track the best same-predecessor non-quick gesture through the frozen beam.

Two distinct analyses:
  (1) original unrestricted width-24 beam: when does that exact alternative
      lineage disappear among OTHER competing paths?
  (2) constrained branch: force the alternative after the actual frozen
      predecessor, then restart the UNMODIFIED scoring/beam rule downstream.
No Motor, learned imitation or human-performance claim.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent, _candidate_gestures,
)
from analyze_dmdogo_v09_style_proxies import _validate
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from audit_dmdogo_v09_h04_beam_decisions import (
    FORMAT as H04_FORMAT, TERMS, State, _expand, _quick_at_join,
    _oracle_events, _chunk_record, audit_case,
)
from dmdogo_v08_provenance import sha256_file
from run_dmdogo_v08_verified_train_case import validate_chart

FORMAT = "dmdogo-v09-h05-nonquick-beam-lineage-v1"


@dataclass(slots=True)
class Branch:
    score: float
    previous: dict
    last_axis: str | None
    exposure: tuple[float, float]
    exposure_t: float
    terms: tuple[float, ...]
    frozen_prefix: bool = False
    tags: frozenset[int] = frozenset()


def _seed_state(event_time: float) -> Branch:
    return Branch(0.0, {}, None, (0.0, 0.0), event_time,
                  (0.0,) * len(TERMS), True)


def _next(state: Branch, window, candidate, previous_due,
          *, frozen_candidate: bool) -> Branch | None:
    value = _expand(
        State(state.score, state.previous, state.last_axis,
              state.exposure, state.exposure_t, None),
        window, candidate, previous_due,
    )
    if value is None:
        return None
    total, last_use, last_axis, exposure, last_time, term_map, _, _ = value
    return Branch(
        total, last_use, last_axis, exposure, last_time,
        tuple(old + term_map[name] for old, name in zip(state.terms, TERMS)),
        state.frozen_prefix and frozen_candidate,
        state.tags,
    )


def _cost_terms(branch: Branch) -> dict:
    return {name: round(value, 9)
            for name, value in zip(TERMS, branch.terms)}


def _same_score(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=0, abs_tol=1e-6)


def trace_unrestricted(events: tuple[GestureEvent, ...],
                       frozen_chunks: list[dict],
                       h04_evidence: list[dict],
                       expected_cost: float,
                       expected_evaluated: int) -> tuple[list[dict], dict[int, Branch], Branch]:
    """Exact global beam with tags propagated from non-quick alternatives.

    Marking a tag NEVER changes candidate generation, scores, sorting or
    pruning; comparison is thus against the genuine frozen global search.
    """
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    if len(windows) != len(frozen_chunks):
        raise ValueError("original frozen chunk boundaries changed")
    if not events:
        raise ValueError("empty Oracle events")

    targeted: dict[int, dict] = {}
    for idx, evidence in enumerate(h04_evidence):
        pair = evidence["flag_pair"]
        hits = [j for j, win in enumerate(windows)
                if win.events[0].event_index == pair[1]]
        if len(hits) != 1 or hits[0] == 0:
            raise ValueError("H0.4 target is not a unique chunk seam")
        win_id = hits[0]
        if win_id in targeted:
            raise ValueError("duplicate H0.4 target")
        if windows[win_id - 1].events[-1].event_index != pair[0]:
            raise ValueError("H0.4 target has nonadjacent chunks")
        targeted[win_id] = {"id": idx, "evidence": evidence}

    states = [_seed_state(events[0].due_s)]
    counted = 0
    records = {}
    seeds = {}
    for win_id, window in enumerate(windows):
        previous_due = (windows[win_id - 1].events[-1].due_s
                        if win_id else None)
        gap = (window.events[0].due_s - previous_due
               if previous_due is not None else float("inf"))
        candidates = _candidate_gestures(window)
        chosen_axes = tuple(tuple(row) for row in frozen_chunks[win_id]["axes"])
        expected_family = frozen_chunks[win_id]["gesture_family"]
        expansions = []
        for parent_rank, parent in enumerate(states, 1):
            for ci, candidate in enumerate(candidates):
                counted += 1
                next_branch = _next(
                    parent, window, candidate, previous_due,
                    frozen_candidate=(candidate.axes == chosen_axes
                                      and candidate.family == expected_family),
                )
                if next_branch is None:
                    continue
                quick = bool(
                    win_id and len(window.events[0:1]) == 1
                    and _quick_at_join(parent.last_axis, candidate.axes[0][0], gap)
                )
                expansions.append((
                    next_branch.score, next_branch, parent_rank, ci,
                    quick, candidate,
                ))
        if not expansions:
            raise ValueError(f"frozen global beam infeasible at chunk {win_id}")
        # Same stable score-only sort as frozen v0.8, including exact ties.
        expansions.sort(key=lambda row: row[0])
        new_tag = targeted.get(win_id)
        if new_tag is not None:
            target_id = new_tag["id"]
            ev = new_tag["evidence"]
            parent_frozen = [i for i, row in enumerate(states, 1)
                             if row.frozen_prefix]
            if len(parent_frozen) != 1 or parent_frozen[0] != ev["prior_beam_rank"]:
                raise ValueError("frozen selected predecessor rank mismatch")
            possible = [(rank, row)
                        for rank, row in enumerate(expansions, 1)
                        if row[2] == parent_frozen[0] and not row[4]]
            if not possible:
                raise ValueError("H0.4 says nonquick candidate exists but replay cannot find one")
            rank, chosen = possible[0]  # expansions are already score-sorted.
            if (ev["best_same_parent_nonquick"] is None
                    or not _same_score(
                        ev["best_same_parent_nonquick"]["cumulative_cost_delta_from_chosen"],
                        chosen[0] - ev["chosen_cumulative_cost"])
                    or ev["best_same_parent_nonquick"]["expanded_rank"] != rank
                    or ev["best_same_parent_nonquick"]["first_axes"]
                    != list(chosen[5].axes[0])
                    or ev["best_same_parent_nonquick"]["gesture_family"]
                    != chosen[5].family):
                raise ValueError("best same-predecessor nonquick choice disagrees with H0.4")
            picked = chosen[1]
            picked.tags = picked.tags | frozenset({target_id})
            seeds[target_id] = picked
            records[target_id] = {
                "flag_pair": ev["flag_pair"],
                "window_index": win_id,
                "original_selected_parent_beam_rank": parent_frozen[0],
                "original_chosen_expanded_rank": ev["chosen_expanded_rank"],
                "nonquick_candidate_index": chosen[3],
                "nonquick_gesture_family": chosen[5].family,
                "nonquick_axes": [list(x) for x in chosen[5].axes],
                "nonquick_join_expanded_rank": rank,
                "nonquick_join_survives_original_beam": rank <= planner.beam_width,
                "nonquick_immediate_cost_delta": round(
                    chosen[0] - ev["chosen_cumulative_cost"], 9),
                "first_lineage_lost_window": None,
                "lineage_loss_reason": None,
                "lineage_terminal_survived": False,
                "lineage_terminal_best_cost": None,
                "lineage_terminal_best_rank": None,
            }
        tags_before = set().union(*(state.tags for state in states))
        # Source tag is introduced at this step, after scoring but before cut.
        after_rows = expansions[:planner.beam_width]
        states = [row[1] for row in after_rows]
        live = set().union(*(state.tags for state in states))
        all_generated_tags = set().union(*(row[1].tags for row in expansions))
        if new_tag is not None:
            tag = new_tag["id"]
            if tag not in live:
                records[tag]["first_lineage_lost_window"] = win_id
                records[tag]["lineage_loss_reason"] = "initial_beam_pruning"
        for tag in tags_before - live:
            records[tag]["first_lineage_lost_window"] = win_id
            records[tag]["lineage_loss_reason"] = (
                "beam_pruning" if tag in all_generated_tags
                else "abstract_reuse_dead_end"
            )
        if len([state for state in states if state.frozen_prefix]) != 1:
            raise ValueError(f"original frozen path pruned at chunk {win_id}")
    if counted != expected_evaluated:
        raise ValueError("global beam evaluated count not identical to frozen result")
    if not _same_score(round(states[0].score, 6), expected_cost):
        raise ValueError("original frozen beam cost differs")
    if not states[0].frozen_prefix:
        raise ValueError("canonical global winner differs from frozen chosen path")
    for tag, info in records.items():
        survivors = [(i, state) for i, state in enumerate(states, 1)
                     if tag in state.tags]
        if survivors:
            info["lineage_terminal_survived"] = True
            info["lineage_terminal_best_rank"] = survivors[0][0]
            info["lineage_terminal_best_cost"] = round(survivors[0][1].score, 9)
        if not survivors and info["first_lineage_lost_window"] is None:
            raise ValueError("tracked original-beam lineage vanished without reason")
    return [records[k] for k in range(len(h04_evidence))], seeds, states[0]


def _force_suffix(seed: Branch, windows, start_window: int, *,
                  beam_width: int = 24) -> dict:
    """Condition on ONE selected nonquick gesture after the real predecessor.

    This is *not* the original unrestricted beam population. All downstream
    alternatives retain that forced seed; they do compete among themselves.
    """
    states = [seed]
    counted = 0
    for win_id in range(start_window + 1, len(windows)):
        window = windows[win_id]
        previous_due = windows[win_id - 1].events[-1].due_s
        candidates = _candidate_gestures(window)
        expanded = []
        for parent in states:
            for candidate in candidates:
                counted += 1
                branch = _next(parent, window, candidate, previous_due,
                               frozen_candidate=False)
                if branch is not None:
                    expanded.append(branch)
        if not expanded:
            return {
                "reached_terminal": False,
                "failure_reason": "no_reuse_safe_beam_continuation",
                "first_failed_chunk_index": win_id,
                "suffix_candidates_evaluated": counted,
                "final_cost": None,
                "final_cost_terms": None,
            }
        expanded.sort(key=lambda item: item.score)
        states = expanded[:beam_width]
    return {
        "reached_terminal": True,
        "failure_reason": None,
        "first_failed_chunk_index": None,
        "suffix_candidates_evaluated": counted,
        "final_cost": round(states[0].score, 9),
        "final_cost_terms": _cost_terms(states[0]),
    }


def audit_case(chart: Path, summary: dict, telemetry: dict, replay: dict,
               h01_case: dict, h04_case: dict) -> dict:
    rows, chunks, _ = _validate(summary, telemetry, replay)
    if summary["chart_sha256"] != h04_case.get("chart_sha256"):
        raise ValueError("saved H0.4 chart identity mismatch")
    # H0.4 recomputation explicitly compares against the physical v0.8 source
    # and canonical frozen planner; no approximate 9-decimal timestamp replay.
    independently_rechecked = audit_case_h04(chart, summary, telemetry,
                                             replay, h01_case)
    if independently_rechecked != h04_case:
        raise ValueError("saved H0.4 report differs from recomputed frozen audit")
    events = _oracle_events(chart)
    expected = summary["abstract_schedule"]
    original_cost = expected["metrics"]["beam_cost"]
    detail, seeds, frozen_winner = trace_unrestricted(
        events, chunks, h04_case["flag_evidence"],
        original_cost, expected["candidates_evaluated"],
    )
    windows = GestureChunkPlannerV08()._windows(events)
    output = []
    for i, item in enumerate(detail):
        forced = _force_suffix(seeds[i], windows, item["window_index"])
        if forced["reached_terminal"]:
            forced["final_cost_delta_from_original_v08"] = round(
                forced["final_cost"] - original_cost, 9)
            forced["final_cost_component_deltas_from_original_v08"] = {
                name: round(forced["final_cost_terms"][name]
                            - _cost_terms(frozen_winner)[name], 9)
                for name in TERMS
            }
        else:
            forced["final_cost_delta_from_original_v08"] = None
            forced["final_cost_component_deltas_from_original_v08"] = None
        output.append({**item, "forced_nonquick_suffix": forced})
    by_type = Counter(item["lineage_loss_reason"] or "terminal_survivor"
                      for item in output)
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(rows),
        "chunk_count": len(windows),
        "original_cost": original_cost,
        "original_global_beam_exactly_reproduced": True,
        "targeted_count": len(output),
        "lineage_outcome_counts": dict(sorted(by_type.items())),
        "forced_suffix_reached_terminal": sum(
            item["forced_nonquick_suffix"]["reached_terminal"]
            for item in output),
        "forced_suffix_lower_final_cost": sum(
            item["forced_nonquick_suffix"]["final_cost_delta_from_original_v08"]
            is not None and
            item["forced_nonquick_suffix"]["final_cost_delta_from_original_v08"] < -1e-7
            for item in output),
        "flag_evidence": output,
    }


# Avoid shadowing the local audit_case function.
from audit_dmdogo_v09_h04_beam_decisions import audit_case as audit_case_h04


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--verified-report", required=True, type=Path)
    p.add_argument("--h0-report", required=True, type=Path)
    p.add_argument("--h01-report", required=True, type=Path)
    p.add_argument("--h04-report", required=True, type=Path)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H0.5 result")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = load(args.manifest)
    verified = load(args.verified_report)
    h0 = load(args.h0_report)
    h01 = load(args.h01_report)
    h04 = load(args.h04_report)
    artifacts = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        prefix = args.results_root / sha[:12]
        summary_path = prefix / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (summary,
                          load(prefix / "telemetry-v08.json"),
                          load(prefix / "replay-v08.json"),
                          load(args.results_root / "_provenance" / f"{sha[:12]}.json"))
    checked = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest),
    )
    if checked != h01:
        raise ValueError("saved H0.1 report does not match verified Train artifacts")
    if (h04.get("format") != H04_FORMAT
            or h04.get("source_v08_commit") != checked["source_v08_commit"]
            or h04.get("frozen_code_digest") != checked["frozen_code_digest"]
            or h04.get("event_count") != checked["aggregate"]["event_count"]
            or h04.get("case_count") != len(manifest["cases"])
            or len(h04.get("cases", [])) != len(manifest["cases"])):
        raise ValueError("H0.4 report source identity mismatch")
    h04_by_sha = {case["chart_sha256"]: case for case in h04["cases"]}
    if set(h04_by_sha) != {case["sha256"] for case in manifest["cases"]}:
        raise ValueError("H0.4 case selection changed")
    train_root = Path(manifest["train_root"]).resolve()
    reports = []
    for case in manifest["cases"]:
        chart = validate_chart(case, train_root)
        summary, telemetry, replay, _ = artifacts[case["sha256"]]
        h01_case = next(item for item in checked["cases"]
                        if item["chart_sha256"] == case["sha256"])
        reports.append(audit_case(
            chart, summary, telemetry, replay, h01_case,
            h04_by_sha[case["sha256"]],
        ))
    agg = Counter()
    reasons = Counter()
    for result in reports:
        agg["targeted_quick_flags"] += result["targeted_count"]
        agg["unrestricted_lineage_terminal_survivors"] += (
            result["lineage_outcome_counts"].get("terminal_survivor", 0))
        agg["forced_suffix_reached_terminal"] += result[
            "forced_suffix_reached_terminal"]
        agg["forced_suffix_lower_final_cost"] += result[
            "forced_suffix_lower_final_cost"]
        reasons.update(result["lineage_outcome_counts"])
    report = {
        "format": FORMAT,
        "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "case_count": len(reports),
        "event_count": sum(item["event_count"] for item in reports),
        "aggregate": {**dict(agg),
                      "unrestricted_lineage_outcome_counts":
                          dict(sorted(reasons.items()))},
        "cases": reports,
        "physical_motor_executed": False,
        "human_likeness_verified": False,
        "causal_effect_established": False,
        "limitations": [
            "The forced suffix is a different beam population from the original unrestricted beam.",
            "Only the least-immediate-cost nonquick choice under the exact frozen predecessor is tested at each flag.",
            "Separate single-flag interventions need not combine into one globally feasible chart.",
            "Negative forced final cost delta is a bounded-beam search gap, not proof of global optimality.",
            "Term totals describe untrained heuristic priors, not human comfort or preferred fingering.",
            "Feasibility is abstract v0.8 reuse only; physical Motor was not executed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "case_count": report["case_count"],
        "aggregate": report["aggregate"],
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
