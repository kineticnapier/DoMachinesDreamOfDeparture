from __future__ import annotations

"""DMDOGO v0.9 H0.3 — controlled interpretation of H0.2 seam experiments.

Frozen v0.8 physical artifacts and original axes stay unchanged.
Compares one and the same H0.2 local surrogate with (A) the original
chunk seam and (B) moved seams. Revalidates original observations and
saved counterfactual axes. No Motor run or human-style judgment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureEvent, _Window, _candidate_gestures,
)
from analyze_dmdogo_v09_style_proxies import _validate
from analyze_dmdogo_v09_h01_conditions import compile_train_report
from analyze_dmdogo_v09_h02_boundary_cf import (
    FORMAT as H02_FORMAT, _chunk_for_event, _full_reuse, _local_replan,
    _partition_boundary_positions, _quick, _rapid_runs,
    _split_lengths, MAX_CHUNK,
)
from dmdogo_v08_provenance import sha256_file


FORMAT = "dmdogo-v09-h03-same-boundary-controls-v1"


def _replaced(rows: list[dict], start: int, end: int,
              axes: list[list[str]]) -> list[dict]:
    if len(axes) != end - start:
        raise ValueError("counterfactual axis count differs from targeted region")
    return (rows[:start] + [
        dict(rows[i], axes=tuple(axes[i - start])) for i in range(start, end)
    ] + rows[end:])


def _verify_candidate_grammar(rows: list[dict], start: int, end: int,
                              lengths: tuple[int, ...],
                              replacement_axes: list[list[str]],
                              join: int) -> dict:
    """Check that the saved candidate actually belongs to the frozen grammar.

    The pair at the *removed* seam is in one of the new chunks.
    Enumerating its grammar gives a structural explanation for flag loss.
    """
    cursor = start
    total = 0
    would_quick = 0
    joint_chunk_has_pair = False
    for length in lengths:
        piece = rows[cursor:cursor + length]
        window = _Window(
            tuple(GestureEvent(r["event_index"], r["due_ms"] / 1000.,
                               len(r["axes"])) for r in piece),
            "h03_grammar_audit", "untrained_local",
        )
        candidates = _candidate_gestures(window)
        desired = tuple(tuple(x) for x in replacement_axes[cursor - start:
                                                           cursor - start + length])
        if desired not in {candidate.axes for candidate in candidates}:
            raise ValueError("saved candidate not present in frozen grammar")
        if cursor < join < cursor + length:
            joint_chunk_has_pair = True
            offset = join - cursor
            for candidate in candidates:
                pair = [dict(rows[join - 1], axes=candidate.axes[offset - 1]),
                        dict(rows[join], axes=candidate.axes[offset])]
                if _quick(pair, 1):
                    would_quick += 1
                total += 1
        cursor += length
    if cursor != end or not joint_chunk_has_pair:
        raise ValueError("counterfactual did not actually remove the original seam")
    return {
        "candidate_sequences_at_join": total,
        "sequences_allowing_quick_jump_at_join": would_quick,
        "grammar_structurally_excludes_jump": total > 0 and would_quick == 0,
    }


def analyze_flag(rows: list[dict], chunks: list[dict], saved: dict) -> dict:
    """One original quick-flag seam, all H0.2 alternatives and one matched control."""
    prev_id, current_id = saved["flag_pair"]
    positions = {r["event_index"]: i for i, r in enumerate(rows)}
    if (len(positions) != len(rows)
            or current_id not in positions or prev_id not in positions):
        raise ValueError("quick flag event IDs missing or duplicated")
    join = positions[current_id]
    if join < 1 or positions[prev_id] != join - 1 or not _quick(rows, join):
        raise ValueError("target does not match original quick flag")

    chunk_for = _chunk_for_event(chunks)
    left_i = chunk_for[prev_id]
    right_i = chunk_for[current_id]
    if (right_i != left_i + 1
            or saved["original_chunk_indices"] != [left_i, right_i]):
        raise ValueError("flag does not cross original adjacent chunks")
    first, second = chunks[left_i], chunks[right_i]
    start = positions[first["start_event_index"]]
    end = positions[second["end_event_index"]] + 1
    original = (join - start, end - join)
    if list(original) != saved["original_lengths"] or max(original) > MAX_CHUNK:
        raise ValueError("original chunk sizes do not match saved experiment")
    if saved["original_quick_flag"] is not True:
        raise ValueError("H0.2 omitted the original flag")

    stored_control = saved["same_boundary_surrogate"]
    # Recompute the control using the same surrogate and identical pinned
    # outside axes; never compare this score to the frozen global beam cost.
    controlled = _local_replan(rows, start, end, original)
    if controlled != stored_control or not controlled["feasible"]:
        raise ValueError("saved same-boundary control changed or is infeasible")
    same_axes = _replaced(rows, start, end, controlled["axes"])
    if _full_reuse(same_axes)["violations"]:
        raise ValueError("same-boundary surrogate failed full-chart abstract reuse")
    baseline_flag = _quick(same_axes, join)
    baseline_rapid = len(_rapid_runs(same_axes))
    old_rapid = len(_rapid_runs(rows))

    alternatives = []
    seen_partitions = set()
    for proposal in saved["alternatives"]:
        lengths = tuple(proposal["lengths"])
        if lengths in seen_partitions or lengths == original:
            raise ValueError("duplicated or unchanged H0.2 partition")
        seen_partitions.add(lengths)
        if (sum(lengths) != end - start
                or len(lengths) not in (2, 3)
                or any(type(x) is not int or not 1 <= x <= MAX_CHUNK
                       for x in lengths)
                or join in _partition_boundary_positions(start, lengths)):
            raise ValueError("invalid moved-seam alternative")
        row = {
            "lengths": list(lengths),
            "same_chunk_count_as_original": len(lengths) == len(original),
            "abstract_feasible": proposal["feasible"],
        }
        if proposal["feasible"]:
            proposal_rows = _replaced(rows, start, end, proposal["axes"])
            reuse = _full_reuse(proposal_rows)
            if reuse["violations"]:
                raise ValueError("H0.2 alternative violates full-chart reuse")
            quick = _quick(proposal_rows, join)
            if quick != proposal["quick_at_original_seam"]:
                raise ValueError("H0.2 alternative quick flag mismatch")
            if (proposal["different_from_v08"] != sum(
                    a["axes"] != b["axes"]
                    for a, b in zip(proposal_rows[start:end], rows[start:end]))):
                raise ValueError("H0.2 changed-axes count mismatch")
            grammar = _verify_candidate_grammar(
                rows, start, end, lengths, proposal["axes"], join)
            row.update({
                "quick_at_original_join": quick,
                "different_from_original_axes": proposal["different_from_v08"],
                "different_from_same_boundary_axes": sum(
                    a["axes"] != b["axes"]
                    for a, b in zip(proposal_rows[start:end], same_axes[start:end])),
                "rapid_run_count_full_chart": len(_rapid_runs(proposal_rows)),
                "surrogate_cost": proposal["cost"],
                "cost_delta_vs_same_boundary": round(
                    proposal["cost"] - controlled["cost"], 8),
                "minimum_abstract_reuse_ms": reuse["minimum_ms"],
                "grammar_audit": grammar,
            })
        alternatives.append(row)

    if set(_split_lengths(end - start)) - {original}:
        expected = {
            x for x in _split_lengths(end - start)
            if join not in _partition_boundary_positions(start, x)
        }
    else:
        expected = set()
    if seen_partitions != expected:
        raise ValueError("H0.2 moved-seam partition population changed")
    feasible = [r for r in alternatives if r["abstract_feasible"]]
    same_chunk_count = [r for r in feasible if r["same_chunk_count_as_original"]]
    extra_chunk_count = [r for r in feasible if not r["same_chunk_count_as_original"]]
    flags_remaining = sum(r["quick_at_original_join"] for r in feasible)
    has_flag_free = flags_remaining < len(feasible)
    if not baseline_flag and has_flag_free:
        category = "replanning_alone_suffices_for_this_flag"
    elif baseline_flag and has_flag_free:
        category = "partition_associated_under_same_surrogate"
    elif baseline_flag:
        category = "flag_persists_across_feasible_studied_conditions"
    else:
        category = "baseline_flag_absent_no_flag_free_alternative"

    return {
        "flag_pair": [prev_id, current_id],
        "region_event_indices": [
            rows[start]["event_index"], rows[end - 1]["event_index"]],
        "original_lengths": list(original),
        "original_quick_flag": True,
        "same_boundary": {
            "abstract_feasible": True,
            "quick_at_original_join": baseline_flag,
            "different_from_frozen_axes": controlled["different_from_v08"],
            "surrogate_cost": controlled["cost"],
            "rapid_run_count_full_chart": baseline_rapid,
            "grammar_not_human_calibrated": True,
        },
        "frozen_full_chart_rapid_runs": old_rapid,
        "alternatives": alternatives,
        "feasible_alternative_count": len(feasible),
        "feasible_same_chunk_count_partitions": len(same_chunk_count),
        "feasible_extra_chunk_partitions": len(extra_chunk_count),
        "alternatives_retaining_quick": flags_remaining,
        "alternatives_removing_quick": len(feasible) - flags_remaining,
        "all_feasible_alternatives_grammar_exclude_quick": all(
            r["grammar_audit"]["grammar_structurally_excludes_jump"]
            for r in feasible),
        "classification": category,
    }


def analyze_case(summary: dict, telemetry: dict, replay: dict,
                 h01_case: dict, h02_case: dict) -> dict:
    rows, chunks, _ = _validate(summary, telemetry, replay)
    if (summary["chart_sha256"] != h01_case.get("chart_sha256")
            or summary["chart_sha256"] != h02_case.get("chart_sha256")
            or len(rows) != h02_case.get("event_count")):
        raise ValueError("source, H0.1 and H0.2 chart identity mismatch")
    observed_pairs = [
        [r["start_event_index"], r["end_event_index"]]
        for r in h01_case["quick_same_hand_depth_jump"]["detected"]]
    stored = h02_case["cases"]
    if [r["flag_pair"] for r in stored] != observed_pairs:
        raise ValueError("H0.2 selected flags do not match H0.1")
    comparisons = [analyze_flag(rows, chunks, entry) for entry in stored]
    classifications = Counter(r["classification"] for r in comparisons)
    return {
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(rows),
        "flag_count": len(comparisons),
        "classification_counts": dict(sorted(classifications.items())),
        "same_boundary_still_flag_count": sum(
            x["same_boundary"]["quick_at_original_join"] for x in comparisons),
        "same_boundary_removed_flag_count": sum(
            not x["same_boundary"]["quick_at_original_join"] for x in comparisons),
        "feasible_moved_seam_variants": sum(
            x["feasible_alternative_count"] for x in comparisons),
        "feasible_equal_chunk_count_variants": sum(
            x["feasible_same_chunk_count_partitions"] for x in comparisons),
        "feasible_extra_chunk_variants": sum(
            x["feasible_extra_chunk_partitions"] for x in comparisons),
        "all_feasible_grammar_exclude_quick": all(
            x["all_feasible_alternatives_grammar_exclude_quick"]
            for x in comparisons),
        "flags": comparisons,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--verified-report", required=True, type=Path)
    p.add_argument("--h0-report", required=True, type=Path)
    p.add_argument("--h01-report", required=True, type=Path)
    p.add_argument("--h02-report", required=True, type=Path)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("refusing to overwrite H0.3 result")
    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = load(a.manifest)
    verified = load(a.verified_report)
    h0 = load(a.h0_report)
    h01 = load(a.h01_report)
    h02 = load(a.h02_report)
    source = {}
    for case in manifest["cases"]:
        sha = case["sha256"]
        directory = a.results_root / sha[:12]
        summary_path = directory / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        source[sha] = (
            summary, load(directory / "telemetry-v08.json"),
            load(directory / "replay-v08.json"),
            load(a.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    checked_h01 = compile_train_report(
        manifest, verified, h0, source,
        manifest_digest=sha256_file(a.manifest))
    if checked_h01 != h01:
        raise ValueError("H0.1 result does not match verified frozen source")
    if (h02.get("format") != H02_FORMAT
            or h02.get("source_v08_commit") != checked_h01["source_v08_commit"]
            or h02.get("frozen_code_digest") != checked_h01["frozen_code_digest"]
            or h02.get("case_count") != len(manifest["cases"])
            or h02.get("event_count") !=
            checked_h01["aggregate"]["event_count"]
            or len(h02.get("cases", [])) != len(manifest["cases"])):
        raise ValueError("H0.2 report source or cohort mismatch")
    h02_by_sha = {case["chart_sha256"]: case for case in h02["cases"]}
    if set(h02_by_sha) != {case["sha256"] for case in manifest["cases"]}:
        raise ValueError("H0.2 cases do not match verified Train selection")
    results = []
    for case in manifest["cases"]:
        sha = case["sha256"]
        summary, telemetry, replay, _ = source[sha]
        h01_case = next(c for c in checked_h01["cases"]
                        if c["chart_sha256"] == sha)
        results.append(analyze_case(
            summary, telemetry, replay, h01_case, h02_by_sha[sha]))
    counts = Counter()
    for report in results:
        counts.update(report["classification_counts"])
    total = sum(x["flag_count"] for x in results)
    output = {
        "format": FORMAT,
        "source_v08_commit": checked_h01["source_v08_commit"],
        "frozen_code_digest": checked_h01["frozen_code_digest"],
        "event_count": sum(x["event_count"] for x in results),
        "case_count": len(results),
        "aggregate": {
            "targeted_flags": total,
            "same_boundary_still_flag": sum(
                x["same_boundary_still_flag_count"] for x in results),
            "same_boundary_flag_removed": sum(
                x["same_boundary_removed_flag_count"] for x in results),
            "feasible_moved_seam_variants": sum(
                x["feasible_moved_seam_variants"] for x in results),
            "feasible_equal_chunk_count_variants": sum(
                x["feasible_equal_chunk_count_variants"] for x in results),
            "feasible_extra_chunk_variants": sum(
                x["feasible_extra_chunk_variants"] for x in results),
            "classification_counts": dict(sorted(counts.items())),
            "grammar_all_eligible_variants_exclude_quick": all(
                x["all_feasible_grammar_exclude_quick"] for x in results),
        },
        "cases": results,
        "same_surrogate_comparison_only": True,
        "human_likeness_verified": False,
        "physical_motor_executed": False,
        "causal_effect_established": False,
        "limitations": [
            "Same-boundary replanning alone may remove a flag; compare it explicitly.",
            "Merging the original flagged pair into one chunk changes the grammar.",
            "Within-chunk gesture grammar may make the 106/106 result structural.",
            "Two- vs three-chunk variants differ in both seam position and chunk count.",
            "Same local surrogate is used, but frozen global beam is not reproduced.",
            "Interventions overlap and are evaluated one at a time, not as a new full run.",
            "Best-of-many feasible variants are a selected, not randomized, cohort.",
            "80ms abstract feasibility is not physical Motor or human feasibility.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "event_count": output["event_count"],
        "aggregate": output["aggregate"],
        "saved": str(a.output),
        "human_likeness_verified": False,
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
