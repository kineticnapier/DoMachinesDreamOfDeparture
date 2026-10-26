from __future__ import annotations

"""H0.2: chunk-seam relabelling and LOCAL counterfactual re-planning.

Exploratory study only. The frozen v0.8 candidate grammar is imported read-only;
its beam planner, Motor, physical controller and files are NEVER mutated/run.
The local surrogate objective is not identical to v0.8's global beam.
"""

import argparse
from collections import Counter
import json
from math import exp
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureEvent, _Window, _candidate_gestures,
)
from analyze_dmdogo_v09_style_proxies import _hand, _depth, _validate, HAND_GAP_MS
from analyze_dmdogo_v09_h01_conditions import (
    analyze_conditions, compile_train_report,
)
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-v09-h02-local-boundary-counterfactual-v1"
BEAM_WIDTH = 24
REUSE_MS = 80.0
MAX_CHUNK = 4
MAX_REPARTITIONS = 32


def _boundaries(chunks: list[dict]) -> set[int]:
    """Event ID of first event after a chunk seam."""
    return {c["start_event_index"] for c in chunks[1:]}


def _split_lengths(n: int) -> list[tuple[int, ...]]:
    """Predeclared variants, no optimization of split thresholds."""
    options = []
    for a in range(1, MAX_CHUNK + 1):
        for b in range(1, MAX_CHUNK + 1):
            if a + b == n:
                options.append((a, b))
            for c in range(1, MAX_CHUNK + 1):
                if a + b + c == n:
                    options.append((a, b, c))
    return sorted(set(options), key=lambda s: (len(s), s))


def _chunk_for_event(chunks: list[dict]) -> dict[int, int]:
    by_id = {}
    for i, chunk in enumerate(chunks):
        for event_id in range(chunk["start_event_index"], chunk["end_event_index"] + 1):
            if event_id in by_id:
                raise ValueError("overlapping chunks")
            by_id[event_id] = i
    return by_id


def _quick(rows: list[dict], i: int) -> bool:
    """Exact original H0 quick flag predicate; independent of chunk labels."""
    a, b = rows[i - 1:i + 1]
    return (0 <= b["due_ms"] - a["due_ms"] <= HAND_GAP_MS
            and len(a["axes"]) == len(b["axes"]) == 1
            and _hand(a["axes"][0]) == _hand(b["axes"][0])
            and abs(_depth(a["axes"][0]) - _depth(b["axes"][0])) >= 2)


def _rapid_runs(rows: list[dict]) -> list[tuple[int, int]]:
    """Maximal H0 same-hand single-axis runs of length >=4."""
    if not rows:
        return []
    result = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i < len(rows):
            a, b = rows[i-1], rows[i]
            same = (len(a["axes"]) == len(b["axes"]) == 1
                    and _hand(a["axes"][0]) == _hand(b["axes"][0])
                    and 0 <= b["due_ms"] - a["due_ms"] <= HAND_GAP_MS)
        else:
            same = False
        if not same:
            if i - start >= 4:
                result.append((start, i - 1))
            start = i
    return result


def _full_reuse(rows: list[dict]) -> dict:
    last = {}
    count = 0
    minimum = float("inf")
    for row in rows:
        for axis in row["axes"]:
            if axis in last:
                gap = row["due_ms"] - last[axis]
                minimum = min(minimum, gap)
                if gap + 1e-6 < REUSE_MS:
                    count += 1
            last[axis] = row["due_ms"]
    return {"violations": count,
            "minimum_ms": round(minimum, 6) if minimum != float("inf") else None}


def _hand_load(rows: list[dict]) -> dict:
    contacts = Counter(_hand(a) for r in rows for a in r["axes"])
    total = sum(contacts.values())
    return {"left": contacts["left"], "right": contacts["right"],
            "left_fraction": round(contacts["left"] / total, 6) if total else None}


def _score_tie(state: tuple) -> tuple:
    return (state[0], state[1])


def _local_replan(rows: list[dict], start: int, end: int,
                  lengths: tuple[int, ...]) -> dict:
    """Independent bounded research optimizer with pinned outside assignments.

    Same candidate grammar and 80ms reuse constraint as v0.8, but a surrogate
    local objective. It does NOT reproduce the global beam state or claim a
    physical valid execution. The returned objective is comparable only to
    another variant of this SAME local optimizer.
    """
    if sum(lengths) != end - start or any(not 1 <= n <= MAX_CHUNK for n in lengths):
        raise ValueError("invalid local partition")
    frozen = [(r["due_ms"], r["axes"]) for r in rows[:start]]
    last = {}
    for when, axes in frozen:
        for axis in axes:
            last[axis] = when
    previous_axis = rows[start - 1]["axes"][-1] if start else None
    previous_due = rows[start - 1]["due_ms"] if start else None
    state = [(0., (), last, previous_axis, (0., 0.),
              rows[start - 1]["due_ms"] if start else rows[start]["due_ms"])]
    evaluated = 0
    cursor = start
    for length in lengths:
        part = rows[cursor:cursor + length]
        window = _Window(
            tuple(GestureEvent(r["event_index"], r["due_ms"] * .001,
                               len(r["axes"])) for r in part),
            "h02_counterfactual", "untrained_local",
        )
        options = _candidate_gestures(window)
        next_state = []
        for cost, assigned, used, prior_axis, exposure, last_clock in state:
            for gesture in options:
                evaluated += 1
                if len(gesture.axes) != length:
                    continue
                new_last = used.copy()
                gain = float(gesture.preference)
                hand_exposure = list(exposure)
                at_last = last_clock
                possible = True
                for row, axes in zip(part, gesture.axes):
                    at = row["due_ms"]
                    decay = exp(-max(0., at - at_last) / 750.)
                    hand_exposure = [v * decay for v in hand_exposure]
                    at_last = at
                    if len(set(axes)) != len(axes):
                        possible = False
                        break
                    for axis in axes:
                        hi = 0 if _hand(axis) == "left" else 1
                        gain += .45 * hand_exposure[hi]
                        hand_exposure[hi] += 1.
                        gap = at - new_last[axis] if axis in new_last else None
                        if gap is not None:
                            if gap <= 1e-9 or gap + 1e-6 < REUSE_MS:
                                possible = False
                                break
                            gain += max(0., 150. - gap) * .002
                        new_last[axis] = at
                    if not possible:
                        break
                if not possible:
                    continue
                flat = [axis for group in gesture.axes for axis in group]
                gain += .35 * sum(_hand(a) != _hand(b)
                                   for a, b in zip(flat, flat[1:]))
                gain += .10 * sum(max(0, abs(_depth(a) - _depth(b)) - 1)
                                   for a, b in zip(flat, flat[1:])
                                   if _hand(a) == _hand(b))
                if prior_axis and flat and previous_due is not None:
                    gap = part[0]["due_ms"] - previous_due
                    if gap <= 50.:
                        gain += (.12 if _hand(prior_axis) != _hand(flat[0])
                                 else .05 * max(0, abs(_depth(prior_axis) -
                                                       _depth(flat[0])) - 1))
                next_state.append((cost + gain, assigned + gesture.axes,
                                   new_last, flat[-1] if flat else prior_axis,
                                   tuple(hand_exposure), at_last))
        if not next_state:
            return {"feasible": False, "reason": "no_abstract_candidate",
                    "candidates_evaluated": evaluated}
        next_state.sort(key=_score_tie)
        state = next_state[:BEAM_WIDTH]
        cursor += length
        previous_due = part[-1]["due_ms"]

    eligible = []
    for state_row in state:
        _, assignments, _, _, _, _ = state_row
        replacement = [dict(r, axes=tuple(axes))
                       for r, axes in zip(rows[start:end], assignments)]
        merged = rows[:start] + replacement + rows[end:]
        if _full_reuse(merged)["violations"] == 0:
            eligible.append((state_row, replacement))
    if not eligible:
        return {"feasible": False, "reason": "suffix_reuse_or_beam_rejection",
                "candidates_evaluated": evaluated}
    best, replacement = min(eligible, key=lambda item: _score_tie(item[0]))
    return {"feasible": True,
            "cost": round(best[0], 8), "candidates_evaluated": evaluated,
            "axes": [list(row["axes"]) for row in replacement],
            "different_from_v08": sum(a["axes"] != b["axes"]
                                      for a, b in zip(replacement, rows[start:end]))}


def _partition_boundary_positions(start: int, lengths: tuple[int, ...]) -> set[int]:
    cursor = start
    result = set()
    for size in lengths[:-1]:
        cursor += size
        result.add(cursor)
    return result


def _counterfactual_at_flag(rows: list[dict], chunks: list[dict],
                            current_id: int) -> dict:
    by_id = _chunk_for_event(chunks)
    if current_id not in by_id or current_id - 1 not in by_id:
        raise ValueError("flag references missing event")
    left_i = by_id[current_id - 1]
    right_i = by_id[current_id]
    if right_i != left_i + 1:
        raise ValueError("quick jump is not at adjacent chunk boundary")
    left, right = chunks[left_i], chunks[right_i]
    start_id = left["start_event_index"]
    end_id = right["end_event_index"]
    positions = {r["event_index"]: i for i, r in enumerate(rows)}
    if any(event not in positions for event in (start_id, end_id, current_id)):
        raise ValueError("chunk IDs not found in ordered events")
    start = positions[start_id]
    end = positions[end_id] + 1
    join = positions[current_id]
    if join == start or join == end or rows[join-1]["event_index"] != current_id - 1:
        raise ValueError("nonconsecutive or invalid chunk boundary")
    original = (join - start, end - join)
    if any(n > MAX_CHUNK for n in original):
        raise ValueError("frozen chunk exceeds 4-event limit")
    options = [split for split in _split_lengths(end - start)
               if join not in _partition_boundary_positions(start, split)]
    if len(options) > MAX_REPARTITIONS:
        raise ValueError("unexpectedly many repartitions")
    baseline = _local_replan(rows, start, end, original)
    baseline_reuse = _full_reuse(rows)
    if baseline_reuse["violations"]:
        raise ValueError("frozen input violates abstract reuse assumption")
    runs_before = _rapid_runs(rows)
    before_rapid = sum(not (b < start or a >= end) for a, b in runs_before)
    baseline_seams = _boundaries(chunks)
    intersecting_rapid = [(a, b) for a, b in runs_before
                          if not (b < start or a >= end)]
    def crossing_count(seams: set[int]) -> int:
        return sum(any(rows[i]["event_index"] in seams
                       for i in range(a + 1, b + 1))
                   for a, b in intersecting_rapid)
    original_rapid_crossing_count = crossing_count(baseline_seams)
    proposals = []
    for lengths in options:
        model = _local_replan(rows, start, end, lengths)
        proposed_seams = (
            baseline_seams - {current_id}
            | {rows[i]["event_index"]
               for i in _partition_boundary_positions(start, lengths)}
        )
        row = {"lengths": list(lengths),
               "original_seam_removed": True,
               "extra_chunk": len(lengths) != len(original),
               "fixed_axes_rapid_crossing_runs": crossing_count(proposed_seams),
               **model}
        if model["feasible"]:
            replacement = [dict(r, axes=tuple(a)) for r, a in zip(
                rows[start:end], model["axes"])]
            merged = rows[:start] + replacement + rows[end:]
            row["quick_at_original_seam"] = _quick(merged, join)
            row["local_rapid_runs_intersecting_region"] = sum(
                not (b < start or a >= end) for a, b in _rapid_runs(merged))
            row["abstract_reuse"] = _full_reuse(merged)
            row["local_hand_contacts"] = _hand_load(replacement)
            if baseline["feasible"]:
                row["surrogate_cost_delta_vs_same_boundary"] = round(
                    model["cost"] - baseline["cost"], 8)
        proposals.append(row)
    valid = [p for p in proposals if p["feasible"]]
    selected = min(valid, key=lambda p: (p["cost"], p["lengths"])) if valid else None
    return {
        "flag_pair": [current_id - 1, current_id],
        "original_chunk_indices": [left_i, right_i],
        "original_lengths": list(original),
        "original_quick_flag": _quick(rows, join),
        "fixed_axes_quick_flag_under_all_relabellings": _quick(rows, join),
        "fixed_axes_rapid_runs_intersecting_region": before_rapid,
        "fixed_axes_original_rapid_crossing_runs": original_rapid_crossing_count,
        "boundary_relabels_tested": len(options),
        "fixed_axes_relabels_making_original_pair_within_chunk": len(options),
        "same_boundary_surrogate": baseline,
        "alternative_feasible_count": len(valid),
        "alternative_infeasible_count": len(options) - len(valid),
        "alternative_quick_flag_still_present": sum(
            p["quick_at_original_seam"] for p in valid),
        "alternative_quick_flag_absent": sum(
            not p["quick_at_original_seam"] for p in valid),
        "alternatives": proposals,
        "lowest_surrogate_cost_alternative": selected,
    }


def analyze_case(summary: dict, telemetry: dict, replay: dict,
                 verified_h01_case: dict) -> dict:
    rows, chunks, _ = _validate(summary, telemetry, replay)
    observed = analyze_conditions(summary, telemetry, replay)
    if (observed["quick_same_hand_depth_jump"]["detected"] !=
            verified_h01_case["quick_same_hand_depth_jump"]["detected"] or
            observed["rapid_single_hand_sequence"]["detected"] !=
            verified_h01_case["rapid_single_hand_sequence"]["detected"]):
        raise ValueError("H0.1 detected regions differ from verified input")
    flagged = observed["quick_same_hand_depth_jump"]["detected"]
    pairs = [_counterfactual_at_flag(rows, chunks, r["end_event_index"])
             for r in flagged]
    fixed_count = sum(_quick(rows, i) for i in range(1, len(rows)))
    if fixed_count != len(flagged):
        raise ValueError("H0.1 quick flags not reproduced from frozen axes")
    return {
        "chart": summary["chart"], "chart_sha256": summary["chart_sha256"],
        "event_count": len(rows), "original_quick_count": len(flagged),
        "original_rapid_count": len(_rapid_runs(rows)),
        "original_chunk_count": len(chunks),
        "original_abstract_reuse": _full_reuse(rows),
        "original_hand_counts": _hand_load(rows),
        "cases": pairs,
        "rollup": {
            "targeted_flags": len(pairs),
            "seams_relabelled_with_fixed_axes": sum(
                bool(p["boundary_relabels_tested"]) for p in pairs),
            "flags_unchanged_when_only_labels_change": len(pairs),
            "local_baseline_abstract_feasible": sum(
                p["same_boundary_surrogate"]["feasible"] for p in pairs),
            "feasible_alternative_repartitions": sum(
                p["alternative_feasible_count"] for p in pairs),
            "feasible_alternatives_without_quick_flag": sum(
                p["alternative_quick_flag_absent"] for p in pairs),
            "eligible_cases_with_some_flag_absent_alternative": sum(
                p["alternative_quick_flag_absent"] > 0 for p in pairs),
        },
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
        p.error("refusing to overwrite H0.2 output")
    load = lambda x: json.loads(x.read_text(encoding="utf-8"))
    manifest = load(a.manifest)
    verified = load(a.verified_report)
    h0 = load(a.h0_report)
    saved_h01 = load(a.h01_report)
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
    if checked != saved_h01:
        raise ValueError("saved H0.1 report mismatch with revalidated source files")
    reports = []
    for case in manifest["cases"]:
        summary, telemetry, replay, _ = case_artifacts[case["sha256"]]
        checked_case = next(c for c in checked["cases"]
                            if c["chart_sha256"] == case["sha256"])
        reports.append(analyze_case(summary, telemetry, replay, checked_case))
    aggregate = Counter()
    for item in reports:
        aggregate.update(item["rollup"])
    report = {
        "format": FORMAT, "source_v08_commit": checked["source_v08_commit"],
        "frozen_code_digest": checked["frozen_code_digest"],
        "case_count": len(reports), "event_count": sum(r["event_count"] for r in reports),
        "cases": reports, "aggregate": dict(aggregate),
        "physical_motor_executed": False, "human_likeness_verified": False,
        "global_v08_planner_reproduced": False,
        "causality_established": False,
        "limitations": [
            "Label-only relabelling cannot change a physical fingering flag.",
            "Local surrogate replanning pins all outside events and is not the frozen global beam.",
            "Alternative partitions may change the number of chunks and candidate sets.",
            "Best-of-many alternatives is selection-biased, not a causal treatment effect.",
            "The frozen candidate grammar is reused read-only; not a calibrated human model.",
            "80ms abstract reuse does not imply physical Motor feasibility or safe play.",
            "No human performance or preferred fingering has been observed.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "event_count": report["event_count"],
        "case_count": report["case_count"],
        "aggregate": report["aggregate"],
        "saved": str(a.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
