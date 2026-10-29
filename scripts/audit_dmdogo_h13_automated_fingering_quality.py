from __future__ import annotations

"""H1.3: automatic, descriptive full-chart fingering-proxy A/B audit.

Reads the H1.1 complete Train counterfactual assignment report. Recomputes
non-calibrated, observable axis/hand pressure indicators on *both* paths,
checks provenance and H0.6 historical metrics, then triages disagreement.
No manual labels, Motor runs, HUD model, training, Validation20 or FINAL.
"""

import argparse
from collections import Counter, deque
import hashlib
import json
from math import isfinite
from pathlib import Path

from audit_dmdogo_h1_quality_evidence import H06_BLOB_SHA, TERMS

FORMAT = "dmdogo-h13-automated-fingering-quality-audit-v1"
H11_FORMAT = "dmdogo-h11-paired-fingering-reconstruction-v1"
SHA_LEN = 64
CHART_SIZES = (1540, 3637, 5054)
TARGET_COUNTS = (10, 10, 2)
EXPECTED_BETTER = 7
EPS = 1e-7
ABS_COST_TOL = 2e-6
QUICK_GAP_MS = 120.0
ONE_SECOND_MS = 1000.0
AXIS_WINDOW_MS = 500.0
AXES = frozenset(f"{hand}_{i}" for hand in ("left", "right") for i in range(1, 5))

# Higher is a *descriptive increased-pressure flag*, NOT a human preference.
PRESSURE_KEYS = (
    "quick_same_hand_depth_jump_pairs_120ms",
    "rapid_single_hand_runs_at_least_four",
    "axis_reuse_under_100ms",
    "axis_reuse_under_150ms",
    "longest_rapid_single_hand_run",
    "max_hand_contacts_in_one_second",
    "max_axis_contacts_in_half_second",
    "absolute_hand_contact_imbalance",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def axis_tuple(axes, chord, label: str) -> tuple[str, ...]:
    if (type(chord) is not int or not 1 <= chord <= 8
            or not isinstance(axes, list) or len(axes) != chord
            or len(set(axes)) != len(axes)
            or any(x not in AXES for x in axes)):
        raise ValueError(f"{label}: invalid frozen 8-axis event assignment")
    return tuple(axes)


def timeline(events: list[dict], *, field: str) -> tuple[tuple, ...]:
    if not isinstance(events, list) or not events:
        raise ValueError("empty or invalid source event timeline")
    checked = []
    ids = set()
    prev_due = float("-inf")
    for row in events:
        if not isinstance(row, dict):
            raise ValueError("invalid event record")
        i = row.get("event_index")
        t = finite(row.get("due_s"), "event due_s")
        chord = row.get("chord_size")
        if (type(i) is not int or i < 0 or i in ids
                or t < prev_due):
            raise ValueError("nonmonotonic or duplicate event identities")
        ids.add(i)
        prev_due = t
        checked.append((i, t * 1000.0,
                        axis_tuple(row.get(field), chord, "original event")))
    return tuple(checked)


def flatten_chunks(chunks, rows: tuple, *, label: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{label}: missing complete frozen chunks")
    result = []
    cursor = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(f"{label}: invalid chunk")
        a, b, n, axes = (
            chunk.get("start_event_index"), chunk.get("end_event_index"),
            chunk.get("event_count"), chunk.get("axes"))
        if (type(a) is not int or type(b) is not int or
                type(n) is not int or n <= 0 or b - a + 1 != n or
                not isinstance(axes, list) or len(axes) != n or
                cursor+n > len(rows) or rows[cursor][0] != a or
                rows[cursor+n-1][0] != b):
            raise ValueError(f"{label}: chunks do not cover identical event IDs")
        for j, value in enumerate(axes):
            chord = len(rows[cursor+j][2])
            result.append(axis_tuple(value, chord, f"{label} chunk"))
        cursor += n
    if cursor != len(rows):
        raise ValueError(f"{label}: chunks omit events")
    return tuple(result)


def metrics(rows: tuple[tuple[int, float, tuple[str, ...]], ...]) -> dict:
    """Pure, O(events) descriptive analysis; all thresholds preregistered.

    Every KeyDown contact is counted once per physical axis. 80ms is
    an abstract reuse criterion; 100/150ms and pressure thresholds are
    exploratory only and are not physiological calibration.
    """
    if not rows:
        raise ValueError("empty axis timeline")
    axis_count = Counter()
    hand_count = Counter()
    hand_queues = {"left": deque(), "right": deque()}
    axis_queues = {axis: deque() for axis in AXES}
    last_use = {}
    peak_hand_1s = {"left": 0, "right": 0}
    max_axis_half = 0
    reuse = []
    below_80 = below_100 = below_150 = 0
    fast_same = fast_depth = 0
    rapid_runs = 0
    longest_rapid = 0
    current_run = 0
    prev_t, prev_hand, prev_depth = None, None, None
    pure_switches = 0
    for event_id, t, event_axes in rows:
        if type(event_id) is not int or not isfinite(t):
            raise ValueError("invalid metrics input")
        hands = {axis.partition("_")[0] for axis in event_axes}
        pure = next(iter(hands)) if len(hands) == 1 else None
        single_axis = event_axes[0] if len(event_axes) == 1 else None
        depth = int(single_axis.partition("_")[2]) if single_axis else None
        if prev_t is not None:
            gap = t-prev_t
            if gap < -1e-7:
                raise ValueError("nonchronological metrics timeline")
            if prev_hand is not None and pure is not None:
                if prev_hand != pure:
                    pure_switches += 1
                if (single_axis is not None and prev_depth is not None
                        and gap <= QUICK_GAP_MS+1e-9
                        and prev_hand == pure):
                    fast_same += 1
                    if abs(depth-prev_depth) >= 2:
                        fast_depth += 1
                    current_run += 1
                else:
                    if current_run >= 4:
                        rapid_runs += 1
                    longest_rapid = max(longest_rapid, current_run)
                    current_run = 1
            else:
                if current_run >= 4:
                    rapid_runs += 1
                longest_rapid = max(longest_rapid, current_run)
                current_run = 1 if single_axis else 0
        else:
            current_run = 1 if single_axis else 0
        prev_t = t
        prev_hand = pure if single_axis else None
        prev_depth = depth

        for axis in event_axes:
            hand = axis.partition("_")[0]
            hand_count[hand] += 1
            axis_count[axis] += 1
            q = hand_queues[hand]
            q.append(t)
            while q and t-q[0] > ONE_SECOND_MS + 1e-9:
                q.popleft()
            peak_hand_1s[hand] = max(peak_hand_1s[hand], len(q))
            aq = axis_queues[axis]
            aq.append(t)
            while aq and t-aq[0] > AXIS_WINDOW_MS + 1e-9:
                aq.popleft()
            max_axis_half = max(max_axis_half, len(aq))
            if axis in last_use:
                interval = t-last_use[axis]
                reuse.append(interval)
                below_80 += interval < 80.0-1e-9
                below_100 += interval < 100.0-1e-9
                below_150 += interval < 150.0-1e-9
            last_use[axis] = t
    longest_rapid = max(longest_rapid, current_run)
    if current_run >= 4:
        rapid_runs += 1
    return {
        "total_contacts": sum(axis_count.values()),
        "hand_contacts": {
            "left": hand_count["left"], "right": hand_count["right"]},
        "axis_contacts": dict(sorted(axis_count.items())),
        "absolute_hand_contact_imbalance":
            abs(hand_count["left"]-hand_count["right"]),
        "pure_hand_event_switches": pure_switches,
        "quick_same_hand_pairs_120ms": fast_same,
        "quick_same_hand_depth_jump_pairs_120ms": fast_depth,
        "rapid_single_hand_runs_at_least_four": rapid_runs,
        "longest_rapid_single_hand_run": longest_rapid,
        "max_hand_contacts_in_one_second": max(peak_hand_1s.values()),
        "per_hand_peak_one_second": peak_hand_1s,
        "max_axis_contacts_in_half_second": max_axis_half,
        "minimum_axis_reuse_ms": round(min(reuse), 6) if reuse else None,
        "axis_reuse_under_80ms": below_80,
        "axis_reuse_under_100ms": below_100,
        "axis_reuse_under_150ms": below_150,
    }


def _validate_source(source: dict) -> list[dict]:
    if (not isinstance(source, dict)
            or source.get("format") != H11_FORMAT
            or source.get("source_h06_git_blob") != H06_BLOB_SHA
            or source.get("source_v08_frozen_commit")
                != "45f30bcaf968281400e2c2913a96dd95515222d2"
            or source.get("chart_count") != 3
            or source.get("target_count") != 22
            or source.get("strict_lower_heuristic_cost_count") != 7
            or source.get("complete_event_axis_assignments") is not True
            or source.get("alternate_motor_executed") is not False
            or source.get("human_preference_observed") is not False
            or source.get("motor_quality_verified") is not False
            or source.get("validation20_or_final_accessed") is not False
            or source.get("training_performed") is not False
            or source.get("frozen_planner_motor_timing_changed") is not False):
        raise ValueError("H1.1 historical provenance/claims mismatch")
    charts = source.get("charts")
    if (not isinstance(charts, list) or
            tuple(c.get("event_count") for c in charts) != CHART_SIZES
            or tuple(c.get("count") for c in charts) != TARGET_COUNTS
            or tuple(len(c.get("reconstructions", ())) for c in charts)
            != TARGET_COUNTS):
        raise ValueError("H1.1 exact frozen chart/target census differs")
    if (source.get("cohort", {}).get("total_events") != sum(CHART_SIZES)
            or source.get("cohort", {}).get("independent_counterfactual_targets") != 22):
        raise ValueError("H1.1 cohort source differs")
    shas = [c.get("chart_sha256") for c in charts]
    if (len(set(shas)) != 3
            or any(not isinstance(s, str) or len(s) != SHA_LEN
                   or any(ch not in "0123456789abcdef" for ch in s)
                   for s in shas)):
        raise ValueError("H1.1 chart SHA invalid/duplicate")
    return charts


def _cost(candidate: dict) -> tuple[float, dict[str, float]]:
    cost = candidate.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("H1.1 candidate cost missing")
    a = finite(cost.get("original_v08_final"), "original cost")
    b = finite(cost.get("forced_final"), "alternative cost")
    delta = finite(cost.get("final_delta"), "cost delta")
    if abs(b-a-delta) > ABS_COST_TOL:
        raise ValueError("H1.1 six-term cost delta disagrees")
    term_deltas = cost.get("six_component_deltas")
    if not isinstance(term_deltas, dict) or set(term_deltas) != set(TERMS):
        raise ValueError("H1.1 six-term cost components missing")
    d = {term: finite(term_deltas[term], term) for term in TERMS}
    if abs(sum(d.values())-delta) > ABS_COST_TOL:
        raise ValueError("H1.1 six-term component sum disagrees")
    return delta, d


def compare_candidate(original_events: list[dict], original_chunks: list[dict],
                      original_metrics: dict, candidate: dict) -> dict:
    """Reconcile full length, changed-event values, chunks and all probes."""
    if (candidate.get("independent_intervention_only") is not True
            or candidate.get("alternate_motor_result") != "not_executed"
            or candidate.get("human_preference") != "unobserved"
            or candidate.get("reached_terminal_abstract") is not True):
        raise ValueError("H1.1 candidate evidence claims not abstract-only")
    original_rows = timeline(original_events, field="original_axes")
    assigned = candidate.get("alternative_axes_by_event_order")
    if not isinstance(assigned, list) or len(assigned) != len(original_rows):
        raise ValueError("H1.1 alternative does not cover every source event")
    new_rows = []
    changes = []
    for ((event_id, due_ms, old), row, assigned_axes) in zip(
            original_rows, original_events, assigned, strict=True):
        updated = axis_tuple(assigned_axes, row["chord_size"], "alternative event")
        new_rows.append((event_id, due_ms, updated))
        if old != updated:
            changes.append({
                "event_index": event_id, "due_s": row["due_s"],
                "original_axes": list(old), "alternative_axes": list(updated),
            })
    if (changes != candidate.get("changed_events")
            or len(changes) != candidate.get("changed_event_count")
            or not changes):
        raise ValueError("H1.1 saved changed-event identity mismatched")
    old_chunk_vector = flatten_chunks(original_chunks, original_rows, label="original")
    new_chunks = candidate.get("alternative_chunks")
    new_chunk_vector = flatten_chunks(new_chunks, original_rows, label="alternative")
    if (old_chunk_vector != tuple(x[2] for x in original_rows)
            or new_chunk_vector != tuple(x[2] for x in new_rows)
            or len(new_chunks) != len(original_chunks)):
        raise ValueError("H1.1 full original/alternative chunk arrays differ")
    chunk_differences = [
        i for i, (a,b) in enumerate(zip(original_chunks,new_chunks,strict=True))
        if a["axes"] != b["axes"] or a["gesture_family"] != b["gesture_family"]
    ]
    if (chunk_differences != candidate.get("changed_chunk_indices")
            or candidate.get("join_window_index") != chunk_differences[0]
            or candidate["changed_event_count"] != len(changes)):
        raise ValueError("H1.1 changed-chunk source differs")
    for a,b in zip(original_chunks,new_chunks,strict=True):
        if any(a.get(field) != b.get(field) for field in (
                "start_event_index","end_event_index","event_count",
                "rhythm","reason")):
            raise ValueError("H1.1 alternative altered frozen segmentation")
    old_metric = metrics(original_rows)
    if old_metric != original_metrics:
        raise ValueError("original chart metrics changed during reconstruction")
    new_metric = metrics(tuple(new_rows))
    if (old_metric["axis_reuse_under_80ms"]
            or new_metric["axis_reuse_under_80ms"]
            or candidate.get("physical_reuse_80ms_proxy_violations") != 0):
        raise ValueError("frozen abstract same-axis 80ms gate violated")
    delta, term_deltas = _cost(candidate)
    signatures = {k: new_metric[k]-old_metric[k] for k in PRESSURE_KEYS}
    increases = sorted(k for k,v in signatures.items() if v>0)
    decreases = sorted(k for k,v in signatures.items() if v<0)
    if increases and decreases:
        comparison = "mixed_descriptive_indicators"
    elif increases:
        comparison = "increased_indicators_only"
    elif decreases:
        comparison = "decreased_indicators_only"
    else:
        comparison = "no_change_in_selected_indicators"
    if delta < -EPS and increases:
        tier = "P1_objective_lower_but_some_pressure_proxies_increase"
    elif increases and decreases:
        tier = "P1_proxy_tradeoff"
    elif len(changes)>=500:
        tier = "P2_large_changed_suffix"
    elif increases:
        tier = "P2_pressure_proxy_increases"
    else:
        tier = "P3_descriptive_only"
    return {
        "candidate_id": candidate["candidate_id"],
        "flag_pair": candidate["flag_pair"],
        "join_window_index": candidate["join_window_index"],
        "changed_event_count": len(changes),
        "changed_event_fraction": round(len(changes)/len(original_rows),6),
        "changed_chunk_count": len(chunk_differences),
        "heuristic_objective": {
            "original_final": candidate["cost"]["original_v08_final"],
            "alternative_final": candidate["cost"]["forced_final"],
            "alternative_minus_original": round(delta,9),
            "alternative_has_lower_frozen_heuristic_cost": delta < -EPS,
            "six_term_deltas": term_deltas,
        },
        "abstract_metrics": {
            "original": old_metric,
            "alternative": new_metric,
        },
        "pressure_proxy_deltas": signatures,
        "increased_pressure_proxy_names": increases,
        "decreased_pressure_proxy_names": decreases,
        "directional_proxy_comparison": comparison,
        "triage_tier": tier,
        "human_review_required": False,
        "optional_review_reasons": (
            ["heuristic_gain_with_increased_pressure_proxies"]
            if delta < -EPS and increases else []
        ) + (["proxy_directions_disagree"] if increases and decreases else []
        ) + (["long_propagated_difference"] if len(changes)>=500 else []),
        "alternate_motor_feasibility": "not_tested",
        "human_preference": "not_measured",
        "conclusions_about_physical_comfort": False,
    }


def compile_audit(source: dict, *, source_sha256: str) -> dict:
    charts = _validate_source(source)
    result_charts = []
    all_cases = []
    known = set()
    for chart in charts:
        original = chart["original_events_by_index"]
        rows = timeline(original, field="original_axes")
        old_chunks = chart["original_chunks"]
        if flatten_chunks(old_chunks, rows, label="original") != tuple(
                row[2] for row in rows):
            raise ValueError("chart source original chunk axes disagree")
        original_metrics = metrics(rows)
        out = []
        for case in chart["reconstructions"]:
            cid = case.get("candidate_id")
            if not isinstance(cid,str) or not cid or cid in known:
                raise ValueError("duplicate/missing candidate ID")
            known.add(cid)
            row = compare_candidate(
                original, old_chunks, original_metrics, case)
            out.append(row)
            all_cases.append(row)
        result_charts.append({
            "chart": chart["chart"],
            "chart_sha256": chart["chart_sha256"],
            "event_count": len(original),
            "original_abstract_metrics": original_metrics,
            "candidate_count": len(out),
            "candidates": out,
        })
    improved = sum(
        r["heuristic_objective"]["alternative_has_lower_frozen_heuristic_cost"]
        for r in all_cases)
    if len(all_cases) != 22 or improved != EXPECTED_BETTER:
        raise ValueError("H1.1 seven-improvement/22-case objective census changed")
    triage = sorted(all_cases, key=lambda c: (
        0 if c["triage_tier"].startswith("P1") else
        1 if c["triage_tier"].startswith("P2") else 2,
        -len(c["increased_pressure_proxy_names"]),
        -len(c["decreased_pressure_proxy_names"]),
        -c["changed_event_count"], c["candidate_id"]))
    return {
        "format": FORMAT,
        "source_h11_sha256": source_sha256,
        "source_h06_git_blob": H06_BLOB_SHA,
        "chart_count": 3,
        "source_event_count": 10231,
        "candidate_count": 22,
        "original_objective_lower_alternatives": improved,
        "protocol": {
            "quick_gap_ms": QUICK_GAP_MS,
            "one_second_window_ms": ONE_SECOND_MS,
            "axis_window_ms": AXIS_WINDOW_MS,
            "axis_reuse_gate_ms": 80,
            "comparison_metrics": list(PRESSURE_KEYS),
            "directionality": (
                "Increased indicators represent more of an abstract "
                "pattern/pressure proxy, NOT demonstrated discomfort."),
            "triage_rule": (
                "P1 for objective gain with increased proxy or mixed "
                "directions; P2 for >=500 changed events or increased "
                "proxies; otherwise P3. No threshold was human-calibrated."),
        },
        "charts": result_charts,
        "triage": {
            "P1": sum(x["triage_tier"].startswith("P1") for x in all_cases),
            "P2": sum(x["triage_tier"].startswith("P2") for x in all_cases),
            "P3": sum(x["triage_tier"].startswith("P3") for x in all_cases),
            "review_not_required": True,
            "optional_first_five_case_ids": [
                x["candidate_id"] for x in triage[:5]],
            "all_cases_ordered_by_diagnostic_priority": [
                {"candidate_id": x["candidate_id"],
                 "tier": x["triage_tier"],
                 "changed_events": x["changed_event_count"],
                 "cost_delta": x["heuristic_objective"][
                     "alternative_minus_original"],
                 "increased_proxy_count":
                     len(x["increased_pressure_proxy_names"]),
                 "decreased_proxy_count":
                     len(x["decreased_pressure_proxy_names"])}
                for x in triage],
        },
        "motor_executed": False,
        "human_reviews_required": False,
        "human_reviews_observed": False,
        "human_likeness_verified": False,
        "training_performed": False,
        "validation20_or_final_accessed": False,
        "frozen_planner_motor_timing_changed": False,
        "limitations": [
            "All data are prior Train seams, not independent unseen charts.",
            "The metrics measure abstract key assignments, not directly observed human fingers.",
            "The objective and descriptive proxies may overlap; this is not an independent learned human-quality model.",
            "Increased/decreased proxy values are not validated human preference labels or physiological fatigue estimates.",
            "A low-cost path with zero reuse violations may still be physically impossible in the frozen Motor or hard for a player.",
            "No alternate Motor simulation was performed, and existing v0.8 original execution cannot stand in for one.",
            "Overlapping counterfactuals from a chart are not independent statistical samples and must not be combined.",
            "Human review is optional, not an invisible requirement; no missing labels are fabricated.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h11-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H1.3 output")
    source_sha = sha256_file(args.h11_report)
    source = json.loads(args.h11_report.read_text(encoding="utf-8"))
    report = compile_audit(source, source_sha256=source_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps({
        "chart_count": report["chart_count"],
        "source_event_count": report["source_event_count"],
        "candidate_count": report["candidate_count"],
        "lower_heuristic_cost": report["original_objective_lower_alternatives"],
        "triage_counts": {
            k: report["triage"][k] for k in ("P1","P2","P3")},
        "optional_first_five_case_ids": report[
            "triage"]["optional_first_five_case_ids"],
        "saved": str(args.output),
        "motor_executed": False,
        "human_reviews_required": False,
        "human_likeness_verified": False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
