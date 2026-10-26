from __future__ import annotations

"""DMDOGO v0.9 H0.1: explain frozen H0 review-flag conditions.

The detector thresholds and frozen v0.8 artifacts are unchanged.  This is a
descriptive association study: it maps flags to Oracle-time intervals and
gesture chunks, then compares them with like-for-like non-flagged
opportunities.  It does not label a gesture as unnatural or causal.
"""

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median

from analyze_dmdogo_v09_style_proxies import (
    HAND_GAP_MS,
    _depth,
    _hand,
    _validate,
)
from dmdogo_v08_provenance import sha256_file
from report_dmdogo_v09_style_proxies import compile_report as compile_h0_report


OUTPUT_FORMAT = "dmdogo-v09-h01-flag-condition-analysis-v1"
TRAIN_FORMAT = "dmdogo-v09-h01-train-condition-report-v1"
RAPID_MIN_EVENTS = 4
QUICK_MIN_DEPTH = 2


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values) -> dict:
    clean = [float(value) for value in values
             if value is not None and isfinite(float(value))]
    return {
        "count": len(clean),
        "min": round(min(clean), 6) if clean else None,
        "p25": round(_quantile(clean, .25), 6) if clean else None,
        "median": round(median(clean), 6) if clean else None,
        "mean": round(mean(clean), 6) if clean else None,
        "p75": round(_quantile(clean, .75), 6) if clean else None,
        "max": round(max(clean), 6) if clean else None,
    }


def _bpm(ioi_ms: float) -> float | None:
    # This is an IOI reciprocal, not the chart BPM or a claim about beat level.
    return round(60000.0 / ioi_ms, 6) if ioi_ms > 0 else None


def _chunk_map(chunks: list[dict]) -> dict[int, dict]:
    output = {}
    for chunk_index, chunk in enumerate(chunks):
        descriptor = {
            "chunk_index": chunk_index,
            "start_event_index": chunk["start_event_index"],
            "end_event_index": chunk["end_event_index"],
            "event_count": chunk["event_count"],
            "gesture_family": chunk.get("gesture_family", "unknown"),
            "rhythm_kind": chunk.get("rhythm_kind", "unknown"),
            "start_reason": chunk.get("start_reason", "unknown"),
        }
        for event_index in range(
                chunk["start_event_index"], chunk["end_event_index"] + 1):
            output[event_index] = descriptor
    return output


def _chunk_context(rows: list[dict], by_event: dict[int, dict]) -> dict:
    chunks = []
    seen = set()
    for row in rows:
        chunk = by_event[row["event_index"]]
        if chunk["chunk_index"] not in seen:
            chunks.append(chunk)
            seen.add(chunk["chunk_index"])
    crossings = max(0, len(chunks) - 1)
    return {
        "chunk_relation": (
            "within_chunk" if crossings == 0 else "cross_chunk_boundary"),
        "chunk_boundary_crossings": crossings,
        "chunks": chunks,
    }


def _movement(rows: list[dict]) -> dict:
    axes = [row["axes"][0] for row in rows]
    distances = [abs(_depth(a) - _depth(b)) for a, b in zip(axes, axes[1:])]
    return {
        "hand": _hand(axes[0]),
        "axes": axes,
        "depth_step_distances": distances,
        "total_depth_distance": sum(distances),
        "maximum_depth_distance": max(distances, default=0),
    }


def _quick_record(previous: dict, current: dict,
                  chunk_by_event: dict[int, dict]) -> dict:
    gap = current["due_ms"] - previous["due_ms"]
    record = {
        "start_event_index": previous["event_index"],
        "end_event_index": current["event_index"],
        "start_due_ms": round(previous["due_ms"], 6),
        "end_due_ms": round(current["due_ms"], 6),
        "ioi_ms": round(gap, 6),
        "local_bpm_equivalent": _bpm(gap),
        "hand": _hand(previous["axes"][0]),
        "from_axis": previous["axes"][0],
        "to_axis": current["axes"][0],
        "depth_distance": abs(
            _depth(previous["axes"][0]) - _depth(current["axes"][0])),
    }
    record.update(_chunk_context([previous, current], chunk_by_event))
    return record


def _rapid_record(rows: list[dict], chunk_by_event: dict[int, dict]) -> dict:
    gaps = [b["due_ms"] - a["due_ms"] for a, b in zip(rows, rows[1:])]
    record = {
        "start_event_index": rows[0]["event_index"],
        "end_event_index": rows[-1]["event_index"],
        "start_due_ms": round(rows[0]["due_ms"], 6),
        "end_due_ms": round(rows[-1]["due_ms"], 6),
        "event_count": len(rows),
        "ioi_ms": [round(value, 6) for value in gaps],
        "local_bpm_equivalent": [_bpm(value) for value in gaps],
        "movement_proxy": _movement(rows),
    }
    record.update(_chunk_context(rows, chunk_by_event))
    return record


def _relations(records: list[dict]) -> dict:
    relation = Counter(record["chunk_relation"] for record in records)
    families = Counter()
    rhythms = Counter()
    starts = Counter()
    for record in records:
        for chunk in record["chunks"]:
            families[chunk["gesture_family"]] += 1
            rhythms[chunk["rhythm_kind"]] += 1
            starts[chunk["start_reason"]] += 1
    return {
        "chunk_relation_counts": dict(sorted(relation.items())),
        "touched_chunk_family_counts": dict(sorted(families.items())),
        "touched_chunk_rhythm_counts": dict(sorted(rhythms.items())),
        "touched_chunk_start_reason_counts": dict(sorted(starts.items())),
    }


def _movement_tendencies(records: list[dict], *, rapid: bool) -> dict:
    hands = Counter()
    patterns = Counter()
    for record in records:
        if rapid:
            movement = record["movement_proxy"]
            hands[movement["hand"]] += 1
            patterns[" -> ".join(movement["axes"])] += 1
        else:
            hands[record["hand"]] += 1
            patterns[f"{record['from_axis']} -> {record['to_axis']}"] += 1
    return {
        "hand_counts": {hand: hands[hand] for hand in ("left", "right")},
        "axis_pattern_counts": dict(sorted(patterns.items())),
    }


def _group_distributions(quick_detected: list[dict], quick_control: list[dict],
                         rapid_detected: list[dict], rapid_control: list[dict]
                         ) -> tuple[dict, dict]:
    quick = {}
    for label, rows in (("detected", quick_detected), ("control", quick_control)):
        quick[f"{label}_ioi_ms"] = _distribution(r["ioi_ms"] for r in rows)
        quick[f"{label}_local_bpm_equivalent"] = _distribution(
            r["local_bpm_equivalent"] for r in rows)
        quick[f"{label}_depth_distance"] = _distribution(
            r["depth_distance"] for r in rows)
    rapid = {}
    for label, rows in (("detected", rapid_detected), ("control", rapid_control)):
        rapid[f"{label}_ioi_ms"] = _distribution(
            value for row in rows for value in row["ioi_ms"])
        rapid[f"{label}_local_bpm_equivalent"] = _distribution(
            value for row in rows for value in row["local_bpm_equivalent"])
        rapid[f"{label}_event_count"] = _distribution(
            row["event_count"] for row in rows)
        rapid[f"{label}_total_depth_distance"] = _distribution(
            row["movement_proxy"]["total_depth_distance"] for row in rows)
    return quick, rapid


def analyze_conditions(summary: dict, telemetry: dict, replay: dict) -> dict:
    events, chunks, _ = _validate(summary, telemetry, replay)
    chunk_by_event = _chunk_map(chunks)

    quick_detected = []
    quick_control = []
    for previous, current in zip(events, events[1:]):
        gap = current["due_ms"] - previous["due_ms"]
        opportunity = (
            0 <= gap <= HAND_GAP_MS
            and len(previous["axes"]) == len(current["axes"]) == 1
            and _hand(previous["axes"][0]) == _hand(current["axes"][0])
        )
        if not opportunity:
            continue
        record = _quick_record(previous, current, chunk_by_event)
        (quick_detected if record["depth_distance"] >= QUICK_MIN_DEPTH
         else quick_control).append(record)

    maximal_runs: list[list[dict]] = []
    run = [events[0]]
    for row in events[1:]:
        previous = run[-1]
        gap = row["due_ms"] - previous["due_ms"]
        same_hand = (
            len(previous["axes"]) == len(row["axes"]) == 1
            and _hand(previous["axes"][0]) == _hand(row["axes"][0])
        )
        if same_hand and 0 <= gap <= HAND_GAP_MS:
            run.append(row)
        else:
            if len(run) >= 2:
                maximal_runs.append(run)
            run = [row]
    if len(run) >= 2:
        maximal_runs.append(run)

    rapid_detected = []
    rapid_control = []
    for rows in maximal_runs:
        record = _rapid_record(rows, chunk_by_event)
        (rapid_detected if len(rows) >= RAPID_MIN_EVENTS
         else rapid_control).append(record)

    quick_ranges = [(r["start_event_index"], r["end_event_index"])
                    for r in quick_detected]
    rapid_ranges = [(r["start_event_index"], r["end_event_index"])
                    for r in rapid_detected]
    pair_overlaps = [
        (quick_index, rapid_index)
        for quick_index, (qa, qb) in enumerate(quick_ranges)
        for rapid_index, (ra, rb) in enumerate(rapid_ranges)
        if ra <= qa and qb <= rb
    ]
    quick_dist, rapid_dist = _group_distributions(
        quick_detected, quick_control, rapid_detected, rapid_control)

    # Cross-check against H0's exact detector rather than silently drifting.
    expected_quick = 0
    for previous, current in zip(events, events[1:]):
        gap = current["due_ms"] - previous["due_ms"]
        if (0 <= gap <= HAND_GAP_MS
                and len(previous["axes"]) == len(current["axes"]) == 1
                and _hand(previous["axes"][0]) == _hand(current["axes"][0])
                and abs(_depth(previous["axes"][0]) -
                        _depth(current["axes"][0])) >= QUICK_MIN_DEPTH):
            expected_quick += 1
    if expected_quick != len(quick_detected):
        raise RuntimeError("H0 quick-jump detector mismatch")

    return {
        "format": OUTPUT_FORMAT,
        "chart": summary["chart"],
        "chart_sha256": summary["chart_sha256"],
        "event_count": len(events),
        "h0_thresholds": {
            "rapid_run_min_events": RAPID_MIN_EVENTS,
            "rapid_run_max_ioi_ms": HAND_GAP_MS,
            "quick_jump_min_depth_indices": QUICK_MIN_DEPTH,
        },
        "thresholds_changed": False,
        "quick_same_hand_depth_jump": {
            "opportunity_definition": (
                "adjacent single-axis events on the same hand with 0..120ms IOI"),
            "detected_count": len(quick_detected),
            "control_count": len(quick_control),
            "detected": quick_detected,
            "non_detected_controls": quick_control,
            "distributions": quick_dist,
            "detected_chunk_tendencies": _relations(quick_detected),
            "control_chunk_tendencies": _relations(quick_control),
            "detected_movement_tendencies": _movement_tendencies(
                quick_detected, rapid=False),
            "control_movement_tendencies": _movement_tendencies(
                quick_control, rapid=False),
        },
        "rapid_single_hand_sequence": {
            "opportunity_definition": (
                "maximal runs of >=2 adjacent single-axis same-hand events with <=120ms IOI"),
            "detected_count": len(rapid_detected),
            "control_count": len(rapid_control),
            "detected": rapid_detected,
            "non_detected_controls": rapid_control,
            "distributions": rapid_dist,
            "detected_chunk_tendencies": _relations(rapid_detected),
            "control_chunk_tendencies": _relations(rapid_control),
            "detected_movement_tendencies": _movement_tendencies(
                rapid_detected, rapid=True),
            "control_movement_tendencies": _movement_tendencies(
                rapid_control, rapid=True),
        },
        "overlap": {
            "quick_flags_inside_rapid_sequence": len({q for q, _ in pair_overlaps}),
            "rapid_sequences_containing_quick_flag": len({r for _, r in pair_overlaps}),
            "pair_region_overlap_count": len(pair_overlaps),
            "pairs": [
                {"quick_detected_index": q, "rapid_detected_index": r}
                for q, r in pair_overlaps
            ],
        },
        "causal_claim": False,
        "human_likeness_verified": False,
        "limitations": [
            "Flags are fixed H0 review proxies, not unnatural-fingering labels.",
            "Controls isolate the detector opportunity set, not every confounder.",
            "IOI reciprocal is not the chart BPM or a uniquely identified beat level.",
            "Depth-index distance is a proxy, not measured anatomical travel.",
            "Chunk associations and distributions are observational, not causal.",
        ],
    }


def _aggregate_case_reports(reports: list[dict]) -> dict:
    quick_detected = [r for report in reports
                      for r in report["quick_same_hand_depth_jump"]["detected"]]
    quick_control = [r for report in reports
                     for r in report["quick_same_hand_depth_jump"]["non_detected_controls"]]
    rapid_detected = [r for report in reports
                      for r in report["rapid_single_hand_sequence"]["detected"]]
    rapid_control = [r for report in reports
                     for r in report["rapid_single_hand_sequence"]["non_detected_controls"]]
    quick_dist, rapid_dist = _group_distributions(
        quick_detected, quick_control, rapid_detected, rapid_control)
    return {
        "event_count": sum(report["event_count"] for report in reports),
        "quick_same_hand_depth_jump": {
            "detected_count": len(quick_detected),
            "control_count": len(quick_control),
            "distributions": quick_dist,
            "detected_chunk_tendencies": _relations(quick_detected),
            "control_chunk_tendencies": _relations(quick_control),
            "detected_movement_tendencies": _movement_tendencies(
                quick_detected, rapid=False),
            "control_movement_tendencies": _movement_tendencies(
                quick_control, rapid=False),
        },
        "rapid_single_hand_sequence": {
            "detected_count": len(rapid_detected),
            "control_count": len(rapid_control),
            "distributions": rapid_dist,
            "detected_chunk_tendencies": _relations(rapid_detected),
            "control_chunk_tendencies": _relations(rapid_control),
            "detected_movement_tendencies": _movement_tendencies(
                rapid_detected, rapid=True),
            "control_movement_tendencies": _movement_tendencies(
                rapid_control, rapid=True),
        },
        "overlap": {
            "quick_flags_inside_rapid_sequence": sum(
                report["overlap"]["quick_flags_inside_rapid_sequence"]
                for report in reports),
            "rapid_sequences_containing_quick_flag": sum(
                report["overlap"]["rapid_sequences_containing_quick_flag"]
                for report in reports),
            "pair_region_overlap_count": sum(
                report["overlap"]["pair_region_overlap_count"]
                for report in reports),
        },
    }


def compile_train_report(manifest: dict, verified: dict, h0: dict,
                         case_artifacts: dict, *, manifest_digest: str) -> dict:
    # Reuse H0's provenance validation and recompute its aggregate counts from
    # the raw artifacts.  max_flags=0 avoids depending on its truncated list.
    checked_h0 = compile_h0_report(
        manifest, verified, case_artifacts,
        manifest_digest=manifest_digest, max_flags_per_chart=0)
    if (checked_h0["aggregate"]["events"] != h0.get("aggregate", {}).get("events")
            or checked_h0["aggregate"]["flag_counts"]
            != h0.get("aggregate", {}).get("flag_counts")):
        raise ValueError("saved H0 report differs from verified source artifacts")
    reports = []
    for case in manifest["cases"]:
        summary, telemetry, replay, _ = case_artifacts[case["sha256"]]
        report = analyze_conditions(summary, telemetry, replay)
        expected = next(row for row in h0["cases"]
                        if row["chart_sha256"] == case["sha256"])["flag_counts"]
        actual = {
            "quick_same_hand_depth_jump":
                report["quick_same_hand_depth_jump"]["detected_count"],
            "rapid_single_hand_sequence":
                report["rapid_single_hand_sequence"]["detected_count"],
        }
        for kind, count in actual.items():
            if count != expected.get(kind, 0):
                raise ValueError(f"H0 detector count mismatch for {case['sha256']}: {kind}")
        reports.append(report)
    return {
        "format": TRAIN_FORMAT,
        "source_h0_format": h0["format"],
        "source_v08_commit": checked_h0["source_v08_commit"],
        "frozen_code_digest": checked_h0["frozen_code_digest"],
        "case_count": len(reports),
        "cases": reports,
        "aggregate": _aggregate_case_reports(reports),
        "thresholds_changed": False,
        "causal_claim": False,
        "human_likeness_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verified-report", required=True, type=Path)
    parser.add_argument("--h0-report", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite H0.1 analysis")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    manifest = load(args.manifest)
    verified = load(args.verified_report)
    h0 = load(args.h0_report)
    artifacts = {}
    for case in manifest.get("cases", []):
        sha = case["sha256"]
        sub = args.results_root / sha[:12]
        summary_path = sub / "summary.json"
        summary = load(summary_path)
        summary["_artifact_sha256"] = sha256_file(summary_path)
        artifacts[sha] = (
            summary,
            load(sub / "telemetry-v08.json"),
            load(sub / "replay-v08.json"),
            load(args.results_root / "_provenance" / f"{sha[:12]}.json"),
        )
    report = compile_train_report(
        manifest, verified, h0, artifacts,
        manifest_digest=sha256_file(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    aggregate = report["aggregate"]
    print(json.dumps({
        "events": aggregate["event_count"],
        "quick": aggregate["quick_same_hand_depth_jump"]["detected_count"],
        "rapid": aggregate["rapid_single_hand_sequence"]["detected_count"],
        "overlap": aggregate["overlap"],
        "saved": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
