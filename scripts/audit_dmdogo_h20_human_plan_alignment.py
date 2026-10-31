from __future__ import annotations

"""H2.0: compare human first-sight input hands with H1.1 PLANNED axes.

Read-only. Human log is kept locally and is never committed. A planned
fingering is not proof of a Motor keydown or human-like execution.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, median

FORMAT = "dmdod-h20-first-sight-vs-planned-hands-v1"
CHART = "02971 - ALL MY FELLAS - RetroBean.adofai"
LINE = re.compile(r"\s*(\d+(?:\.\d+)?)\s+(K\d+):([A-Z][A-Z0-9]*)\s*")


def parse_human(text):
    rows = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line == "time_ms    hits":
            continue
        m = LINE.fullmatch(line)
        if not m:
            raise ValueError(f"invalid log line {line_no}: {line[:100]}")
        t, key, finger = m.groups()
        if finger[0] not in "LR":
            raise ValueError(f"unmapped finger on line {line_no}")
        item = {"time_s": float(t) / 1000, "key": key,
                "finger": finger, "hand": finger[0]}
        if rows and item["time_s"] < rows[-1]["time_s"]:
            raise ValueError("input timestamps are not chronological")
        rows.append(item)
    if not rows:
        raise ValueError("no keydowns in log")
    return rows


def sparse_align(human_times, event_times, offset, gap=2.75,
                 tolerance=.050, sigma=.028):
    """Exact minimum-cost monotone alignment using sparse weighted matches.

    Every unpaired item costs gap; a valid paired item costs squared
    normalized time error. Hard pairing window is tolerance seconds.
    """
    m = len(event_times)
    bit = [(0., -1)] * (m + 1)
    nodes = []

    def query(j):
        best = (0., -1)
        while j:
            if bit[j][0] > best[0]:
                best = bit[j]
            j -= j & -j
        return best

    def update(j, pair):
        while j <= m:
            if pair[0] > bit[j][0]:
                bit[j] = pair
            j += j & -j

    for i, t in enumerate(human_times):
        lo = bisect_left(event_times, t - offset - tolerance - 1e-12)
        hi = bisect_right(event_times, t - offset + tolerance + 1e-12)
        pending = []
        for j in range(lo, hi):
            delta = t - event_times[j] - offset
            if abs(delta) > tolerance + 1e-10:
                continue
            previous_score, previous_id = query(j)
            weight = 2 * gap - (delta / sigma) ** 2
            nodes.append((i, j, previous_id))
            pending.append((j + 1, (previous_score + weight, len(nodes)-1)))
        for j, state in pending:
            update(j, state)

    score, node_id = query(m)
    matches = []
    while node_id >= 0:
        i, j, node_id = nodes[node_id]
        matches.append((i, j))
    matches.reverse()
    return gap * (len(human_times) + m) - score, matches


def best_offset(human_times, event_times):
    start = human_times[0] - event_times[0]
    coarse = [(sparse_align(human_times, event_times, start + k*.002)[0],
               start + k*.002) for k in range(-60, 61)]
    center = min(coarse)[1]
    fine = [(sparse_align(human_times, event_times, center + k*.0001)[0],
             center + k*.0001) for k in range(-25, 26)]
    return round(min(fine)[1], 6)


def axis_hand(axis):
    return "L" if axis.startswith("left_") else "R"


def analyze(human, source, forced_offset=None):
    if source.get("format") != "dmdogo-h11-paired-fingering-reconstruction-v1":
        raise ValueError("unexpected H1.1 format")
    matching = [c for c in source["charts"] if c["chart"] == CHART]
    if len(matching) != 1:
        raise ValueError("ALL MY FELLAS missing or duplicated")
    chart = matching[0]
    events = chart["original_events_by_index"]
    if len(events) != 1540 or any(x["chord_size"] != 1 or
                                 x["event_index"] != i
                                 for i,x in enumerate(events)):
        raise ValueError("requires complete single-input 1540-event timeline")
    times = [x["due_s"] for x in events]
    if times != sorted(times):
        raise ValueError("unordered source timeline")
    plans = [("original", [r["original_axes"][0] for r in events])]
    for rec in chart["reconstructions"]:
        axes = rec["alternative_axes_by_event_order"]
        if len(axes) != len(events) or any(len(x) != 1 for x in axes):
            raise ValueError("incomplete alternate fingering")
        plans.append((rec["candidate_id"], [x[0] for x in axes]))
    if len(plans) != 11:
        raise ValueError("not exactly 11 separate plans")
    human_times = [row["time_s"] for row in human]
    offset = (best_offset(human_times, times) if forced_offset is None
              else forced_offset)
    cost, paired = sparse_align(human_times, times, offset)
    human_index = {j: i for i,j in paired}
    aligned_human = {i for i,j in paired}
    residuals = sorted(abs(1000*(human_times[i] - times[j] - offset))
                       for i,j in paired)
    comparisons = []
    original = plans[0][1]
    for name, axes in plans:
        count = sum(human[i]["hand"] == axis_hand(axes[j]) for i,j in paired)
        comparisons.append({
            "id": name,
            "hand_agreements": count,
            "hand_agreement_percent": round(100 * count / len(paired), 4),
            "changed_axes_vs_original": sum(a!=b for a,b in zip(original,axes)),
            "changed_hands_vs_original": sum(axis_hand(a)!=axis_hand(b)
                                             for a,b in zip(original,axes))
        })
    transitions = [(a,b) for a,b in zip(paired,paired[1:])
                   if b[0] == a[0]+1 and b[1] == a[1]+1]
    transition_agreement = sum(
        (human[a[0]]["hand"] == human[b[0]]["hand"]) ==
        (axis_hand(original[a[1]]) == axis_hand(original[b[1]]))
        for a,b in transitions)
    suspected = []
    last_key = {}
    for i,row in enumerate(human):
        earlier = last_key.get(row["key"])
        if earlier is not None:
            gap_ms = 1000*(row["time_s"] - human[earlier]["time_s"])
            if gap_ms < 50:
                suspected.append({
                    "key": row["key"], "earlier_human_index": earlier,
                    "later_human_index": i, "gap_ms": round(gap_ms, 3),
                    "later_input_unpaired": i not in aligned_human,
                    "chattering_unconfirmed": True})
        last_key[row["key"]] = i
    by_event = []
    for j,event in enumerate(events):
        i = human_index.get(j)
        by_event.append({
            "event_index": j,
            "due_s": times[j],
            "original_axis": original[j],
            "human_index": i,
            "human_key": human[i]["key"] if i is not None else None,
            "human_hand": human[i]["hand"] if i is not None else None,
            "timing_residual_ms": round(
                1000*(human_times[i] - times[j] - offset),3
            ) if i is not None else None,
            "changed_alternatives": {
                name: axes[j] for name,axes in plans[1:]
                if axes[j] != original[j]}})
    return {
        "format": FORMAT,
        "chart": CHART,
        "chart_sha256": chart["chart_sha256"],
        "alignment": {
            "method": "time-only ordered optimal matching; 50ms hard window",
            "estimated_log_minus_reference_offset_s": offset,
            "matched_events": len(paired),
            "unmatched_chart_events": len(events)-len(paired),
            "unmatched_human_keydowns": len(human)-len(paired),
            "median_abs_residual_ms": round(median(residuals),3),
            "mean_abs_residual_ms": round(mean(residuals),3),
            "cost": round(cost,6)},
        "human_hands": dict(Counter(x["hand"] for x in human)),
        "dmd_planned_hands": dict(Counter(axis_hand(x) for x in original)),
        "planned_hand_agreement": comparisons,
        "same_hand_vs_switch_transition_agreement": {
            "count": transition_agreement,
            "pairs": len(transitions),
            "percent": round(100 * transition_agreement / len(transitions),4)},
        "suspected_duplicate_keydowns": suspected,
        "unpaired_human_indices": [i for i in range(len(human))
                                   if i not in aligned_human],
        "unpaired_chart_indices": [j for j in range(len(events))
                                   if j not in human_index],
        "per_event": by_event,
        "limitations": {
            "human_fingers_are_configured_labels_not_measured": True,
            "dmd_actual_motor_trace_compared": False,
            "time_alignment_is_not_game_judgment": True,
            "human_preference_observed": False,
            "human_likeness_verified": False,
            "possible_chatter_kept_in_raw_input": True}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-log", type=Path, required=True)
    parser.add_argument("--h11-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-s", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior output")
    human = parse_human(args.human_log.read_text(encoding="utf-8-sig"))
    report = json.loads(args.h11_report.read_text(encoding="utf-8-sig"))
    output = analyze(human, report, args.offset_s)
    output["source_sha256"] = {
        "human_log": sha256(args.human_log.read_bytes()).hexdigest(),
        "h11": sha256(args.h11_report.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({
        "alignment": output["alignment"],
        "human_hands": output["human_hands"],
        "dmd_planned_hands": output["dmd_planned_hands"],
        "original": output["planned_hand_agreement"][0],
        "transition": output["same_hand_vs_switch_transition_agreement"],
        "saved": str(args.output),
        "human_likeness_verified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
