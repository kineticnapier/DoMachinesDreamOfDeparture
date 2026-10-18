from __future__ import annotations

"""Read-only posthoc fingering-reuse cause audit from a saved v0.1.1 replay.

Diagnoses reservation/pressed causes from chosen-event timing and 10ms
physical frames; no checkpoint, inference, simulator, or training required.
The historical eligibility mask/reasons are not saved in replay-v011.json,
so reconstruction is cross-checked against the saved short-reuse summary.
"""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean, median

AXES = tuple(f"left_{i}" for i in range(6, 0, -1)) + tuple(
    f"right_{i}" for i in range(1, 7)
)
WINDOWS = (35, 60, 100, 200, 500)
PREVIEW_TAIL_S = .035


def _frame_at(frames: list, timestamps: list[float], decision_s: float):
    idx = bisect_left(timestamps, decision_s - .000002)
    if idx >= len(frames) or abs(timestamps[idx] - decision_s) > .000002:
        return None
    return frames[idx]


def _hit_time(row: dict) -> float | None:
    if row.get("outcome") != "Hit" or row.get("timing_error_ms") is None:
        return None
    return float(row["due_s"]) + float(row["timing_error_ms"]) / 1000.0


def _pending_reservation(axis: str, current: dict, older: list[dict]) -> bool | None:
    now = float(current["decision_s"])
    unknown = False
    for old in older:
        if axis not in old.get("axes", ()):
            continue
        chosen_at = float(old["decision_s"])
        due = float(old["due_s"])
        if chosen_at > now + 1e-7 or due + PREVIEW_TAIL_S < now - 1e-7:
            continue
        used_at = _hit_time(old)
        if used_at is None:
            unknown = True
        elif used_at > now + 1e-7:
            return True
    return None if unknown else False


def _replay_policy(payload: dict, policy: str) -> tuple[dict, list[dict]]:
    pol = payload["policies"][policy]
    frames = pol["trace"]
    timestamps = [float(frame[0]) for frame in frames]
    event_rows = []
    for event in payload["events"]:
        decision = (event.get("decisions") or {}).get(policy)
        if decision is not None:
            event_rows.append(dict(decision))
    event_rows.sort(key=lambda d: (float(d["decision_s"]), int(d["event_index"])))
    if any(len(row.get("axes", ())) != 1 for row in event_rows):
        raise ValueError("this audit requires one physical axis per event")
    if len(set(r["event_index"] for r in event_rows)) != len(event_rows):
        raise ValueError("duplicate event_index")
    outcomes = []
    for index, row in enumerate(event_rows):
        if index == 0:
            continue
        prev = event_rows[index - 1]
        axis = prev["axes"][0]
        gap_ms = (float(row["due_s"]) - float(prev["due_s"])) * 1000
        if gap_ms < -1e-5:
            raise ValueError("nonchronological due times")
        frame = _frame_at(frames, timestamps, float(row["decision_s"]))
        if axis not in AXES:
            raise ValueError(f"unsupported axis: {axis}")
        pressed = bool(frame[3][AXES.index(axis)]) if frame is not None else None
        reserved = _pending_reservation(axis, row, event_rows[:index])
        depth = int(axis.rsplit("_", 1)[-1])
        if depth == 6:
            reason = "not_offered_for_single_note"
        elif reserved is True:
            reason = "reserved"
        elif reserved is None:
            reason = "reservation_unknown"
        elif pressed is True:
            reason = "pressed"
        elif pressed is None:
            reason = "frame_missing"
        elif depth == 5:
            hand_idx = 0 if axis.startswith("left") else 1
            modes = frame[4]
            reason = "available" if modes[hand_idx] == "tip" else "topology_unknown"
        else:
            reason = "available"
        selected_same = row["axes"][0] == axis
        if selected_same and reason != "available":
            raise RuntimeError(
                f"reconstruction contradiction at {policy} event {row['event_index']}: "
                f"selected {axis}, but reconstructed {reason}"
            )
        outcomes.append({
            "event_index": row["event_index"], "previous_event_index": prev["event_index"],
            "event_gap_ms": round(gap_ms, 3), "decision_s": float(row["decision_s"]),
            "previous_axis": axis, "chosen_axis": row["axes"][0],
            "selected_same": selected_same, "pressed_at_decision": pressed,
            "pending_reservation": reserved, "reason": reason,
            "timing_error_ms": row.get("timing_error_ms"),
            "outcome": row.get("outcome"),
        })
    diagnostics = {}
    for threshold in WINDOWS:
        applicable = [r for r in outcomes if r["event_gap_ms"] <= threshold + 1e-4]
        categories = Counter(r["reason"] for r in applicable)
        diagnostics[str(threshold)] = {
            "pairs": len(applicable), "reasons": dict(sorted(categories.items())),
            "previous_axis_available": categories["available"],
            "available_and_reused": sum(r["selected_same"] for r in applicable if r["reason"] == "available"),
            "available_but_switched": sum(not r["selected_same"] for r in applicable if r["reason"] == "available"),
            "pressed_snapshot_total": sum(r["pressed_at_decision"] is True for r in applicable),
            "reserved_and_pressed_overlap": sum(r["pressed_at_decision"] is True and r["pending_reservation"] is True for r in applicable),
        }
    last_hit = {}
    bins = defaultdict(list)
    joint = defaultdict(list)
    for row in event_rows:
        axis = row["axes"][0]
        hit_at = _hit_time(row)
        if hit_at is None:
            continue
        last = last_hit.get(axis)
        gap = None if last is None else (hit_at - last) * 1000
        if gap is None:
            label = "first_use"
        elif gap < 100:
            label = "reuse_lt_100ms"
        elif gap < 200:
            label = "reuse_100_200ms"
        elif gap < 500:
            label = "reuse_200_500ms"
        else:
            label = "reuse_ge_500ms"
        err = abs(float(row["timing_error_ms"]))
        bins[label].append(err)
        last_hit[axis] = hit_at
    for row in outcomes:
        err = row.get("timing_error_ms")
        if err is None:
            continue
        density = "dense_le_35ms" if row["event_gap_ms"] <= 35 else (
            "medium_35_100ms" if row["event_gap_ms"] <= 100 else "sparse_gt_100ms"
        )
        joint[(density, row["reason"])].append(abs(float(err)))
    def aggregate(values):
        return {"n": len(values), "mean_abs_timing_error_ms": round(mean(values), 3),
                "median_abs_timing_error_ms": round(median(values), 3)} if values else {"n": 0}
    return {
        "events": len(event_rows), "observed_pairs": len(outcomes),
        "within_due_gap_ms": diagnostics,
        "timing_by_axis_reuse_interval": {name: aggregate(vals) for name, vals in sorted(bins.items())},
        "timing_by_density_and_reason": {"/".join(k): aggregate(v) for k, v in sorted(joint.items())},
        "limitations": [
            "Eligibility mask and explicit reserved/pressed reasons were not retained in the replay JSON.",
            "Reservation reconstructed from Hit timing and committed-event lifetime; check against saved eligibility counts.",
            "For thumb, topology switching cannot be fully diagnosed from normalized frames; unknown cases remain unknown.",
            "Pressed snapshots reflect exact recorded frame flags; if pending too, reserved takes priority.",
            "Timing-error association does not establish a causal benefit or harm of reusing an axis.",
        ],
    }, outcomes


def analyze(payload: dict, old_summary: dict | None) -> dict:
    if not all(p in payload.get("policies", {}) for p in ("free", "human_proxy")):
        raise ValueError("expected Free/Human-Proxy comparison replay")
    result = {
        "format": "dmdod-fingering-v011-posthoc-reuse-causes",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": payload.get("chart"), "retrained": False, "physics_rerun": False,
        "policies": {},
    }
    for policy in ("free", "human_proxy"):
        report, rows = _replay_policy(payload, policy)
        report["event_details"] = rows
        checks = {}
        if old_summary is not None:
            source = old_summary["policy_results"][policy]["reuse"]["previous_axis_eligible_at_decision_within_ms"]
            for threshold in WINDOWS:
                key = str(threshold)
                actual = source[key]
                reconstructed = report["within_due_gap_ms"][key]
                checks[key] = {
                    "saved_eligible": actual["previous_axis_candidate_eligible"],
                    "reconstructed_eligible": reconstructed["previous_axis_available"],
                    "saved_pairs": actual["adjacent_single_note_pairs"],
                    "reconstructed_pairs": reconstructed["pairs"],
                    "match": actual["previous_axis_candidate_eligible"] == reconstructed["previous_axis_available"]
                    and actual["adjacent_single_note_pairs"] == reconstructed["pairs"],
                }
        report["verification_against_saved_summary"] = {
            "checked": old_summary is not None,
            "all_counts_match": all(row["match"] for row in checks.values()) if checks else None,
            "thresholds": checks,
        }
        result["policies"][policy] = report
    return result


def render_html(report: dict) -> str:
    rows = []
    for policy, results in report["policies"].items():
        for threshold, summary in results["within_due_gap_ms"].items():
            count = summary["reasons"]
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                policy, threshold, summary["pairs"], count.get("reserved", 0),
                count.get("pressed", 0), count.get("available", 0),
                count.get("topology_unknown", 0) + count.get("reservation_unknown", 0) + count.get("frame_missing", 0),
                summary["available_and_reused"], summary["available_but_switched"],
            )) + "</tr>")
    validation = []
    for policy, row in report["policies"].items():
        r = row["verification_against_saved_summary"]
        validation.append(f"<p>{escape(policy)}: counts check = {escape(str(r['all_counts_match']))}</p>")
    body = ("<h1>Fingering v0.1.1 — 同じ指を使えない理由</h1>"
            "<p>保存済みReplayの打鍵結果・Motor状態を使った事後復元。"
            "元の全候補判定ログはReplayに保存されていないため、照合結果を必ず確認。</p>"
            + "".join(validation) + "<table><thead><tr>"
            "<th>方式</th><th>間隔≤ms</th><th>組数</th><th>予約</th><th>押下中</th>"
            "<th>使用可能</th><th>不明</th><th>再利用</th><th>別軸選択</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
            "<p>予約と押下が重なる場合、候補判定の優先順位に従い「予約」に分類。</p>"
            "<h2>同一軸再利用間隔とTiming誤差（観測上の相関）</h2>"
            "<pre>" + escape(json.dumps({p: v["timing_by_axis_reuse_interval"] for p, v in report["policies"].items()}, indent=2, ensure_ascii=False)) + "</pre>"
            "<h2>打鍵密度 × 前の指の不適格理由</h2>"
            "<pre>" + escape(json.dumps({p: v["timing_by_density_and_reason"] for p, v in report["policies"].items()}, indent=2, ensure_ascii=False)) + "</pre>")
    return ("<!doctype html><html lang='ja'><meta charset='utf-8'><title>Fingering reuse causes</title>"
            "<style>body{font:15px/1.6 system-ui;margin:2rem auto;max-width:1200px;background:#131925;color:#eee;padding:0 1rem}"
            "table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #455062}"
            "pre{white-space:pre-wrap;background:#202a39;padding:16px;border-radius:8px}</style>"
            "<main>" + body + "</main></html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", required=True, type=Path)
    parser.add_argument("--short-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite: {args.output_dir}")
    payload = json.loads(args.replay_json.read_text(encoding="utf-8"))
    saved = json.loads(args.short_summary_json.read_text(encoding="utf-8"))
    if payload.get("chart") != saved.get("chart") or payload.get("checkpoint_sha256") != saved.get("checkpoint_sha256"):
        raise ValueError("saved report and replay originate from different chart/checkpoint")
    result = analyze(payload, saved)
    html = render_html(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "reuse-cause-report.json").open("x", encoding="utf-8") as dest:
        json.dump(result, dest, ensure_ascii=False, indent=2)
    with (args.output_dir / "reuse-cause-report.html").open("x", encoding="utf-8") as dest:
        dest.write(html)
    for policy, row in result["policies"].items():
        print(policy, "counts verified:", row["verification_against_saved_summary"]["all_counts_match"])
        for gap in ("100", "200", "500"):
            summary = row["within_due_gap_ms"][gap]
            print(f"  <= {gap} ms: {summary['pairs']} pairs; categories {summary['reasons']}")
    print("viewer:", args.output_dir / "reuse-cause-report.html")


if __name__ == "__main__":
    main()
