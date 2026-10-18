from __future__ import annotations

"""Read-only signed Timing-error audit for saved Fingering v0.1.1 replays.

The event pairing, actual KeyDown times and motor selection are immutable.
No simulator run, checkpoint, or model is needed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean, median


def band(gap_ms: float | None) -> str:
    if gap_ms is None:
        return "first_use"
    if gap_ms < 100:
        return "reuse_lt_100ms"
    if gap_ms < 200:
        return "reuse_100_200ms"
    if gap_ms < 500:
        return "reuse_200_500ms"
    return "reuse_ge_500ms"


def summary(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    neg = sum(e < -1e-7 for e in errors)
    pos = sum(e > 1e-7 for e in errors)
    return {
        "n": len(errors),
        "early": neg,
        "late": pos,
        "exact": len(errors) - pos - neg,
        "early_rate": round(neg / len(errors), 6),
        "late_rate": round(pos / len(errors), 6),
        "mean_signed_ms": round(mean(errors), 3),
        "median_signed_ms": round(median(errors), 3),
        "mean_abs_ms": round(mean(abs(e) for e in errors), 3),
        "median_abs_ms": round(median(abs(e) for e in errors), 3),
    }


def collect_policy(payload: dict, name: str) -> tuple[dict, dict[int, dict]]:
    chronological = sorted(payload["events"], key=lambda e: (e["episode_s"], e["event_index"]))
    last_keydown = {}
    groups = defaultdict(list)
    density = defaultdict(list)
    points = {}
    for event in chronological:
        decision = (event.get("decisions") or {}).get(name)
        if not decision:
            continue
        axes = list(decision.get("axes") or [])
        if len(axes) != 1:
            raise ValueError(f"only single-axis notes supported: {name} event {event['event_index']}")
        error = decision.get("timing_error_ms")
        if error is None or decision.get("outcome") != "Hit":
            continue
        axis = axes[0]
        hit_time_ms = float(event["episode_s"]) * 1000 + float(error)
        delta = None if axis not in last_keydown else hit_time_ms - last_keydown[axis]
        if delta is not None and delta < -1e-5:
            raise ValueError("signed KeyDown chronology drift")
        last_keydown[axis] = hit_time_ms
        tag = band(delta)
        error = float(error)
        groups[tag].append(error)
        points[event["event_index"]] = {
            "event_index": event["event_index"], "axis": axis,
            "due_ms": round(float(event["episode_s"]) * 1000, 3),
            "signed_timing_error_ms": round(error, 3),
            "last_axis_reuse_gap_ms": None if delta is None else round(delta, 3),
            "reuse_band": tag,
        }
    sorted_events = sorted(points)
    previous = None
    for ix in sorted_events:
        item = points[ix]
        if previous is None:
            d = "first_event"
        else:
            gap = item["due_ms"] - previous["due_ms"]
            d = ("event_interval_le_35ms" if gap <= 35 else
                 "event_interval_35_100ms" if gap <= 100 else
                 "event_interval_gt_100ms")
        density[d].append(item["signed_timing_error_ms"])
        item["event_density_band"] = d
        previous = item
    return {
        "eligible_hit_events": len(points),
        "by_axis_reuse_gap": {label: summary(groups[label]) for label in
                              ("first_use", "reuse_lt_100ms", "reuse_100_200ms",
                               "reuse_200_500ms", "reuse_ge_500ms")},
        "by_event_density": {label: summary(errs) for label, errs in sorted(density.items())},
        "all_events": summary([p["signed_timing_error_ms"] for p in points.values()]),
    }, points


def analyze(payload: dict) -> dict:
    if not all(p in payload.get("policies", {}) for p in ("free", "human_proxy")):
        raise ValueError("requires a saved Free/Human-Proxy v0.1.1 comparison replay")
    data, points = {}, {}
    for policy in ("free", "human_proxy"):
        data[policy], points[policy] = collect_policy(payload, policy)
    common = sorted(points["free"].keys() & points["human_proxy"].keys())
    paired = defaultdict(list)
    details = []
    for ix in common:
        a, b = points["free"][ix], points["human_proxy"][ix]
        d_abs = abs(b["signed_timing_error_ms"]) - abs(a["signed_timing_error_ms"])
        paired[a["reuse_band"]].append(d_abs)
        details.append({
            "event_index": ix,
            "free_axis": a["axis"], "human_proxy_axis": b["axis"],
            "free_reuse_gap_ms": a["last_axis_reuse_gap_ms"],
            "human_proxy_reuse_gap_ms": b["last_axis_reuse_gap_ms"],
            "free_signed_error_ms": a["signed_timing_error_ms"],
            "human_proxy_signed_error_ms": b["signed_timing_error_ms"],
            "human_minus_free_abs_error_ms": round(d_abs, 3),
        })
    return {
        "format": "dmdod-fingering-v011-signed-error-audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": payload.get("chart"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "policies": data,
        "paired": {
            "common_events": len(common),
            "mean_human_minus_free_abs_error_ms":
                round(mean(x["human_minus_free_abs_error_ms"] for x in details), 3) if details else None,
            "by_free_reuse_band": {
                tag: {"n": len(values), "mean_human_minus_free_abs_error_ms": round(mean(values), 3)}
                for tag, values in sorted(paired.items())
            },
        },
        "details": details,
        "limitations": [
            "Signed errors are measured from nearest actual KeyDown within the saved event match.",
            "Reuse gap uses recovered physical-axis KeyDown time; no fatigue-state inference is made.",
            "The paired comparison is observational and does not isolate causation.",
            "AutoPlay supplies exact target times; no HUD-only Timing System is tested.",
        ],
    }


def render_html(result: dict) -> str:
    rows = []
    labels = ("first_use", "reuse_lt_100ms", "reuse_100_200ms",
              "reuse_200_500ms", "reuse_ge_500ms")
    for label in labels:
        for name in ("free", "human_proxy"):
            x = result["policies"][name]["by_axis_reuse_gap"][label]
            rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
                name, label, x["n"], x.get("early", "—"), x.get("late", "—"),
                x.get("mean_signed_ms", "—"), x.get("mean_abs_ms", "—"),
                x.get("median_signed_ms", "—"),
            )) + "</tr>")
    table = ("<table><thead><tr><th>方式</th><th>再利用間隔</th><th>件数</th>"
             "<th>早い</th><th>遅い</th><th>平均符号付き誤差(ms)</th>"
             "<th>平均絶対誤差(ms)</th><th>中央値(ms)</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")
    paired = result["paired"]
    return ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
            "<title>Signed Fingering Timing Error</title>"
            "<style>body{background:#111827;color:#e5e7eb;font:15px/1.65 system-ui;"
            "margin:24px auto;max-width:1200px;padding:0 16px}"
            "table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;"
            "border-bottom:1px solid #3b4658}th{background:#202b3c}"
            "pre{white-space:pre-wrap;background:#202b3c;padding:16px}</style>"
            f"<h1>短時間の指再利用とTiming誤差の方向</h1><p>{escape(str(result['chart']))}</p>"
            + table +
            f"<h2>同じイベントの比較</h2><p>{paired['common_events']}イベント。"
            f"Human-Proxy − Free の絶対誤差差: "
            f"{paired['mean_human_minus_free_abs_error_ms']} ms</p>"
            "<pre>" + escape(json.dumps(paired["by_free_reuse_band"],
                                        ensure_ascii=False, indent=2)) + "</pre>"
            "<p>早い・遅いの方向は実際のKeyDown時刻基準。相関であり因果関係ではない。</p></html>\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error(f"refusing to overwrite: {args.output_dir}")
    payload = json.loads(args.replay_json.read_text(encoding="utf-8"))
    result = analyze(payload)
    html = render_html(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "signed-error-audit.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "signed-error-audit.html").open("x", encoding="utf-8") as f:
        f.write(html)
    for name in ("free", "human_proxy"):
        print(f"\n{name}")
        for label, group in result["policies"][name]["by_axis_reuse_gap"].items():
            print(f"  {label}: {group}")
    print("viewer:", args.output_dir / "signed-error-audit.html")


if __name__ == "__main__":
    main()
