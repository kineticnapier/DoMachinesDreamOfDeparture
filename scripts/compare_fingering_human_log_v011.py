from __future__ import annotations

"""Compare a human Fingering sequence with saved Free/Human-Proxy replay.

Reads the user's log and saved replay JSON; does not run inference or alter
checkpoints. Human physical-key names are *not* the same as model contact
names. Comparison uses anatomical proxy / physical-axis equivalence only.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from statistics import median

# The human input mapping in the supplied recorder log. Not OS scan codes.
HUMAN_KEY_AXIS = {
    "K01": "left_4", "K02": "left_3", "K03": "left_2", "K04": "left_1",
    "K05": "left_6", "K06": "left_6",
    "K07": "left_5", "K08": "left_5",
    "K09": "right_1", "K10": "right_2", "K11": "right_3", "K12": "right_4",
    "K13": "right_5", "K14": "right_5",
    "K15": "right_6", "K16": "right_6",
}
EXPECTED_FINGER = {
    "K01": "LP", "K02": "LR", "K03": "LM", "K04": "LI",
    "K05": "LP", "K06": "LP", "K07": "LT", "K08": "LT",
    "K09": "RI", "K10": "RM", "K11": "RR", "K12": "RP",
    "K13": "RT", "K14": "RT", "K15": "RP", "K16": "RP",
}
AXIS_FINGER = {
    "left_1": "LI", "left_2": "LM", "left_3": "LR", "left_4": "LP",
    "left_5": "LT", "left_6": "LP/side",
    "right_1": "RI", "right_2": "RM", "right_3": "RR", "right_4": "RP",
    "right_5": "RT", "right_6": "RP/side",
}
HIT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(K\d{2}):([A-Za-z0-9_]+)\s*$")
MAPPING = re.compile(r"^#\s*(K\d{2})\s*=\s*(.*?)\s*->\s*(\S+)\s*$")
THRESHOLDS_MS = (35, 60, 100, 200, 500)


def parse_human_log(path: Path) -> list[dict]:
    values = []
    headers = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        match = MAPPING.match(line)
        if match:
            headers[match.group(1)] = (match.group(2).strip(), match.group(3))
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = HIT.match(line)
        if not match:
            raise ValueError(f"unrecognized human log line {line_no}: {line[:90]!r}")
        ms, key, finger = match.groups()
        if key not in HUMAN_KEY_AXIS:
            raise ValueError(f"human key {key} has no equivalent 12-axis contact; line {line_no}")
        if EXPECTED_FINGER[key] != finger:
            raise ValueError(f"human key {key}: expected finger {EXPECTED_FINGER[key]}, got {finger}; line {line_no}")
        if key in headers and headers[key][1] != finger:
            raise ValueError(f"human header disagrees with {key}:{finger}; line {line_no}")
        time_ms = float(ms)
        if values and time_ms < values[-1]["time_ms"]:
            raise ValueError(f"human log is not chronological at line {line_no}")
        values.append({
            "time_ms": time_ms, "key": key, "key_label": headers.get(key, ("", ""))[0],
            "finger": finger, "axis": HUMAN_KEY_AXIS[key],
        })
    if not values:
        raise ValueError("human log has no recorded events")
    return values


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = p * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def chosen_axes(event: dict, policy: str) -> list[str]:
    choice = (event.get("decisions") or {}).get(policy) or {}
    axes = list(choice.get("axes") or [])
    if any(axis not in AXIS_FINGER for axis in axes):
        raise ValueError(f"unknown model axis in event {event.get('event_index')}: {axes}")
    return axes


def summarize_strokes(strokes: list[tuple[float, str]]) -> dict:
    strokes = sorted(strokes)
    # Coupled logical contacts on the same physical axis can share one stroke.
    unique = []
    seen = set()
    for ms, axis in strokes:
        key = (round(ms, 3), axis)
        if key not in seen:
            unique.append((ms, axis))
            seen.add(key)
    last_time = {}
    gaps = []
    for at_ms, axis in unique:
        if axis in last_time:
            gaps.append(at_ms - last_time[axis])
        last_time[axis] = at_ms
    return {
        "strokes": len(unique),
        "axis_counts": dict(sorted(Counter(axis for _, axis in unique).items())),
        "short_same_axis_repeats": {
            str(k): sum(gap <= k + 1e-6 for gap in gaps) for k in THRESHOLDS_MS
        },
        "minimum_repeat_ms": round(min(gaps), 3) if gaps else None,
        "median_repeat_ms": round(median(gaps), 3) if gaps else None,
    }


def analyze(payload: dict, human: list[dict]) -> tuple[dict, list[dict]]:
    policies = payload.get("policies") or {}
    if not all(name in policies for name in ("free", "human_proxy")):
        raise ValueError("requires v0.1.1 Free/Human-Proxy comparison replay")
    events = sorted(payload.get("events") or [], key=lambda e: e["event_index"])
    if len(events) != len(human):
        raise ValueError(f"event count mismatch: model={len(events)}, human={len(human)}. "
                         "Refusing unsafe event-order alignment.")
    if any(int(e.get("chord_size", 1)) != 1 for e in events):
        raise ValueError("human 0-ms-chord log requires all comparison events to be single-contact")
    if len({e["event_index"] for e in events}) != len(events):
        raise ValueError("duplicate model event index")
    if any(events[i]["episode_s"] > events[i + 1]["episode_s"] + 1e-9
           for i in range(len(events) - 1)):
        raise ValueError("model events are not chronological")

    model_times = [1000.0 * float(e["episode_s"]) for e in events]
    human_times = [row["time_ms"] for row in human]
    offsets = [h - m for h, m in zip(human_times, model_times)]
    shift = median(offsets)
    errors = [abs(v - shift) for v in offsets]
    interval_errors = [abs((human_times[i] - human_times[i - 1]) -
                           (model_times[i] - model_times[i - 1]))
                       for i in range(1, len(events))]
    p95_error = quantile(errors, 0.95) or 0.0
    p95_interval_error = quantile(interval_errors, 0.95) or 0.0
    drift = offsets[-1] - offsets[0]
    alignment = {
        "method": "event order, fixed median clock offset; not verified solely by equal count",
        "human_events": len(human),
        "model_events": len(events),
        "estimated_human_minus_oracle_offset_ms": round(shift, 3),
        "median_absolute_residual_ms": round(median(errors), 3),
        "p95_absolute_residual_ms": round(p95_error, 3),
        "p95_interval_difference_ms": round(p95_interval_error, 3),
        "first_to_last_offset_drift_ms": round(drift, 3),
        "status": "intervals_consistent" if (p95_error <= 50 and p95_interval_error <= 40
                                              and abs(drift) <= 100) else "unverified",
        "warning": ("Event-order alignment is a hypothesis. Inspect timing differences before "
                    "interpreting per-event finger matches if status is unverified."),
    }

    rows = []
    for i, (event, h) in enumerate(zip(events, human)):
        choices = {name: chosen_axes(event, name) for name in ("free", "human_proxy")}
        rows.append({
            "ordinal": i, "event_index": event["event_index"],
            "human_time_ms": h["time_ms"], "oracle_time_ms": round(model_times[i], 3),
            "interval_ms": round(h["time_ms"] - human[i - 1]["time_ms"], 3) if i else None,
            "human": h,
            "choices": choices,
            "match": {name: choices[name] == [h["axis"]] for name in choices},
            "hand_match": {name: len(choices[name]) == 1 and
                          choices[name][0].split("_")[0] == h["axis"].split("_")[0]
                          for name in choices},
        })

    per_policy = {}
    for name in ("free", "human_proxy"):
        matches = sum(row["match"][name] for row in rows)
        hands = sum(row["hand_match"][name] for row in rows)
        ordinary = [row for row in rows if int(row["human"]["axis"].split("_")[1]) <= 4]
        by_dense = {}
        for threshold in THRESHOLDS_MS:
            group = [row for row in rows if row["interval_ms"] is not None and
                     0 <= row["interval_ms"] <= threshold]
            by_dense[str(threshold)] = {
                "events": len(group), "axis_matches": sum(row["match"][name] for row in group),
                "hand_matches": sum(row["hand_match"][name] for row in group),
            }
        raw_strokes = [(float(down.get("time_ms", float(down["episode_s"]) * 1000)),
                        str(down["axis"]))
                       for down in policies[name].get("keydowns", [])]
        per_policy[name] = {
            "axis_matches": matches, "axis_match_rate": round(matches / len(rows), 5),
            "hand_matches": hands, "hand_match_rate": round(hands / len(rows), 5),
            "ordinary_axis_matches": sum(row["match"][name] for row in ordinary),
            "ordinary_human_events": len(ordinary),
            "axis_choices": dict(sorted(Counter(axis for row in rows
                                                 for axis in row["choices"][name]).items())),
            "dense_event_matches": by_dense,
            "physical_keydown_reuse": summarize_strokes(raw_strokes),
            "saved_metrics": policies[name].get("saved_metrics", {}),
        }

    human_strokes = [(row["time_ms"], row["axis"]) for row in human]
    bands = []
    start = human_times[0]
    for begin_idx in range(int((human_times[-1] - start) / 4000) + 1):
        subset = [row for row in rows if begin_idx * 4000 <=
                  row["human_time_ms"] - start < (begin_idx + 1) * 4000]
        if not subset:
            continue
        bands.append({
            "start_ms": round(start + begin_idx * 4000, 1),
            "end_ms": round(start + (begin_idx + 1) * 4000, 1),
            "events": len(subset),
            "human_distinct_axes": len(set(row["human"]["axis"] for row in subset)),
            "free_distinct_axes": len(set(axis for row in subset for axis in row["choices"]["free"])),
            "human_proxy_distinct_axes": len(set(axis for row in subset
                                                 for axis in row["choices"]["human_proxy"])),
            "free_match": sum(row["match"]["free"] for row in subset),
            "human_proxy_match": sum(row["match"]["human_proxy"] for row in subset),
        })

    report = {
        "format": "dmdod-fingering-human-vs-network-v011",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": payload.get("chart"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "human_input_count": len(human),
        "alignment": alignment,
        "human": summarize_strokes(human_strokes),
        "policies": per_policy,
        "four_second_windows": bands,
        "limitations": [
            "Human input keys are matched to proxy axes, NOT identical logical contacts.",
            "Model v0.1.1 axis 1 is an index proxy, not calibrated anatomical geometry.",
            "Model thumb/side contacts may produce two logical keys from one physical stroke.",
            "KeyDown reuse rates are from the actual recorded physical strokes per source.",
            "Event-order pairing is valid only if recording has exactly one hit per chart event.",
            "Model timing uses privileged AutoPlay; human recording uses real player timing.",
        ],
    }
    return report, rows


def axis_text(axes: list[str]) -> str:
    if not axes:
        return "—"
    return " + ".join(AXIS_FINGER[axis] for axis in axes)


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def render_html(report: dict, rows: list[dict]) -> str:
    align = report["alignment"]
    model = report["policies"]
    sections = []
    for group in report["four_second_windows"]:
        n = group["events"]
        sections.append(
            "<tr>" +
            "".join(f"<td>{value}</td>" for value in (
                f"{group['start_ms']/1000:.1f}–{group['end_ms']/1000:.1f}s", n,
                group["human_distinct_axes"], group["free_distinct_axes"],
                group["human_proxy_distinct_axes"],
                pct(group["free_match"], n), pct(group["human_proxy_match"], n))) +
            "</tr>"
        )
    events_html = []
    for row in rows:
        human = row["human"]
        free = row["choices"]["free"]
        proxy = row["choices"]["human_proxy"]
        dense = row["interval_ms"] is not None and row["interval_ms"] <= 35
        status = "both" if all(row["match"].values()) else (
            "free" if row["match"]["free"] else
            "human_proxy" if row["match"]["human_proxy"] else "neither")
        format_axis = lambda axes, ok: (
            f'<span class="finger {"hit" if ok else "miss"}">{escape(axis_text(axes))}</span>'
        )
        events_html.append(
            f'<tr data-dense="{int(dense)}" data-status="{status}">'
            f'<td>{row["event_index"]}</td><td>{human["time_ms"]/1000:.3f}</td>'
            f'<td>{"—" if row["interval_ms"] is None else f"{row['interval_ms']:.1f}"}</td>'
            f'<td><span class="finger human">{escape(human["finger"])}</span> '
            f'<span class="muted">{escape(human["key"])}</span></td>'
            f'<td>{format_axis(free, row["match"]["free"])}</td>'
            f'<td>{format_axis(proxy, row["match"]["human_proxy"])}</td></tr>'
        )
    header = ("<tr><th>時間</th><th>打鍵</th><th>人間の使用軸</th>"
              "<th>Freeの使用軸</th><th>Human-Proxyの使用軸</th>"
              "<th>Free一致率</th><th>Human-Proxy一致率</th></tr>")
    full_head = ("<tr><th>Event</th><th>人間時刻(s)</th><th>前間隔(ms)</th>"
                 "<th>人間</th><th>Free</th><th>Human-Proxy</th></tr>")
    status = align["status"]
    caution = ("時刻差分の整合性は良好。ただしイベント順の対応は記録の欠落がない前提。" if
               status == "intervals_consistent" else
               "時刻差分の検証に失敗。以下の運指一致率は暫定値で、同一譜面・欠落なしを確認する必要がある。")
    css = """*{box-sizing:border-box}body{margin:0;background:#101521;color:#e8edf5;font:14px/1.55 system-ui,sans-serif}
main{max-width:1280px;padding:24px;margin:auto}h1{font-size:25px;margin:0 0 6px}h2{font-size:18px;margin:28px 0 10px}
small,.muted{color:#a7b3c6}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}
.card{background:#1c2738;border:1px solid #334052;border-radius:12px;padding:16px}
.big{font-weight:700;font-size:25px}.warn{padding:12px 16px;border-left:3px solid #ebb25e;background:#272535;margin:16px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{padding:7px 10px;border-bottom:1px solid #2e3c4e;text-align:left}
th{position:sticky;top:0;background:#1c2738;z-index:1}.scroll{overflow:auto;max-height:650px;border:1px solid #334052;border-radius:9px}
.finger{display:inline-block;min-width:33px;border-radius:5px;padding:2px 6px;text-align:center}
.finger.human{background:#4b3d77}.finger.hit{background:#245e51}.finger.miss{background:#473643}
button{background:#26354c;color:white;border:1px solid #516078;padding:8px 12px;border-radius:7px;cursor:pointer;margin-right:8px}
button.active{outline:2px solid #87a5e0}p{max-width:95ch}"""
    script = """<script>
let selected='all';function filter(mode){selected=mode;
document.querySelectorAll('button[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===mode));
let visible=0;document.querySelectorAll('#events tbody tr').forEach(tr=>{
 const show=mode==='all'||mode==='dense'&&tr.dataset.dense==='1'||mode==='disagree'&&tr.dataset.status!=='both';
 tr.hidden=!show;if(show)visible++;
});document.getElementById('visible').textContent=visible;}
</script>"""
    count = len(rows)
    return (f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'<title>Human / Free / Human-Proxy fingering audit</title><style>{css}</style><main>'
            f'<h1>運指比較 — Human / Free / Human-Proxy</h1><small>{escape(str(report["chart"]))}</small>'
            f'<div class="warn"><strong>照合: {escape(status)}</strong> — {caution} '
            f'推定時刻差 {align["estimated_human_minus_oracle_offset_ms"]:+.1f}ms、'
            f'残差p95 {align["p95_absolute_residual_ms"]:.1f}ms、'
            f'連続間隔差p95 {align["p95_interval_difference_ms"]:.1f}ms。</div>'
            f'<div class="cards"><div class="card"><small>人間の打鍵</small><div class="big">{count:,}</div>'
            f'<small>同一指再利用中央値 {report["human"]["median_repeat_ms"]}ms</small></div>'
            f'<div class="card"><small>Free: 人間との軸一致</small>'
            f'<div class="big">{pct(model["free"]["axis_matches"],count)}</div>'
            f'<small>{model["free"]["axis_matches"]:,}/{count:,}、手の一致 {pct(model["free"]["hand_matches"],count)}</small></div>'
            f'<div class="card"><small>Human-Proxy: 人間との軸一致</small>'
            f'<div class="big">{pct(model["human_proxy"]["axis_matches"],count)}</div>'
            f'<small>{model["human_proxy"]["axis_matches"]:,}/{count:,}、手の一致 '
            f'{pct(model["human_proxy"]["hand_matches"],count)}</small></div></div>'
            f'<h2>4秒区間別の指分散と一致率</h2><div class="scroll"><table>{header}{"".join(sections)}</table></div>'
            f'<h2>全イベント照合</h2><p>色のついたFree / Human-Proxy列は、人間と同じ物理軸相当なら緑、'
            f'異なれば赤。指の解剖学的な校正ではなく、12軸Motorとの対応による比較。</p>'
            f'<button class="active" data-filter="all" onclick="filter(&quot;all&quot;)">全件</button>'
            f'<button data-filter="dense" onclick="filter(&quot;dense&quot;)">直前35ms以内</button>'
            f'<button data-filter="disagree" onclick="filter(&quot;disagree&quot;)">どちらかが異なる</button>'
            f'<small>表示件数: <span id="visible">{count}</span></small>'
            f'<div class="scroll" id="events"><table>{full_head}<tbody>{"".join(events_html)}</tbody></table></div>'
            f'<h2>解釈上の制約</h2><p>{" / ".join(escape(note) for note in report["limitations"])}</p>'
            f'</main>{script}</html>\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--human-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    human = parse_human_log(args.human_log)
    payload = json.loads(args.replay_json.read_text(encoding="utf-8"))
    report, rows = analyze(payload, human)
    html = render_html(report, rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "human-model-comparison.json").open("x", encoding="utf-8") as out:
        json.dump({"summary": report, "events": rows}, out, ensure_ascii=False, indent=2)
    with (args.output_dir / "human-model-comparison.html").open("x", encoding="utf-8") as out:
        out.write(html)
    for name in ("free", "human_proxy"):
        policy = report["policies"][name]
        print(f"{name}: axis matching human {policy['axis_matches']}/{len(rows)}; "
              f"hand match {policy['hand_matches']}/{len(rows)}")
    print("alignment:", report["alignment"]["status"])
    print("HTML:", args.output_dir / "human-model-comparison.html")


if __name__ == "__main__":
    main()
