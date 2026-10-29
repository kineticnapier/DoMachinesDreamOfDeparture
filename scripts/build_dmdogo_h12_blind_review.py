from __future__ import annotations

"""H1.2: create a truly separated blinded A/B HTML review and private answer key.

Offline observation interface, not a Motor experiment or human-likeness score.
The public self-contained HTML deliberately carries no original/alternate
identity, score, source candidate IDs, review seed, or secret mapping.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import secrets

FORMAT_H11 = "dmdogo-h11-paired-fingering-reconstruction-v1"
FORMAT_PUBLIC = "dmdogo-h12-blind-review-packet-v1"
FORMAT_KEY = "dmdogo-h12-private-answer-key-v1"
AXES = {f"{hand}_{i}" for hand in ("left", "right") for i in range(1, 5)}
EXPECTED_COUNTS = (10, 10, 2)
EXPECTED_EVENTS = 10231
EXPECTED_TARGETS = 22
EXPECTED_BETTER = 7


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_axes(raw, *, chord: int, label: str) -> list[str]:
    if (not isinstance(raw, list) or len(raw) != chord
            or len(set(raw)) != len(raw)
            or any(axis not in AXES for axis in raw)):
        raise ValueError(f"bad {label} axis assignment")
    return raw


def _source_valid(source: dict) -> None:
    if (source.get("format") != FORMAT_H11
            or source.get("chart_count") != 3
            or source.get("target_count") != EXPECTED_TARGETS
            or source.get("strict_lower_heuristic_cost_count") != EXPECTED_BETTER
            or source.get("complete_event_axis_assignments") is not True
            or source.get("alternate_motor_executed") is not False
            or source.get("human_preference_observed") is not False
            or source.get("training_performed") is not False
            or source.get("validation20_or_final_accessed") is not False
            or source.get("frozen_planner_motor_timing_changed") is not False):
        raise ValueError("not an unmodified, complete H1.1 source")
    charts = source.get("charts")
    if not isinstance(charts, list) or len(charts) != 3:
        raise ValueError("H1.1 chart census missing")
    if tuple(len(c.get("reconstructions", ())) for c in charts) != EXPECTED_COUNTS:
        raise ValueError("H1.1 per-chart target census changed")
    if sum(c.get("event_count", 0) for c in charts) != EXPECTED_EVENTS:
        raise ValueError("H1.1 full Train event census changed")
    if sum(len(c["reconstructions"]) for c in charts) != EXPECTED_TARGETS:
        raise ValueError("H1.1 22 target identities changed")


def _validate_chart(chart: dict) -> tuple[list[dict], list[str]]:
    rows = chart["original_events_by_index"]
    if (not isinstance(rows, list) or not rows
            or len(rows) != chart["event_count"]
            or not isinstance(chart.get("chart_sha256"), str)
            or len(chart["chart_sha256"]) != 64):
        raise ValueError("invalid H1.1 chart identity and event count")
    ids = []
    last_due = float("-inf")
    for row in rows:
        idx, at, chord = (
            row.get("event_index"), row.get("due_s"), row.get("chord_size"))
        if (type(idx) is not int or
                type(at) not in (int, float) or not math.isfinite(at) or
                at < last_due or type(chord) is not int or not 1 <= chord <= 8):
            raise ValueError("bad H1.1 event time or identity")
        ids.append(idx)
        last_due = at
        validate_axes(row.get("original_axes"), chord=chord, label="original")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original event IDs")
    return rows, ids


def _entry(chart: dict, candidate: dict, events: list[dict],
           id_order: list[int]) -> tuple[dict, dict]:
    full = candidate.get("alternative_axes_by_event_order")
    if not isinstance(full, list) or len(full) != len(events):
        raise ValueError("incomplete H1.1 alternative event assignments")
    original = [row["original_axes"] for row in events]
    for i, axes in enumerate(full):
        validate_axes(axes, chord=events[i]["chord_size"], label="alternative")
    diff = [i for i, (left, right) in enumerate(zip(original, full))
            if left != right]
    saved_changes = candidate["changed_events"]
    if (len(diff) != candidate["changed_event_count"]
            or len(saved_changes) != len(diff)
            or [x["event_index"] for x in saved_changes] !=
            [id_order[i] for i in diff]):
        raise ValueError("H1.1 change-set identity differs")
    for offset, record in zip(diff, saved_changes):
        if (record["original_axes"] != original[offset]
                or record["alternative_axes"] != full[offset]
                or abs(record["due_s"] - events[offset]["due_s"]) > 1e-9):
            raise ValueError("H1.1 changed-event values differ")
    # Each candidate MUST cover the whole original timeline, including
    # cases where a single change propagates over thousands of events.
    candidate_id = candidate["candidate_id"]
    if (not isinstance(candidate_id, str) or not candidate_id
            or not candidate.get("independent_intervention_only")
            or candidate.get("alternate_motor_result") != "not_executed"
            or candidate.get("human_preference") != "unobserved"):
        raise ValueError("H1.1 candidate evidence boundary changed")
    delta = candidate["cost"]["final_delta"]
    if type(delta) not in (int, float) or not math.isfinite(delta):
        raise ValueError("bad H1.1 objective cost")
    if len(candidate["changed_chunk_indices"]) == 0:
        raise ValueError("empty changed-chunk path")
    if candidate["join_window_index"] != candidate[
            "changed_chunk_indices"][0]:
        raise ValueError("target join is not first changed chunk")

    public = {
        # source identity and cost are assigned only AFTER randomization;
        # never include candidate_id, chart SHA or original-vs-forced.
        "events": [
            {"i": row["event_index"], "t": round(row["due_s"], 9),
             "a": old, "b": new}
            for row, old, new in zip(events, original, full)
        ],
    }
    private = {
        "candidate_id": candidate_id,
        "chart_sha256": chart["chart_sha256"],
        "source_event_pair": candidate["flag_pair"],
        "changed_event_count": len(diff),
        "changed_chunk_count": len(candidate["changed_chunk_indices"]),
        "original_minus_forced_not_used": True,
        "forced_minus_original_objective_cost": delta,
        "forced_lower_heuristic_cost": delta < -1e-7,
        "original_side": None,
        "alternative_side": None,
    }
    return public, private


def compile_blind(source: dict, *, source_sha256: str,
                  seed: bytes) -> tuple[dict, dict]:
    """Balancing A/B within each source chart; independent of cost direction.

    All random ordering comes from SHA256(seed|fixed context|candidate id).
    Public packet has no seed, case source IDs, chart names/sha or costs.
    """
    _source_valid(source)
    if not isinstance(seed, bytes) or len(seed) < 16:
        raise ValueError("a private random seed of at least 128 bits is required")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("invalid source digest")
    candidates = []
    seen = set()
    for chart_index, chart in enumerate(source["charts"]):
        events, ids = _validate_chart(chart)
        for item in chart["reconstructions"]:
            public, private = _entry(chart, item, events, ids)
            cid = private["candidate_id"]
            if cid in seen:
                raise ValueError("duplicate H1.1 candidate identity")
            seen.add(cid)
            digest = hashlib.sha256(
                seed + b"|h12-case|" + cid.encode("utf-8")).hexdigest()
            candidates.append({
                "chart": chart_index,
                "priority": digest, "public": public,
                "private": private,
            })
    if len(candidates) != EXPECTED_TARGETS or len(seen) != EXPECTED_TARGETS:
        raise ValueError("H1.1 candidate census changed")
    if sum(c["private"]["forced_lower_heuristic_cost"]
           for c in candidates) != EXPECTED_BETTER:
        raise ValueError("H1.1 strictly improved candidate census changed")
    for chart_index, count in enumerate(EXPECTED_COUNTS):
        cohort = sorted(
            (c for c in candidates if c["chart"] == chart_index),
            key=lambda c: c["priority"])
        if len(cohort) != count:
            raise ValueError("source chart A/B balancing mismatch")
        # Exactly half original on A, half on B *per chart*.
        for idx, candidate in enumerate(cohort):
            original_side = "A" if idx < count//2 else "B"
            candidate["private"]["original_side"] = original_side
            candidate["private"]["alternative_side"] = (
                "B" if original_side == "A" else "A")
            if original_side == "B":
                for row in candidate["public"]["events"]:
                    row["a"], row["b"] = row["b"], row["a"]
    candidates.sort(key=lambda c: c["priority"])
    packet_id = hashlib.sha256(
        b"H1.2-public-packet|" + source_sha256.encode() +
        b"|" + seed).hexdigest()[:20]
    public_cases = []
    private_cases = []
    for idx, item in enumerate(candidates, 1):
        code = f"Case-{idx:02d}"
        public_cases.append({
            "case_code": code,
            "chart_alias": f"Chart-{item['chart'] + 1}",
            "event_count": len(item["public"]["events"]),
            "events": item["public"]["events"],
        })
        private_cases.append({
            "case_code": code,
            **item["private"],
        })
    public = {
        "format": FORMAT_PUBLIC,
        "packet_id": packet_id,
        "case_count": len(public_cases),
        "charts_in_packet": 3,
        "method": "locally_generated_counterbalanced_blind_review",
        "annotation_origin": "reviewer_input_only_no_prefilled_judgments",
        "observation_type": "visual_inspection_of_abstract_axis_assignments",
        "human_preference_verification": "not_collected",
        "physical_motor_performed": False,
        "cases": public_cases,
    }
    key = {
        "format": FORMAT_KEY,
        "packet_id": packet_id,
        "source_h11_sha256": source_sha256,
        "seed_hex": seed.hex(),
        "case_count": len(private_cases),
        "case_mapping": private_cases,
        "purpose": "private_unblinding_only_after_independent_review",
        "motor_executed": False,
        "human_preference_observed": False,
        "original_on_A_count": sum(
            x["original_side"] == "A" for x in private_cases),
        "original_on_B_count": sum(
            x["original_side"] == "B" for x in private_cases),
    }
    if (key["original_on_A_count"] != 11 or
            key["original_on_B_count"] != 11):
        raise ValueError("22-case order not balanced")
    return public, key


def render_html(packet: dict) -> str:
    """A single-file offline viewer, no fetch, CDN, network or key embedding."""
    if packet.get("format") != FORMAT_PUBLIC:
        raise ValueError("unrecognized blinded packet")
    raw = json.dumps(packet, ensure_ascii=False,
                     separators=(",", ":")).replace("<", "\\u003c")
    return HTML_TEMPLATE.replace("__PACKET_JSON__", raw)


# Embedded JSON is source/answer-key redacted before reaching this template.
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DMDOGO H1.2 — Masked A/B fingering review</title>
<style>
:root { color-scheme: dark; font-family: system-ui,-apple-system,"Segoe UI",sans-serif; background:#0c111b; color:#e5eaf2; }
* {box-sizing:border-box}
body {margin:0}
header {padding:22px clamp(14px,4vw,50px); border-bottom:1px solid #29344a; background:#141b29}
h1 {margin:0 0 7px;font-size:1.45rem;font-weight:700} h2 {font-size:1.12rem;margin:0 0 13px}
p,small {color:#aab6cb;line-height:1.5} p {margin:6px 0}
.layout {display:grid;grid-template-columns:240px minmax(0,1fr); min-height:calc(100vh - 105px)}
aside {border-right:1px solid #283448;padding:18px;background:#101622}
main {min-width:0;padding:20px clamp(12px,3vw,35px)}
button,select,input,textarea {font:inherit;color:inherit}
button,select,input,textarea {border-radius:9px;border:1px solid #3b4a65;background:#1b2637}
button {padding:8px 12px;cursor:pointer} button:hover {background:#2b3b52}
button:disabled {opacity:.4;cursor:default}
button:focus-visible,select:focus-visible,textarea:focus-visible {outline:2px solid #88b8ed}
nav button {display:block;width:100%;margin:4px 0;text-align:left}
nav button.active {border-color:#85b4fb;background:#263957}
.toolbar {display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:13px 0}
select,input {padding:7px 9px}
.panel {border:1px solid #29354b;border-radius:12px;background:#121b29;padding:17px;margin:15px 0}
.badge {display:inline-block;background:#26364e;color:#cbd9ed;padding:3px 8px;border-radius:8px;font-size:.77rem}
.table-wrap {max-height:65vh;overflow:auto;border-radius:11px;border:1px solid #30415a}
table {border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.88rem}
th {text-align:left;position:sticky;top:0;background:#202b3f;z-index:1}
td,th {padding:8px 10px;border-bottom:1px solid #263449}
td:nth-child(3),td:nth-child(4) {font-family:ui-monospace,monospace}
tr.changed {background:#1f3340}
tr.changed td:nth-child(3),tr.changed td:nth-child(4) {color:#a7e9d5}
.muted {color:#aab6cb} .row {display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.choice {display:flex;flex-wrap:wrap;gap:12px;padding:7px 0}
.choice label {cursor:pointer} textarea {width:100%;min-height:80px;padding:10px}
footer {padding:20px;font-size:.78rem;color:#95a4bc}
@media(max-width:760px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid #29344a}nav{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}nav button{margin:0;font-size:.77rem}main{padding:12px}}
</style>
</head>
<body>
<header>
<h1>DMDOGO H1.2 — 運指 A/B 比較</h1>
<p>コスト・元のPlanner側・変更候補側は非表示です。これは抽象的な軸割当の視覚レビューで、身体的な実行可能性やヒット率の試験ではありません。</p>
</header>
<div class="layout">
<aside>
<h2>比較対象 <span id="completed" class="badge">0/22</span></h2>
<nav id="caselist" aria-label="比較対象一覧"></nav>
</aside>
<main>
<h2 id="name">Case</h2>
<p id="summary"></p>
<div class="toolbar">
<label for="mode">表示</label>
<select id="mode">
<option value="context">差分の前後8イベント</option>
<option value="changes">変更イベントのみ</option>
<option value="all">全イベント（長大な譜面も閲覧可能）</option>
</select>
<button id="firstchange">最初の差分へ</button>
<button id="prev">前の80行</button>
<span id="pager" class="badge"></span>
<button id="next">次の80行</button>
</div>
<div class="table-wrap">
<table>
<thead><tr><th>Event</th><th>時刻 (秒)</th><th>A軸</th><th>B軸</th></tr></thead>
<tbody id="rows"></tbody>
</table>
</div>
<p class="muted">色付き行はA/Bの軸が異なる位置。時間・イベント番号は両側で共通です。変更が長距離に及ぶ場合は「全イベント」で前後を確認できます。</p>
<section class="panel">
<h2>この比較の評価（実際に確認した内容だけを入力）</h2>
<p>プレイ結果ではなく、提示された軸割当からの印象です。「不明」を選んでも構いません。</p>
<form id="form">
<div><strong>見た限りで好ましい運指</strong></div>
<div class="choice" id="preference">
<label><input type="radio" name="preference" value="A"> A</label>
<label><input type="radio" name="preference" value="B"> B</label>
<label><input type="radio" name="preference" value="tie"> 同程度</label>
<label><input type="radio" name="preference" value="uncertain"> 判断不能</label>
</div>
<div><strong>見た限りの叩きやすさの印象（実機検証ではない）</strong></div>
<div class="choice" id="plausibility">
<label><input type="radio" name="plausibility" value="A"> A</label>
<label><input type="radio" name="plausibility" value="B"> B</label>
<label><input type="radio" name="plausibility" value="tie"> 同程度</label>
<label><input type="radio" name="plausibility" value="uncertain"> 不明</label>
</div>
<div class="toolbar">
<label for="confidence">判断の確信度</label>
<select id="confidence">
<option value="">未回答</option><option value="low">低い</option>
<option value="medium">中程度</option><option value="high">高い</option>
</select>
</div>
<label for="notes">理由・気になった運指（任意）</label>
<textarea id="notes" placeholder="実際に確認した事実や判断の限界を記入"></textarea>
</form>
</section>
<div class="toolbar">
<button id="export" type="button">記入した評価をJSONに保存</button>
<span class="muted">記入内容は自動送信されません。未回答は未回答のまま保存します。</span>
</div>
</main></div>
<footer>H1.2 | Offline only | Original/alternate mapping stored in a separate private file | No physical Motor, no training</footer>
<script id="review-data" type="application/json">__PACKET_JSON__</script>
<script>
"use strict";
const packet=JSON.parse(document.getElementById("review-data").textContent);
const $=id=>document.getElementById(id);
const choices={};
let active=0,page=0;
const PER_PAGE=80;
function same(a,b){return JSON.stringify(a)===JSON.stringify(b)}
function current(){return packet.cases[active]}
function filtered(){
  const all=current().events;
  const mode=$("mode").value;
  if(mode==="all")return all;
  const changePositions=[];
  for(let i=0;i<all.length;i++)if(!same(all[i].a,all[i].b))changePositions.push(i);
  if(mode==="changes")return changePositions.map(i=>all[i]);
  const use=new Set();
  for(const i of changePositions)for(let j=Math.max(0,i-8);j<=Math.min(all.length-1,i+8);j++)use.add(j);
  return [...use].sort((a,b)=>a-b).map(i=>all[i]);
}
function saveForm(){
  if(!packet.cases[active])return;
  const code=current().case_code;
  const obj=choices[code]||{};
  for(const field of ["preference","plausibility"]){
    const selected=document.querySelector('input[name="'+field+'"]:checked');
    obj[field]=selected?selected.value:null;
  }
  obj.confidence=$("confidence").value||null;
  obj.notes=$("notes").value;
  choices[code]=obj;
}
function restoreForm(){
  const obj=choices[current().case_code]||{};
  $("form").reset();
  for(const field of ["preference","plausibility"]){
    if(obj[field]){
      const hit=[...document.querySelectorAll('input[name="'+field+'"]')].find(x=>x.value===obj[field]);
      if(hit)hit.checked=true;
    }
  }
  $("confidence").value=obj.confidence||"";
  $("notes").value=obj.notes||"";
}
function refreshNav(){
  const nav=$("caselist");nav.replaceChildren();
  let rated=0;
  for(let i=0;i<packet.cases.length;i++){
    const c=packet.cases[i];
    const obj=choices[c.case_code]||{};
    if(obj.preference)rated++;
    const b=document.createElement("button");
    b.textContent=c.case_code+(obj.preference?" · 記入済":"");
    b.className=i===active?"active":"";
    b.onclick=()=>{saveForm();active=i;page=0;draw()};
    nav.append(b);
  }
  $("completed").textContent=rated+"/"+packet.case_count;
}
function drawTable(){
  const rows=filtered();
  const totalPages=Math.max(1,Math.ceil(rows.length/PER_PAGE));
  page=Math.min(Math.max(0,page),totalPages-1);
  $("pager").textContent=(page+1)+"/"+totalPages+"ページ · "+rows.length+"行表示対象";
  $("prev").disabled=page===0;$("next").disabled=page>=totalPages-1;
  const tbody=$("rows");tbody.replaceChildren();
  for(const r of rows.slice(page*PER_PAGE,(page+1)*PER_PAGE)){
    const tr=document.createElement("tr");
    if(!same(r.a,r.b))tr.className="changed";
    for(const text of [String(r.i),String(r.t),r.a.join("+"),r.b.join("+")]){
      const td=document.createElement("td");td.textContent=text;tr.append(td);
    }
    tbody.append(tr);
  }
}
function draw(){
  const c=current();
  $("name").textContent=c.case_code+" · "+c.chart_alias;
  const changed=c.events.filter(e=>!same(e.a,e.b)).length;
  $("summary").textContent=c.event_count+"イベント · 比較可能な差分 "+changed+"イベント";
  restoreForm();refreshNav();drawTable();
}
$("mode").addEventListener("change",()=>{page=0;drawTable()});
$("prev").addEventListener("click",()=>{page--;drawTable()});
$("next").addEventListener("click",()=>{page++;drawTable()});
$("firstchange").addEventListener("click",()=>{
  const rows=filtered();
  const i=rows.findIndex(r=>!same(r.a,r.b));
  if(i>=0){page=Math.floor(i/PER_PAGE);drawTable()}
});
$("form").addEventListener("change",()=>{saveForm();refreshNav()});
$("notes").addEventListener("input",()=>{saveForm()});
$("export").addEventListener("click",()=>{
  saveForm();
  const records=packet.cases.map(c=>{
    const obj=choices[c.case_code]||{};
    return {case_code:c.case_code,
      preference:obj.preference||null,
      playability_impression:obj.plausibility||null,
      confidence:obj.confidence||null,notes:obj.notes||""};
  });
  const result={
    format:"dmdogo-h12-independent-visual-review-input-v1",
    packet_id:packet.packet_id,
    review_basis:"visual_abstract_axis_assignments_only",
    not_physical_motor_evidence:true,
    no_auto_generated_preferences:true,
    records:records
  };
  const blob=new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;a.download="dmdogo-h12-review-"+packet.packet_id+".json";
  document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
draw();
</script>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h11-report", required=True, type=Path)
    p.add_argument("--viewer-output", required=True, type=Path)
    p.add_argument("--private-key-output", required=True, type=Path)
    args = p.parse_args()
    if (args.viewer_output.suffix.lower() != ".html"
            or args.private_key_output.suffix.lower() != ".json"):
        p.error("viewer must be HTML and private key must be JSON")
    if args.viewer_output.parent.resolve() == args.private_key_output.parent.resolve():
        p.error("private unblinding key must be in a different directory")
    if args.viewer_output.exists() or args.private_key_output.exists():
        p.error("refusing to overwrite an existing viewer or private key")
    source_sha = file_sha256(args.h11_report)
    source = json.loads(args.h11_report.read_text(encoding="utf-8"))
    packet, key = compile_blind(
        source, source_sha256=source_sha, seed=secrets.token_bytes(32))
    viewer = render_html(packet)
    if (key["packet_id"] != packet["packet_id"]
            or key["case_count"] != packet["case_count"]
            or key["original_on_A_count"] != 11):
        raise ValueError("public/private packet identities differ")
    args.viewer_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_key_output.parent.mkdir(parents=True, exist_ok=True)
    # Both data products are created only after all validation succeeds.
    # Since no source artifacts are mutated, any unexpected I/O failure
    # must be addressed without overwriting an existing partial result.
    with args.private_key_output.open("x", encoding="utf-8") as out:
        json.dump(key, out, ensure_ascii=False, indent=2)
        out.write("\n")
    with args.viewer_output.open("x", encoding="utf-8") as out:
        out.write(viewer)
    print(json.dumps({
        "packet_id": packet["packet_id"],
        "cases": packet["case_count"],
        "chart_count": packet["charts_in_packet"],
        "original_on_A": key["original_on_A_count"],
        "original_on_B": key["original_on_B_count"],
        "viewer": str(args.viewer_output),
        "private_key": str(args.private_key_output),
        "human_reviews_recorded": 0,
        "motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
