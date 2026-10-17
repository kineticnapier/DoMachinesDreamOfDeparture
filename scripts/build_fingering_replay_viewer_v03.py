from __future__ import annotations

"""Build Viewer v0.3 from saved replay data without inference or simulation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_fingering_replay_viewer_v01 as v01

from dmdod.adofai.chart import AdoFaiAction, load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.contact_16k import LOGICAL_KEYS, PHYSICAL_AXES


FORMAT = "dmdod-full-chart-fingering-replay-v0.3"
KEY_LABELS = ("Tab", "1", "2", "E", "P", "^", "|", "Enter",
              "LShift", "LCtrl", "C", "無変換", "RAlt", ".", "RCtrl", "RShift")
KEY_DISPLAY_MAP = tuple(zip(KEY_LABELS, LOGICAL_KEYS))


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def key_overlay_snapshot(keydowns: list[dict], at_s: float) -> dict:
    """Recompute seek-safe counters exclusively from saved logical KeyDowns."""

    selected = [row for row in keydowns if float(row["episode_s"]) <= at_s + 1e-12]
    counts = Counter(row.get("contact") for row in selected if row.get("contact") in LOGICAL_KEYS)
    recent = [row for row in selected if float(row["episode_s"]) > at_s - 1.0]
    return {"total": len(selected), "kps": len(recent),
            "counts": {contact: counts[contact] for contact in LOGICAL_KEYS}}


def _new_bpm(actions: list[AdoFaiAction], base_bpm: float, current_bpm: float) -> float:
    value = 0.0
    for index, action in enumerate(actions):
        if (action.speed_type or "").lower() == "bpm":
            value = float(action.beats_per_minute or current_bpm)
        else:
            multiplier = float(action.bpm_multiplier or 1.0)
            value = value * multiplier if index > 0 else current_bpm * multiplier
    return value if value > 0 else current_bpm


def rotation_profiles(chart, compiled, segment) -> list[dict]:
    """Export exact arc geometry and piecewise SetSpeed timing for the viewer."""

    actions_by_floor: dict[int, list[AdoFaiAction]] = {}
    for action in chart.actions:
        if action.active:
            actions_by_floor.setdefault(action.floor, []).append(action)
    base_bpm = float(chart.initial_bpm if chart.initial_bpm > 0 else 100.0)
    current_bpm = base_bpm
    pitch = segment.pitch_ratio
    identity = 0
    result = []
    for floor in compiled.floors:
        actions = actions_by_floor.get(floor.index, [])
        speeds = [a for a in actions if a.event_type == "SetSpeed"]
        moved_deg = math.degrees(floor.angle_moved_rad)
        cursor_deg = 0.0
        pieces = []
        by_offset: dict[float, list[AdoFaiAction]] = {}
        for action in speeds:
            offset = max(0.0, min(moved_deg, float(action.angle_offset or 0.0)))
            by_offset.setdefault(offset, []).append(action)
        for offset in sorted(by_offset):
            if offset > cursor_deg + 1e-12:
                pieces.append((cursor_deg, offset, current_bpm))
            current_bpm = _new_bpm(by_offset[offset], base_bpm, current_bpm)
            cursor_deg = offset
        if moved_deg > cursor_deg + 1e-12:
            pieces.append((cursor_deg, moved_deg, current_bpm))

        entry = segment.episode_time_from_chart(floor.target_time_s)
        exit_at = segment.episode_time_from_chart(floor.exit_time_s)
        pause = floor.pause_s / pitch
        nominal = sum((end - start) / 180.0 * 60.0 / bpm / pitch
                      for start, end, bpm in pieces)
        rotation_duration = max(0.0, exit_at - entry - pause)
        pre_hold = max(0.0, rotation_duration - nominal)
        rotation_start = entry + pause + pre_hold
        rows = []
        elapsed = rotation_start
        for index, (start, end, bpm) in enumerate(pieces):
            duration = (end - start) / 180.0 * 60.0 / bpm / pitch
            finish = exit_at if index == len(pieces) - 1 else elapsed + duration
            rows.append({"start_s": round(elapsed, 9), "end_s": round(finish, 9),
                         "from_rad": round(math.radians(start), 9),
                         "to_rad": round(math.radians(end), 9), "bpm": round(bpm, 6)})
            elapsed = finish
        result.append({
            "index": floor.index, "x": round(floor.x, 6), "y": round(floor.y, 6),
            "episode_s": round(entry, 9), "exit_episode_s": round(exit_at, 9),
            "chart_s": round(floor.target_time_s, 9), "bpm": round(floor.bpm, 6),
            "entry_angle_rad": round(floor.entry_angle_rad, 9),
            "exit_angle_rad": round(floor.exit_angle_rad, 9),
            "orbit_start_angle_rad": round(floor.orbit_start_angle_rad, 9),
            "orbit_end_angle_rad": round(floor.orbit_end_angle_rad, 9),
            "angle_moved_rad": round(floor.angle_moved_rad, 9),
            "rotation_sign": -1 if floor.is_ccw else 1,
            "rotation_start_s": round(rotation_start, 9),
            "rotation_end_s": round(exit_at, 9),
            "motion_segments": rows,
            "midspin": floor.midspin, "previous_midspin": floor.previous_midspin,
            "is_ccw": floor.is_ccw, "num_planets": floor.num_planets,
            "fixed_identity": identity % max(2, floor.num_planets),
            "pause_chart_s": round(floor.pause_s, 9),
            "markers": list(floor.event_markers),
        })
        if not floor.midspin and floor.angle_moved_rad > 1e-8:
            identity += 1
    return result


def orbit_point(floor: dict, at_s: float, *, radius: float = 1.5) -> tuple[float, float]:
    """Primary moving planet position; Python oracle for JS fixture tests."""

    moved = 0.0
    segments = floor["motion_segments"]
    if at_s >= floor["rotation_end_s"]:
        moved = floor["angle_moved_rad"]
    elif at_s > floor["rotation_start_s"]:
        for row in segments:
            span = row["end_s"] - row["start_s"]
            if at_s >= row["end_s"]:
                moved = row["to_rad"]
            elif at_s > row["start_s"] and span > 0:
                moved = row["from_rad"] + (row["to_rad"] - row["from_rad"]) * (
                    (at_s - row["start_s"]) / span)
                break
    theta = floor["orbit_start_angle_rad"] + floor["rotation_sign"] * moved
    return floor["x"] + math.sin(theta) * radius, floor["y"] + math.cos(theta) * radius


def build_payload(source: dict, chart_path: Path) -> dict:
    if v01.file_sha256(chart_path) != source["chart_sha256"]:
        raise RuntimeError("chart SHA256 differs from saved replay")
    chart = load_adofai(chart_path)
    compiled = compile_adofai(chart)
    segment = build_playable_segment(compiled)
    if abs(segment.duration_s - float(source["timebase"]["full_chart_duration_s"])) > 1e-6:
        raise RuntimeError("compiled full-chart duration drift")
    payload = dict(source)
    payload["format"] = FORMAT
    payload["floors"] = rotation_profiles(chart, compiled, segment)
    payload["key_overlay"] = {
        "mapping": [{"label": label, "contact": contact}
                    for label, contact in KEY_DISPLAY_MAP],
        "mapping_semantics": "Viewer label only; not a claim about physical keyboard binding.",
        "counter_source": "policies[policy].keydowns[].contact and episode_s",
        "pressed_source": "saved 12-axis trace plus thumb/edge topology",
    }
    payload["rendering_limitations"] = [
        "Gameplay timing/geometry only; decorations, camera events and Unity easing are not rendered.",
        "Two-planet circular orbit, direction, pause and arrival geometry are reproduced from compiled timing.",
        "Three-planet mode uses the compiled primary-orbit correction and diagnostic equally-spaced auxiliary planets; it is not a claim of Unity-identical artwork.",
        "Display key labels are configurable aliases for logical contacts, not verified OS keyboard bindings.",
        "Saved physical metrics are copied unchanged and are not recomputed.",
    ]
    return payload


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def render_html(payload: dict) -> str:
    title = escape(payload["chart"])
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMDOD Replay v0.3 — {title}</title><style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#080d18;color:#edf0fa}}main{{max-width:1500px;margin:auto;padding:12px}}h1{{font-size:19px;margin:0}}.sub,.note,.tag{{color:#9da9bd;font-size:12px}}.controls{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:9px 0}}button,select,input{{background:#17152b;color:#eeeaff;border:1px solid #58478b;border-radius:6px;padding:7px}}#seek{{flex:1;min-width:220px}}.layout{{display:grid;grid-template-columns:minmax(560px,1.55fr) minmax(420px,1fr);gap:10px}}.panel{{background:#101626;border:1px solid #292c48;border-radius:9px;padding:9px}}#chart{{width:100%;height:min(57vh,610px);background:#070d18;border-radius:7px;display:block}}.times{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:7px}}.metric{{background:#0a1020;padding:6px;border-radius:6px;font-variant-numeric:tabular-nums}}
.keyhead{{display:flex;justify-content:space-between;align-items:end;margin:7px 0 5px}}.keytotals b{{color:#d7b7ff;font-size:20px}}.keygrid{{display:grid;grid-template-columns:repeat(8,minmax(48px,1fr));gap:5px}}.key{{min-width:0;background:#080a13;border:1px solid #7254b7;border-bottom:3px solid #4a337d;border-radius:5px;padding:5px 3px;text-align:center;box-shadow:inset 0 0 10px #291a44}}.key .label{{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#d8c8f3}}.key .count{{font:700 17px ui-monospace,monospace}}.key.down{{background:#6d36b4;border-color:#d7a8ff;box-shadow:0 0 15px #a85cff,inset 0 0 12px #d394ff;color:#fff}}
.axisgrid{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}.axis{{display:grid;grid-template-columns:62px 1fr 48px;align-items:center;gap:5px;padding:4px;background:#0a1020;border-radius:5px}}.bar{{height:8px;background:#283048;border-radius:5px;overflow:hidden}}.fill{{height:100%;background:#8b63cf}}.axis.down{{outline:2px solid #d193ff}}.axis.recover{{outline:1px solid #eead57}}details{{margin-top:8px}}summary{{cursor:pointer;color:#cbb9ea}}.card{{background:#0a1020;padding:7px;border-radius:6px}}.Hit{{color:#7ce197}}.Miss{{color:#ff718a}}.events{{max-height:145px;overflow:auto;font:12px/1.5 ui-monospace,monospace}}.now{{color:#f0c8ff}}@media(max-width:980px){{.layout{{grid-template-columns:1fr}}#chart{{height:50vh}}.keygrid{{gap:3px}}.key .label{{font-size:9px}}}}
</style></head><body><main><h1>{title}</h1><div class="sub">Saved Network replay / Viewer v0.3 / no inference, training or physics rerun</div>
<div class="controls"><button id="play">▶ 再生</button><button id="prev">← Event</button><button id="next">Event →</button><label>速度 <select id="speed"><option>.25</option><option>.5</option><option selected>1</option><option>2</option></select>×</label><label>Event <input id="jump" type="number" min="0" style="width:82px"></label><button id="jumpBtn">移動</button><input id="seek" type="range" min="0" step="0.001"><span id="clock"></span></div>
<section class="panel"><div class="keyhead"><b>16 Contact KeyDown Counter</b><div class="keytotals">KPS <b id="kps">0</b>　Total <b id="total">0</b></div></div><div id="keys" class="keygrid"></div><div class="note">表示名は論理接点への変更可能なalias。実キーボード配線を表しません。</div></section>
<div class="layout"><section class="panel"><canvas id="chart" width="920" height="600"></canvas><div class="times"><div class="metric">episode<br><b id="episode"></b></div><div class="metric">chart<br><b id="chartTime"></b></div><div class="metric">audio<br><b id="audioTime"></b></div><div class="metric">floor/event<br><b id="floorEvent"></b></div></div><p class="note">円軌道はcompiled entry/exit angle・回転量・方向・Pause・SetSpeed区間を使用。装飾・camera・Unity easingは未対応。</p></section>
<aside class="panel"><div id="metrics" class="metric"></div><h3>現在イベント</h3><div id="decision"></div><h3>実KeyDown</h3><div id="keydowns" class="events"></div><details open><summary>12物理軸 Motor詳細</summary><div id="axes" class="axisgrid"></div></details></aside></div>
<script>const D={_json(payload)},AXES={_json(list(PHYSICAL_AXES))},LABELS={_json(v01.AXIS_LABELS)};const $=x=>document.getElementById(x),canvas=$('chart'),ctx=canvas.getContext('2d'),P=D.policies.network,duration=D.timebase.episode_duration_s;let t=0,playing=false,lastWall=0;$('seek').max=duration;
function lastLE(rows,time,get=x=>x[0]){{let lo=0,hi=rows.length-1;if(!rows.length||get(rows[0])>time)return-1;while(lo<hi){{const m=(lo+hi+1)>>1;if(get(rows[m])<=time)lo=m;else hi=m-1}}return lo}}function lowerGT(rows,time,get=x=>x.episode_s){{let lo=0,hi=rows.length;while(lo<hi){{const m=(lo+hi)>>1;if(get(rows[m])<=time)lo=m+1;else hi=m}}return lo}}function frameAt(time){{return P.trace[Math.max(0,lastLE(P.trace,time))]}}function floorAt(time){{return D.floors[Math.max(0,lastLE(D.floors,time,x=>x.episode_s))]}}function eventAt(time){{let i=D.events.findIndex(e=>e.episode_s>=time-1e-9);return D.events[i<0?D.events.length-1:i]}}function timeValues(time){{const chart=D.timebase.segment_start_chart_s+time*D.timebase.pitch_ratio;return{{chart,audio:(chart+D.timebase.offset_ms/1000-D.timebase.separate_countdown_chart_s)/D.timebase.pitch_ratio}}}}
function arcMoved(f,time){{if(time<=f.rotation_start_s)return 0;if(time>=f.rotation_end_s)return f.angle_moved_rad;for(const s of f.motion_segments){{if(time>=s.end_s)continue;if(time<=s.start_s)return s.from_rad;return s.from_rad+(s.to_rad-s.from_rad)*(time-s.start_s)/Math.max(1e-12,s.end_s-s.start_s)}}return f.angle_moved_rad}}function orbitPoint(f,time,extra=0){{const a=f.orbit_start_angle_rad+f.rotation_sign*arcMoved(f,time)+extra;return{{x:f.x+Math.sin(a)*1.5,y:f.y+Math.cos(a)*1.5}}}}
function pressedContacts(frame){{const out=new Set(),down=frame[3],modes=frame[4],edges=frame[5];for(let h=0;h<2;h++){{const hand=h?'right':'left',base=h*6;for(let i=0;i<4;i++)if(down[base+i])out.add(`${{hand}}_${{i+1}}`);if(down[base+4]){{if(modes[h]==='tip'||modes[h]==='flat')out.add(`${{hand}}_5`);if(modes[h]==='flat')out.add(`${{hand}}_6`)}}if(down[base+5]&&edges[h]){{out.add(`${{hand}}_7`);out.add(`${{hand}}_8`)}}}}return out}}function renderKeys(){{const end=lowerGT(P.keydowns,t),start=lowerGT(P.keydowns,t-1),shown=P.keydowns.slice(0,end),counts=new Map();for(const k of shown)counts.set(k.contact,(counts.get(k.contact)||0)+1);const down=pressedContacts(frameAt(t));$('kps').textContent=end-start;$('total').textContent=end;$('keys').innerHTML=D.key_overlay.mapping.map(k=>`<div class="key ${{down.has(k.contact)?'down':''}}" title="${{k.contact}}"><div class="label">${{k.label}}</div><div class="count">${{counts.get(k.contact)||0}}</div></div>`).join('')}}
function drawChart(){{const f=floorAt(t),p=orbitPoint(f,t),W=canvas.width,H=canvas.height,scale=54,cx=W/2,cy=H/2,wx=x=>cx+(x-f.x)*scale,wy=y=>cy-(y-f.y)*scale,around=D.floors.filter(x=>Math.abs(x.index-f.index)<=20);ctx.clearRect(0,0,W,H);ctx.fillStyle='#070d18';ctx.fillRect(0,0,W,H);ctx.beginPath();around.forEach((x,i)=>i?ctx.lineTo(wx(x.x),wy(x.y)):ctx.moveTo(wx(x.x),wy(x.y)));ctx.strokeStyle='#343851';ctx.lineWidth=4;ctx.stroke();for(const x of around){{ctx.beginPath();ctx.arc(wx(x.x),wy(x.y),x.midspin?5:8,0,Math.PI*2);ctx.fillStyle=x.index===f.index?'#f0c66c':'#59647a';ctx.fill();if(x.midspin){{ctx.strokeStyle='#e893d1';ctx.stroke()}}if(x.markers.length){{ctx.fillStyle='#d8ddef';ctx.font='10px sans-serif';ctx.fillText(x.markers.map(v=>v==='SetSpeed'?'S':v==='Twirl'?'T':v==='Pause'?'P':v==='MultiPlanet'?'M':'').join(''),wx(x.x)-5,wy(x.y)-12)}}}}const colors=['#ef6f88','#73dbef','#c18cff'];const fixedColor=colors[f.fixed_identity%colors.length],movingColor=colors[(f.fixed_identity+1)%colors.length];ctx.beginPath();ctx.arc(wx(f.x),wy(f.y),12,0,Math.PI*2);ctx.fillStyle=fixedColor;ctx.fill();ctx.beginPath();ctx.arc(wx(p.x),wy(p.y),11,0,Math.PI*2);ctx.fillStyle=movingColor;ctx.fill();if(f.num_planets>2){{const q=orbitPoint(f,t,Math.PI*2/3);ctx.beginPath();ctx.arc(wx(q.x),wy(q.y),10,0,Math.PI*2);ctx.fillStyle=colors[(f.fixed_identity+2)%colors.length];ctx.fill()}}const ev=eventAt(t),target=ev&&D.floors.find(x=>x.index===ev.floor_indices[0]);if(target){{ctx.strokeStyle='#8af0a0';ctx.lineWidth=3;ctx.strokeRect(wx(target.x)-13,wy(target.y)-13,26,26)}}ctx.fillStyle='#e3e6f2';ctx.font='14px ui-monospace,monospace';ctx.fillText(`floor ${{f.index}} | BPM ${{f.bpm.toFixed(2)}} | ${{f.is_ccw?'CCW':'CW'}} | arc ${{(f.angle_moved_rad*180/Math.PI).toFixed(1)}}°`,15,24)}}
function renderAxes(){{const s=frameAt(t);$('axes').innerHTML=AXES.map((axis,i)=>{{const pos=s[1][i],vel=s[2][i],down=!!s[3][i],recover=!down&&(pos>.04||vel<-.04);return`<div class="axis ${{down?'down':recover?'recover':''}}"><span>${{LABELS[axis]}}</span><div class="bar"><div class="fill" style="width:${{Math.max(0,Math.min(100,pos*100))}}%"></div></div><span class="tag">${{down?'押下':recover?'復帰':'待機'}}</span></div>`}}).join('')}}function render(){{t=Math.max(0,Math.min(duration,t));$('seek').value=t;$('clock').textContent=t.toFixed(3)+'s';const tv=timeValues(t),f=floorAt(t),e=eventAt(t),m=P.saved_metrics;$('episode').textContent=t.toFixed(6)+'s';$('chartTime').textContent=tv.chart.toFixed(6)+'s';$('audioTime').textContent=tv.audio.toFixed(6)+'s';$('floorEvent').textContent=`#${{f.index}} / ${{e?'E'+e.event_index:'—'}}`;$('metrics').textContent=`Hit ${{m.hits}}/${{m.targets}} | Miss ${{m.misses}} | TooEarly ${{m.too_early}} | Overload ${{m.overloaded}} | X ${{Number(m.x_accuracy_percent).toFixed(3)}}% | MAE ${{Number(m.timing_mae_ms).toFixed(3)}}ms`;$('decision').innerHTML=e?.network?`<div class="card"><b>${{e.network.chosen||'未割当'}}</b><br><span class="${{e.network.outcome}}">${{e.network.outcome}}</span> / error ${{e.network.timing_error_ms??'—'}}ms</div>`:'<div class="card">decisionなし</div>';const a=Math.max(0,lowerGT(P.keydowns,t-.18)-1),b=lowerGT(P.keydowns,t+.18);$('keydowns').innerHTML=P.keydowns.slice(a,b).map(k=>`<div class="${{Math.abs(k.episode_s-t)<.012?'now':''}}">${{k.episode_s.toFixed(3)}}s ${{k.contact}} / ${{LABELS[k.axis]}}</div>`).join('')||'<span class="tag">±180msにKeyDownなし</span>';renderKeys();drawChart();renderAxes()}}
function seekEvent(d){{const e=eventAt(t),i=Math.max(0,Math.min(D.events.length-1,(e?.event_index??0)+d));t=D.events[i].episode_s;playing=false;render()}}function tick(now){{if(!playing)return;if(!lastWall)lastWall=now;t+=(now-lastWall)/1000*Number($('speed').value);lastWall=now;if(t>=duration){{t=duration;playing=false;$('play').textContent='▶ 再生'}}render();if(playing)requestAnimationFrame(tick)}}$('play').onclick=()=>{{playing=!playing;$('play').textContent=playing?'⏸ 一時停止':'▶ 再生';lastWall=0;if(playing)requestAnimationFrame(tick)}};$('seek').oninput=e=>{{t=Number(e.target.value);playing=false;render()}};$('prev').onclick=()=>seekEvent(-1);$('next').onclick=()=>seekEvent(1);$('jumpBtn').onclick=()=>{{const n=Number($('jump').value)||0,e=D.events.find(x=>x.event_index===n);if(e)t=e.episode_s;render()}};render();</script></main></body></html>'''


def run(source_path: Path, chart_path: Path, output_dir: Path) -> dict:
    source_text = source_path.read_text(encoding="utf-8")
    source = json.loads(source_text)
    before_metrics = json.dumps(source["policies"]["network"]["saved_metrics"], sort_keys=True)
    payload = build_payload(source, chart_path)
    after_metrics = json.dumps(payload["policies"]["network"]["saved_metrics"], sort_keys=True)
    if before_metrics != after_metrics:
        raise RuntimeError("saved physical metrics changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "full-chart-replay-v03.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "full-chart-replay-v03.html", render_html(payload))
    manifest = {"format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_replay": str(source_path), "source_replay_sha256": v01.file_sha256(source_path),
                "chart": str(chart_path), "chart_sha256": v01.file_sha256(chart_path),
                "events_reused": len(payload["events"]),
                "motor_frames_reused": len(payload["policies"]["network"]["trace"]),
                "inference_run": False, "physics_run": False, "training_run": False,
                "saved_metrics_unchanged": True}
    _write_once(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite output")
    run(args.source_replay, args.chart, args.output_dir)


if __name__ == "__main__":
    main()
