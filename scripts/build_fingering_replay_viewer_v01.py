from __future__ import annotations

"""Build self-contained synchronized real-chart fingering replay viewers.

The saved pilot decisions are immutable inputs.  This script replays those
fixed allocations only to export 12-axis Motor state; it never re-runs Network
inference, changes labels, trains a model, or selects a checkpoint.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdod-real-chart-fingering-replay-v0.1"
AXIS_LABELS = {
    **{f"left_{index}": f"左指{index}" for index in range(1, 5)},
    "left_5": "左親指", "left_6": "左側面",
    **{f"right_{index}": f"右指{index}" for index in range(1, 5)},
    "right_5": "右親指", "right_6": "右側面",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _finger_from_token(token: str) -> teacher.Finger:
    axis, separator, mode = token.partition("/")
    if axis not in PHYSICAL_AXES:
        raise ValueError(f"unknown physical axis: {axis}")
    return teacher.Finger(axis, mode if separator else "single")


def parse_allocation(value: str | None) -> tuple[teacher.Finger, ...]:
    if not value:
        return ()
    return tuple(_finger_from_token(token.strip()) for token in value.split(" + "))


def _physical_axis(contact: str) -> str:
    hand, depth_text = contact.split("_")
    depth = int(depth_text)
    return f"{hand}_{6 if depth in (7, 8) else 5 if depth in (5, 6) else depth}"


def _trace_frame(env: CoupledContactRealChartEnv) -> list:
    state = env.motor.motor.observe()
    actuation = env.motor.motor.keyboard.config.actuation_m
    return [
        round(env.privileged_episode_time_s(), 6),
        [round(float(value / actuation), 6) for value in state.positions_m],
        [round(float(value / max(actuation / teacher.CTRL, 1e-9)), 6)
         for value in state.velocities_m_s],
        [int(value) for value in state.pressed_flags],
        [mode.value for mode in env.motor._modes],
        [int(value) for value in env.motor._edge],
    ]


def replay_fixed_decisions(segment, decisions: list[dict]) -> tuple[list[list], list[dict], dict]:
    """Replay saved allocations and export state without doing inference."""

    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_by_index = {event.event_index: event for event in oracle.events}
    schedule = sorted(decisions, key=lambda row: (float(row["decision_ms"]), row["event_index"]))
    if len({row["event_index"] for row in schedule}) != len(schedule):
        raise ValueError("duplicate fixed decision event")
    env = CoupledContactRealChartEnv(segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS)
    env.reset()
    committed: dict[int, teacher.PlanItem] = {}
    installed: set[int] = set()
    frames = [_trace_frame(env)]
    keydowns = []
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    for _ in range(maximum):
        now = env.privileged_episode_time_s()
        committed = {
            index: item for index, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[ordinal] for ordinal in event_by_index[index].target_ordinals)
        }
        for row in schedule:
            index = int(row["event_index"])
            if index in installed or float(row["decision_ms"]) * .001 > now + 1e-9:
                continue
            if index not in event_by_index:
                raise ValueError(f"decision references missing event {index}")
            allocation = parse_allocation(row.get("chosen"))
            event = event_by_index[index]
            item = teacher.PlanItem(index, event.episode_time_s, allocation)
            installed.add(index)
            if allocation:
                committed[index] = item
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                keydowns.append({
                    "episode_s": round(event.time_s, 6),
                    "axis": _physical_axis(event.key),
                })
        frames.append(_trace_frame(env))
        if step.done:
            break
    else:
        raise RuntimeError("fixed-decision replay exceeded step budget")
    stats = env.stats
    result = {
        "targets": stats.targets, "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses, "overloaded": stats.overloaded,
        "x_accuracy_percent": stats.x_accuracy_percent,
        "timing_mae_ms": stats.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
    }
    return frames, keydowns, result


def _assert_metric_match(policy: str, replay: dict, saved: dict) -> None:
    exact = ("targets", "hits", "misses", "too_early", "overloaded", "physical_keydowns")
    for key in exact:
        if replay[key] != saved[key]:
            raise RuntimeError(f"{policy} replay drift: {key} {replay[key]} != {saved[key]}")
    for key in ("x_accuracy_percent", "timing_mae_ms"):
        left, right = replay[key], saved[key]
        if left is None or right is None:
            if left != right:
                raise RuntimeError(f"{policy} replay drift: {key}")
        elif abs(float(left) - float(right)) > 1e-9:
            raise RuntimeError(f"{policy} replay drift: {key} {left} != {right}")


def _decision_view(row: dict | None, other: dict | None) -> dict | None:
    if row is None:
        return None
    keydowns = row.get("keydowns", [])
    errors = []
    for down in keydowns:
        time_ms = down.get("time_ms")
        if time_ms is not None:
            errors.append(float(time_ms) - float(row["due_ms"]))
    return {
        "event_index": int(row["event_index"]),
        "due_s": round(float(row["due_ms"]) * .001, 6),
        "decision_s": round(float(row["decision_ms"]) * .001, 6),
        "chosen": row.get("chosen"),
        "axes": [finger.axis for finger in parse_allocation(row.get("chosen"))],
        "teacher_choice": row.get("teacher_chosen") if "teacher_chosen" in row
                          else row.get("chosen"),
        "different": bool(other and row.get("chosen") != other.get("chosen")),
        "outcome": str(row.get("judgement", row.get("outcome", "unknown"))),
        "timing_error_ms": (round(min(errors, key=abs), 3) if errors else row.get("actual_error_ms")),
        "too_early_keydowns": int(row.get("too_early_keydowns", 0)),
    }


def _floor_rows(compiled, segment) -> list[dict]:
    target_indices = [target.floor_index for target in segment.targets]
    first = max(0, min(target_indices) - 18)
    last = min(len(compiled.floors) - 1, max(target_indices) + 18)
    rows = []
    for floor in compiled.floors[first:last + 1]:
        rows.append({
            "index": floor.index, "x": round(floor.x, 6), "y": round(floor.y, 6),
            "episode_s": round(segment.episode_time_from_chart(floor.target_time_s), 9),
            "exit_episode_s": round(segment.episode_time_from_chart(floor.exit_time_s), 9),
            "chart_s": round(floor.target_time_s, 9), "bpm": round(floor.bpm, 6),
            "midspin": floor.midspin, "is_ccw": floor.is_ccw,
            "num_planets": floor.num_planets, "pause_chart_s": round(floor.pause_s, 9),
            "markers": list(floor.event_markers),
        })
    return rows


def build_case_payload(case: dict, *, checkpoint: Path) -> dict:
    chart_path = Path(case["path"])
    if file_sha256(chart_path) != case["sha256"]:
        raise RuntimeError("source chart SHA256 changed")
    compiled = compile_adofai(load_adofai(chart_path))
    pitch_ratio = max(1e-6, compiled.pitch_percent * .01)
    start_chart_s = float(case["source_start_episode_s"]) * pitch_ratio
    end_chart_s = float(case["source_end_episode_s"]) * pitch_ratio
    segment = build_playable_segment(compiled, start_s=start_chart_s, end_s=end_chart_s)
    oracle = AutoplayTimingTeacher.from_segment(segment)

    network_rows = list(case["network_v0_1"]["decisions"])
    teacher_focus = {int(row["event_index"]): row for row in case["teacher"]["decisions"]}
    first_focus = min(teacher_focus)
    teacher_rows = [row for row in case["center_first"]["decisions"]
                    if int(row["event_index"]) < first_focus]
    teacher_rows.extend(case["teacher"]["decisions"])

    network_frames, network_downs, network_replay = replay_fixed_decisions(segment, network_rows)
    teacher_frames, teacher_downs, teacher_replay = replay_fixed_decisions(segment, teacher_rows)
    _assert_metric_match("Network", network_replay, case["network_v0_1"])
    _assert_metric_match("Teacher", teacher_replay, case["teacher"]["stats"])

    network_by_event = {int(row["event_index"]): row for row in network_rows}
    teacher_by_event = {int(row["event_index"]): row for row in teacher_rows}
    events = []
    for event in oracle.events:
        network_row = network_by_event.get(event.event_index)
        teacher_row = teacher_by_event.get(event.event_index)
        events.append({
            "event_index": event.event_index,
            "episode_s": round(event.episode_time_s, 9),
            "chart_s": round(event.chart_time_s, 9),
            "audio_s": round(compiled.chart_to_audio_time(event.chart_time_s), 9),
            "floor_indices": list(event.floor_indices), "chord_size": event.chord_size,
            "teacher": _decision_view(teacher_row, network_row),
            "network": _decision_view(network_row, teacher_row),
        })
    return {
        "format": FORMAT,
        "chart": case["chart"], "chart_sha256": case["sha256"],
        "difficulty": case.get("difficulty"),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "timebase": {
            "episode_duration_s": segment.duration_s,
            "segment_start_chart_s": segment.start_chart_s,
            "segment_end_chart_s": segment.end_chart_s,
            "pitch_percent": compiled.pitch_percent, "pitch_ratio": pitch_ratio,
            "offset_ms": compiled.offset_ms,
            "initial_bpm": compiled.initial_bpm,
            "countdown_ticks": compiled.countdown_ticks,
            "separate_countdown_time": compiled.separate_countdown_time,
            "separate_countdown_chart_s": compiled._separate_countdown_chart_seconds(),
            "formula": {
                "chart": "segment_start_chart_s + episode_s * pitch_ratio",
                "audio": "(chart_s + offset_ms/1000 - separate_countdown_chart_s) / pitch_ratio",
            },
        },
        "floors": _floor_rows(compiled, segment), "events": events,
        "policies": {
            "teacher": {"trace": teacher_frames, "keydowns": teacher_downs,
                        "saved_metrics": case["teacher"]["stats"]},
            "network": {"trace": network_frames, "keydowns": network_downs,
                        "saved_metrics": {key: case["network_v0_1"][key] for key in (
                            "targets", "hits", "misses", "too_early", "overloaded",
                            "x_accuracy_percent", "timing_mae_ms", "physical_keydowns")}},
        },
        "rendering_limitations": [
            "Gameplay floor geometry/timing only; decorations and camera events are not rendered.",
            "Planet travel is an explicitly labelled diagnostic interpolation, not Unity rendering.",
            "Saved physical metrics are displayed verbatim and are not recomputed by the HTML.",
        ],
    }


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def render_html(payload: dict) -> str:
    data = _json_for_script(payload)
    axis_labels = _json_for_script(AXIS_LABELS)
    title = escape(payload["chart"])
    policy_options = "".join(
        f'<option value="{escape(name)}">{escape(name.title())}</option>'
        for name in payload["policies"]
    )
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMDOD Fingering Replay — {title}</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#09101d;color:#e5edf8}}
main{{max-width:1500px;margin:auto;padding:14px}}h1{{font-size:20px;margin:0}}.sub,.note{{color:#91a3ba;font-size:12px}}.top{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}}
.controls{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:10px 0}}button,select,input{{background:#17243a;color:#e5edf8;border:1px solid #334968;border-radius:6px;padding:7px}}button.active{{background:#226b83}}#seek{{flex:1;min-width:220px}}
.layout{{display:grid;grid-template-columns:minmax(560px,1.55fr) minmax(420px,1fr);gap:12px}}.panel{{background:#101b2d;border:1px solid #243a58;border-radius:10px;padding:10px}}
#chart{{width:100%;height:min(62vh,650px);background:#08111f;border-radius:7px;display:block}}.times{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}}.metric{{background:#0a1423;padding:7px;border-radius:6px;font-variant-numeric:tabular-nums}}
.axisgrid{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}.axis{{display:grid;grid-template-columns:62px 1fr 52px;align-items:center;gap:6px;padding:5px;background:#0a1423;border-radius:6px}}.bar{{height:9px;background:#25334a;border-radius:5px;overflow:hidden}}.fill{{height:100%;background:#48b7d4}}.axis.down{{outline:2px solid #75db8a}}.axis.recover{{outline:1px solid #eead57}}.tag{{font-size:11px;color:#9eb0c6}}
.decision{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}.card{{background:#0a1423;padding:8px;border-radius:7px}}.different{{border:1px solid #e0a85b}}.judgement{{font-weight:700}}.Hit,.hit{{color:#79df91}}.Miss,.miss{{color:#ff718a}}.events{{max-height:190px;overflow:auto;font:12px/1.5 ui-monospace,monospace}}.now{{color:#ffd36d}}
@media(max-width:980px){{.layout{{grid-template-columns:1fr}}#chart{{height:52vh}}}}
</style></head><body><main>
<div class="top"><div><h1>{title}</h1><div class="sub">Oracle-assisted fixed-decision replay / training disabled</div></div><div id="source" class="sub"></div></div>
<div class="controls"><button id="play">▶ 再生</button><button id="prev">← 1イベント</button><button id="next">1イベント →</button>
<label>速度 <select id="speed"><option>.25</option><option>.5</option><option selected>1</option><option>2</option></select>×</label>
<label>表示 <select id="policy">{policy_options}</select></label>
<label>Event <input id="jump" type="number" min="0" style="width:80px"></label><button id="jumpBtn">移動</button>
<button id="prevBad">← 異常</button><button id="nextBad">異常 →</button>
<input id="seek" type="range" min="0" step="0.001"><span id="clock"></span></div>
<div class="layout"><section class="panel"><canvas id="chart" width="920" height="620"></canvas>
<div class="times"><div class="metric">episode<br><b id="episode"></b></div><div class="metric">chart<br><b id="chartTime"></b></div><div class="metric">audio<br><b id="audioTime"></b></div><div class="metric">floor/event<br><b id="floorEvent"></b></div></div>
<p class="note">惑星移動は診断用補間。床geometry・target時刻・Pause/SetSpeed/Twirl/Midspin markerはcompiled chart由来。装飾・camera・Unity easingは未対応。</p></section>
<aside class="panel"><div id="metrics" class="metric"></div><h3>12物理軸</h3><div id="axes" class="axisgrid"></div>
<h3>現在イベント</h3><div id="decision" class="decision"></div><h3>実KeyDown</h3><div id="keydowns" class="events"></div></aside></div>
<script>
const D={data}, LABELS={axis_labels}, AXES={_json_for_script(list(PHYSICAL_AXES))};
const $=id=>document.getElementById(id), canvas=$('chart'),ctx=canvas.getContext('2d'),duration=D.timebase.episode_duration_s;
let t=0,playing=false,lastWall=0; $('seek').max=duration; $('source').textContent=`${{D.difficulty||'difficulty unknown'}} | ${{D.chart_sha256.slice(0,12)}} | pitch ${{D.timebase.pitch_percent}}%`;
function binaryLast(rows,time,get=x=>x[0]){{let lo=0,hi=rows.length-1;if(!rows.length||get(rows[0])>time)return -1;while(lo<hi){{const mid=(lo+hi+1)>>1;if(get(rows[mid])<=time)lo=mid;else hi=mid-1}}return lo}}
function traceAt(policy,time){{const rows=D.policies[policy].trace,i=binaryLast(rows,time);return rows[Math.max(0,i)]}}
function eventAt(time){{let next=D.events.findIndex(e=>e.episode_s>=time-1e-9);if(next<0)next=D.events.length-1;return D.events[next]}}
function floorAt(time){{const i=binaryLast(D.floors,time,f=>f.episode_s);return D.floors[Math.max(0,i)]}}
function timeValues(time){{const chart=D.timebase.segment_start_chart_s+time*D.timebase.pitch_ratio;const audio=(chart+D.timebase.offset_ms/1000-D.timebase.separate_countdown_chart_s)/D.timebase.pitch_ratio;return{{chart,audio}}}}
function planetPoint(current,next,time){{if(!next)return{{x:current.x,y:current.y,p:1}};const pause=current.pause_chart_s/D.timebase.pitch_ratio,start=current.episode_s+pause,span=Math.max(1e-9,next.episode_s-start),p=Math.max(0,Math.min(1,(time-start)/span));const dx=next.x-current.x,dy=next.y-current.y,len=Math.hypot(dx,dy)||1,bow=Math.sin(Math.PI*p)*.38*(current.is_ccw?-1:1);return{{x:current.x+dx*p-dy/len*bow,y:current.y+dy*p+dx/len*bow,p}}}}
function drawChart(){{const current=floorAt(t),next=D.floors.find(f=>f.episode_s>t+1e-9)||current,planet=planetPoint(current,next,t),ev=eventAt(t);const W=canvas.width,H=canvas.height,scale=54,cx=W/2,cy=H/2,wx=x=>cx+(x-current.x)*scale,wy=y=>cy-(y-current.y)*scale;ctx.clearRect(0,0,W,H);ctx.fillStyle='#08111f';ctx.fillRect(0,0,W,H);const around=D.floors.filter(f=>Math.abs(f.index-current.index)<=20);ctx.beginPath();around.forEach((f,i)=>i?ctx.lineTo(wx(f.x),wy(f.y)):ctx.moveTo(wx(f.x),wy(f.y)));ctx.strokeStyle='#34455d';ctx.lineWidth=4;ctx.stroke();for(const f of around){{const target=D.events.some(e=>e.floor_indices.includes(f.index));ctx.beginPath();ctx.arc(wx(f.x),wy(f.y),f.midspin?5:8,0,Math.PI*2);ctx.fillStyle=f.index===current.index?'#ffd36d':target?'#55c5e2':'#657287';ctx.fill();if(f.midspin){{ctx.strokeStyle='#e893d1';ctx.stroke()}}if(f.markers.length){{ctx.fillStyle='#d9e6f7';ctx.font='10px sans-serif';ctx.fillText(f.markers.map(x=>x==='SetSpeed'?'S':x==='Twirl'?'T':x==='Pause'?'P':x==='MultiPlanet'?'M':'').join(''),wx(f.x)-5,wy(f.y)-12)}}}}
ctx.beginPath();ctx.arc(wx(current.x),wy(current.y),12,0,Math.PI*2);ctx.fillStyle='#ef6f88';ctx.fill();ctx.beginPath();ctx.arc(wx(planet.x),wy(planet.y),11,0,Math.PI*2);ctx.fillStyle='#68d6ec';ctx.fill();if(ev){{const target=D.floors.find(f=>f.index===ev.floor_indices[0]);if(target){{ctx.strokeStyle='#8af0a0';ctx.lineWidth=3;ctx.strokeRect(wx(target.x)-13,wy(target.y)-13,26,26)}}}}ctx.fillStyle='#dce8f6';ctx.font='14px ui-monospace,monospace';ctx.fillText(`floor ${{current.index}} | BPM ${{current.bpm.toFixed(2)}} | ${{current.num_planets}} planets`,15,24)}}
function renderAxes(policy){{const s=traceAt(policy,t),root=$('axes');root.innerHTML='';AXES.forEach((axis,i)=>{{const pos=s[1][i],vel=s[2][i],down=!!s[3][i],recover=!down&&(pos>.04||vel<-.04),el=document.createElement('div');el.className='axis'+(down?' down':recover?' recover':'');el.innerHTML=`<span>${{LABELS[axis]}}</span><div class="bar"><div class="fill" style="width:${{Math.max(0,Math.min(100,pos*100))}}%"></div></div><span class="tag">${{down?'押下':recover?'復帰':'待機'}}</span>`;root.appendChild(el)}})}}
function card(label,d){{if(!d)return`<div class="card"><b>${{label}}</b><p>decisionなし</p></div>`;return`<div class="card ${{d.different?'different':''}}"><b>${{label}}</b><p>${{d.chosen||'未割当'}}</p><span class="judgement ${{d.outcome}}">${{d.outcome}}</span><br><span class="tag">KeyDown誤差 ${{d.timing_error_ms??'—'}}ms / decision ${{d.decision_s.toFixed(3)}}s</span></div>`}}
function fmt(v,n=3){{return v==null?'—':Number(v).toFixed(n)}}
function render(){{t=Math.max(0,Math.min(duration,t));const policy=$('policy').value,ev=eventAt(t),tv=timeValues(t),m=D.policies[policy].saved_metrics,s=traceAt(policy,t),live=s[6];$('seek').value=t;$('clock').textContent=t.toFixed(3)+'s';$('episode').textContent=t.toFixed(6)+'s';$('chartTime').textContent=tv.chart.toFixed(6)+'s';$('audioTime').textContent=tv.audio.toFixed(6)+'s';const floor=floorAt(t);$('floorEvent').textContent=`#${{floor.index}} / ${{ev?'E'+ev.event_index:'—'}}`;const cumulative=live?` | 現在 Hit ${{live.hits}} Miss ${{live.misses}} TooEarly ${{live.too_early}}${{live.overloaded?' OVERLOAD':''}}`:'';$('metrics').textContent=`${{policy.toUpperCase()}} (saved): Hit ${{m.hits}}/${{m.targets}} | Miss ${{m.misses}} | TooEarly ${{m.too_early}} | Overload ${{m.overloaded}} | X ${{fmt(m.x_accuracy_percent)}}% | MAE ${{fmt(m.timing_mae_ms)}}ms${{cumulative}}`;drawChart();renderAxes(policy);$('decision').innerHTML=(ev?.teacher?card('Teacher',ev.teacher):'')+card('Network',ev?.network);const near=D.policies[policy].keydowns.filter(k=>Math.abs(k.episode_s-t)<.18);$('keydowns').innerHTML=near.map(k=>`<div class="${{Math.abs(k.episode_s-t)<.012?'now':''}}">${{k.episode_s.toFixed(3)}}s ${{LABELS[k.axis]}}</div>`).join('')||'<span class="tag">±180msにKeyDownなし</span>'}}
function seekEvent(delta){{const ev=eventAt(t),index=Math.max(0,Math.min(D.events.length-1,(ev?.event_index??0)+delta));t=D.events[index].episode_s;playing=false;$('play').textContent='▶ 再生';render()}}
function tick(now){{if(!playing)return;if(!lastWall)lastWall=now;t+=(now-lastWall)/1000*Number($('speed').value);lastWall=now;if(t>=duration){{t=duration;playing=false;$('play').textContent='▶ 再生'}}render();if(playing)requestAnimationFrame(tick)}}
function jumpBad(direction){{const bad=D.events.filter(e=>{{const d=e[$('policy').value];return d&&(d.outcome!=='Hit'||d.too_early_keydowns>0)}});const found=direction>0?bad.find(e=>e.episode_s>t+1e-6):[...bad].reverse().find(e=>e.episode_s<t-1e-6);if(found){{t=found.episode_s;playing=false;render()}}}}
$('play').onclick=()=>{{playing=!playing;$('play').textContent=playing?'⏸ 一時停止':'▶ 再生';lastWall=0;if(playing)requestAnimationFrame(tick)}};$('seek').oninput=e=>{{t=Number(e.target.value);playing=false;$('play').textContent='▶ 再生';render()}};$('prev').onclick=()=>seekEvent(-1);$('next').onclick=()=>seekEvent(1);$('prevBad').onclick=()=>jumpBad(-1);$('nextBad').onclick=()=>jumpBad(1);$('jumpBtn').onclick=()=>{{const wanted=Number($('jump').value)||0,ev=D.events.find(e=>e.event_index===wanted)||D.events[Math.max(0,Math.min(D.events.length-1,wanted))];if(ev)t=ev.episode_s;render()}};$('policy').onchange=render;render();
</script></main></body></html>'''


def render_index(entries: list[dict]) -> str:
    rows = "".join(
        f'<li><a href="{escape(entry["file"])}">{escape(entry["chart"])}</a> — '
        f'{escape(entry["difficulty"] or "unknown")} / {entry["duration_s"]:.3f}s</li>'
        for entry in entries
    )
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>DMDOD Fingering Replay</title>
<style>body{{font:15px/1.6 system-ui;background:#09101d;color:#e5edf8;max-width:1000px;margin:auto;padding:24px}}a{{color:#67d7ed}}li{{margin:10px}}</style>
<h1>DMDOD Real-Chart Fingering Replay v0.1</h1><p>固定済みpilot decisionsのローカル同期再生。学習・再推論なし。</p><ol>{rows}</ol></html>'''


def run(report_path: Path, checkpoint: Path, output_dir: Path, *, case_limit: int | None) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = list(report["cases"])
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive")
        cases = cases[:case_limit]
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, case in enumerate(cases, 1):
        payload = build_case_payload(case, checkpoint=checkpoint)
        stem = f"replay-{index:02d}-{case['sha256'][:10]}"
        _write_once(output_dir / f"{stem}.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        _write_once(output_dir / f"{stem}.html", render_html(payload))
        entries.append({"chart": case["chart"], "difficulty": case.get("difficulty"),
                        "duration_s": payload["timebase"]["episode_duration_s"],
                        "file": f"{stem}.html", "data": f"{stem}.json"})
        print(f"replay {index}/{len(cases)}: {case['sha256'][:12]}", flush=True)
    _write_once(output_dir / "index.html", render_index(entries))
    manifest = {
        "format": FORMAT, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path), "source_report_sha256": file_sha256(report_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "cases": entries, "training_performed": False,
        "physical_metrics_recomputed_for_display": False,
        "fixed_decision_replay_verified_against_saved_metrics": True,
        "external_dependencies": [],
    }
    _write_once(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        manifest = run(args.report, args.checkpoint, args.output_dir, case_limit=args.case_limit)
        _write_once(args.status_file, json.dumps({
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "output_dir": str(args.output_dir),
            "cases": len(manifest["cases"]),
        }, indent=2) + "\n")
    except BaseException as error:
        if not args.status_file.exists():
            _write_once(args.status_file, json.dumps({
                "pid": os.getpid(), "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1, "error_type": type(error).__name__, "error": str(error),
            }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
