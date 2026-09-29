from __future__ import annotations

import argparse
import json
import math
import webbrowser
from pathlib import Path

import torch

import eval_real_chart as evaluator
import eval_real_chart_v070 as v070_eval
import train_real_chart_v054 as v054
from fatal_diagnostics import FatalTrackingMixin, fatal_summary, format_fatal_summary
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.keyboard import KeyEvent
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.real_chart_hud import DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import (
    HUD_REAL_CHART_INPUT_DIM,
    encode_hud_real_chart_observation,
)


DEFAULT_OUTPUT = "artifacts/real_chart_replay.html"


class _ReplayRecorderMixin:
    """Record physical KeyDown judgements without changing environment semantics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.replay_events: list[dict] = []

    def reset(self):
        self.replay_events = []
        return super().reset()

    def _score_event(self, event):
        if event.event is not KeyEvent.DOWN:
            return super()._score_event(event)

        target = self.privileged_next_target()
        before = len(self.hit_margins)
        reward = super()._score_event(event)
        judgement = None
        if len(self.hit_margins) > before:
            judgement = self.hit_margins[-1].value
        self.replay_events.append(
            {
                "t": round(float(event.time_s), 6),
                "key": str(event.key),
                "floor": None if target is None else int(target.floor_index),
                "target_t": None if target is None else round(float(target.episode_time_s), 6),
                "judgement": judgement,
            }
        )
        return reward


class ReplayRealChartEnv(_ReplayRecorderMixin, FatalTrackingMixin, v054.DiagnosticRealChartMotorEnv):
    """Replay recorder for the legacy 233D observation."""


class ReplayHudRealChartEnv(_ReplayRecorderMixin, FatalTrackingMixin, DiagnosticHudRealChartMotorEnv):
    """Replay recorder for the v0.7+ 245D human-visible HUD observation."""


def _finite(value: float, digits: int = 6) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _record_frame(env, observation, action_left: float, action_right: float) -> list:
    now = env.privileged_episode_time_s()
    chart_time = env.segment.chart_time_from_episode(now)
    floor_index = env._floor_index_at_chart_time(chart_time)
    diagnostics = env.motor.diagnostics()
    return [
        _finite(now, 4),
        int(floor_index),
        _finite(observation.orbiting_x, 5),
        _finite(observation.orbiting_y, 5),
        _finite(action_left, 4),
        _finite(action_right, 4),
        _finite(observation.motor.left_position_m * 1000.0, 4),
        _finite(observation.motor.right_position_m * 1000.0, 4),
        1 if observation.motor.left_pressed else 0,
        1 if observation.motor.right_pressed else 0,
        _finite(diagnostics.left_activation, 4),
        _finite(diagnostics.right_activation, 4),
        _finite(diagnostics.left_fatigue, 4),
        _finite(diagnostics.right_fatigue, 4),
        _finite(env._overload.value, 4),
        int(env._hits),
        int(env._misses),
        int(env._too_early),
    ]


def _collect_replay(
    model,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    env_cls=ReplayRealChartEnv,
    encoder=encode_real_chart_observation,
) -> tuple[dict, object, object]:
    env = env_cls(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    frames: list[list] = [_record_frame(env, observation, 0.0, 0.0)]

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(
                encoder(observation),
                dtype=torch.float32,
                device=device,
            )
            action, state = model.deterministic_action(x, state)
            step = env.step(action)
            observation = step.observation
            frames.append(_record_frame(env, observation, action.left, action.right))
            if step.done:
                break
        else:
            raise RuntimeError("visualizer episode exceeded step budget")

    target_floors = {target.floor_index for target in segment.targets}
    floors = [
        [
            int(floor.index),
            _finite(floor.x, 5),
            _finite(floor.y, 5),
            1 if floor.midspin else 0,
            1 if floor.index in target_floors else 0,
        ]
        for floor in segment.chart.floors
    ]
    stats = env.stats
    fatal = fatal_summary(env)
    data = {
        "frames": frames,
        "floors": floors,
        "events": env.replay_events,
        "segment": {
            "start": _finite(segment.start_chart_s, 4),
            "end": _finite(segment.end_chart_s, 4),
            "duration": _finite(segment.duration_s, 4),
            "targets": len(segment.targets),
        },
        "result": {
            "hits": int(stats.hits),
            "misses": int(stats.misses),
            "xacc": _finite(stats.x_accuracy_percent, 3),
            "pp": _finite(stats.perfect_rate * 100.0, 3),
            "mae": None if stats.mean_abs_error_ms is None else _finite(stats.mean_abs_error_ms, 3),
            "early": int(stats.too_early_presses),
            "overload": bool(stats.overloaded),
            "keydowns": int(env.physical_keydowns),
            "clear": bool(fatal["clear"]),
            "fatal_reason": fatal["fatal_reason"],
            "fatal_floor": fatal["fatal_floor"],
            "fatal_time_s": None if fatal["fatal_time_s"] is None else _finite(fatal["fatal_time_s"], 4),
            "survived_targets": int(fatal["survived_targets"]),
        },
    }
    return data, v054.StudentEvalResult(stats, env.physical_keydowns), env


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _html_document(data: dict, *, checkpoint: str, chart: str) -> str:
    payload = _json_for_script(data)
    meta = _json_for_script({"checkpoint": checkpoint, "chart": chart})
    return f"""<!doctype html>
<html lang=\"ja\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>DMDOD Real Chart Replay</title>
<style>
:root {{ color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
body {{ margin:0; background:#101217; color:#e8eaf0; }}
main {{ max-width:1180px; margin:auto; padding:18px; }}
.top {{ display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-bottom:12px; }}
.title {{ font-size:20px; font-weight:700; margin-right:auto; }}
.small {{ color:#9aa3b2; font-size:12px; overflow-wrap:anywhere; }}
.grid {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(260px,1fr); gap:12px; }}
.panel {{ background:#181c24; border:1px solid #2c3340; border-radius:10px; padding:12px; }}
canvas {{ width:100%; aspect-ratio:16/10; display:block; background:#0d1016; border-radius:7px; }}
.controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:10px; }}
button,select {{ color:inherit; background:#252c38; border:1px solid #3a4556; border-radius:7px; padding:7px 10px; }}
input[type=range] {{ flex:1; min-width:180px; }}
.metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
.metric {{ background:#11151c; padding:8px; border-radius:7px; }}
.metric b {{ font-size:17px; }}
.finger {{ margin-top:10px; }}
.row {{ display:grid; grid-template-columns:58px 1fr 66px; gap:8px; align-items:center; margin:7px 0; }}
.track {{ height:12px; background:#0c0f14; border-radius:999px; position:relative; overflow:hidden; }}
.fill {{ position:absolute; height:100%; left:50%; width:0; background:#7aa2f7; }}
.fill.neg {{ right:50%; left:auto; background:#bb9af7; }}
.key {{ display:inline-block; min-width:22px; text-align:center; padding:3px 5px; border:1px solid #434d5e; border-radius:5px; }}
.key.on {{ background:#9ece6a; color:#101217; }}
.events {{ max-height:210px; overflow:auto; font-size:12px; line-height:1.55; }}
.event-current {{ color:#f4bf75; }}
.legend {{ display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:#aab3c2; margin-top:8px; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }}
@media(max-width:780px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<div class=\"top\">
  <div><div class=\"title\">DMDOD Real Chart Replay</div><div id=\"source\" class=\"small\"></div></div>
  <div class=\"small\">training=DISABLED / deterministic policy replay</div>
</div>
<div class=\"grid\">
  <section class=\"panel\">
    <canvas id=\"chart\" width=\"900\" height=\"560\"></canvas>
    <div class=\"legend\"><span><i class=\"dot\" style=\"background:#565f70\"></i>floor</span><span><i class=\"dot\" style=\"background:#7dcfff\"></i>playable target</span><span><i class=\"dot\" style=\"background:#f7768e\"></i>current</span><span><i class=\"dot\" style=\"background:#e0af68\"></i>orbiting</span></div>
    <div class=\"controls\">
      <button id=\"play\" type=\"button\">▶ Play</button>
      <select id=\"speed\" aria-label=\"Playback speed\"><option value=\"0.25\">0.25×</option><option value=\"0.5\">0.5×</option><option value=\"1\" selected>1×</option><option value=\"2\">2×</option><option value=\"4\">4×</option><option value=\"8\">8×</option></select>
      <input id=\"scrub\" type=\"range\" min=\"0\" max=\"1\" value=\"0\" step=\"1\" aria-label=\"Replay position\">
      <span id=\"clock\">0.00s</span>
    </div>
  </section>
  <aside class=\"panel\">
    <div class=\"metrics\">
      <div class=\"metric\">Hits<br><b id=\"hits\">0</b></div>
      <div class=\"metric\">Miss<br><b id=\"misses\">0</b></div>
      <div class=\"metric\">Early<br><b id=\"early\">0</b></div>
      <div class=\"metric\">Overload<br><b id=\"overload\">0.00</b></div>
    </div>
    <div class=\"finger\"><b>Motor output / body</b>
      <div class=\"row\"><span>Left</span><div class=\"track\"><div id=\"laP\" class=\"fill\"></div><div id=\"laN\" class=\"fill neg\"></div></div><span><span id=\"lk\" class=\"key\">L</span></span></div>
      <div class=\"row\"><span>Right</span><div class=\"track\"><div id=\"raP\" class=\"fill\"></div><div id=\"raN\" class=\"fill neg\"></div></div><span><span id=\"rk\" class=\"key\">R</span></span></div>
      <div id=\"motorText\" class=\"small\"></div>
    </div>
    <hr style=\"border:0;border-top:1px solid #2c3340;margin:12px 0\">
    <b>Physical KeyDowns</b>
    <div id=\"events\" class=\"events\"></div>
    <hr style=\"border:0;border-top:1px solid #2c3340;margin:12px 0\">
    <div id=\"final\" class=\"small\"></div>
  </aside>
</div>
<script>
const D={payload};
const M={meta};
const F=D.frames, floors=D.floors, events=D.events;
const byIndex=new Map(floors.map(f=>[f[0],f]));
const canvas=document.getElementById('chart'), ctx=canvas.getContext('2d');
const scrub=document.getElementById('scrub'), play=document.getElementById('play'), speed=document.getElementById('speed');
let idx=0, playing=false, lastWall=0, replayClock=F.length?F[0][0]:0;
scrub.max=Math.max(0,F.length-1);
document.getElementById('source').textContent=M.chart+'  ←  '+M.checkpoint;
const r=D.result;
const fatal=r.clear?'clear=True':`clear=False fatal=${{r.fatal_reason}} floor=${{r.fatal_floor}} t=${{r.fatal_time_s.toFixed(3)}}s survived=${{r.survived_targets}}/${{D.segment.targets}}`;
document.getElementById('final').textContent=`FINAL: H=${{r.hits}}/${{D.segment.targets}}  X=${{r.xacc.toFixed(2)}}%  PP=${{r.pp.toFixed(1)}}%  MAE=${{r.mae===null?'—':r.mae.toFixed(2)+'ms'}}  early=${{r.early}}  overload=${{r.overload}}  keydowns=${{r.keydowns}} | ADOFAI-style ${{fatal}}`;
function setBar(pos,neg,v){{ const a=Math.min(1,Math.abs(v))*50; pos.style.width=(v>=0?a:0)+'%'; neg.style.width=(v<0?a:0)+'%'; }}
function eventText(e){{ const delta=e.target_t===null?'':` Δ=${{((e.t-e.target_t)*1000).toFixed(1)}}ms`; return `${{e.t.toFixed(3)}}s ${{e.key}} floor=${{e.floor??'—'}} ${{e.judgement??'NO_TARGET'}}${{delta}}`; }}
function draw(){{
  if(!F.length)return;
  const f=F[idx], current=byIndex.get(f[1])||floors[0];
  const W=canvas.width,H=canvas.height, scale=48, cx=W*0.5, cy=H*0.5;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0d1016'; ctx.fillRect(0,0,W,H);
  const wx=x=>cx+(x-current[1])*scale, wy=y=>cy-(y-current[2])*scale;
  ctx.lineWidth=3; ctx.strokeStyle='#333b49'; ctx.beginPath();
  let started=false;
  for(const p of floors){{ if(Math.abs(p[0]-f[1])>22)continue; const x=wx(p[1]),y=wy(p[2]); if(!started){{ctx.moveTo(x,y);started=true}}else ctx.lineTo(x,y); }} ctx.stroke();
  for(const p of floors){{ if(Math.abs(p[0]-f[1])>22)continue; const x=wx(p[1]),y=wy(p[2]); ctx.beginPath(); ctx.arc(x,y,p[3]?5:8,0,Math.PI*2); ctx.fillStyle=p[4]?'#7dcfff':'#565f70'; ctx.fill(); if(p[3]){{ctx.strokeStyle='#c0caf5';ctx.lineWidth=2;ctx.stroke();}} }}
  const sx=wx(current[1]),sy=wy(current[2]); ctx.beginPath();ctx.arc(sx,sy,13,0,Math.PI*2);ctx.fillStyle='#f7768e';ctx.fill();
  ctx.beginPath();ctx.arc(sx+f[2]*scale,sy-f[3]*scale,11,0,Math.PI*2);ctx.fillStyle='#e0af68';ctx.fill();
  ctx.fillStyle='#c0caf5';ctx.font='16px ui-monospace,monospace';ctx.fillText(`t=${{f[0].toFixed(2)}}s  floor=${{f[1]}}`,16,26);
  document.getElementById('clock').textContent=f[0].toFixed(2)+'s';
  document.getElementById('hits').textContent=f[15]; document.getElementById('misses').textContent=f[16]; document.getElementById('early').textContent=f[17]; document.getElementById('overload').textContent=f[14].toFixed(3);
  setBar(document.getElementById('laP'),document.getElementById('laN'),f[4]); setBar(document.getElementById('raP'),document.getElementById('raN'),f[5]);
  document.getElementById('lk').classList.toggle('on',!!f[8]); document.getElementById('rk').classList.toggle('on',!!f[9]);
  document.getElementById('motorText').textContent=`cmd L=${{f[4].toFixed(3)}} R=${{f[5].toFixed(3)}} | pos=${{f[6].toFixed(2)}}/${{f[7].toFixed(2)}}mm | activation=${{f[10].toFixed(2)}}/${{f[11].toFixed(2)}} | fatigue=${{f[12].toFixed(3)}}/${{f[13].toFixed(3)}}`;
  const visible=events.filter(e=>e.t<=f[0]+1e-9).slice(-12); const box=document.getElementById('events'); box.innerHTML=''; visible.forEach((e,j)=>{{const d=document.createElement('div');d.textContent=eventText(e);if(j===visible.length-1)d.className='event-current';box.appendChild(d);}});
  scrub.value=idx;
}}
function seekByTime(t){{ let lo=0,hi=F.length-1; while(lo<hi){{const m=(lo+hi+1)>>1;if(F[m][0]<=t)lo=m;else hi=m-1;}} idx=lo; replayClock=F[idx][0]; draw(); }}
function tick(now){{ if(!playing)return; if(!lastWall)lastWall=now; const dt=(now-lastWall)/1000*Number(speed.value);lastWall=now;replayClock+=dt; if(replayClock>=F[F.length-1][0]){{idx=F.length-1;playing=false;play.textContent='▶ Play';draw();return;}} seekByTime(replayClock); requestAnimationFrame(tick); }}
play.addEventListener('click',()=>{{playing=!playing;play.textContent=playing?'⏸ Pause':'▶ Play';lastWall=0;if(playing){{if(idx>=F.length-1){{idx=0;replayClock=F[0][0];}}requestAnimationFrame(tick);}}}});
scrub.addEventListener('input',()=>{{idx=Number(scrub.value);replayClock=F[idx][0];lastWall=0;draw();}});
draw();
</script>
</main></body></html>"""


def _load_visualizer_backend(payload: dict, *, device: torch.device):
    input_dim = int(payload.get("input_dim", -1))
    if input_dim == HUD_REAL_CHART_INPUT_DIM:
        return (
            v070_eval._load_hud_model(payload, device=device),
            ReplayHudRealChartEnv,
            encode_hud_real_chart_observation,
            f"HUD {HUD_REAL_CHART_INPUT_DIM}D",
        )
    if input_dim == REAL_CHART_INPUT_DIM:
        return (
            evaluator._load_model(payload, device=device),
            ReplayRealChartEnv,
            encode_real_chart_observation,
            f"legacy {REAL_CHART_INPUT_DIM}D",
        )
    raise SystemExit(
        f"checkpoint input dimension {input_dim} is unsupported by visualizer "
        f"({REAL_CHART_INPUT_DIM}D legacy / {HUD_REAL_CHART_INPUT_DIM}D HUD)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained browser replay of a DMDOD real-chart checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--no-open", action="store_true", help="write HTML without opening the default browser")
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument("--same-hand", dest="same_hand_override", action="store_const", const=True, default=None)
    hand_group.add_argument("--cross-hand", dest="same_hand_override", action="store_const", const=False)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")
    model, env_cls, encoder, observation_label = _load_visualizer_backend(payload, device=device)
    same_hand, control_dt_s, hand_source, control_source = evaluator._resolve_eval_config(
        payload,
        same_hand_override=args.same_hand_override,
        control_dt_override=args.control_dt,
    )

    compiled = load_compiled_adofai(args.chart)
    start_s, end_s = evaluator._resolve_range(compiled.duration_s, args.start, args.end)
    segment = build_playable_segment(compiled, start_s=start_s, end_s=end_s)
    if not segment.targets:
        raise SystemExit("visualization segment contains no playable targets")

    print("=== DMDOD / Real Chart Replay Visualizer ===")
    print(f"checkpoint={checkpoint_path} chart={args.chart}")
    print(
        f"segment={start_s:g}..{end_s:g}s targets={len(segment.targets)} "
        f"observation={observation_label} "
        f"body={'same-hand' if same_hand else 'cross-hand'}({hand_source}) "
        f"control={control_dt_s * 1000.0:.1f}ms({control_source}) training=DISABLED"
    )
    data, result, env = _collect_replay(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
        env_cls=env_cls,
        encoder=encoder,
    )
    print(v054._format_eval("replay eval", result))
    print(format_fatal_summary(env))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _html_document(data, checkpoint=str(checkpoint_path), chart=args.chart),
        encoding="utf-8",
    )
    print(f"replay-html={output.resolve()} frames={len(data['frames'])} events={len(data['events'])}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
