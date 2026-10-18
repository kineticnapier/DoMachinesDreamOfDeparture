from __future__ import annotations

"""Read-only audit of the fixed 44ms lead controller and physical recovery.

Runs exact saved Free/Human-Proxy allocations without Network inference.
Reports motor position, velocity, activation and recovery at the FIRST +1
command, then the physically observed KeyDown latency. No training or edits.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import audit_fingering_physical_reuse_counterfactual_v011 as fixed
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


BANDS = ("first_use", "reuse_lt_100ms", "reuse_100_200ms",
         "reuse_200_500ms", "reuse_ge_500ms")


def reuse_band(gap_ms: float | None) -> str:
    if gap_ms is None:
        return "first_use"
    if gap_ms < 100:
        return "reuse_lt_100ms"
    if gap_ms < 200:
        return "reuse_100_200ms"
    if gap_ms < 500:
        return "reuse_200_500ms"
    return "reuse_ge_500ms"


def _state(env: CoupledContactRealChartEnv, axis: str) -> dict:
    motor = env.motor.motor
    index = motor.key_names.index(axis)
    physical = motor.body.states[index]
    obs = motor.observe()
    hand = motor.body.left_hand_state if axis.startswith("left_") else motor.body.right_hand_state
    actuation = motor.keyboard.config.actuation_m
    reset = motor.keyboard.config.reset_m
    return {
        "position_mm": round(physical.position_m * 1000, 6),
        "distance_to_actuation_mm": round((actuation - physical.position_m) * 1000, 6),
        "distance_to_reset_mm": round((reset - physical.position_m) * 1000, 6),
        "velocity_mm_s": round(physical.velocity_m_s * 1000, 6),
        "activation": round(physical.activation, 8),
        "finger_fatigue": round(physical.fatigue, 8),
        "hand_fatigue": round(hand.fatigue, 8),
        "hand_coordination": round(hand.coordination, 8),
        "pressed": bool(obs.pressed(axis)),
    }


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def average(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    errors = [float(r["signed_timing_error_ms"]) for r in rows]
    return {
        "n": len(rows),
        "late": sum(v > 0 for v in errors),
        "early": sum(v < 0 for v in errors),
        "mean_signed_error_ms": round(mean(errors), 3),
        "mean_absolute_error_ms": round(mean(abs(v) for v in errors), 3),
        "median_signed_error_ms": round(median(errors), 3),
        "mean_command_lead_ms": average("command_lead_ms"),
        "mean_command_to_keydown_ms": average("command_to_keydown_ms"),
        "mean_decision_to_command_ms": average("decision_to_command_ms"),
        "mean_position_at_command_mm": average("position_at_command_mm"),
        "mean_velocity_at_command_mm_s": average("velocity_at_command_mm_s"),
        "mean_activation_at_command": average("activation_at_command"),
        "mean_finger_fatigue_at_command": average("finger_fatigue_at_command"),
        "mean_hand_fatigue_at_command": average("hand_fatigue_at_command"),
        "mean_previous_keyup_to_command_ms": average("previous_keyup_to_command_ms"),
        "previous_keyup_observed": sum(r["previous_keyup_to_command_ms"] is not None for r in rows),
        "moving_toward_actuation_at_command": sum(r["velocity_at_command_mm_s"] > 0 for r in rows),
        "pressed_at_command": sum(r["pressed_at_command"] for r in rows),
    }


def run_policy(segment, payload: dict, name: str) -> dict:
    rows = fixed._events(payload, name)
    event_by_index = {
        event.event_index: event
        for event in AutoplayTimingTeacher.from_segment(segment).events
    }
    saved_policy = payload["policies"][name]
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS
    )
    env.reset()
    committed = {}
    selected = 0
    case_records = {}
    actual_downs = []
    physical_downs = []
    latest_down = {}
    latest_up = {}
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    for _ in range(maximum):
        now = env.privileged_episode_time_s()
        committed = {
            index: item for index, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[o] for o in event_by_index[index].target_ordinals)
        }
        while selected < len(rows) and float(rows[selected]["decision_s"]) <= now + 1e-7:
            row = rows[selected]
            selected += 1
            event = event_by_index[row["event_index"]]
            allocation = viewer.parse_allocation(row["chosen"])
            if len(allocation) != 1:
                raise ValueError("only one physical axis per chart event is supported")
            axis = allocation[0].axis
            case_records[row["event_index"]] = {
                "event_index": row["event_index"], "axis": axis,
                "due_ms": float(event.episode_time_s) * 1000,
                "decision_ms": now * 1000,
                "saved_error_ms": row.get("timing_error_ms"),
                "outcome": row.get("outcome"),
                "state_at_decision": _state(env, axis),
            }
            committed[row["event_index"]] = teacher.PlanItem(
                row["event_index"], event.episode_time_s, allocation
            )
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        for index, item in committed.items():
            record = case_records[index]
            if "first_positive_command_ms" in record:
                continue
            axis = record["axis"]
            if action.commands[teacher.AXES.index(axis)] > 0:
                record["first_positive_command_ms"] = now * 1000
                record["state_at_command"] = _state(env, axis)
                record["previous_keydown_before_command_ms"] = latest_down.get(axis)
                record["previous_keyup_before_command_ms"] = latest_up.get(axis)
        step = env.step(action)
        # This raw event stream records all physical KeyDown/KeyUp transitions
        # at 1ms resolution, before the 12-axis -> 16-contact mapping.
        raw = env.motor.motor.last_transition.evaluator_events
        for physical in raw:
            ms = float(physical.time_s) * 1000
            if physical.event is KeyEvent.DOWN:
                physical_downs.append((ms, physical.key))
                latest_down[physical.key] = ms
            elif physical.event is KeyEvent.UP:
                latest_up[physical.key] = ms
        for logical in env.last_events:
            if logical.event is KeyEvent.DOWN:
                actual_downs.append((logical.time_s * 1000, logical.key))
        if step.done:
            break
    else:
        raise RuntimeError(f"{name}: fixed replay exceeded frame budget")
    if selected != len(rows):
        raise RuntimeError(f"{name}: saved decisions not fully replayed: {selected}/{len(rows)}")
    verification = fixed._verify_run(env, saved_policy, actual_downs, name)

    used_down_indices = set()
    cases = []
    for row in rows:
        rec = case_records[row["event_index"]]
        if rec["outcome"] != "Hit" or rec["saved_error_ms"] is None:
            continue
        if "first_positive_command_ms" not in rec:
            raise RuntimeError(f"{name} event {row['event_index']} had no positive command")
        expected_ms = rec["due_ms"] + float(rec["saved_error_ms"])
        choices = [(abs(down_ms - expected_ms), i, down_ms)
                   for i, (down_ms, down_axis) in enumerate(physical_downs)
                   if down_axis == rec["axis"] and i not in used_down_indices]
        if not choices:
            raise RuntimeError(f"{name} event {row['event_index']}: physical KeyDown missing")
        distance, best_index, observed_ms = min(choices)
        if distance > .005:
            raise RuntimeError(
                f"{name} event {row['event_index']}: physical KeyDown pairing mismatch "
                f"{distance:.3f}ms"
            )
        used_down_indices.add(best_index)
        onset = float(rec["first_positive_command_ms"])
        at_command = rec["state_at_command"]
        previous_down = rec["previous_keydown_before_command_ms"]
        previous_up = rec["previous_keyup_before_command_ms"]
        gap = observed_ms - previous_down if previous_down is not None else None
        # The last UP is meaningful only if it follows the last prior DOWN.
        reset_at = (previous_up if previous_up is not None and previous_down is not None
                    and previous_up >= previous_down else None)
        cases.append({
            "event_index": rec["event_index"], "axis": rec["axis"],
            "reuse_band": reuse_band(gap),
            "same_axis_keydown_gap_ms": None if gap is None else round(gap, 3),
            "signed_timing_error_ms": round(observed_ms - rec["due_ms"], 3),
            "decision_ms": round(rec["decision_ms"], 3),
            "first_positive_command_ms": round(onset, 3),
            "actual_keydown_ms": round(observed_ms, 3),
            "decision_to_command_ms": round(onset - rec["decision_ms"], 3),
            "command_lead_ms": round(rec["due_ms"] - onset, 3),
            "command_to_keydown_ms": round(observed_ms - onset, 3),
            "previous_keyup_to_command_ms":
                None if reset_at is None else round(onset - reset_at, 3),
            "previous_keydown_to_keyup_ms":
                None if reset_at is None else round(reset_at - previous_down, 3),
            "position_at_command_mm": at_command["position_mm"],
            "distance_to_actuation_at_command_mm": at_command["distance_to_actuation_mm"],
            "velocity_at_command_mm_s": at_command["velocity_mm_s"],
            "activation_at_command": at_command["activation"],
            "finger_fatigue_at_command": at_command["finger_fatigue"],
            "hand_fatigue_at_command": at_command["hand_fatigue"],
            "hand_coordination_at_command": at_command["hand_coordination"],
            "pressed_at_command": at_command["pressed"],
            "state_at_decision": rec["state_at_decision"],
        })
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["reuse_band"]].append(case)
    return {
        "fixed_replay_verification": verification,
        "decision_count": len(rows),
        "paired_hit_count": len(cases),
        "overall": _stats(cases),
        "by_reuse_interval": {key: _stats(grouped[key]) for key in BANDS},
        "events": cases,
    }


def analyze(payload: dict, chart: Path) -> dict:
    if chart.name != payload.get("chart"):
        raise ValueError("chart filename differs from replay")
    if viewer.file_sha256(chart) != payload.get("chart_sha256"):
        raise ValueError("chart SHA256 differs from replay")
    if not all(name in payload.get("policies", {}) for name in ("free", "human_proxy")):
        raise ValueError("Free and Human-Proxy replays required")
    compiled = compile_adofai(load_adofai(chart))
    segment = build_playable_segment(compiled)
    result = {
        "format": "dmdod-fingering-v011-motor-onset-recovery",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart.name, "chart_sha256": payload["chart_sha256"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "fixed_lead_ms": teacher.LEAD * 1000,
        "control_step_ms": teacher.CTRL * 1000,
        "physics_step_ms": teacher.PHYS * 1000,
        "policies": {},
        "limitations": [
            "Fixed-policy physical replay; no Network inference or retraining.",
            "Physical profile and action_for controller are unchanged.",
            "Only true physical KeyDown/KeyUp on 12 axes are used for repeat intervals.",
            "Each event is paired to its saved timing error within 0.005ms.",
            "The first +1 action is a command onset, not instantaneous muscle activation.",
            "Recent physical recovery, finger fatigue and hand coordination may be correlated.",
            "This diagnostic is observational; varying lead or force would require a separate controlled experiment.",
        ],
    }
    for name in ("free", "human_proxy"):
        result["policies"][name] = run_policy(segment, payload, name)
    return result


def render_html(result: dict) -> str:
    table = []
    for bucket in BANDS:
        for policy in ("free", "human_proxy"):
            data = result["policies"][policy]["by_reuse_interval"][bucket]
            table.append("<tr>" + "".join(
                f"<td>{escape(str(x) if x is not None else '—')}</td>"
                for x in (policy, bucket, data["n"], data.get("mean_signed_error_ms"),
                          data.get("mean_command_lead_ms"),
                          data.get("mean_command_to_keydown_ms"),
                          data.get("mean_position_at_command_mm"),
                          data.get("mean_velocity_at_command_mm_s"),
                          data.get("mean_activation_at_command"),
                          data.get("mean_previous_keyup_to_command_ms"))
            ) + "</tr>")
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>DMDOD Motor onset audit</title><style>"
        "body{max-width:1280px;margin:24px auto;padding:0 15px;background:#101723;"
        "color:#edf2fa;font:14px/1.65 system-ui}table{border-collapse:collapse;width:100%}"
        "th,td{text-align:left;border-bottom:1px solid #33455a;padding:9px}"
        "th{position:sticky;top:0;background:#1d2b3c}</style>"
        "<h1>44ms制御開始とMotor復帰状態</h1>"
        f"<p>{escape(result['chart'])} / 10ms制御・1ms物理 / 学習なし</p>"
        "<table><thead><tr><th>方式</th><th>再利用間隔</th><th>Hit数</th>"
        "<th>符号付き誤差(ms)</th><th>指令先行(ms)</th>"
        "<th>指令からKeyDown(ms)</th><th>開始位置(mm)</th>"
        "<th>開始速度(mm/s)</th><th>開始活性化</th>"
        "<th>前回KeyUpからの時間(ms)</th></tr></thead><tbody>"
        + "".join(table) +
        "</tbody></table><p>平均値は観測上の関係であり、原因の切り分けには追加の対照実験が必要。</p>"
        "</html>\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart", type=Path, required=True)
    p.add_argument("--replay-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error(f"refusing to overwrite existing directory: {args.output_dir}")
    payload = json.loads(args.replay_json.read_text(encoding="utf-8"))
    report = analyze(payload, args.chart)
    html = render_html(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "motor-onset-audit.json").open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    with (args.output_dir / "motor-onset-audit.html").open("x", encoding="utf-8") as out:
        out.write(html)
    for policy, value in report["policies"].items():
        print(f"\n{policy}: fixed replay={value['fixed_replay_verification']['replay_exact']}, "
              f"paired={value['paired_hit_count']}")
        for bucket in BANDS:
            group = value["by_reuse_interval"][bucket]
            if group["n"]:
                print(f"  {bucket}: n={group['n']} "
                      f"error={group['mean_signed_error_ms']}ms; "
                      f"command->KeyDown={group['mean_command_to_keydown_ms']}ms; "
                      f"position={group['mean_position_at_command_mm']}mm; "
                      f"velocity={group['mean_velocity_at_command_mm_s']}mm/s")
    print("viewer:", args.output_dir / "motor-onset-audit.html")


if __name__ == "__main__":
    main()
