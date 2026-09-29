from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_real_chart as evaluator
import train_real_chart_v053 as v053
import visualize_real_chart as visualizer
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


DEFAULT_OUTPUT = "artifacts/teacher_student_motion.json"


def _target_fields(segment, t: float) -> dict:
    target_index = None
    target_floor = None
    target_time = None
    for i, target in enumerate(segment.targets):
        if float(target.episode_time_s) >= t - 1e-12:
            target_index = i
            target_floor = int(target.floor_index)
            target_time = float(target.episode_time_s)
            break
    return {
        "next_target_index": target_index,
        "next_target_floor": target_floor,
        "next_target_time_s": target_time,
        "time_to_target_ms": None if target_time is None else round((target_time - t) * 1000.0, 3),
    }


def _collect_teacher(segment, *, env_cls, same_hand: bool, control_dt_s: float, lead_s: float) -> dict:
    env = env_cls(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    frames = [visualizer._record_frame(env, observation, 0.0, 0.0)]
    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        action = v053._teacher_action(env, observation, lead_s)
        step = env.step(action)
        observation = step.observation
        frames.append(visualizer._record_frame(env, observation, action.left, action.right))
        if step.done:
            break
    else:
        raise RuntimeError("teacher episode exceeded step budget")
    return {"frames": frames, "events": getattr(env, "replay_events", []), "stats": env.stats}


def _first_miss_time(data: dict) -> float | None:
    frames = data["frames"]
    if not frames:
        return None
    prev = int(frames[0][16])
    for frame in frames[1:]:
        miss = int(frame[16])
        if miss > prev:
            return float(frame[0])
        prev = miss
    return None


def _frame_dict(frame: list, segment) -> dict:
    t = float(frame[0])
    row = {
        "t_s": t,
        "floor": int(frame[1]),
        "action_left": float(frame[4]),
        "action_right": float(frame[5]),
        "left_pos_mm": float(frame[6]),
        "right_pos_mm": float(frame[7]),
        "left_pressed": bool(frame[8]),
        "right_pressed": bool(frame[9]),
        "left_activation": float(frame[10]),
        "right_activation": float(frame[11]),
        "left_fatigue": float(frame[12]),
        "right_fatigue": float(frame[13]),
        "overload": float(frame[14]),
        "hits": int(frame[15]),
        "misses": int(frame[16]),
        "too_early": int(frame[17]),
    }
    row.update(_target_fields(segment, t))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare privileged teacher and student motion around the student's first miss.")
    parser.add_argument("checkpoint")
    parser.add_argument("chart")
    parser.add_argument("--before", type=float, default=1.0)
    parser.add_argument("--after", type=float, default=0.5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--control-dt", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")

    model, env_cls, encoder, observation_label = visualizer._load_visualizer_backend(payload, device=device)
    same_hand, control_dt_s, _, _ = evaluator._resolve_eval_config(
        payload, same_hand_override=None, control_dt_override=args.control_dt
    )
    compiled = load_compiled_adofai(args.chart)
    segment = build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)
    if not segment.targets:
        raise SystemExit("chart contains no playable targets")

    student_data, _student_result, _student_env = visualizer._collect_replay(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
        env_cls=env_cls,
        encoder=encoder,
    )
    miss_t = _first_miss_time(student_data)
    if miss_t is None:
        center = float(student_data["frames"][-1][0])
    else:
        center = miss_t

    calibration = calibrate_single_press_lead(control_dt_s=control_dt_s, same_hand=same_hand)
    teacher_data = _collect_teacher(
        segment,
        env_cls=env_cls,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        lead_s=calibration.lead_s,
    )

    lo = center - max(0.0, args.before)
    hi = center + max(0.0, args.after)
    student_frames = [f for f in student_data["frames"] if lo <= float(f[0]) <= hi]
    teacher_frames = [f for f in teacher_data["frames"] if lo <= float(f[0]) <= hi]
    teacher_by_tick = {round(float(f[0]) / control_dt_s): f for f in teacher_frames}

    rows = []
    for sf in student_frames:
        tick = round(float(sf[0]) / control_dt_s)
        tf = teacher_by_tick.get(tick)
        s = _frame_dict(sf, segment)
        t = None if tf is None else _frame_dict(tf, segment)
        row = {
            "t_s": s["t_s"],
            "dt_from_student_miss_ms": None if miss_t is None else round((s["t_s"] - miss_t) * 1000.0, 3),
            "target": {
                "floor": s["next_target_floor"],
                "time_s": s["next_target_time_s"],
                "time_to_target_ms": s["time_to_target_ms"],
            },
            "student": s,
            "teacher": t,
        }
        if t is not None:
            row["delta"] = {
                "action_left": round(s["action_left"] - t["action_left"], 4),
                "action_right": round(s["action_right"] - t["action_right"], 4),
                "left_pos_mm": round(s["left_pos_mm"] - t["left_pos_mm"], 4),
                "right_pos_mm": round(s["right_pos_mm"] - t["right_pos_mm"], 4),
            }
        rows.append(row)

    output = {
        "meta": {
            "checkpoint": str(checkpoint_path),
            "chart": args.chart,
            "observation": observation_label,
            "same_hand": same_hand,
            "control_dt_ms": control_dt_s * 1000.0,
            "teacher_lead_ms": calibration.lead_s * 1000.0,
            "student_first_miss_time_s": miss_t,
            "window_before_s": args.before,
            "window_after_s": args.after,
        },
        "student_result": student_data["result"],
        "teacher_result": {
            "hits": int(teacher_data["stats"].hits),
            "misses": int(teacher_data["stats"].misses),
            "xacc": float(teacher_data["stats"].x_accuracy_percent),
            "early": int(teacher_data["stats"].too_early_presses),
            "overload": bool(teacher_data["stats"].overloaded),
        },
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"student_first_miss={miss_t}")
    print(f"teacher_lead={calibration.lead_s * 1000.0:.1f}ms")
    print(f"rows={len(rows)}")
    print(f"output={out.resolve()}")


if __name__ == "__main__":
    main()
