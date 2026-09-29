from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

import eval_real_chart as evaluator
import visualize_real_chart as visualizer
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai


DEFAULT_JSON = "artifacts/fatal_motion.json"
DEFAULT_CSV = "artifacts/fatal_motion.csv"


def _fatal_from_frames(data: dict) -> dict | None:
    frames = data["frames"]
    if not frames:
        return None

    prev_miss = int(frames[0][16])
    prev_overload = float(frames[0][14])
    overload_latched = False

    for i, frame in enumerate(frames):
        miss = int(frame[16])
        overload = float(frame[14])
        if miss > prev_miss:
            return {
                "index": i,
                "reason": "miss",
                "time_s": float(frame[0]),
                "floor": int(frame[1]),
                "survived_targets": int(frame[15]),
            }
        if not overload_latched and overload >= 6.0 and prev_overload < 6.0:
            return {
                "index": i,
                "reason": "overload",
                "time_s": float(frame[0]),
                "floor": int(frame[1]),
                "survived_targets": int(frame[15]),
            }
        prev_miss = miss
        prev_overload = overload
        overload_latched = overload_latched or overload >= 6.0
    return None


def _window_rows(data: dict, *, fatal: dict | None, before_s: float, after_s: float) -> list[dict]:
    frames = data["frames"]
    events = data["events"]
    if not frames:
        return []

    center = float(fatal["time_s"]) if fatal is not None else float(frames[-1][0])
    lo = center - before_s
    hi = center + after_s

    event_i = 0
    latest_event = None
    rows: list[dict] = []
    for frame in frames:
        t = float(frame[0])
        while event_i < len(events) and float(events[event_i]["t"]) <= t + 1e-12:
            latest_event = events[event_i]
            event_i += 1
        if t < lo or t > hi:
            continue

        row = {
            "t_s": t,
            "dt_from_fatal_ms": None if fatal is None else round((t - center) * 1000.0, 3),
            "floor": int(frame[1]),
            "orbit_x": float(frame[2]),
            "orbit_y": float(frame[3]),
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
            "latest_keydown_t_s": None,
            "latest_key": None,
            "latest_keydown_floor": None,
            "latest_judgement": None,
            "latest_timing_error_ms": None,
        }
        if latest_event is not None:
            row["latest_keydown_t_s"] = float(latest_event["t"])
            row["latest_key"] = latest_event.get("key")
            row["latest_keydown_floor"] = latest_event.get("floor")
            row["latest_judgement"] = latest_event.get("judgement")
            target_t = latest_event.get("target_t")
            if target_t is not None:
                row["latest_timing_error_ms"] = round(
                    (float(latest_event["t"]) - float(target_t)) * 1000.0,
                    3,
                )
        rows.append(row)
    return rows


def _summary(data: dict, fatal: dict | None, rows: list[dict]) -> dict:
    result = data["result"]
    summary = {
        "result": result,
        "segment": data["segment"],
        "fatal": fatal,
        "window_frames": len(rows),
    }
    if not rows:
        return summary

    left_drive = sum(abs(float(r["action_left"])) for r in rows) / len(rows)
    right_drive = sum(abs(float(r["action_right"])) for r in rows) / len(rows)
    left_motion = max(float(r["left_pos_mm"]) for r in rows) - min(float(r["left_pos_mm"]) for r in rows)
    right_motion = max(float(r["right_pos_mm"]) for r in rows) - min(float(r["right_pos_mm"]) for r in rows)
    summary["window"] = {
        "mean_abs_action_left": round(left_drive, 6),
        "mean_abs_action_right": round(right_drive, 6),
        "left_position_span_mm": round(left_motion, 6),
        "right_position_span_mm": round(right_motion, 6),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a compact fatal-window motion log for DMDOD diagnosis."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--before", type=float, default=1.5, help="seconds before first fatal event")
    parser.add_argument("--after", type=float, default=0.5, help="seconds after first fatal event")
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--control-dt", type=float, default=None)
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument("--same-hand", dest="same_hand_override", action="store_const", const=True, default=None)
    hand_group.add_argument("--cross-hand", dest="same_hand_override", action="store_const", const=False)
    args = parser.parse_args()

    if args.before < 0 or args.after < 0:
        raise SystemExit("--before/--after must be non-negative")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")

    model, env_cls, encoder, observation_label = visualizer._load_visualizer_backend(
        payload, device=device
    )
    same_hand, control_dt_s, _, _ = evaluator._resolve_eval_config(
        payload,
        same_hand_override=args.same_hand_override,
        control_dt_override=args.control_dt,
    )

    compiled = load_compiled_adofai(args.chart)
    start_s, end_s = evaluator._resolve_range(compiled.duration_s, args.start, args.end)
    segment = build_playable_segment(compiled, start_s=start_s, end_s=end_s)
    if not segment.targets:
        raise SystemExit("diagnostic segment contains no playable targets")

    data, result = visualizer._collect_replay(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
        env_cls=env_cls,
        encoder=encoder,
    )

    fatal = _fatal_from_frames(data)
    rows = _window_rows(data, fatal=fatal, before_s=args.before, after_s=args.after)
    summary = _summary(data, fatal, rows)
    output = {
        "meta": {
            "checkpoint": str(checkpoint_path),
            "chart": args.chart,
            "observation": observation_label,
            "same_hand": same_hand,
            "control_dt_ms": control_dt_s * 1000.0,
            "window_before_s": args.before,
            "window_after_s": args.after,
        },
        "summary": summary,
        "rows": rows,
    }

    json_path = Path(args.json)
    csv_path = Path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    if fatal is None:
        fatal_text = "none"
    else:
        fatal_text = (
            f"{fatal['reason']} floor={fatal['floor']} t={fatal['time_s']:.3f}s "
            f"survived={fatal['survived_targets']}/{data['segment']['targets']}"
        )
    print(f"fatal={fatal_text}")
    print(f"observation={observation_label} frames={len(rows)}")
    print(f"json={json_path.resolve()}")
    print(f"csv={csv_path.resolve()}")
    print(result)


if __name__ == "__main__":
    main()
