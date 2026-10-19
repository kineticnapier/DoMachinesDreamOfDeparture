from __future__ import annotations

"""Audit paired stock SetSpeed-icon recordings without privileged inputs."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess


FORMAT = "dmdod-stock-setspeed-icon-capture-audit-v0.1"
DEADLINE_S = 3.9761634902576737
ALLOWED_STATES = {
    "DOUBLE_RABBIT_VISIBLE",
    "TILE_VISIBLE_NO_SPEED_ICON",
    "UNKNOWN_OR_OCCLUDED",
}
EXPECTED = {
    "original-8x": "DOUBLE_RABBIT_VISIBLE",
    "counterfactual-1x": "TILE_VISIBLE_NO_SPEED_ICON",
}


def _rate(text: str) -> float:
    value = float(Fraction(text))
    if value <= 0:
        raise ValueError(f"invalid frame rate: {text}")
    return value


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}")
    stream = streams[0]
    average = _rate(stream["avg_frame_rate"])
    nominal = _rate(stream["r_frame_rate"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": float(stream["duration"]),
        "average_fps": average,
        "nominal_fps": nominal,
        "constant_rate_reported": abs(average - nominal) <= 1e-6,
    }


def pixel_difference(left: bytes, right: bytes) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("frames must have the same nonzero byte size")
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {
        "mean_absolute_rgb_difference": sum(differences) / len(differences),
        "max_absolute_rgb_difference": max(differences),
        "fraction_rgb_channels_over_16": sum(x > 16 for x in differences) / len(differences),
    }


def _decode_crop(path: Path, crop: tuple[int, int, int, int], start_s: float,
                 duration_s: float, fps: float) -> list[bytes]:
    x, y, width, height = crop
    if min(x, y) < 0 or not 1 <= width <= 1024 or not 1 <= height <= 1024:
        raise ValueError("invalid crop_xywh")
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-ss", f"{start_s:.9f}",
        "-t", f"{duration_s:.9f}", "-vf", f"crop={width}:{height}:{x}:{y}",
        "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_size = width * height * 3
    if len(result.stdout) % frame_size:
        raise ValueError("ffmpeg returned a partial raw frame")
    frames = [
        result.stdout[offset:offset + frame_size]
        for offset in range(0, len(result.stdout), frame_size)
    ]
    expected = duration_s * fps
    if abs(len(frames) - expected) > 2.1:
        raise ValueError("decoded frame count is inconsistent with reported CFR")
    return frames


def annotation_interval(annotation: dict, capture: dict, fps: float) -> dict:
    state = annotation.get("state")
    if state not in ALLOWED_STATES:
        raise ValueError(f"invalid annotation state: {state}")
    video_s = float(annotation["video_time_s"])
    sync = capture["sync"]
    latency = capture["capture_latency"]
    anchor_video = float(sync["anchor_video_s"])
    anchor_episode = float(sync.get("anchor_episode_s", 0.0))
    sync_uncertainty = float(sync["uncertainty_s"])
    latency_min = float(latency["min_s"])
    latency_max = float(latency["max_s"])
    if min(sync_uncertainty, latency_min, latency_max) < 0 or latency_min > latency_max:
        raise ValueError("invalid sync uncertainty or capture latency")
    center = video_s - anchor_video + anchor_episode
    quantization = 0.5 / fps
    return {
        "state": state,
        "video_time_s": video_s,
        "episode_center_s": center,
        "frame_quantization_half_width_s": quantization,
        "arrival_lower_s": center - sync_uncertainty - quantization + latency_min,
        "arrival_upper_s": center + sync_uncertainty + quantization + latency_max,
        "evidence": annotation.get("evidence"),
        "first_occurrence_confirmed": bool(annotation.get("first_occurrence_confirmed", False)),
    }


def judge(intervals: dict[str, dict], deadline_s: float = DEADLINE_S) -> tuple[str, str]:
    if set(intervals) != set(EXPECTED):
        return "UNVERIFIED", "both counterfactual worlds need valid observations"
    for world, expected in EXPECTED.items():
        if intervals[world]["state"] != expected:
            return "UNVERIFIED", f"{world} lacks required state {expected}"
    guaranteed = max(row["arrival_upper_s"] for row in intervals.values())
    earliest = max(row["arrival_lower_s"] for row in intervals.values())
    if guaranteed <= deadline_s:
        return "OBSERVABLE_BEFORE_DEADLINE", "both state intervals arrive before the deadline"
    if earliest > deadline_s:
        if all(row.get("first_occurrence_confirmed", False) for row in intervals.values()):
            return "TOO_LATE", "exhaustive review puts the first paired distinction after the deadline"
        return "UNVERIFIED", "late annotations are not confirmed as the first occurrences"
    return "UNVERIFIED", "arrival uncertainty overlaps the deadline"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def audit(manifest_path: Path, *, analyze_pixels: bool = True) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deadline = float(manifest.get("deadline_episode_s", DEADLINE_S))
    if abs(deadline - DEADLINE_S) > 1e-9:
        raise ValueError("deadline differs from the fixed Phase-4 value")
    captures = manifest.get("captures", [])
    if {row.get("world") for row in captures} != set(EXPECTED):
        raise ValueError("manifest must contain exactly the 8x and 1x worlds")

    missing = []
    probes = {}
    intervals = {}
    capture_by_world = {row["world"]: row for row in captures}
    for world, capture in capture_by_world.items():
        video_value = capture.get("video_path")
        if not video_value:
            missing.append(f"{world}: video_path")
            continue
        video = _resolve(manifest_path.parent, video_value)
        if not video.is_file():
            missing.append(f"{world}: video file")
            continue
        required_recording = (
            "recording_utc", "game_build", "capture_software", "capture_mode",
            "reported_fps", "resolution", "camera_and_display_conditions",
        )
        for key in required_recording:
            if capture.get("recording", {}).get(key) in (None, ""):
                missing.append(f"{world}: recording.{key}")
        for section, keys in (("sync", ("anchor_video_s", "uncertainty_s")),
                              ("capture_latency", ("min_s", "max_s", "basis"))):
            for key in keys:
                if capture.get(section, {}).get(key) in (None, ""):
                    missing.append(f"{world}: {section}.{key}")
        probe = probe_video(video)
        probes[world] = {"path": str(video), **probe}
        if not probe["constant_rate_reported"]:
            missing.append(f"{world}: constant frame rate")
        reported_fps = capture.get("recording", {}).get("reported_fps")
        if reported_fps is not None and abs(float(reported_fps) - probe["average_fps"]) > 0.01:
            missing.append(f"{world}: reported_fps does not match ffprobe")
        candidates = []
        for annotation in capture.get("annotations", []):
            if all(capture.get("sync", {}).get(k) is not None for k in ("anchor_video_s", "uncertainty_s")) \
                    and all(capture.get("capture_latency", {}).get(k) is not None for k in ("min_s", "max_s")):
                candidates.append(annotation_interval(annotation, capture, probe["average_fps"]))
        expected = EXPECTED[world]
        candidates = [row for row in candidates if row["state"] == expected]
        if candidates:
            intervals[world] = min(candidates, key=lambda row: row["arrival_upper_s"])
        else:
            missing.append(f"{world}: annotation {expected}")

    if len(probes) == 2:
        recording_keys = (
            "game_build", "capture_software", "capture_mode", "reported_fps",
            "resolution", "mods_enabled", "camera_and_display_conditions",
        )
        left = capture_by_world["original-8x"].get("recording", {})
        right = capture_by_world["counterfactual-1x"].get("recording", {})
        for key in recording_keys:
            if left.get(key) != right.get(key):
                missing.append(f"paired recording condition mismatch: {key}")

    pixel_analysis = {"performed": False, "reason": "videos or crop unavailable"}
    crop_value = manifest.get("crop_xywh")
    if analyze_pixels and not missing and isinstance(crop_value, list) and len(crop_value) == 4:
        crop = tuple(int(x) for x in crop_value)
        starts = {}
        decoded = {}
        for world, capture in capture_by_world.items():
            probe = probes[world]
            anchor = float(capture["sync"]["anchor_video_s"])
            start = max(0.0, anchor - 0.05)
            duration = deadline + 0.10
            starts[world] = start
            decoded[world] = _decode_crop(
                Path(probe["path"]), crop, start, duration, probe["average_fps"]
            )
        left_world, right_world = "original-8x", "counterfactual-1x"
        left_probe, right_probe = probes[left_world], probes[right_world]
        rows = []
        for left_index, left_frame in enumerate(decoded[left_world]):
            left_video_s = starts[left_world] + left_index / left_probe["average_fps"]
            episode_s = left_video_s - float(capture_by_world[left_world]["sync"]["anchor_video_s"])
            right_video_s = episode_s + float(capture_by_world[right_world]["sync"]["anchor_video_s"])
            right_index = round((right_video_s - starts[right_world]) * right_probe["average_fps"])
            if 0 <= right_index < len(decoded[right_world]) and episode_s <= deadline:
                rows.append({
                    "episode_s": episode_s,
                    "original_frame_index_in_window": left_index,
                    "counterfactual_frame_index_in_window": right_index,
                    **pixel_difference(left_frame, decoded[right_world][right_index]),
                })
        top = sorted(rows, key=lambda row: row["mean_absolute_rgb_difference"], reverse=True)[:20]
        pixel_analysis = {
            "performed": True,
            "crop_xywh": crop,
            "paired_frames": len(rows),
            "top_differences": top,
            "decision_use": "diagnostic only; pixel threshold does not determine the verdict",
        }
    elif analyze_pixels and crop_value is None:
        missing.append("crop_xywh")

    verdict, reason = judge(intervals, deadline)
    if missing:
        verdict, reason = "UNVERIFIED", "capture evidence or metadata is incomplete"
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "deadline_episode_s": deadline,
        "verdict": verdict,
        "reason": reason,
        "missing_or_invalid_evidence": sorted(set(missing)),
        "video_probes": probes,
        "first_valid_observation_intervals": intervals,
        "pixel_difference_analysis": pixel_analysis,
        "policy_boundary": {
            "allowed_states": sorted(ALLOWED_STATES),
            "no_icon_rule": "valid only when tile and expected icon site are visible",
            "forbidden": "icon enum, multiplier, compiled speed, target time, or evaluator labels as Student input",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-pixel-analysis", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing overwrite")
    result = audit(args.manifest, analyze_pixels=not args.no_pixel_analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("verdict:", result["verdict"])
    if result["missing_or_invalid_evidence"]:
        print("missing:", "; ".join(result["missing_or_invalid_evidence"]))


if __name__ == "__main__":
    main()
