#!/usr/bin/env python3
"""MCCXVI constrained-key research preflight.

Usage inside the DMDOD checkout (authoritative DMDOD timing):
    python scripts/mccxvi_4k_benchmark.py --zip "MCCXVI - JinoBeats.zip" --out reports/mccxvi-preflight.json --timing official

Usage as a portable exploratory report (timing port is *not* authoritative):
    python scripts/mccxvi_4k_benchmark.py --zip "MCCXVI - JinoBeats.zip" --out mccxvi-preflight.json --timing approximate

Does not read audio or reproduce chart contents; stores only aggregated statistics.
Window calculations are necessary bounds for exact K physical keys, not
sufficient proof of motor reachability/release. This script does not execute
Motor or implement stock gameplay key-limit semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile


PHASES = (
    (0, 235, None, "未表示"),
    (235, 662, 4, "4K"),
    (662, 744, 2, "2K"),
    (744, 970, None, "非表示"),
    (970, 1089, 2, "2K"),
    (1089, 1276, 4, "4K"),
    (1276, 1420, 2, "2K"),
    (1420, 1646, 4, "4K"),
    (1646, 1776, 2, "2K"),
    (1776, 1837, 4, "4K"),
)
COOLDOWNS_MS = (0, 40, 50, 60, 80, 100, 120, 150, 180)


@dataclass(frozen=True)
class Event:
    floor: int
    due_s: float
    contacts: int = 1


def extract_chart(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            info for info in archive.infolist()
            if info.filename.endswith("/level.adofai") or info.filename == "level.adofai"
        ]
        if len(matches) != 1:
            raise ValueError("ZIP must contain exactly one level.adofai")
        if matches[0].file_size > 5 * 1024 * 1024:
            raise ValueError("chart file too large")
        data = archive.read(matches[0])
    raw = json.loads(data.decode("utf-8-sig"), strict=False)
    if not isinstance(raw.get("angleData"), list):
        raise ValueError("angleData missing")
    return data, raw


def verify_limit_markers(raw):
    actions = raw.get("actions", [])
    observed = []
    for action in actions:
        if (
            action.get("eventType") != "MoveDecorations"
            or action.get("tag") != "limit"
            or action.get("duration") != 0
        ):
            continue
        floor = action.get("floor")
        if floor not in [phase[0] for phase in PHASES[1:]]:
            continue
        label = action.get("decorationImage")
        visible = action.get("visible")
        # Display visibility is not verified gameplay enforcement.
        if label not in ("2K.png", "4k.png") and visible is False and floor == 744:
            label = "hidden"
        elif not label and floor == 235:
            label = "4k.png (from AddDecoration at 235)"
        observed.append({"floor": floor, "image": label, "visible": visible})
    expected = [235, 662, 744, 970, 1089, 1276, 1420, 1646, 1776]
    ids = [row["floor"] for row in observed]
    if sorted(ids) != expected:
        raise ValueError(f"limit markers unexpectedly changed: {ids}")
    return observed


def load_events(raw, timing: str, zip_path: Path):
    if timing == "official":
        from dmdod.adofai.chart import parse_adofai_bytes
        from dmdod.adofai.timing import compile_adofai
        from dmdod.adofai.playable import build_playable_segment
        from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher

        data, _ = extract_chart(zip_path)
        chart = parse_adofai_bytes(data, source_path=str(zip_path) + "!level.adofai")
        compiled = compile_adofai(chart)
        segment = build_playable_segment(compiled)
        oracle = AutoplayTimingTeacher.from_segment(segment)
        rows = [Event(e.floor_indices[0], e.episode_time_s, e.chord_size) for e in oracle.events]
        return rows, {
            "timing_authority": "dmdod_stock_compile",
            "compiled_duration_s": segment.duration_s,
            "playable_target_count": len(segment.targets),
            "oracle_event_count": len(rows),
            "hold_count": len(segment.holds),
            "chord_count": sum(e.contacts > 1 for e in rows),
        }
    # Independent port for exploration only; not proof of game/DMDOD timing.
    try:
        from mccxvi_stock_timing_approx import compile_times
    except ImportError as error:
        raise RuntimeError(
            "exploratory timing helper missing; use --timing official inside DMDOD"
        ) from error

    rows = compile_times(raw)
    pitch = float(raw.get("settings", {}).get("pitch", 100)) / 100
    data = [Event(row["floor"], row["time_s"] / pitch) for row in rows[1:] if not row["midspin"]]
    return data, {
        "timing_authority": "independent_exploratory_port_NOT_OFFICIAL",
        "compiled_duration_s": rows[-1]["end_s"] / pitch,
        "playable_target_count": len(data),
        "oracle_event_count": None,
        "hold_count": sum(row["hold"] > 0 for row in rows),
        "chord_count": None,
    }


def demand_window(events, tau_s):
    """Maximum press contacts in a half-open tau window.

    A key used at t may be reused at exactly t+tau.
    Necessary simultaneous-axis lower bound; unit events in a fixed
    pool also have an interval-coloring interpretation.
    """
    if not events:
        return 0, None
    active = deque()
    total = 0
    maximum = 0
    offender = None
    for event in events:
        while active and event.due_s - active[0].due_s >= tau_s - 1e-10:
            total -= active.popleft().contacts
        active.append(event)
        total += event.contacts
        if total > maximum:
            maximum = total
            offender = {
                "first_floor": active[0].floor,
                "last_floor": event.floor,
                "span_ms": round(1000 * (event.due_s - active[0].due_s), 6),
                "first_time_s": round(active[0].due_s, 6),
                "last_time_s": round(event.due_s, 6),
                "required_distinct_keys": maximum,
            }
    return maximum, offender


def min_k_spacing(events, k):
    """Necessary maximum feasible cooldown in this phase (unit events).

    This does not check physical dynamics, holds, or phase boundaries.
    """
    if any(event.contacts != 1 for event in events) or len(events) <= k:
        return None
    delta, i = min(
        (events[i + k].due_s - events[i].due_s, i)
        for i in range(len(events) - k)
    )
    return {
        "critical_key_reuse_upper_bound_ms": round(delta * 1000, 6),
        "witness_first_floor": events[i].floor,
        "witness_last_floor": events[i + k].floor,
    }


def compute_report(zip_path: Path, timing="official"):
    data, raw = extract_chart(zip_path)
    markers = verify_limit_markers(raw)
    events, compile_meta = load_events(raw, timing, zip_path)
    if len(raw["angleData"]) != 1836:
        raise ValueError("this benchmark is pinned to MCCXVI angle count 1836")
    if not events or any(
        second.due_s + 1e-9 < first.due_s
        for first, second in zip(events, events[1:])
    ):
        raise ValueError("compiled onsets are empty or unsorted")

    phases = []
    for start, end, key_limit, label in PHASES:
        phase = [event for event in events if start <= event.floor < end]
        result = {
            "floor_start_inclusive": start,
            "floor_end_exclusive": end,
            "display_label": label,
            "modeled_key_limit": key_limit,
            "note_events": len(phase),
            "first_due_s": round(phase[0].due_s, 6) if phase else None,
            "last_due_s": round(phase[-1].due_s, 6) if phase else None,
            "critical": min_k_spacing(phase, key_limit) if key_limit else None,
        }
        if key_limit:
            trials = []
            for ms in COOLDOWNS_MS:
                if ms == 0:
                    continue
                required, witness = demand_window(phase, ms / 1000)
                trials.append({
                    "cooldown_ms": ms,
                    "minimum_keys_demanded": required,
                    "violates_key_limit": required > key_limit,
                    "witness": witness if required > key_limit else None,
                })
            result["cooldown_tests"] = trials
        phases.append(result)
    return {
        "format": "dmdod-mccxvi-keylimit-preflight-v1",
        "chart": "MCCXVI - JinoBeats",
        "sha256_level_adofai": hashlib.sha256(data).hexdigest(),
        "source": "user-supplied ZIP; no chart/audio included in this report",
        "angle_count": len(raw["angleData"]),
        "initial_bpm": raw.get("settings", {}).get("bpm"),
        "action_counts": dict(sorted(
            Counter(action["eventType"] for action in raw.get("actions", [])).items()
        )),
        "limit_markers": markers,
        "compiled": compile_meta,
        "phases": phases,
        "scope": {
            "key_limit_source": "visual decoration cues, not a game-enforced rule",
            "limit_hidden_intervals": "not assumed to be 4K",
            "pool_choice": "not fixed; static phase-local necessary bounds",
            "motor_executed": False,
            "human_log_trained_on": False,
            "source_chart_redistributed": False,
            "notes": (
                "Per-phase feasibility ignores reservations/recovery across "
                "phase boundaries; it does not prove physical reachability."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timing", choices=("official", "approximate"), default="official")
    args = parser.parse_args()
    result = compute_report(args.zip, args.timing)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({
        "sha256_level_adofai": result["sha256_level_adofai"],
        "timing_authority": result["compiled"]["timing_authority"],
        "playable_targets": result["compiled"]["playable_target_count"],
        "phases": [
            {"start": p["floor_start_inclusive"], "end": p["floor_end_exclusive"],
             "limit": p["modeled_key_limit"], "critical": p["critical"]}
            for p in result["phases"] if p["modeled_key_limit"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
