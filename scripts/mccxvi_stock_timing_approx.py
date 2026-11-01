"""Independent port for exploratory MCCXVI timings; DMDOD timing is authoritative.

Derived from DMDOD's stock timing source. Float operations may differ; do not
claim authoritative results until compared against compile_adofai.
"""
from __future__ import annotations

from collections import defaultdict
import math
import struct

PI = 3.1415927410125732
TWOPI = 6.2831854820251465
FIRST = 4.71238899230957
DTR = 0.017453292
RTD = 57.29578


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def mod(x, n):
    r = math.fmod(x, n)
    return r + n if r < 0 else r


def moved(entr, ex, cw):
    return mod((ex - entr) * (1 if cw else -1), TWOPI)


def compile_times(raw):
    angles = raw["angleData"]
    settings = raw.get("settings") or {}
    base = f32(settings.get("bpm", 100))
    tick = int(settings.get("countdownTicks", 4))
    acts = defaultdict(list)
    for a in raw.get("actions", []):
        acts[a.get("floor")].append(a)
    speed = f32(1.0)
    cw = False
    entry = FIRST
    prev_mid = False
    time = 0.0
    pending = None
    rows = []
    for floor in range(len(angles) + 1):
        has_angle = floor < len(angles)
        angle = f32(angles[floor]) if has_angle else 0
        mid = has_angle and angle == f32(999)
        if not has_angle:
            ex = entry + PI
        elif mid:
            ex = float(f32(entry))
        else:
            ex = (-float(angle) + 90.0) * f32(DTR)
        nxt = mod(ex + PI, TWOPI) if floor + 1 < len(angles) + 1 else entry
        state = {
            "floor": floor, "entry": entry, "exit": ex, "mid": mid,
            "prev_mid": prev_mid, "ccw": cw, "planets": 2,
            "speed": speed, "pause": 0, "hold": 0, "hold_action": -1,
        }
        old = speed
        actions = acts[floor]
        speeds = [
            a for a in actions
            if a.get("active", True) and a.get("eventType") == "SetSpeed"
        ]
        for a in actions:
            if not a.get("active", True):
                continue
            if a.get("eventType") == "Twirl":
                cw = not cw
        if speeds:
            offsets = sorted({f32(a.get("angleOffset") or 0) for a in speeds})
            initial_bpm = f32(base * speed)
            crotchet = f32(60 / initial_bpm)
            total = f32(offsets[0] / 180 * crotchet)
            has_mid = False
            floor_angle = (
                f32(360) if abs(entry - ex) < .001
                else f32(moved(entry, ex, not cw) * f32(RTD))
            )
            for oi, offset in enumerate(offsets):
                at = [a for a in speeds if f32(a.get("angleOffset") or 0) == offset]
                last = at[-1]
                next_off = offsets[oi + 1] if oi + 1 < len(offsets) else floor_angle
                if (last.get("speedType") or "").lower() == "bpm":
                    new = f32(last.get("beatsPerMinute") or (base * speed))
                else:
                    new = f32(0)
                    for i, a in enumerate(at):
                        if (a.get("speedType") or "").lower() == "bpm":
                            new = f32(a.get("beatsPerMinute") or (base * speed))
                        else:
                            mult = f32(a.get("bpmMultiplier") or 1.)
                            new = f32(new * mult) if i > 0 else f32(base * speed * mult)
                if new <= 0:
                    new = f32(base * speed)
                seg = f32(60 / new)
                total = f32(total + (next_off - offset) / 180 * seg)
                if offset > 0 and offset <= floor_angle:
                    has_mid = True
                speed = f32(new / base)
            state["speed"] = (
                f32(60 / total * (floor_angle / 180) / base)
                if has_mid and total > 0 else speed
            )
        else:
            state["speed"] = speed
        state["ccw"] = cw
        bpm = f32(base * old)
        pause = 0.0
        for a in actions:
            if not a.get("active", True):
                continue
            typ = a.get("eventType")
            if typ == "SetSpeed" and f32(a.get("angleOffset") or 0) <= 0:
                if (a.get("speedType") or "").lower() == "bpm" and (a.get("beatsPerMinute") or 0) > 0:
                    bpm = f32(a["beatsPerMinute"])
                elif (a.get("bpmMultiplier") or 0) > 0:
                    bpm = f32(bpm * f32(a["bpmMultiplier"]))
            elif typ == "Pause" and (a.get("duration") or 0) > 0:
                pause += a["duration"] * 60 / max(1e-9, bpm)
            elif typ == "Hold":
                d = a.get("duration")
                if d is None or d != int(d):
                    raise ValueError("invalid Hold duration")
                state["hold"] = int(d) if int(d) >= 0 and floor < len(angles) else 0
        state["pause"] = pause
        if pending is not None:
            p = pending
            if p["floor"] == 0:
                m = moved(p["entry"], p["exit"], not p["ccw"])
                seconds = (
                    max(0, tick - 1) * float(f32(60 / base))
                    + m / PI * (60 / base / p["speed"])
                )
            else:
                inv = 3.1415926 * (p["planets"] - 2.) / p["planets"]
                offset = inv * (-1 if p["ccw"] else 1)
                if p["mid"]:
                    offset = 0.
                if p["prev_mid"] and p["planets"] > 2:
                    offset -= (TWOPI + inv) * (-1 if p["ccw"] else 1)
                orbit_start = p["entry"] + offset
                orbit_end = p["exit"] + (offset if p["mid"] else 0)
                m = moved(orbit_start, orbit_end, not p["ccw"])
                seconds = m / PI * (60 / base / p["speed"])
                if m <= 1e-6 or m >= 6.283184482025146:
                    if p["mid"]:
                        m = 0.
                        seconds = 0.
                    else:
                        m = float(f32(6.2831855))
                        seconds = 2 * (60 / base / p["speed"])
                if p["hold"] > 0:
                    ha = float(p["hold"] * 2) * PI
                    seconds += ha / PI * (60 / base / p["speed"])
            rows.append({
                "floor": p["floor"], "time_s": time,
                "end_s": time + p["pause"] + seconds,
                "bpm": base * p["speed"], "midspin": p["mid"], "hold": p["hold"],
            })
            time += p["pause"] + seconds
        pending = state
        prev_mid = mid
        entry = nxt
    if pending:
        # Terminal floor; original chart does not use special terminal events.
        p = pending
        inv = 3.1415926 * (p["planets"] - 2.) / p["planets"]
        offset = inv * (-1 if p["ccw"] else 1)
        if p["mid"]:
            offset = 0.
        orbit_start = p["entry"] + offset
        orbit_end = p["exit"] + (offset if p["mid"] else 0)
        m = moved(orbit_start, orbit_end, not p["ccw"])
        seconds = m / PI * (60 / base / p["speed"])
        if m <= 1e-6 or m >= 6.283184482025146:
            seconds = 0. if p["mid"] else 2 * (60 / base / p["speed"])
        rows.append({
            "floor": p["floor"], "time_s": time,
            "end_s": time + p["pause"] + seconds,
            "bpm": base * p["speed"], "midspin": p["mid"], "hold": p["hold"],
        })
    return rows
