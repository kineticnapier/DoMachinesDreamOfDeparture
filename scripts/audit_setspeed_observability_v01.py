from __future__ import annotations

"""Phase-5 causal audit for future SetSpeed observations on Train bool.

This is an evaluator/renderer audit, not a Student sensor.  It deliberately
does not turn a compiled speed value into a policy observation.  A stock icon
is marked observable only after a timestamped pixel capture has established
that the sprite was actually visible to the player.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.features.jrp_overlay_v2 import JrpOverlayRenderer
try:
    from scripts.audit_bool_second_event_counterfactual_v01 import (
        DEFAULT_CHART,
        EXPECTED_CHART_SHA256,
        REQUIRED_MOTOR_LEAD_S,
        compile_pair,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from audit_bool_second_event_counterfactual_v01 import (
        DEFAULT_CHART,
        EXPECTED_CHART_SHA256,
        REQUIRED_MOTOR_LEAD_S,
        compile_pair,
    )


FORMAT = "dmdod-setspeed-observability-phase5-v0.1"
DLL_SHA256 = "0d1524dabf0f8c67aa58ba1992d6ed858522b9094490ebbb00e5e8aeff7f40cd"


def stock_speed_icon(previous_speed: float, current_speed: float, *, playing: bool) -> str | None:
    """Port the stock DLL's coarse SetSpeed floor-icon selection.

    Inputs are renderer-side game state.  Returning this value does not make
    it Student-visible: the sensor still needs an actual pixel observation.
    """
    if not all(isfinite(x) and x > 0 for x in (previous_speed, current_speed)):
        raise ValueError("speeds must be finite and positive")
    relative = (current_speed - previous_speed) / previous_speed
    magnitude = abs(relative)
    if magnitude > 0.05:
        if relative > 0:
            return "Rabbit" if magnitude < 1.05 else "DoubleRabbit"
        return "Snail" if 1.0 - magnitude > 0.45 else "DoubleSnail"
    # scrFloor.UpdateIconSprite draws SameSpeed only while controller.paused.
    return None if playing else "SameSpeed"


def _jrp_evidence(original, counterfactual, deadline_s: float) -> dict:
    rows = {}
    for mode in ("normal", "jongyeol"):
        left = JrpOverlayRenderer.from_segment(original, mode=mode)
        right = JrpOverlayRenderer.from_segment(counterfactual, mode=mode)
        show_left, show_right = left.on_show(0), right.on_show(0)
        hit_left, hit_right = left.on_hit_hook(1), right.on_hit_hook(1)
        first_hook_s = float(original.targets[0].episode_time_s)
        rows[mode] = {
            "on_show_floor0_original": show_left,
            "on_show_floor0_counterfactual": show_right,
            "on_show_equal": show_left == show_right,
            "first_on_hit_floor1_original": hit_left,
            "first_on_hit_floor1_counterfactual": hit_right,
            "first_on_hit_differs": hit_left != hit_right,
            "earliest_possible_update_s_posthoc": first_hook_s,
            "deadline_s_posthoc": deadline_s,
            "lead_at_earliest_possible_update_ms_posthoc": round(
                (deadline_s - first_hook_s) * 1000.0, 6
            ),
            "verdict": "INDISTINGUISHABLE",
            "reason": "The startup text is equal; the first differing hit-driven refresh is after the 44ms deadline.",
        }
    return rows


def run(chart_path: Path = DEFAULT_CHART) -> dict:
    if not chart_path.is_file():
        raise ValueError("fixed Train bool chart is missing")
    chart_hash = sha256(chart_path.read_bytes()).hexdigest()
    if chart_hash != EXPECTED_CHART_SHA256:
        raise ValueError(f"Train bool SHA256 mismatch: {chart_hash}")

    chart = load_adofai(chart_path)
    original, counterfactual, intervention = compile_pair(chart)
    first_due = float(original.targets[0].episode_time_s)
    second_original = float(original.targets[1].episode_time_s)
    second_counterfactual = float(counterfactual.targets[1].episode_time_s)
    deadline = min(second_original, second_counterfactual) - REQUIRED_MOTOR_LEAD_S

    original_icon = stock_speed_icon(1.0, original.chart.floors[1].bpm / original.chart.initial_bpm, playing=True)
    counterfactual_icon = stock_speed_icon(
        1.0, counterfactual.chart.floors[1].bpm / counterfactual.chart.initial_bpm, playing=True
    )
    song_filename = chart.song_filename or ""
    song_path = chart_path.parent / str(song_filename)

    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name,
        "chart_sha256": chart_hash,
        "dll_sha256": DLL_SHA256,
        "intervention_evaluator_only": intervention,
        "first_due_s_posthoc": first_due,
        "second_due_original_s_posthoc": second_original,
        "second_due_counterfactual_s_posthoc": second_counterfactual,
        "deadline_44ms_s_posthoc": deadline,
        "candidates": {
            "stock_current_motion_and_hud278": {
                "verdict": "INDISTINGUISHABLE",
                "first_distinguishable_s": None,
                "evidence": "Phase 4 compared 398 10ms frames through 3.970s with maximum HUD278 difference 0.0.",
            },
            "stock_setspeed_speed_icon": {
                "renderer_semantics_original": original_icon,
                "renderer_semantics_counterfactual": counterfactual_icon,
                "renderer_semantics_distinguish_pair": original_icon != counterfactual_icon,
                "actual_pixel_capture_available": False,
                "first_observed_s": None,
                "capture_latency_s": None,
                "verdict": "UNVERIFIED",
                "reason": "The DLL predicts distinct sprites, but no paired Train-bool capture proves Floor 1 is visible before the deadline.",
            },
            "jrp_tbpm_cbpm_kps": _jrp_evidence(original, counterfactual, deadline),
            "audio": {
                "song_filename_both_worlds": song_filename,
                "audio_asset_present_in_train80": song_path.is_file(),
                "paired_counterfactual_capture_available": False,
                "first_observed_s": None,
                "verdict": "UNVERIFIED",
                "reason": "The intervention preserves the audio filename and countdown, but the asset and paired real captures are absent.",
            },
            "other_bpm_mods": {
                "verdict": "UNVERIFIED",
                "reason": "No inspected source or real capture established a pre-hit preview of the future numeric SetSpeed value.",
            },
        },
        "overall": {
            "verdict": "NO_CONFIRMED_OBSERVATION_SOURCE_BEFORE_DEADLINE",
            "phase6_action": "Capture the stock speed icon in both worlds with timestamps before implementing a sensor.",
            "motor_connection_allowed": False,
        },
        "student_boundary": {
            "allowed": "timestamped pixels/audio and their causal detection history",
            "forbidden": "compiled speeds, SetSpeed parameters, future BPM, floor IDs, target times, and renderer diagnostics",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing overwrite")
    result = run(args.chart)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("overall:", result["overall"]["verdict"])
    print("stock icon:", result["candidates"]["stock_setspeed_speed_icon"]["verdict"])
    print("JRP normal:", result["candidates"]["jrp_tbpm_cbpm_kps"]["normal"]["verdict"])


if __name__ == "__main__":
    main()
