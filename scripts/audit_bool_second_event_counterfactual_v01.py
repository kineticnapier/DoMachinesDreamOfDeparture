from __future__ import annotations

"""Phase-4: causal HUD278 counterfactual of Train bool's second event.

Change only the numerical value of the existing SetSpeed on floor 1.
Compare complete observed Mod-HUD263 + stock-countdown text prefixes before
the *earlier* world's second-target 44ms motor preparation deadline.

Renderer internals may draw the current HUD, but policy-facing capture never
reads target times, SetSpeed values, future actions or scorer results.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import floor, isfinite
from pathlib import Path

from dmdod.adofai.chart import AdoFaiChart, load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor, StockCountdownRenderer, encode_hud_v3,
)
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import (
    ContactTimingHUD263Adapter, MOD_VISIBLE_BPM_CONTRACT,
)

FORMAT = "dmdod-bool-second-event-counterfactual-v0.1"
DEFAULT_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)
EXPECTED_CHART_SHA256 = "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
CONTROL_DT_S = .010
REQUIRED_MOTOR_LEAD_S = .044
SAME_HUD_TOLERANCE = 5e-5
SPEED_FACTOR = .125
TARGET_WINDOW_S = 5.0


def intervene_floor1_setspeed(chart: AdoFaiChart) -> tuple[AdoFaiChart, dict]:
    """Change only an existing floor-1 SetSpeed numeric value, preserving markers.

    No floor-1 SetSpeed means this hypothesis does not apply; do not silently
    add a new visible marker or misreport an indistinguishable counterfactual.
    """
    candidates = [
        (i, action) for i, action in enumerate(chart.actions)
        if action.floor == 1 and action.active and action.event_type == "SetSpeed"
    ]
    if not candidates:
        raise ValueError("NO_FLOOR1_SETSPEED: different hypothesis required")
    index, original = candidates[-1]
    if (original.speed_type or "").lower() == "bpm":
        old = original.beats_per_minute
        if old is None or not isfinite(old) or old <= 0:
            raise ValueError("INVALID_FLOOR1_BPM")
        changed = replace(original, beats_per_minute=old * SPEED_FACTOR)
        field = "beatsPerMinute"
    else:
        old = original.bpm_multiplier
        if old is None or not isfinite(old) or old <= 0:
            raise ValueError("INVALID_FLOOR1_MULTIPLIER")
        changed = replace(original, bpm_multiplier=old * SPEED_FACTOR)
        field = "bpmMultiplier"
    actions = list(chart.actions)
    actions[index] = changed
    counterpart = replace(chart, actions=tuple(actions))
    assert all(
        a == b or i == index
        for i, (a, b) in enumerate(zip(chart.actions, counterpart.actions, strict=True))
    )
    return counterpart, {
        "action_source_index": original.source_index,
        "action_floor": 1,
        "modified_field": field,
        "original_value_evaluator_only": old,
        "counterfactual_value_evaluator_only": old * SPEED_FACTOR,
        "factor_predeclared": SPEED_FACTOR,
        "visible_marker_preserved": True,
    }


def compile_pair(chart: AdoFaiChart):
    alt_chart, intervention = intervene_floor1_setspeed(chart)
    left = build_playable_segment(compile_adofai(chart), end_s=TARGET_WINDOW_S)
    right = build_playable_segment(compile_adofai(alt_chart), end_s=TARGET_WINDOW_S)
    if len(left.targets) < 2 or len(right.targets) < 2:
        raise ValueError("COUNTERFACTUAL_NOT_IN_FIRST_5S")
    return left, right, intervention


def capture_causal_prefix(segment, *, through_step: int) -> tuple[dict, ...]:
    """Renderer / sensor only; no target truth or future action values."""
    chart = segment.chart
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    env = HudCoupledContactRealChartEnv(
        segment, control_dt_s=CONTROL_DT_S, physics_dt_s=.001, tail_s=.040,
    )
    observation = env.reset()
    rows = []
    for step in range(through_step + 1):
        now = step * CONTROL_DT_S
        cue = sensor.observe(step=step, observed_at_s=now, frame=renderer.render(now))
        hud263 = ContactTimingHUD263Adapter.encode(step, observation).features
        hud278 = encode_hud_v3(hud263, cue)
        rows.append({
            "step": step,
            "observed_at_s": now,
            "features": hud278.features,
            "cue_valid": cue.valid,
            "cue_kind": cue.kind.value,
            "cue_digit": cue.digit,
            "rendered_at_s": cue.rendered_at_s,
            "observed_latency_s": cue.latency_s,
        })
        if step < through_step:
            result = env.step(ContactAction((0.0,) * 12))
            if result.done:
                raise RuntimeError("environment ended before cutoff")
            observation = result.observation
    return tuple(rows)


def compare_pair(original, alternate, intervention: dict) -> dict:
    """Evaluator-only posthoc comparison of *already chosen* observations."""
    first_a, second_a = (float(x.episode_time_s) for x in original.targets[:2])
    first_b, second_b = (float(x.episode_time_s) for x in alternate.targets[:2])
    deadline = min(second_a, second_b) - REQUIRED_MOTOR_LEAD_S
    if deadline <= 0:
        raise ValueError("NO_PREDEADLINE_PREFIX")
    last_step = floor(deadline / CONTROL_DT_S)
    a_rows = capture_causal_prefix(original, through_step=last_step)
    b_rows = capture_causal_prefix(alternate, through_step=last_step)
    diffs = [
        max(abs(a-b) for a, b in zip(left["features"], right["features"], strict=True))
        for left, right in zip(a_rows, b_rows, strict=True)
    ]
    diverged = [step for step, delta in enumerate(diffs) if delta > SAME_HUD_TOLERANCE]
    cues_equal = all(
        (a["cue_valid"], a["cue_kind"], a["cue_digit"], a["rendered_at_s"])
        == (b["cue_valid"], b["cue_kind"], b["cue_digit"], b["rendered_at_s"])
        for a, b in zip(a_rows, b_rows, strict=True)
    )
    first_same = abs(first_a - first_b) < 1e-6
    topology_same = all(
        a.visible() == b.visible()
        for a, b in zip(original.chart.floors[:3], alternate.chart.floors[:3], strict=True)
    )
    due_gap = abs(second_a - second_b)
    prefix_equal = max(diffs) <= SAME_HUD_TOLERANCE and cues_equal
    status = (
        "INDISTINGUISHABLE_BEFORE_44MS_DEADLINE"
        if first_same and topology_same
        and due_gap >= REQUIRED_MOTOR_LEAD_S and prefix_equal
        else "NOT_PROVEN"
    )
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bpm_observation_contract": MOD_VISIBLE_BPM_CONTRACT,
        "intervention_evaluator_only": intervention,
        "first_due_original_s_posthoc": first_a,
        "first_due_counterfactual_s_posthoc": first_b,
        "first_due_unchanged": first_same,
        "first_three_floors_visible_topology_equal": topology_same,
        "second_due_original_s_posthoc": second_a,
        "second_due_counterfactual_s_posthoc": second_b,
        "second_due_gap_ms_posthoc": round(due_gap * 1000, 6),
        "earlier_second_due_s_posthoc": min(second_a, second_b),
        "deadline_44ms_s_posthoc": deadline,
        "last_compared_frame": last_step,
        "last_compared_observation_s": last_step * CONTROL_DT_S,
        "frames_compared": len(diffs),
        "max_absolute_hud278_difference": max(diffs),
        "same_hud_tolerance": SAME_HUD_TOLERANCE,
        "cue_metadata_equal": cues_equal,
        "first_distinguishable_frame_in_prefix": diverged[0] if diverged else None,
        "indistinguishable_prefix": prefix_equal,
        "conclusion_limit": (
            "Proves only exact due-time non-identifiability in the modeled "
            "Mod-HUD278 observation and fixed intervention. Not proof every "
            "physical strategy misses; does not cover audio, stock-only, OCR "
            "or real capture latency."
        ),
    }


def run(chart_path: Path = DEFAULT_CHART) -> dict:
    if not chart_path.is_file() or "train" not in {p.lower() for p in chart_path.parts}:
        raise ValueError("expected existing fixed Train bool chart")
    observed_sha = sha256(chart_path.read_bytes()).hexdigest()
    if observed_sha != EXPECTED_CHART_SHA256:
        raise ValueError(f"Train bool SHA256 mismatch: {observed_sha}")
    original, alternate, intervention = compile_pair(load_adofai(chart_path))
    result = compare_pair(original, alternate, intervention)
    result["chart"] = chart_path.name
    result["chart_sha256"] = observed_sha
    return result


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
    print("status:", result["status"])
    print("first due unchanged:", result["first_due_unchanged"])
    print("second due gap ms:", result["second_due_gap_ms_posthoc"])
    print("last compared s:", result["last_compared_observation_s"])
    print("max HUD278 delta:", result["max_absolute_hud278_difference"])
    if result["status"] != "INDISTINGUISHABLE_BEFORE_44MS_DEADLINE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
