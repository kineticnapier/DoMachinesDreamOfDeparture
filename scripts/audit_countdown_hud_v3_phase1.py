from __future__ import annotations

"""Causal phase-1 audit for the stock-countdown HUD v3 sensor.

The capture pass receives rendered countdown text only.  Target times are
joined later by the evaluator, so they cannot influence Student observations.
"""

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hud263_nonidentifiability_v01 import analyze, paired_charts
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind,
    CountdownStudentObservation,
    CountdownVisualSensor,
    StockCountdownRenderer,
)


FORMAT = "dmdod-countdown-hud-v3-phase1-audit-v0.1"
CONTROL_DT_S = .010
FIRST_TICK_S = .250
REQUIRED_MOTOR_LEAD_S = .044


def _capture(segment) -> tuple[CountdownStudentObservation, ...]:
    """Renderer/sensor pass; intentionally has no target-time argument."""
    chart = segment.chart
    period_s = 60.0 / chart.initial_bpm / segment.pitch_ratio
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=FIRST_TICK_S,
        tick_period_s=period_s,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    # Capture through the first post-Go frame. This duration is renderer-side
    # schedule data, not a target or playable-event time.
    end_s = (
        FIRST_TICK_S + (chart.countdown_ticks - 1) * period_s
        + 60.0 / chart.initial_bpm + CONTROL_DT_S
    )
    count = int(round(end_s / CONTROL_DT_S)) + 1
    result = []
    for step in range(count):
        now_s = step * CONTROL_DT_S
        result.append(sensor.observe(
            step=step, observed_at_s=now_s, frame=renderer.render(now_s)
        ))
    return tuple(result)


def _public(observation: CountdownStudentObservation) -> dict:
    value = asdict(observation)
    value["kind"] = observation.kind.value
    value["latency_s"] = observation.latency_s
    return value


def _first_difference(left, right):
    for a, b in zip(left, right, strict=False):
        if (a.valid, a.kind, a.digit) != (b.valid, b.kind, b.digit):
            return a, b
    raise AssertionError("HUD v3 countdown observations never became distinct")


def _first_go(observations):
    return next(x for x in observations if x.kind is CountdownCueKind.GO)


def _simple_segment(*, bpm: float, countdown: int = 4):
    raw = json.dumps({
        "angleData": [0.0, 90.0, 180.0, 270.0],
        "settings": {
            "bpm": bpm, "pitch": 100.0, "countdownTicks": countdown,
        },
        "actions": [],
    })
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


def _go_gap_row(bpm: float) -> dict:
    segment = _simple_segment(bpm=bpm)
    observations = _capture(segment)
    go = _first_go(observations)
    # Evaluator-only join after capture has completed.
    target_s = FIRST_TICK_S + float(segment.targets[0].episode_time_s)
    return {
        "bpm": bpm,
        "first_observed_go_s": go.observed_at_s,
        "first_target_s_posthoc": target_s,
        "go_to_first_target_ms_posthoc": round(
            (target_s - go.observed_at_s) * 1000.0, 6
        ),
    }


def run() -> dict:
    legacy = analyze("countdown_count")
    short, long = paired_charts("countdown_count")
    short_obs, long_obs = _capture(short), _capture(long)
    first_a, first_b = _first_difference(short_obs, long_obs)

    # Truth is first read here, after both Student-visible sequences exist.
    due_a_s = FIRST_TICK_S + float(short.targets[0].episode_time_s)
    due_b_s = FIRST_TICK_S + float(long.targets[0].episode_time_s)
    lead_a_s = due_a_s - first_a.observed_at_s
    lead_b_s = due_b_s - first_b.observed_at_s
    go_gaps = (_go_gap_row(120.0), _go_gap_row(240.0))
    forbidden = {
        "countdown_ticks", "countdownTicks", "target_time_s", "target_index",
        "target_ordinal", "future_tick_s", "first_tick_s", "tick_period_s",
    }
    public_fields = {item.name for item in fields(CountdownStudentObservation)}
    no_truth_leak = public_fields.isdisjoint(forbidden)
    go_gap_is_not_fixed = (
        go_gaps[0]["go_to_first_target_ms_posthoc"]
        != go_gaps[1]["go_to_first_target_ms_posthoc"]
    )
    passed = all((
        legacy["status"] == "INDISTINGUISHABLE",
        first_a.digit == 2,
        first_b.digit == 4,
        min(lead_a_s, lead_b_s) >= REQUIRED_MOTOR_LEAD_S,
        no_truth_leak,
        go_gap_is_not_fixed,
    ))
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "capture_contract": {
            "control_interval_ms": CONTROL_DT_S * 1000.0,
            "synthetic_capture_latency_ms": 0.0,
            "student_fields": sorted(public_fields),
            "forbidden_truth_fields_absent": no_truth_leak,
            "note": (
                "Display capture is completed before evaluator-only target "
                "times are read. Real display/capture latency is not measured."
            ),
        },
        "legacy_hud263": legacy,
        "countdown_3_vs_5": {
            "first_distinguishable_observation_a": _public(first_a),
            "first_distinguishable_observation_b": _public(first_b),
            "first_distinguishable_s": first_a.observed_at_s,
            "first_tick_boundary_s": FIRST_TICK_S,
            "sampling_delay_after_strict_boundary_ms": round(
                (first_a.observed_at_s - FIRST_TICK_S) * 1000.0, 6
            ),
            "first_target_a_s_posthoc": due_a_s,
            "first_target_b_s_posthoc": due_b_s,
            "lead_a_ms_posthoc": round(lead_a_s * 1000.0, 6),
            "lead_b_ms_posthoc": round(lead_b_s * 1000.0, 6),
            "required_motor_lead_ms": REQUIRED_MOTOR_LEAD_S * 1000.0,
            "required_lead_met_in_fixture": (
                min(lead_a_s, lead_b_s) >= REQUIRED_MOTOR_LEAD_S
            ),
        },
        "go_to_first_target_is_not_fixed": {
            "status": "PASS" if go_gap_is_not_fixed else "FAIL",
            "fixtures": go_gaps,
            "note": (
                "These target times are evaluator-only. Countdown text alone "
                "does not supply an exact first-hit deadline."
            ),
        },
        "scope": (
            "Controlled normal-start fixtures only. Fast takeoff, checkpoints, "
            "localized/occluded text, and real capture latency remain unverified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output file already exists, refusing overwrite")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    pair = result["countdown_3_vs_5"]
    print("status:", result["status"])
    print("first distinguishable:", pair["first_distinguishable_s"], "s")
    print("lead:", pair["lead_a_ms_posthoc"], pair["lead_b_ms_posthoc"], "ms")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
