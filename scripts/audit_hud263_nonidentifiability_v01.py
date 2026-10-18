from __future__ import annotations

"""Counterfactual proof of HUD-only event-time non-identifiability.

Compare paired compiled charts with the SAME entire visible HUD263 prefix
but DIFFERENT future target due times. These paired charts are evaluator-side
experimental interventions, NEVER information supplied to a policy.

Two possible hidden causes:
  (1) future SetSpeed number (marker only, no future numeric speed in HUD)
  (2) initial countdown tick count (orbit phase repeats every two ticks)

A passing result establishes an information-theoretic barrier to
deterministic exact-event forecast with only this HUD contract; it does
not imply every missed target in every chart is unobservable.
"""

import argparse
from datetime import datetime, timezone
import json
from math import floor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import ContactTimingHUD263Adapter
from dmdod.planning.hud_timing_tracker_transition_v14 import forecast_visible_window
from audit_hud_timing_tracker_transition_v14_matrix import generated


FORMAT = "dmdod-hud263-counterfactual-observability-v0.1"
SAME_HUD_EPS = 5e-5
MIN_DIFFERENT_DUE_S = .044
PROBE_LEAD_S = .055


def _compiled(angle_data: list[float], bpm: float, countdown: int,
              speed_floor: int, future_bpm: float):
    raw = json.dumps({
        "angleData": angle_data,
        "settings": {"bpm": bpm, "pitch": 100.0,
                     "countdownTicks": countdown},
        "actions": [{
            "floor": speed_floor,
            "eventType": "SetSpeed", "speedType": "Bpm",
            "beatsPerMinute": future_bpm,
        }],
    })
    return build_playable_segment(
        compile_adofai(parse_adofai_text(raw))
    )


def paired_charts(case: str):
    if case == "set_speed_value":
        return (
            generated((0., 90., 180., 270.), 496., "speed_double"),
            generated((0., 90., 180., 270.), 496., "speed_half"),
        )
    if case == "countdown_count":
        angle_data = [0., 90., 180., 270.] * 12
        return (
            _compiled(angle_data, 60.01, 3, 1, 480.08),
            _compiled(angle_data, 60.01, 5, 1, 480.08),
        )
    raise ValueError("unsupported predeclared counterfactual case")


def _collect_visible_prefix(chart, through_step: int):
    env = HudCoupledContactRealChartEnv(
        chart, control_dt_s=.01, physics_dt_s=.001, tail_s=.04,
    )
    obs = env.reset()
    frames = []
    for step in range(through_step + 1):
        frame = ContactTimingHUD263Adapter.encode(step, obs)
        frames.append(frame.features)
        if step == through_step:
            break
        result = env.step(ContactAction((0.0,) * 12))
        if result.done:
            raise RuntimeError("counterfactual target was reached before probe")
        obs = result.observation
    return tuple(frames)


def analyze(case: str) -> dict:
    left, right = paired_charts(case)
    truth_a = tuple(float(x.episode_time_s) for x in left.targets)
    truth_b = tuple(float(x.episode_time_s) for x in right.targets)
    if len(truth_a) != len(truth_b):
        raise AssertionError("paired charts should share visible event topology")
    different = tuple(
        i for i, (a, b) in enumerate(zip(truth_a, truth_b))
        if abs(a-b) > MIN_DIFFERENT_DUE_S
    )
    if not different:
        raise AssertionError("paired hidden interventions did not change due times")
    first_difference = different[0]
    first_soon = min(truth_a[first_difference], truth_b[first_difference])
    probe_step = floor((first_soon - PROBE_LEAD_S) / .010)
    if probe_step < 1:
        raise AssertionError("insufficient pre-event HUD evidence")
    before_a = _collect_visible_prefix(left, probe_step)
    before_b = _collect_visible_prefix(right, probe_step)
    diffs = tuple(
        max(abs(x-y) for x, y in zip(a, b, strict=True))
        for a,b in zip(before_a, before_b, strict=True)
    )
    maximum = max(diffs)
    eligible_a = forecast_visible_window(
        before_a[-1], horizon_s=.080, previous_hud=before_a[-2]
    )
    eligible_b = forecast_visible_window(
        before_b[-1], horizon_s=.080, previous_hud=before_b[-2]
    )
    if maximum > SAME_HUD_EPS:
        status = "DISPROVED_VISIBLE_PREFIX_EQUALITY"
    elif eligible_a != eligible_b:
        status = "FAIL_TRACKER_DETERMINISM"
    else:
        status = "INDISTINGUISHABLE"
    return {
        "case": case, "status": status,
        "first_different_target_ordinal_posthoc": first_difference,
        "first_different_due_a_s_posthoc": truth_a[first_difference],
        "first_different_due_b_s_posthoc": truth_b[first_difference],
        "future_due_difference_ms_posthoc": round(
            abs(truth_a[first_difference] - truth_b[first_difference])*1000, 3
        ),
        "last_identical_frame": probe_step,
        "last_identical_local_clock_s": round(probe_step*.010, 6),
        "earliest_due_remaining_at_probe_ms": round(
            (first_soon - probe_step*.010)*1000, 3
        ),
        "compared_prefix_frames": len(before_a),
        "maximum_visible_hud263_feature_difference": maximum,
        "same_hud_tolerance": SAME_HUD_EPS,
        "same_deterministic_v14_prediction": eligible_a == eligible_b,
        "v14_probe_status": eligible_a.status,
        "v14_probe_forecasts_s_relative": list(eligible_a.times_s),
        "limitation": (
            "This proves indistinguishable current/past HUD263 features in "
            "a constructed pair, not a verified inability to infer the "
            "timing of any other user chart. No oracle fallback allowed."
        ),
    }


def run():
    rows = tuple(analyze(x) for x in ("set_speed_value", "countdown_count"))
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(
            row["status"] == "INDISTINGUISHABLE" for row in rows
        ) else "FAIL",
        "results": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("output file already exists, refusing overwrite")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for row in result["results"]:
        print(row["case"], row["status"],
              "sameHUDmax=", row["maximum_visible_hud263_feature_difference"],
              "probe=", row["last_identical_local_clock_s"],
              "future_due_gap_ms=", row["future_due_difference_ms_posthoc"],
              "v14_status=", row["v14_probe_status"], flush=True)
    print("counterfactual status:", result["status"], flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
