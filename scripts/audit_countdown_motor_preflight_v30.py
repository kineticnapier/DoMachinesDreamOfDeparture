from __future__ import annotations

"""Train-bool HUD v3 Timing preflight before any Motor command is issued."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor,
    StockCountdownRenderer,
)
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
)


FORMAT = "dmdod-countdown-motor-preflight-v0.3"
CONTROL_DT_S = .010
NORMAL_LEAD_S = .044
DECISION_LEAD_S = .069
DEFAULT_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)


def collect(chart_path: Path = DEFAULT_CHART) -> tuple[list[dict], object]:
    if not chart_path.is_file() or "train" not in {
        part.lower() for part in chart_path.parts
    }:
        raise ValueError("preflight requires the fixed Train bool chart")
    segment = build_playable_segment(
        compile_adofai(load_adofai(chart_path)), end_s=5.0
    )
    chart = segment.chart
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    source = HUDV3CompositeForecastSource(horizon_s=.080)
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S,
        physics_dt_s=.001,
        tail_s=.040,
    )
    observation = env.reset()
    rows = []
    for step in range(501):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        update, raw = source.observe(step, observation, cue)
        rows.append({
            "step": step,
            "local_clock_s": now_s,
            "cue": cue.kind.value,
            "cue_valid": cue.valid,
            "composite_status": raw.status,
            "composite_abstention_reason": raw.abstention_reason,
            "timing_status": update.status,
            "timing_abstention_reason": update.abstention_reason,
            "new_ids": list(update.new_ids),
            "reused_ids": list(update.reused_ids),
            "canceled_ids": list(update.canceled_ids),
            "elapsed_ids": list(update.elapsed_ids),
            "pending": [{
                "local_id": item.local_id,
                "due_local_s": item.due_local_s,
                "uncertainty_s": item.uncertainty_s,
                "due_lower_s": (
                    item.due_local_s
                    - (0.0 if item.uncertainty_s is None else item.uncertainty_s)
                ),
                "due_upper_s": (
                    item.due_local_s
                    + (0.0 if item.uncertainty_s is None else item.uncertainty_s)
                ),
                "confirmed_now": item.confirmed_now,
                "confirmations": item.confirmations,
            } for item in update.pending],
            "raw_v14_pending": [{
                "local_id": item.local_id,
                "due_s": item.due_s,
                "confirmed_now": item.confirmed_now,
            } for item in raw.v14_update.pending],
        })
        outcome = env.step(ContactAction((0.0,) * 12))
        if outcome.done:
            break
        observation = outcome.observation
    return rows, segment


def evaluate(rows: list[dict], segment, chart_path: Path) -> dict:
    """Truth is joined only after collection has completely finished."""
    truth_s = float(segment.targets[0].episode_time_s)
    new = [
        (row, item)
        for row in rows
        for item in row["pending"]
        if item["local_id"] in row["new_ids"]
    ]
    matching = [
        (row, item) for row, item in new
        if item["due_lower_s"] - 1e-9 <= truth_s <= item["due_upper_s"] + 1e-9
    ]
    first = matching[0] if matching else None
    timing_id = None if first is None else first[1]["local_id"]
    tracked = [
        (row, item)
        for row in rows
        for item in row["pending"]
        if item["local_id"] == timing_id
    ]
    decision = next((
        (row, item) for row, item in tracked
        if item["confirmed_now"]
        and item["due_local_s"] - row["local_clock_s"] <= DECISION_LEAD_S + 1e-9
        and item["due_lower_s"] - row["local_clock_s"] >= NORMAL_LEAD_S - 1e-9
    ), None)
    due_intervals = [
        (item["due_lower_s"], item["due_upper_s"])
        for _, item in tracked
    ]
    nonnested_changes = sum(
        not (current[0] + 1e-9 >= previous[0]
             and current[1] <= previous[1] + 1e-9)
        for previous, current in zip(due_intervals, due_intervals[1:])
    )
    decision_step = None if decision is None else decision[0]["step"]
    final_required_step = int(truth_s // CONTROL_DT_S)
    maintained = (
        decision is not None
        and all(any(
            item["local_id"] == timing_id and item["confirmed_now"]
            for item in rows[step]["pending"]
        ) for step in range(decision_step, final_required_step + 1))
    )
    conflicts = sum(
        row["composite_status"] in (
            "source_conflict", "countdown_interval_conflict"
        ) for row in rows
    )
    pre_due_new_ids = {
        item["local_id"]
        for row in rows if row["local_clock_s"] < truth_s
        for item in row["pending"] if item["local_id"] in row["new_ids"]
    }
    decision_record = None if decision is None else {
        "step": decision[0]["step"],
        "observed_at_s": decision[0]["local_clock_s"],
        "due_local_s": decision[1]["due_local_s"],
        "due_lower_s": decision[1]["due_lower_s"],
        "midpoint_lead_ms": round(
            (decision[1]["due_local_s"] - decision[0]["local_clock_s"]) * 1000,
            6,
        ),
        "lower_bound_lead_ms": round(
            (decision[1]["due_lower_s"] - decision[0]["local_clock_s"]) * 1000,
            6,
        ),
        "confirmed_now": decision[1]["confirmed_now"],
    }
    gates = {
        "first_target_forecast_exists": first is not None,
        "one_pre_due_timing_id": len(pre_due_new_ids) == 1,
        "decision_window_confirmed": decision is not None,
        "maintained_through_pre_due_frame": maintained,
        "no_source_or_interval_conflict": conflicts == 0,
        "due_interval_only_narrows": nonnested_changes == 0,
        "no_pre_due_cancellation": not any(
            timing_id in row["canceled_ids"]
            for row in rows if row["local_clock_s"] < truth_s
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "chart": chart_path.name,
        "chart_sha256": sha256(chart_path.read_bytes()).hexdigest(),
        "scope_s": 5.0,
        "first_target_due_s_posthoc": truth_s,
        "first_target_timing_id": timing_id,
        "first_prediction": None if first is None else {
            "step": first[0]["step"],
            "observed_at_s": first[0]["local_clock_s"],
            **first[1],
        },
        "decision_window": decision_record,
        "tracked_frame_count": len(tracked),
        "first_tracked_step": None if not tracked else tracked[0][0]["step"],
        "last_tracked_step": None if not tracked else tracked[-1][0]["step"],
        "source_conflict_count": conflicts,
        "nonnested_due_interval_changes": nonnested_changes,
        "pre_due_new_ids": sorted(pre_due_new_ids),
        "gates": gates,
        "frames": rows,
        "causality": (
            "Policy collection used current/past HUD278 only. AutoPlay truth "
            "was joined by evaluate() after collect() returned."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; refusing overwrite")
    rows, segment = collect(args.chart)
    report = evaluate(rows, segment, args.chart)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("status:", report["status"])
    print("gates:", report["gates"])
    print("decision:", report["decision_window"])
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
