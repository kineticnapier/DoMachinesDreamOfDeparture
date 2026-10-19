from __future__ import annotations

"""Fixed, causal Phase 7 audit for HUD Timing v1.4 and HUD v3 composite."""

import argparse
from collections import Counter
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
    encode_hud_v3,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import (
    ContactTimingHUD263Adapter,
    MOD_VISIBLE_BPM_CONTRACT,
)
from dmdod.planning.hud_timing_composite_v20 import HUDTimingComposite
from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)


FORMAT = "dmdod-hud-timing-generalization-phase7-v0.1"
CONTROL_DT_S = 0.010
AUDIT_END_S = 12.0
MATCH_TOLERANCE_S = 0.001
MIN_FORECAST_LEAD_S = 0.010
FIXED_CHARTS = (
    "02334 - Time Leaper - Jipper.adofai",
    "02521 - Triple Cross - Milkyway.adofai",
    "07322 - Runaway - smy258.adofai",
    "07827 - 24eeev0-$ - hotduck _ COMMON.adofai",
)
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")


def _capture_policy_stream(path: Path) -> tuple[list[dict], object]:
    """Run policy-side inference before consulting any target timestamps."""
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart, end_s=AUDIT_END_S)
    pitch = segment.pitch_ratio
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / pitch,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    composite = HUDTimingComposite(
        bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT,
        horizon_s=0.080,
    )
    improved = HUDTimingComposite(
        bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT,
        horizon_s=0.080,
        v14_tracker=QuantizedLifecycleCausalTimingTracker(horizon_s=0.080),
    )
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S,
        physics_dt_s=0.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=0.040,
    )
    observation = env.reset()
    idle = ContactAction((0.0,) * 12)
    rows: list[dict] = []
    for step in range(int(AUDIT_END_S / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        hud263 = ContactTimingHUD263Adapter.encode(step, observation).features
        update = composite.observe(encode_hud_v3(hud263, cue), cue)
        improved_update = improved.observe(encode_hud_v3(hud263, cue), cue)
        rows.append({
            "step": step,
            "time_s": now_s,
            "cue": cue.kind.value,
            "v14_status": update.v14_update.status,
            "v14_pending": [
                {
                    "id": f"v14:{item.local_id}",
                    "due_s": item.due_s,
                    "confirmed_now": item.confirmed_now,
                }
                for item in update.v14_update.pending
            ],
            "v14_canceled": [f"v14:{item.local_id}" for item in update.v14_update.canceled],
            "composite_status": update.status,
            "composite_pending": [
                {
                    "id": item.local_id,
                    "due_s": item.due_local_s,
                    "due_lower_s": item.due_interval_s.lower_s,
                    "due_upper_s": item.due_interval_s.upper_s,
                    "confirmed_now": item.confirmed_now,
                    "source": item.source,
                }
                for item in update.pending
            ],
            "composite_canceled": [item.local_id for item in update.canceled],
            "improved_status": improved_update.status,
            "improved_pending": [
                {
                    "id": item.local_id,
                    "due_s": item.due_local_s,
                    "due_lower_s": item.due_interval_s.lower_s,
                    "due_upper_s": item.due_interval_s.upper_s,
                    "confirmed_now": item.confirmed_now,
                    "source": item.source,
                }
                for item in improved_update.pending
            ],
            "improved_canceled": [
                item.local_id for item in improved_update.canceled
            ],
        })
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows, segment


def _method_rows(rows: list[dict], prefix: str) -> list[dict]:
    pending_key = f"{prefix}_pending"
    canceled_key = f"{prefix}_canceled"
    status_key = f"{prefix}_status"
    result = []
    for row in rows:
        result.append({
            "step": row["step"],
            "time_s": row["time_s"],
            "status": row[status_key],
            "pending": row[pending_key],
            "canceled": row[canceled_key],
        })
    return result


def _evaluate(rows: list[dict], truths: tuple[float, ...]) -> dict:
    """Evaluator-only association; rows were already collected causally."""
    forecast_by_id: dict[str, list[dict]] = {}
    canceled_at: dict[str, float] = {}
    for row in rows:
        for local_id in row["canceled"]:
            canceled_at.setdefault(local_id, row["time_s"])
        for token in row["pending"]:
            forecast_by_id.setdefault(token["id"], []).append({
                **token,
                "observed_at_s": row["time_s"],
                "status": row["status"],
            })

    false_ids = []
    for local_id, trace in forecast_by_id.items():
        if not any(
            (
                sample.get("due_lower_s", sample["due_s"]) - 1e-9
                <= truth
                <= sample.get("due_upper_s", sample["due_s"]) + 1e-9
            )
            or abs(sample["due_s"] - truth) <= MATCH_TOLERANCE_S
            for sample in trace for truth in truths
        ):
            false_ids.append(local_id)

    abstention_statuses = {
        "pause_motion_unobservable",
        "pause_motion_alias",
        "countdown_before_special_arrival",
        "special_future_floor",
        "special_event",
        "midspin_or_multiplanet",
        "full_turn_no_motion_history",
        "full_turn_motion_alias",
        "insufficient_lookahead",
    }
    events = []
    for index, truth in enumerate(truths):
        window = [
            row for row in rows
            if truth - 0.080 - 1e-9 <= row["time_s"] < truth - 1e-9
        ]
        matches = []
        wrong = []
        for row in window:
            for token in row["pending"]:
                sample = {**token, "observed_at_s": row["time_s"]}
                lower = token.get("due_lower_s", token["due_s"])
                upper = token.get("due_upper_s", token["due_s"])
                interval_match = lower - 1e-9 <= truth <= upper + 1e-9
                if interval_match or abs(token["due_s"] - truth) <= MATCH_TOLERANCE_S:
                    matches.append(sample)
                else:
                    nearest = min(
                        truths, key=lambda candidate: abs(token["due_s"] - candidate)
                    )
                    if nearest == truth and abs(token["due_s"] - truth) <= 0.010:
                        wrong.append(sample)
        ids = sorted({sample["id"] for sample in matches})
        lifecycle_bad = len(ids) > 1 or any(
            canceled_at.get(local_id, float("inf")) < truth - 1e-9
            for local_id in ids
        )
        if lifecycle_bad:
            category = 5
            label = "event_id_or_lifecycle_inconsistency"
        elif matches:
            category = 1
            label = "observable_prediction_success"
        elif wrong:
            category = 2
            label = "observable_estimation_failure"
        elif window and any(row["status"] in abstention_statuses for row in window):
            category = 3
            label = "justified_observation_abstention"
        else:
            category = 4
            label = "observability_unverified"
        best = max(matches, key=lambda item: truth - item["observed_at_s"]) if matches else None
        events.append({
            "event_index": index,
            "truth_s_posthoc": truth,
            "category": category,
            "classification": label,
            "matching_ids": ids,
            "first_forecast_lead_ms": None if best is None else round(
                (truth - best["observed_at_s"]) * 1000.0, 6
            ),
            "first_forecast_error_ms": None if best is None else round(
                (best["due_s"] - truth) * 1000.0, 6
            ),
            "first_forecast_interval_ms": None if best is None else [
                round((best.get("due_lower_s", best["due_s"]) - truth) * 1000.0, 6),
                round((best.get("due_upper_s", best["due_s"]) - truth) * 1000.0, 6),
            ],
            "pre_event_statuses": sorted({row["status"] for row in window}),
            "wrong_candidate_count": len(wrong),
        })

    successful = [event for event in events if event["category"] == 1]
    errors = [abs(event["first_forecast_error_ms"]) for event in successful]
    leads = [event["first_forecast_lead_ms"] for event in successful]
    counts = Counter(event["category"] for event in events)
    return {
        "events": events,
        "metrics": {
            "target_count": len(events),
            "category_counts": {str(key): counts.get(key, 0) for key in range(1, 6)},
            "coverage": len(successful) / len(events) if events else 0.0,
            "timing_mae_ms": sum(errors) / len(errors) if errors else None,
            "mean_forecast_lead_ms": sum(leads) / len(leads) if leads else None,
            "false_positive_ids": len(false_ids),
            "false_positive_id_list": sorted(false_ids),
            "abstention": counts.get(3, 0) + counts.get(4, 0),
            "event_id_continuity_failures": counts.get(5, 0),
            "status_frame_counts": dict(sorted(Counter(
                row["status"] for row in rows
            ).items())),
        },
    }


def run(train_dir: Path = TRAIN_DIR) -> dict:
    chart_results = []
    for filename in FIXED_CHARTS:
        path = train_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, segment = _capture_policy_stream(path)
        # Evaluator boundary: target truth is read only after inference ends.
        truths = tuple(
            float(target.episode_time_s)
            for target in segment.targets
            if target.episode_time_s <= AUDIT_END_S + 1e-9
        )
        chart_results.append({
            "chart": filename,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "audit_end_s": AUDIT_END_S,
            "captured_frames": len(rows),
            "constructs_posthoc": {
                "countdown_ticks": segment.chart.countdown_ticks,
                "set_speed_floors": sum(
                    "SetSpeed" in floor.event_markers
                    for floor in segment.chart.floors
                    if floor.target_time_s <= AUDIT_END_S + 1e-9
                ),
                "pause_floors": sum(
                    "Pause" in floor.event_markers
                    for floor in segment.chart.floors
                    if floor.target_time_s <= AUDIT_END_S + 1e-9
                ),
                "twirl_floors": sum(
                    "Twirl" in floor.event_markers
                    for floor in segment.chart.floors
                    if floor.target_time_s <= AUDIT_END_S + 1e-9
                ),
                "midspin_floors": sum(
                    floor.midspin for floor in segment.chart.floors
                    if floor.target_time_s <= AUDIT_END_S + 1e-9
                ),
            },
            "v14": _evaluate(_method_rows(rows, "v14"), truths),
            "composite": _evaluate(_method_rows(rows, "composite"), truths),
            "improved_lifecycle_v15": _evaluate(
                _method_rows(rows, "improved"), truths
            ),
        })
        baseline_events = chart_results[-1]["composite"]["events"]
        improved_events = chart_results[-1]["improved_lifecycle_v15"]["events"]
        recovered = [
            baseline["event_index"]
            for baseline, candidate in zip(baseline_events, improved_events)
            if baseline["category"] == 5 and candidate["category"] == 1
        ]
        chart_results[-1]["recovered_by_lifecycle_rule"] = recovered
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_before_audit": True,
        "charts": chart_results,
        "causality": {
            "student_input": "current/past HUD263 plus observed countdown pixels only",
            "truth_materialized": "after each complete policy capture",
            "truth_use": "posthoc association and metrics only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    result = run(args.train_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    for chart in result["charts"]:
        print(chart["chart"])
        print(" v1.4", chart["v14"]["metrics"])
        print(" composite", chart["composite"]["metrics"])
        print(" improved", chart["improved_lifecycle_v15"]["metrics"])


if __name__ == "__main__":
    main()
