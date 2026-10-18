from __future__ import annotations

"""Predeclared synthetic and Train-bool Timing-only HUD v3 Phase 2 audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai, parse_adofai_text
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


FORMAT = "dmdod-countdown-timing-phase2-audit-v0.1"
CONTROL_DT_S = .010
MOTOR_LEAD_S = .044
DEFAULT_TRAIN_BOOL = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)


def _synthetic_segment(*, bpm: float, countdown: int, pitch: float = 100.0):
    raw = json.dumps({
        "angleData": [0.0, 90.0, 180.0, 270.0] * 3,
        "settings": {
            "bpm": bpm, "pitch": pitch, "countdownTicks": countdown,
            "separateCountdownTime": True,
        },
        "actions": [],
    })
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


def _collect_visible_stream(segment, *, max_s: float) -> list[dict]:
    """Policy collection pass. No target schedule is consulted here."""
    chart = segment.chart  # renderer-side state only
    pitch = segment.pitch_ratio
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / pitch,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    composite = HUDTimingComposite(
        bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT
    )
    env = HudCoupledContactRealChartEnv(
        segment, control_dt_s=CONTROL_DT_S, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    rows = []
    for step in range(int(max_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        rendered = renderer.render(now_s)
        cue = sensor.observe(
            step=step, observed_at_s=now_s, frame=rendered
        )
        legacy = ContactTimingHUD263Adapter.encode(step, observation).features
        update = composite.observe(encode_hud_v3(legacy, cue), cue)
        rows.append({
            "step": step,
            "observed_at_s": now_s,
            "cue": cue.kind.value,
            "digit": cue.digit,
            "phase_status": update.countdown_phase.status,
            "timing_status": update.status,
            "pending": [{
                "local_id": token.local_id,
                "due_lower_s": token.due_interval_s.lower_s,
                "due_upper_s": token.due_interval_s.upper_s,
                "due_mid_s": token.due_local_s,
                "source": token.source,
                "confirmed_now": token.confirmed_now,
                "lead_guaranteed": token.lead_guaranteed,
            } for token in update.pending],
            "new_ids": list(update.new_ids),
            "reused_ids": list(update.reused_ids),
            "canceled_ids": [x.local_id for x in update.canceled],
            "raw_v14_status": update.v14_update.status,
        })
        outcome = env.step(ContactAction((0.0,) * 12))
        if outcome.done:
            break
        observation = outcome.observation
    return rows


def _posthoc_first_event(rows: list[dict], segment) -> dict:
    truth_s = float(segment.targets[0].episode_time_s)
    candidates = {}
    for row in rows:
        for item in row["pending"]:
            if item["source"].startswith("countdown_first_event"):
                candidates.setdefault(item["local_id"], []).append((row, item))
    firsts = [values[0] for values in candidates.values()]
    matching = [
        pair for pair in firsts
        if pair[1]["due_lower_s"] - 1e-9 <= truth_s
        <= pair[1]["due_upper_s"] + 1e-9
        and abs(pair[1]["due_mid_s"] - truth_s) <= .010 + 1e-9
    ]
    false_ids = sorted(set(candidates) - {
        item["local_id"] for _, item in matching
    })
    first = min(matching, key=lambda pair: pair[0]["step"]) if matching else None
    return {
        "true_first_due_s_posthoc": truth_s,
        "unique_countdown_candidate_ids": len(candidates),
        "false_candidate_ids_posthoc": false_ids,
        "first_matching_prediction": None if first is None else {
            "step": first[0]["step"],
            "observed_at_s": first[0]["observed_at_s"],
            **first[1],
            "midpoint_error_ms_posthoc": round(
                (first[1]["due_mid_s"] - truth_s) * 1000.0, 6
            ),
            "lead_lower_ms_posthoc": round(
                (first[1]["due_lower_s"] - first[0]["observed_at_s"])
                * 1000.0,
                6,
            ),
        },
    }


def _case(segment, name: str, *, max_s: float) -> dict:
    rows = _collect_visible_stream(segment, max_s=max_s)
    result = _posthoc_first_event(rows, segment)
    first = result["first_matching_prediction"]
    result.update({
        "name": name,
        "hud_frames": len(rows),
        "status": "PASS" if (
            first is not None
            and result["unique_countdown_candidate_ids"] == 1
            and not result["false_candidate_ids_posthoc"]
            and first["lead_lower_ms_posthoc"] >= MOTOR_LEAD_S * 1000.0
        ) else "FAIL",
    })
    return result


def run(train_bool: Path = DEFAULT_TRAIN_BOOL) -> dict:
    synthetic = []
    for countdown in (3, 5):
        for bpm in (120.0, 240.0, 480.0):
            segment = _synthetic_segment(bpm=bpm, countdown=countdown)
            synthetic.append(_case(
                segment, f"N{countdown}-{int(bpm)}bpm-pitch100",
                max_s=min(segment.duration_s, 4.0),
            ))
    pitch_segment = _synthetic_segment(bpm=120.0, countdown=4, pitch=200.0)
    pitch_case = _case(
        pitch_segment, "N4-120bpm-pitch200",
        max_s=min(pitch_segment.duration_s, 3.0),
    )

    if not train_bool.is_file() or "train" not in {
        part.lower() for part in train_bool.parts
    }:
        raise ValueError("Train bool path must be an existing Train chart")
    train_segment = build_playable_segment(
        compile_adofai(load_adofai(train_bool)), end_s=5.0
    )
    train_case = _case(train_segment, "Train bool first 5s", max_s=5.0)
    train_case["chart"] = train_bool.name
    train_case["sha256"] = hashlib.sha256(train_bool.read_bytes()).hexdigest()

    all_cases = synthetic + [pitch_case, train_case]
    gates = {
        "synthetic_n3_n5_120_240_480": all(
            item["status"] == "PASS" for item in synthetic
        ),
        "pitch_200": pitch_case["status"] == "PASS",
        "false_candidate_zero": all(
            not item["false_candidate_ids_posthoc"] for item in all_cases
        ),
        "unique_first_event_id": all(
            item["unique_countdown_candidate_ids"] == 1 for item in all_cases
        ),
        "due_midpoint_within_10ms": all(
            item["first_matching_prediction"] is not None for item in all_cases
        ),
        "lead_lower_at_least_44ms": all(
            item["first_matching_prediction"] is not None
            and item["first_matching_prediction"]["lead_lower_ms_posthoc"] >= 44.0
            for item in all_cases
        ),
        "train_bool_first_event": train_case["status"] == "PASS",
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "synthetic": synthetic,
        "pitch": pitch_case,
        "train_bool": train_case,
        "limitations": [
            "Renderer settings are used only to draw the observed cue stream.",
            "Target truth is joined only after the full HUD stream is collected.",
            "No learning, checkpoint update, Validation20, or Motor run was performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bool", type=Path, default=DEFAULT_TRAIN_BOOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output file already exists, refusing overwrite")
    result = run(args.train_bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("status:", result["status"])
    print("gates:", result["gates"])
    print("Train bool:", result["train_bool"])
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
