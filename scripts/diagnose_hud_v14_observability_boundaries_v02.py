from __future__ import annotations

"""Audit exactly when HUD-v1.4 first has observable Timing on failed cases.

Uses an idle Contact HUD renderer only: deterministic v1.4 timing does not
read Motor/feedback slices. The sole Timing input is each current 263D HUD
frame; chart-target truth is consulted POSTHOC after collecting the stream.

No Oracle fallback, no scorer changes, no Timing tuning, no training.
Renderer geometry parity with the old integration is a documented
assumption: --compare-report checks actual first-seen HUD event times.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hud_timing_tracker_transition_v14_matrix import generated
from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.hud_v14_forecast_source_v02 import HUDV14ForecastSource
from dmdod.planning import hud_timing_tracker_transition_v14 as v14
from run_hud_v14_fingering_motor_integration_v02 import DEFAULT_TRAIN_CHART


CASES = ("special-speed_double", "train-bool-5s")
REPORT_FORMAT = "dmdod-hud-v14-observability-boundary-v0.1"
TARGET_TOLERANCE_S = .020
MIN_MOTOR_LEAD_S = .044
PRE_WINDOW_S = .110
POST_WINDOW_S = .020


def load_case(name: str, train_chart: Path):
    if name == "special-speed_double":
        return generated((0., 90., 180., 270.), 496., "speed_double")
    if name == "train-bool-5s":
        if ("train" not in (p.lower() for p in train_chart.parts)
                or not train_chart.is_file()):
            raise ValueError("must supply existing Train chart, not Validation")
        return build_playable_segment(
            compile_adofai(load_adofai(train_chart)), end_s=5.
        )
    raise ValueError("unsupported predeclared audit case")


def _visible_floors(observation) -> list[dict]:
    return [
        {
            "relative_index": f.relative_index,
            "entry_angle": f.entry_angle_rad,
            "exit_angle": f.exit_angle_rad,
            "markers": list(f.event_markers),
            "midspin": f.midspin,
            "num_planets": f.num_planets,
        }
        for f in observation.contact.floors
        if f.relative_index in (-1, 0, 1, 2)
    ]


def collect_visible_hud(segment) -> list[dict]:
    """No privileged target list access inside this collection loop."""
    env = HudCoupledContactRealChartEnv(
        segment, control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    source = HUDV14ForecastSource(horizon_s=.080)
    collected = []
    max_steps = int((segment.duration_s + 2.0)/.010) + 200
    for step in range(max_steps):
        hud = source.hud.encode(step, observation)
        # Raw reason (otherwise normalized to 'abstain' by the bridge).
        window = v14.forecast_visible_window(
            hud.features, horizon_s=.080, previous_hud=source._previous_hud
        )
        update = source.observe(hud)
        expected = window.status if window.eligible else "abstain"
        if update.status != expected:
            raise RuntimeError("raw/bridged v1.4 eligibility mismatch")
        now = step * .010
        collected.append({
            "step": step,
            "local_clock_s": now,
            "raw_status": window.status,
            "status": update.status,
            "covered_until_s": window.covered_until_s,
            "new_ids": list(update.new_ids),
            "canceled_ids": list(update.canceled_ids),
            "elapsed_ids": list(update.elapsed_ids),
            "predicted_due_s": [
                round(now + delta, 9) for delta in window.times_s
            ],
            "pending": [
                {"local_id": token.local_id,
                 "due_local_s": token.due_local_s,
                 "confirmed_now": token.confirmed_now}
                for token in update.pending
            ],
            "tile_bpm": observation.tile_bpm,
            "real_bpm": observation.real_bpm,
            "orbit_xy": [
                observation.contact.orbiting_x,
                observation.contact.orbiting_y
            ],
            "visible_floors": _visible_floors(observation),
        })
        outcome = env.step(ContactAction((0.0,) * 12))
        if outcome.done:
            break
        observation = outcome.observation
    else:
        raise RuntimeError("HUD-only observation collection did not terminate")
    return collected


def _first_seen(rows: list[dict], truth_due: float):
    observed = [
        (row, token)
        for row in rows
        for token in row["pending"]
        if token["confirmed_now"]
        and abs(token["due_local_s"] - truth_due) <= TARGET_TOLERANCE_S
        and row["local_clock_s"] <= truth_due + 1e-9
    ]
    if not observed:
        return None
    first, token = min(observed, key=lambda pair: pair[0]["step"])
    return {
        "frame": first["step"],
        "clock_s": first["local_clock_s"],
        "lead_ms": round((truth_due - first["local_clock_s"]) * 1000., 3),
        "due_error_ms": round(
            (token["due_local_s"] - truth_due) * 1000., 6
        ),
        "local_id": token["local_id"],
    }


def summarize_posthoc(rows: list[dict], segment, name: str,
                      selected: tuple[int, ...] | None = None) -> dict:
    """Truth used here only AFTER rendering the entire HUD prefix stream."""
    targets = tuple(float(t.episode_time_s) for t in segment.targets)
    chosen = (
        tuple(range(len(targets))) if selected is None else tuple(selected)
    )
    summaries = []
    for ordinal in chosen:
        if ordinal < 0 or ordinal >= len(targets):
            raise ValueError("selected target ordinal out of range")
        due = targets[ordinal]
        leading = [
            row for row in rows
            if due - PRE_WINDOW_S <= row["local_clock_s"] <= due + POST_WINDOW_S
        ]
        reasons_before = Counter(
            row["raw_status"] for row in leading
            if row["local_clock_s"] <= due
        )
        first = _first_seen(rows, due)
        summaries.append({
            "target_ordinal": ordinal,
            "true_due_s_posthoc": due,
            "first_confirmed": first,
            "normal_motor_lead_sufficient": (
                first is not None and first["lead_ms"] >= MIN_MOTOR_LEAD_S * 1000
            ),
            "raw_status_counts_in_110ms_before": dict(reasons_before),
            "frames": leading,
        })
    return {
        "format": REPORT_FORMAT,
        "case": name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hud_sample_count": len(rows),
        "total_target_count_posthoc": len(targets),
        "horizon_s": .080,
        "frozen_motor_normal_lead_ms": MIN_MOTOR_LEAD_S * 1000,
        "selected_target_results": summaries,
        "interpretation": (
            "One idle-Motor HUD-only stream; v1.4 reads no Motor state. "
            "Target dues were used only after the HUD stream was collected. "
            "No Oracle timing was passed to the tracker or planner."
        ),
    }


def compare_to_saved_arm(report: dict, arm_path: Path) -> dict:
    """Flag possible disagreement with actual Motor-run HUD first-seen records."""
    data = json.loads(arm_path.read_text(encoding="utf-8"))
    if data.get("arm") not in ("hud_network", "hud_center_first"):
        raise ValueError("comparison requires a saved HUD-only arm JSON")
    existing = data["first_seen_predictions"]
    mismatched = []
    for item in report["selected_target_results"]:
        due = item["true_due_s_posthoc"]
        candidates = [
            row for row in existing
            if abs(row["due_local_s"] - due) <= TARGET_TOLERANCE_S
        ]
        stored = (
            min(candidates, key=lambda row: row["first_seen_s"])
            if candidates else None
        )
        now = item["first_confirmed"]
        if (stored is None) != (now is None):
            mismatched.append(item["target_ordinal"])
        elif stored is not None and now is not None:
            if (abs(stored["first_seen_s"] - now["clock_s"]) > 1e-9
                or abs(stored["due_local_s"] - due) > TARGET_TOLERANCE_S):
                mismatched.append(item["target_ordinal"])
    return {
        "reference": str(arm_path),
        "matched": not mismatched,
        "mismatched_target_ordinals": mismatched,
        "note": (
            "First-seen clocks must match the real HUD Motor run; "
            "otherwise idle-renderer equivalence is not established."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--train-chart", type=Path, default=DEFAULT_TRAIN_CHART)
    parser.add_argument("--targets", default=None,
                        help="comma-separated posthoc target ordinals; default all")
    parser.add_argument("--compare-report", type=Path, default=None,
                        help="saved hud_network or hud_center_first arm JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; refusing overwrite")
    segment = load_case(args.case, args.train_chart)
    chosen = None if args.targets is None else tuple(
        int(i.strip()) for i in args.targets.split(",") if i.strip()
    )
    frames = collect_visible_hud(segment)
    report = summarize_posthoc(frames, segment, args.case, chosen)
    if args.compare_report is not None:
        report["idle_to_motor_hud_parity"] = compare_to_saved_arm(
            report, args.compare_report
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    for target in report["selected_target_results"]:
        first = target["first_confirmed"]
        print("TARGET", target["target_ordinal"],
              "due", round(target["true_due_s_posthoc"], 6),
              "first", first,
              "lead_44ms_ok", target["normal_motor_lead_sufficient"],
              "reasons", target["raw_status_counts_in_110ms_before"],
              flush=True)
        for row in target["frames"]:
            if row["raw_status"] != "eligible" or row["new_ids"]:
                print("  OBS",
                      round(row["local_clock_s"], 4),
                      row["raw_status"],
                      "tile", round(row["tile_bpm"], 3),
                      "real", round(row["real_bpm"], 3),
                      "new", row["new_ids"],
                      "forecast", row["predicted_due_s"],
                      "floors", [
                          (f["relative_index"], f["markers"])
                          for f in row["visible_floors"]
                          if f["markers"] or f["relative_index"] in (0, 1)
                      ], flush=True)
    if "idle_to_motor_hud_parity" in report:
        print("HUD parity:", report["idle_to_motor_hud_parity"], flush=True)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
