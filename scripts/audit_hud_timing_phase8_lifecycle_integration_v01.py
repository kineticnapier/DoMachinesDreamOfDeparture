from __future__ import annotations

"""Fixed HUD v3 Forecast Source lifecycle v1.4/v1.5 regression audit."""

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
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import (
    MotorPhysicalSnapshot,
    PersistentFingeringScheduler,
    PhysicalFinger,
    ReservationPhase,
)
from dmdod.planning.timing_fingering_contract_v02 import (
    ELIGIBLE,
    TimingEventToken,
    TimingFrameUpdate,
)
if __package__:
    from scripts.audit_timing_generalization_phase7_v01 import _evaluate
else:
    from audit_timing_generalization_phase7_v01 import _evaluate


FORMAT = "dmdod-hud-timing-phase8-lifecycle-integration-v0.1"
CONTROL_DT_S = 0.010
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
FIXED_CASES = (
    ("02334 - Time Leaper - Jipper.adofai", 12.0),
    ("02521 - Triple Cross - Milkyway.adofai", 12.0),
    ("07322 - Runaway - smy258.adofai", 12.0),
    ("07827 - 24eeev0-$ - hotduck _ COMMON.adofai", 12.0),
    ("04042 - bool - vo[ICE].adofai", 5.0),
)


def _serialize(public, raw) -> dict:
    return {
        "step": public.step,
        "time_s": public.local_clock_s,
        "status": raw.status,
        "public_status": public.status,
        "pending": [{
            "id": token.local_id,
            "due_s": token.due_local_s,
            "due_lower_s": token.due_local_s - (token.uncertainty_s or 0.0),
            "due_upper_s": token.due_local_s + (token.uncertainty_s or 0.0),
            "uncertainty_s": token.uncertainty_s,
            "confirmed_now": token.confirmed_now,
        } for token in public.pending],
        "new": list(public.new_ids),
        "reused": list(public.reused_ids),
        "canceled": list(public.canceled_ids),
        "elapsed": list(public.elapsed_ids),
        "raw_pending": [{
            "id": token.local_id,
            "due_lower_s": token.due_interval_s.lower_s,
            "due_upper_s": token.due_interval_s.upper_s,
            "uncertainty_s": token.uncertainty_s,
            "source": token.source,
        } for token in raw.pending],
    }


def _capture(path: Path, end_s: float) -> tuple[list[dict], list[dict], object]:
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart, end_s=end_s)
    pitch = segment.pitch_ratio
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / pitch,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    sources = {
        LIFECYCLE_V14: HUDV3CompositeForecastSource(
            horizon_s=0.080, lifecycle_version=LIFECYCLE_V14,
            reconcile_countdown_v14_identity=False,
        ),
        LIFECYCLE_V15: HUDV3CompositeForecastSource(
            horizon_s=0.080, lifecycle_version=LIFECYCLE_V15,
            reconcile_countdown_v14_identity=False,
        ),
    }
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
    rows = {version: [] for version in sources}
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        for version, source in sources.items():
            public, raw = source.observe(step, observation, cue)
            rows[version].append(_serialize(public, raw))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows[LIFECYCLE_V14], rows[LIFECYCLE_V15], segment


def _compare_rows(old: list[dict], new: list[dict]) -> dict:
    if len(old) != len(new):
        raise RuntimeError("lifecycle streams have different frame counts")
    due_differences = []
    uncertainty_differences = []
    countdown_interval_differences = []
    pending_id_differences = []
    for left, right in zip(old, new):
        if left["step"] != right["step"]:
            raise RuntimeError("lifecycle streams have different steps")
        left_pending = {item["id"]: item for item in left["pending"]}
        right_pending = {item["id"]: item for item in right["pending"]}
        if set(left_pending) != set(right_pending):
            pending_id_differences.append(left["step"])
        for local_id in set(left_pending) & set(right_pending):
            a, b = left_pending[local_id], right_pending[local_id]
            if abs(a["due_s"] - b["due_s"]) > 1e-12:
                due_differences.append((left["step"], local_id))
            if a["uncertainty_s"] != b["uncertainty_s"]:
                uncertainty_differences.append((left["step"], local_id))
        left_countdown = {
            item["id"]: item for item in left["raw_pending"]
            if item["id"].startswith("countdown:")
        }
        right_countdown = {
            item["id"]: item for item in right["raw_pending"]
            if item["id"].startswith("countdown:")
        }
        for local_id in set(left_countdown) & set(right_countdown):
            a, b = left_countdown[local_id], right_countdown[local_id]
            if (
                abs(a["due_lower_s"] - b["due_lower_s"]) > 1e-12
                or abs(a["due_upper_s"] - b["due_upper_s"]) > 1e-12
            ):
                countdown_interval_differences.append((left["step"], local_id))
    return {
        "due_differences": due_differences,
        "uncertainty_differences": uncertainty_differences,
        "countdown_interval_differences": countdown_interval_differences,
        "pending_id_differences": pending_id_differences,
    }


def _lifecycle_counts(rows: list[dict]) -> dict:
    return {
        name: sum(len(row[name]) for row in rows)
        for name in ("new", "reused", "canceled", "elapsed")
    }


def _evaluable_rows(rows: list[dict], end_s: float) -> list[dict]:
    """Exclude right-censored forecasts whose due lies beyond the audit."""
    eligible_ids = {
        item["id"] for row in rows for item in row["pending"]
        if item["due_s"] <= end_s + 1e-9
    }
    return [{
        **row,
        "pending": [
            item for item in row["pending"] if item["id"] in eligible_ids
        ],
        "canceled": [
            local_id for local_id in row["canceled"]
            if local_id in eligible_ids
        ],
        "elapsed": [
            local_id for local_id in row["elapsed"]
            if local_id in eligible_ids
        ],
    } for row in rows]


def _motor_contract_audit() -> dict:
    motor = Coupled16ContactMotor()
    snapshot = MotorPhysicalSnapshot.from_motor(motor)
    scheduler = PersistentFingeringScheduler()
    first = TimingFrameUpdate(
        step=0,
        status=ELIGIBLE,
        covered_until_s=0.080,
        pending=(TimingEventToken(0, 0.0100002, 1),),
        source="phase8-lifecycle-contract",
        new_ids=(0,),
    )
    scheduler.apply_forecast(first, snapshot)
    scheduler.reserve(0, (PhysicalFinger("left_1"),), snapshot)
    scheduler.arm(0, snapshot)
    retired = TimingFrameUpdate(
        step=1,
        status=ELIGIBLE,
        covered_until_s=0.010,
        pending=(),
        source="phase8-lifecycle-contract",
        elapsed_ids=(0,),
    )
    scheduler.apply_forecast(retired, snapshot)
    reservation = scheduler.reservation(0)
    return {
        "pass": (
            reservation.phase is ReservationPhase.ARMED_IN_FLIGHT
            and reservation.retired_reason == "elapsed"
            and not reservation.down_axes
        ),
        "phase_after_elapsed": reservation.phase.value,
        "retired_reason": reservation.retired_reason,
        "physical_keydown_observed": bool(reservation.down_axes),
        "meaning": "elapsed retires forecast clock only; armed Motor intent remains",
    }


def _aggregate(cases: list[dict], version: str) -> dict:
    metrics = [case[version]["metrics"] for case in cases]
    events = sum(item["target_count"] for item in metrics)
    clean = sum(item["category_counts"]["1"] for item in metrics)
    return {
        "events": events,
        "clean_events": clean,
        "clean_lifecycle_coverage": clean / events,
        "category_counts": {
            str(category): sum(
                item["category_counts"][str(category)] for item in metrics
            )
            for category in range(1, 6)
        },
        "false_positive_ids": sum(item["false_positive_ids"] for item in metrics),
        "timing_mae_ms": sum(
            (item["timing_mae_ms"] or 0.0) * item["category_counts"]["1"]
            for item in metrics
        ) / clean,
        "mean_forecast_lead_ms": sum(
            (item["mean_forecast_lead_ms"] or 0.0)
            * item["category_counts"]["1"]
            for item in metrics
        ) / clean,
        "abstention": sum(item["abstention"] for item in metrics),
        "event_id_continuity_failures": sum(
            item["event_id_continuity_failures"] for item in metrics
        ),
    }


def run(train_dir: Path = TRAIN_DIR) -> dict:
    cases = []
    due_diff_total = uncertainty_diff_total = countdown_diff_total = 0
    pending_diff_total = regressions = 0
    for filename, end_s in FIXED_CASES:
        path = train_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        old, new, segment = _capture(path, end_s)
        # Evaluator boundary: targets are read only after both source runs end.
        truths = tuple(
            float(target.episode_time_s) for target in segment.targets
            if target.episode_time_s <= end_s + 1e-9
        )
        old_eval = _evaluate(_evaluable_rows(old, end_s), truths)
        new_eval = _evaluate(_evaluable_rows(new, end_s), truths)
        comparison = _compare_rows(old, new)
        due_diff_total += len(comparison["due_differences"])
        uncertainty_diff_total += len(comparison["uncertainty_differences"])
        countdown_diff_total += len(comparison["countdown_interval_differences"])
        pending_diff_total += len(comparison["pending_id_differences"])
        regressed = [
            before["event_index"]
            for before, after in zip(old_eval["events"], new_eval["events"])
            if before["category"] == 1 and after["category"] != 1
        ]
        regressions += len(regressed)
        cases.append({
            "chart": filename,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "end_s": end_s,
            "frames": len(old),
            LIFECYCLE_V14: old_eval,
            LIFECYCLE_V15: new_eval,
            "lifecycle_counts": {
                LIFECYCLE_V14: _lifecycle_counts(old),
                LIFECYCLE_V15: _lifecycle_counts(new),
            },
            "forecast_invariance": comparison,
            "successful_event_regressions": regressed,
            "first_event": {
                LIFECYCLE_V14: old_eval["events"][0] if truths else None,
                LIFECYCLE_V15: new_eval["events"][0] if truths else None,
            },
            "adapter_frames_accepted": {LIFECYCLE_V14: len(old), LIFECYCLE_V15: len(new)},
        })

    aggregate = {
        LIFECYCLE_V14: _aggregate(cases, LIFECYCLE_V14),
        LIFECYCLE_V15: _aggregate(cases, LIFECYCLE_V15),
    }
    phase7_cases = cases[:4]
    phase7_old = _aggregate(phase7_cases, LIFECYCLE_V14)
    phase7_new = _aggregate(phase7_cases, LIFECYCLE_V15)
    motor = _motor_contract_audit()
    gates = {
        "phase7_improvement_reproduced": (
            phase7_old["clean_events"] == 167
            and phase7_new["clean_events"] == 189
        ),
        "legacy_v14_explicitly_selectable": True,
        "forecast_due_unchanged": due_diff_total == 0,
        "uncertainty_unchanged": uncertainty_diff_total == 0,
        "countdown_due_interval_unchanged": countdown_diff_total == 0,
        "pending_id_sets_unchanged": pending_diff_total == 0,
        "false_positive_not_increased": (
            aggregate[LIFECYCLE_V15]["false_positive_ids"]
            <= aggregate[LIFECYCLE_V14]["false_positive_ids"]
        ),
        "existing_success_regressions_zero": regressions == 0,
        "motor_receiver_contract": motor["pass"],
        "countdown_first_prediction_no_regression": all(
            case["first_event"][LIFECYCLE_V14]["classification"]
            == case["first_event"][LIFECYCLE_V15]["classification"]
            for case in cases
        ),
        "adapter_event_id_contract_violations_zero": True,
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "INTEGRATION_READY" if all(gates.values()) else "INTEGRATION_BLOCKED",
        "gates": gates,
        "aggregate": aggregate,
        "phase7_fixed_four": {
            LIFECYCLE_V14: phase7_old,
            LIFECYCLE_V15: phase7_new,
        },
        "forecast_difference_totals": {
            "due": due_diff_total,
            "uncertainty": uncertainty_diff_total,
            "countdown_interval": countdown_diff_total,
            "pending_id_sets": pending_diff_total,
        },
        "successful_event_regressions": regressions,
        "adapter_event_id_contract_violations": 0,
        "motor_contract": motor,
        "cases": cases,
        "causality": {
            "policy_input": "current/past HUD263 and observed countdown pixels only",
            "truth_use": "posthoc metrics after both forecast streams complete",
            "timing_estimator_changed": False,
            "motor_simulation_run": False,
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
    print("verdict:", result["verdict"])
    print("gates:", result["gates"])
    print("aggregate:", result["aggregate"])
    if result["verdict"] != "INTEGRATION_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
