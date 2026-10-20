from __future__ import annotations

"""Fixed Train-bool HUD v1.4 vs HUD v3 Countdown Motor comparison."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor,
    StockCountdownRenderer,
)
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.hud_v14_forecast_source_v02 import HUDV14ForecastSource
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
)
from dmdod.planning.timing_fingering_motor_policy_v02 import (
    TimingFingeringMotorPolicy,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_chart_fingering_replay_v02 as frozen


FORMAT = "dmdod-countdown-hud-v3-motor-phase3-v0.1"
DEFAULT_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)
DEFAULT_CHECKPOINT = Path(
    "checkpoints/fingering/fingering_network_v01-4fa2ae1.pt"
)
ARMS = ("hud_v14", "hud_v3_composite")
CONTROL_DT_S = .010
PHYSICS_DT_S = .001


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _reservation_rows(policy: TimingFingeringMotorPolicy) -> list[dict]:
    return [{
        "local_id": item.local_id,
        "due_local_s": item.due_local_s,
        "axes": list(item.axes),
        "phase": item.phase.value,
        "confirmed": item.confirmed,
        "down_axes": sorted(item.down_axes),
        "up_axes": sorted(item.up_axes),
        "retired_reason": item.retired_reason,
    } for item in policy.scheduler.reservations]


def _stats(env) -> dict:
    stats = env.stats
    return {
        "targets": stats.targets,
        "hits": stats.hits,
        "misses": stats.misses,
        "too_early": stats.too_early_presses,
        "overloaded": bool(stats.overloaded),
        "overload_counter": stats.overload_counter,
        "x_accuracy_percent": stats.x_accuracy_percent,
        "timing_mae_ms": stats.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
    }


def _target_outcomes(segment, env) -> list[dict]:
    """Evaluator-only join after the policy/Motor loop is complete."""
    accepted = {}
    for row in env.judgement_trace:
        margin = row["margin"]
        if margin not in (None, "too_early", "fail_overload"):
            accepted.setdefault(row["target_index"], row)
    return [{
        "ordinal": ordinal,
        "due_s": float(target.episode_time_s),
        "hit": bool(env._used[ordinal]),
        "miss": bool(env._missed[ordinal]),
        "keydown_s": (
            None if ordinal not in accepted
            else accepted[ordinal]["time_s"]
        ),
        "timing_error_ms": (
            None if ordinal not in accepted
            else round(
                (accepted[ordinal]["time_s"] - target.episode_time_s) * 1000,
                6,
            )
        ),
        "judgement": (
            "miss" if env._missed[ordinal]
            else None if ordinal not in accepted
            else accepted[ordinal]["margin"]
        ),
        "contact": (
            None if ordinal not in accepted else accepted[ordinal]["key"]
        ),
    } for ordinal, target in enumerate(segment.targets)]


def simulate(segment, model, arm: str, *, wall_timeout_s: float,
             lifecycle_version: str = LIFECYCLE_V14,
             midspin_recovery: bool = False,
             twirl_recovery: bool = False,
             checkpoint_recovery: bool = False,
             capture_judgement_trace: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown fixed comparison arm")
    if arm != "hud_v3_composite" and (
        midspin_recovery or twirl_recovery or checkpoint_recovery
    ):
        raise ValueError("recovery flags require hud_v3_composite arm")
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S,
        physics_dt_s=PHYSICS_DT_S,
        tail_s=.040,
    )
    observation = env.reset()
    policy = TimingFingeringMotorPolicy(
        model=model,
        style_alpha=.5,
        selector="network",
        early_lead=True,
    )
    v14 = HUDV14ForecastSource(horizon_s=.080) if arm == "hud_v14" else None
    v3 = (
        HUDV3CompositeForecastSource(
            horizon_s=.080,
            lifecycle_version=lifecycle_version,
            midspin_recovery=midspin_recovery,
            twirl_recovery=twirl_recovery,
            checkpoint_recovery=checkpoint_recovery,
        )
        if arm == "hud_v3_composite" else None
    )
    renderer = sensor = None
    if v3 is not None:
        renderer = StockCountdownRenderer(
            countdown_ticks=segment.chart.countdown_ticks,
            first_tick_s=0.0,
            tick_period_s=(
                60.0 / segment.chart.initial_bpm / segment.pitch_ratio
            ),
            go_duration_s=60.0 / segment.chart.initial_bpm,
        )
        sensor = CountdownVisualSensor()

    frames = []
    physical_keydowns = []
    logical_keydowns = []
    status_counts = Counter()
    started = time.monotonic()
    maximum = int((segment.duration_s + 2.0) / CONTROL_DT_S) + 200
    for step in range(maximum):
        if time.monotonic() - started > wall_timeout_s:
            raise TimeoutError(f"{arm} exceeded {wall_timeout_s}s")
        now_s = step * CONTROL_DT_S
        raw = cue = None
        if v14 is not None:
            forecast = v14.observe(v14.hud.encode(step, observation))
        else:
            cue = sensor.observe(
                step=step,
                observed_at_s=now_s,
                frame=renderer.render(now_s),
            )
            forecast, raw = v3.observe(step, observation, cue)
        status_counts[forecast.status] += 1
        decision_count_before = len(policy.decisions)
        action = policy.command(forecast, env.motor)
        reservations_after_command = _reservation_rows(policy)
        outcome = env.step(action)
        transition = env.motor.motor.last_transition
        physical_edges = [{
            "time_s": event.time_s,
            "axis": event.key,
            "edge": event.event.value,
        } for event in transition.evaluator_events]
        logical_edges = [{
            "time_s": event.time_s,
            "contact": event.key,
            "edge": event.event.value,
        } for event in env.last_events]
        physical_keydowns.extend(
            item for item in physical_edges if item["edge"] == KeyEvent.DOWN.value
        )
        logical_keydowns.extend(
            item for item in logical_edges if item["edge"] == KeyEvent.DOWN.value
        )
        policy.observe_physics(env.motor)
        frames.append({
            "step": step,
            "local_clock_s": now_s,
            "countdown_cue": None if cue is None else cue.kind.value,
            "countdown_valid": None if cue is None else cue.valid,
            "timing_status": forecast.status,
            "timing_abstention_reason": forecast.abstention_reason,
            "new_ids": list(forecast.new_ids),
            "reused_ids": list(forecast.reused_ids),
            "canceled_ids": list(forecast.canceled_ids),
            "elapsed_ids": list(forecast.elapsed_ids),
            "pending": [{
                "local_id": item.local_id,
                "due_local_s": item.due_local_s,
                "uncertainty_s": item.uncertainty_s,
                "due_lower_s": (
                    item.due_local_s
                    - (0.0 if item.uncertainty_s is None else item.uncertainty_s)
                ),
                "confirmed_now": item.confirmed_now,
                "confirmations": item.confirmations,
            } for item in forecast.pending],
            "composite_status": None if raw is None else raw.status,
            "composite_abstention_reason": (
                None if raw is None else raw.abstention_reason
            ),
            "reservations_after_command": reservations_after_command,
            "new_decisions": [{
                "local_id": item.local_id,
                "step": item.step,
                "due_local_s": item.due_local_s,
                "choice": item.choice,
                "reason": item.reason,
                "extra_lead_s": item.extra_lead_s,
            } for item in policy.decisions[decision_count_before:]],
            "physical_edges": physical_edges,
            "logical_edges": logical_edges,
            "reservations_after_physics": _reservation_rows(policy),
        })
        if outcome.done:
            break
        observation = outcome.observation
    else:
        raise RuntimeError(f"{arm} exceeded bounded step count")

    result = {
        "arm": arm,
        "recovery_flags": {
            "midspin": midspin_recovery,
            "twirl": twirl_recovery,
            "checkpoint": checkpoint_recovery,
        },
        "lifecycle_version": (
            lifecycle_version if arm == "hud_v3_composite" else None
        ),
        "stats": _stats(env),
        "frames": frames,
        "frame_count": len(frames),
        "run_wall_s": round(time.monotonic() - started, 6),
        "timing_status_counts": dict(status_counts),
        "decisions": [{
            "local_id": item.local_id,
            "step": item.step,
            "decision_s": item.step * CONTROL_DT_S,
            "due_local_s": item.due_local_s,
            "choice": item.choice,
            "reason": item.reason,
            "extra_lead_s": item.extra_lead_s,
        } for item in policy.decisions],
        "physical_keydowns": physical_keydowns,
        "logical_keydowns": logical_keydowns,
        "scheduling": {
            "insufficient_lead": policy.insufficient_observable_lead,
            "unassigned": policy.unassigned,
            "reservation_collisions": policy.finger_collisions,
            "inflight_cancellations": policy.inflight_cancellations,
            "abstained_frames": policy.hud_abstentions,
            "pending_reservations_at_end": len(policy.scheduler.reservations),
        },
    }
    # Evaluator truth is joined only after the final action and physical step.
    result["target_outcomes_posthoc"] = _target_outcomes(segment, env)
    if capture_judgement_trace:
        # Evaluator-owned target indices are only exposed AFTER the complete
        # physical replay. Never feed these rows into Timing or Motor.
        result["judgement_trace_posthoc"] = [
            dict(row) for row in env.judgement_trace
        ]
    result["source_conflict_count"] = sum(
        row["composite_status"] in (
            "source_conflict", "countdown_interval_conflict"
        ) for row in frames
    )
    result["first_target_timing_id_posthoc"] = next((
        item["local_id"]
        for row in frames
        for item in row["pending"]
        if item["due_lower_s"] - 1e-9
        <= float(segment.targets[0].episode_time_s)
        <= item["due_local_s"]
           + (0.0 if item["uncertainty_s"] is None else item["uncertainty_s"])
           + 1e-9
    ), None)
    return result


def run(chart: Path, checkpoint: Path, *, wall_timeout_s: float) -> dict:
    if not chart.is_file() or "train" not in {p.lower() for p in chart.parts}:
        raise ValueError("fixed comparison accepts Train bool only")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    segment = build_playable_segment(
        compile_adofai(load_adofai(chart)), end_s=5.0
    )
    model = frozen.load_model(checkpoint)
    arms = {}
    for arm in ARMS:
        print("starting", arm, flush=True)
        arms[arm] = simulate(
            segment, model, arm, wall_timeout_s=wall_timeout_s
        )
        print(arm, arms[arm]["stats"], flush=True)
    baseline = arms["hud_v14"]["stats"]
    candidate = arms["hud_v3_composite"]["stats"]
    gates = {
        "same_target_count": baseline["targets"] == candidate["targets"],
        "v3_first_target_hit": bool(
            arms["hud_v3_composite"]["target_outcomes_posthoc"]
            and arms["hud_v3_composite"]["target_outcomes_posthoc"][0]["hit"]
        ),
        "v3_no_source_conflict": (
            arms["hud_v3_composite"]["source_conflict_count"] == 0
        ),
        "v3_no_overload": not candidate["overloaded"],
        "v3_too_early_not_worse": candidate["too_early"] <= baseline["too_early"],
        "hit_improvement": candidate["hits"] > baseline["hits"],
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "chart": chart.name,
        "chart_sha256": _hash(chart),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _hash(checkpoint),
        "scope_s": 5.0,
        "control_dt_s": CONTROL_DT_S,
        "physics_dt_s": PHYSICS_DT_S,
        "fixed_policy": {
            "selector": "network",
            "style_alpha": .5,
            "normal_lead_s": .044,
            "decision_lead_s": .069,
        },
        "gates": gates,
        "delta_v3_minus_v14": {
            "hits": candidate["hits"] - baseline["hits"],
            "misses": candidate["misses"] - baseline["misses"],
            "too_early": candidate["too_early"] - baseline["too_early"],
            "physical_keydowns": (
                candidate["physical_keydowns"] - baseline["physical_keydowns"]
            ),
        },
        "arms": arms,
        "causality": (
            "Both policies received only TimingFrameUpdate plus current Motor. "
            "Target truth and judgement_trace were read only after simulation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = run(
        args.chart, args.checkpoint, wall_timeout_s=args.wall_timeout_s
    )
    output = args.output_dir / "comparison.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)
    print("status:", result["status"], "gates:", result["gates"])
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
