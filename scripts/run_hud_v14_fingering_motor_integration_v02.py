from __future__ import annotations

"""Frozen HUD-v1.4/Oracle Timing -> Fingering -> 12-axis Motor smoke runner.

Only the evaluator/renderer and *Oracle comparator* can read chart truth.
The shared TimingFingeringMotorPolicy accepts ONLY TimingFrameUpdate+Motor.
No validation/test/qualification/reserve/final chart is selected by default.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_chart_fingering_replay_v02 as frozen

from dmdod.adofai.chart import load_adofai, parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.planning.hud_v14_forecast_source_v02 import HUDV14ForecastSource
from dmdod.planning.oracle_forecast_source_v02 import OracleDue, OracleForecastSource
from dmdod.planning.timing_fingering_motor_policy_v02 import TimingFingeringMotorPolicy
from dmdod.planning.persistent_fingering_scheduler_v01 import MotorPhysicalSnapshot
from dmdod.motor.contact_16k import PHYSICAL_AXES


FORMAT = "dmdod-hud-v14-fingering-motor-integration-v0.2"
FIXTURE_BPMS = (120.0, 240.0, 480.0)
ARMS = (
    ("oracle_network", "oracle", "network"),
    ("hud_network", "hud", "network"),
    ("hud_center_first", "hud", "center_first"),
)
DEFAULT_CHECKPOINT = Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt")
DEFAULT_TRAIN_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)


def synthetic_segment(bpm: float):
    """Predeclared ordinary, single-input fixture, no hidden special actions."""
    if bpm not in FIXTURE_BPMS:
        raise ValueError("fixture BPM must be one of 120, 240, 480")
    raw = json.dumps({
        "angleData": [0.0, 90.0, 180.0, 270.0] * 12,
        "settings": {"bpm": bpm, "pitch": 100.0, "countdownTicks": 0},
        "actions": [],
    })
    return build_playable_segment(
        compile_adofai(parse_adofai_text(raw))
    )


def _arm_stats(env) -> dict:
    s = env.stats
    return {
        "targets": s.targets, "hits": s.hits, "misses": s.misses,
        "too_early": s.too_early_presses,
        "overloaded": bool(s.overloaded),
        "x_accuracy_percent": s.x_accuracy_percent,
        "timing_mae_ms": s.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
    }


def _forecast_posthoc(tokens: list[dict], segment) -> dict:
    """Never executed before the entire policy loop has finished."""
    truth = sorted(float(target.episode_time_s) for target in segment.targets)
    predicted = sorted(tokens, key=lambda x: x["first_seen_s"])
    unique = {}
    for item in predicted:
        unique.setdefault(item["local_id"], item)
    predicted = list(unique.values())
    actual = [row["due_local_s"] for row in predicted]
    out = {"unique_forecasts": len(predicted), "true_targets": len(truth)}
    # Distinct-ID forecasts count once. Greedy one-to-one matching is sorted
    # by error, never repeated per-frame reconfirmation.
    for tolerance_ms in (1, 5, 10, 20):
        candidates = sorted(
            (abs(t - p), pi, ti)
            for pi, p in enumerate(actual)
            for ti, t in enumerate(truth)
            if abs(t - p) <= tolerance_ms * .001 + 1e-12
        )
        used_p, used_t = set(), set()
        for _, pi, ti in candidates:
            if pi not in used_p and ti not in used_t:
                used_p.add(pi)
                used_t.add(ti)
        out[f"matched_{tolerance_ms}ms"] = len(used_p)
        out[f"precision_{tolerance_ms}ms"] = (
            len(used_p) / len(actual) if actual else None
        )
        out[f"recall_{tolerance_ms}ms"] = (
            len(used_t) / len(truth) if truth else None
        )
    out["first_seen_lead_ms"] = [
        round((row["due_local_s"] - row["first_seen_s"]) * 1000, 3)
        for row in predicted
    ]
    return out


def _supported_posthoc(first_seen: list[dict], decisions: list,
                        env, segment) -> dict:
    """Scorer-only attribution after actions; do not pass truth to the policy.

    Success means the associated truth ordinal was scored Hit by the ordered
    evaluator. It does NOT claim the intended physical finger caused that Hit.
    """
    truth = [(i, float(event.episode_time_s))
             for i, event in enumerate(segment.targets)]
    seen_truth: set[int] = set()
    decided = {d.local_id: d for d in decisions}
    eligible = []
    for forecast in sorted(first_seen, key=lambda x: x["due_local_s"]):
        due = forecast["due_local_s"]
        lead = due - forecast["first_seen_s"]
        choice = decided.get(forecast["local_id"])
        if lead + 1e-9 < .044 or choice is None or choice.reason != "assigned":
            continue
        matches = sorted((abs(due - target_due), index) for index, target_due in truth
                         if index not in seen_truth and abs(due-target_due) <= .020)
        if not matches:
            continue
        _, index = matches[0]
        seen_truth.add(index)
        eligible.append(index)
    hits = sum(bool(env._used[i]) for i in eligible)
    return {
        "sufficient_lead_supported_truth_events": len(eligible),
        "supported_truth_hit_by_ordered_scorer": hits,
        "supported_truth_misses": len(eligible) - hits,
        "sufficient_lead_supported_hit_fraction": (
            hits / len(eligible) if eligible else None
        ),
        "truth_matching_tolerance_ms": 20,
        "warning": (
            "Posthoc only; ordered scorer may assign a KeyDown from another "
            "physical finger to this ordinal."
        ),
    }


def _diagnose_misses_posthoc(segment, env, first_seen: list[dict],
                              decisions: list, physical_switch_trace: list[dict],
                              forecast_lifecycle_trace: list[dict]) -> list[dict]:
    """Cross-reference missed target ordinals AFTER policy and Motor finish.

    The nearest forecast is a diagnostic *candidate*, not a proof that a
    particular local ID or finger was intended to hit this scorer target.
    """
    decisions_by_id = {item.local_id: item for item in decisions}
    rows = []
    for index, target in enumerate(segment.targets):
        if not env._missed[index]:
            continue
        due = float(target.episode_time_s)
        nearest = min(
            first_seen,
            key=lambda row: abs(row["due_local_s"] - due),
            default=None,
        )
        if nearest is not None and abs(nearest["due_local_s"] - due) > .020:
            nearest = None
        decision = (
            decisions_by_id.get(nearest["local_id"])
            if nearest is not None else None
        )
        surrounding_keys = [
            dict(item) for item in physical_switch_trace
            if abs(item["time_s"] - due) <= .10
            and "down" in str(item["edge"]).lower()
        ]
        statuses = [
            {"step": item["step"], "status": item["status"],
             "reason": item["abstention_reason"]}
            for item in forecast_lifecycle_trace
            if due - .100 <= item["local_clock_s"] <= due
        ]
        if nearest is None:
            category = "no_forecast_within_20ms"
        elif decision is None:
            category = "forecast_without_decision"
        elif decision.reason != "assigned":
            category = decision.reason
        else:
            category = "assigned_but_scorer_missed_check_physical_trace"
        rows.append({
            "target_ordinal": index, "target_due_s": due,
            "category": category,
            "nearest_local_forecast": (
                {"local_id": nearest["local_id"],
                 "forecast_due_s": nearest["due_local_s"],
                 "due_error_ms": round(
                     (nearest["due_local_s"] - due) * 1000, 3
                 ),
                 "first_seen_lead_ms": round(
                     (nearest["due_local_s"] - nearest["first_seen_s"]) * 1000, 3
                 )} if nearest is not None else None
            ),
            "decision": (
                {"choice": decision.choice, "reason": decision.reason,
                 "step": decision.step, "extra_lead_s": decision.extra_lead_s}
                if decision is not None else None
            ),
            "nearby_physical_keydowns": surrounding_keys,
            "recent_forecast_statuses": statuses,
        })
    return rows


def simulate(segment, model, arm: str, *, wall_timeout_s: float = 180.0,
             trace_axis: str | None = None,
             trace_window_s: tuple[float, float] | None = None):
    source_kind = next((src for name, src, _ in ARMS if name == arm), None)
    selector = next((sel for name, _, sel in ARMS if name == arm), None)
    if source_kind is None or selector is None:
        raise ValueError(f"unknown comparison arm: {arm}")
    if (trace_axis is None) != (trace_window_s is None):
        raise ValueError("diagnostic trace requires both axis and window")
    if trace_axis is not None:
        if trace_axis not in PHYSICAL_AXES:
            raise ValueError("diagnostic trace axis must be a 12-axis Motor actuator")
        from math import isfinite
        if (len(trace_window_s) != 2 or not all(isfinite(x) for x in trace_window_s)
                or trace_window_s[0] < 0 or trace_window_s[1] <= trace_window_s[0]):
            raise ValueError("invalid diagnostic trace time window")
    env = HudCoupledContactRealChartEnv(
        segment, control_dt_s=.010, physics_dt_s=.001,
        tail_s=.040,
    )
    obs = env.reset()
    if source_kind == "oracle":
        # One-way privileged boundary. ONLY this constructor sees target truth.
        source = OracleForecastSource(tuple(
            OracleDue(float(target.episode_time_s))
            for target in segment.targets
        ))
    else:
        source = HUDV14ForecastSource(horizon_s=.080)
    policy = TimingFingeringMotorPolicy(
        model=model if selector == "network" else None,
        style_alpha=.5, selector=selector, early_lead=True,
    )
    physical_switch_trace = []
    logical_switch_trace = []
    forecast_lifecycle_trace = []
    raw_new_ids = []
    observed = Counter()
    control_trace = []
    started = time.monotonic()
    max_steps = int((segment.duration_s + 2.0) / .010) + 200
    for step in range(max_steps):
        if time.monotonic() - started > wall_timeout_s:
            raise TimeoutError(f"{arm}: exceeded {wall_timeout_s}s")
        if source_kind == "oracle":
            forecast = source.observe(step)
        else:
            visible = source.hud.encode(step, obs)
            forecast = source.observe(visible)
        for token in forecast.pending:
            if token.local_id in forecast.new_ids:
                raw_new_ids.append({
                    "local_id": token.local_id,
                    "due_local_s": token.due_local_s,
                    "first_seen_s": forecast.local_clock_s,
                    "chord_size": token.chord_size,
                })
        observed[forecast.status] += 1
        if (forecast.new_ids or forecast.canceled_ids or forecast.elapsed_ids
                or forecast.status != "eligible"):
            forecast_lifecycle_trace.append({
                "step": step,
                "local_clock_s": forecast.local_clock_s,
                "status": forecast.status,
                "abstention_reason": forecast.abstention_reason,
                "new_ids": list(forecast.new_ids),
                "canceled_ids": list(forecast.canceled_ids),
                "elapsed_ids": list(forecast.elapsed_ids),
                "pending": [
                    {"local_id": t.local_id, "due_local_s": t.due_local_s,
                     "confirmed_now": t.confirmed_now}
                    for t in forecast.pending
                ],
            })
        capture_axis = (
            trace_axis is not None
            and trace_window_s[0] - 1e-9 <= forecast.local_clock_s
            <= trace_window_s[1] + 1e-9
        )
        before_axis = (MotorPhysicalSnapshot.from_motor(env.motor).axis(trace_axis)
                       if capture_axis else None)
        action = policy.command(forecast, env.motor)
        reserved = ([
            {"local_id": reservation.local_id,
             "phase": reservation.phase.value,
             "confirmed": reservation.confirmed,
             "due_local_s": reservation.due_local_s,
             "retired_reason": reservation.retired_reason,
             "down_axes": sorted(reservation.down_axes)}
            for reservation in policy.scheduler.reservations
            if trace_axis in reservation.axes
        ] if capture_axis else [])
        outcome = env.step(action)
        # These are raw physical contacts, not scorer judgements.
        transition = env.motor.motor.last_transition
        physical_switch_trace.extend({
            "time_s": e.time_s, "axis": e.key, "edge": e.event.value
        } for e in transition.evaluator_events)
        logical_switch_trace.extend({
            "time_s": e.time_s, "key": e.key, "edge": e.event.value
        } for e in env.last_events)
        # The shared policy reads only the physical NKeyMotor transition.
        policy.observe_physics(env.motor)
        if capture_axis:
            after_axis = MotorPhysicalSnapshot.from_motor(env.motor).axis(trace_axis)
            index = PHYSICAL_AXES.index(trace_axis)
            control_trace.append({
                "step": step,
                "local_clock_s": round(forecast.local_clock_s, 6),
                "forecast_status": forecast.status,
                "forecast_pending": [
                    {"local_id": t.local_id, "due_local_s": t.due_local_s,
                     "confirmed_now": t.confirmed_now}
                    for t in forecast.pending
                    if abs(t.due_local_s - forecast.local_clock_s) <= .100
                ],
                "axis": trace_axis,
                "axis_before": {
                    "position_m": before_axis.position_m,
                    "velocity_m_s": before_axis.velocity_m_s,
                    "activation": before_axis.activation,
                    "pressed": before_axis.pressed,
                },
                "axis_command": action.commands[index],
                "reservation_after_command": reserved,
                "axis_after": {
                    "position_m": after_axis.position_m,
                    "velocity_m_s": after_axis.velocity_m_s,
                    "activation": after_axis.activation,
                    "pressed": after_axis.pressed,
                },
                "physical_edges": [
                    {"time_s": event.time_s, "edge": event.event.value}
                    for event in transition.evaluator_events
                    if event.key == trace_axis
                ],
                "reservation_after_physics": [
                    {"local_id": q.local_id, "phase": q.phase.value,
                     "confirmed": q.confirmed, "retired_reason": q.retired_reason}
                    for q in policy.scheduler.reservations if trace_axis in q.axes
                ],
            })
        if outcome.done:
            break
        obs = outcome.observation
    else:
        raise RuntimeError(f"{arm}: bounded simulation did not terminate")
    result = {
        "arm": arm,
        "diagnostic_control_trace": control_trace,
        "timing_source": source_kind,
        "fingering_selector": selector,
        "stats": _arm_stats(env),
        "run_wall_s": round(time.monotonic() - started, 3),
        "frames": step + 1,
        "forecast_status_counts": dict(observed),
        "forecast_new_token_count": len(raw_new_ids),
        "first_seen_predictions": raw_new_ids,
        "forecast_lifecycle_trace": forecast_lifecycle_trace,
        "physical_switch_trace": physical_switch_trace,
        "logical_switch_trace": logical_switch_trace,
        "decisions": [
            {"local_id": d.local_id, "step": d.step,
             "due_local_s": d.due_local_s, "choice": d.choice,
             "reason": d.reason, "extra_lead_s": d.extra_lead_s}
            for d in policy.decisions
        ],
        "scheduling": {
            "insufficient_observable_lead": policy.insufficient_observable_lead,
            "unassigned": policy.unassigned,
            "reservation_collisions": policy.finger_collisions,
            "inflight_cancellations": policy.inflight_cancellations,
            "abstained_control_frames": policy.hud_abstentions,
            "pending_physical_reservations_at_end": len(policy.scheduler.reservations),
        },
        "limitation": (
            "The current v1.4 estimator emits single-contact ordinary event "
            "tokens only; unsupported chords/specials abstain."
        ),
    }
    # Exact target list is used ONLY posthoc, after the final Motor step.
    # Scorer truth remains a separate posthoc record, not policy input.
    result["target_outcomes_posthoc"] = [
        {"ordinal": i, "due_s": float(item.episode_time_s),
         "hit": bool(env._used[i]), "miss": bool(env._missed[i])}
        for i, item in enumerate(segment.targets)
    ]
    result["miss_diagnostics_posthoc"] = _diagnose_misses_posthoc(
        segment, env, raw_new_ids, policy.decisions,
        physical_switch_trace, forecast_lifecycle_trace,
    )
    result["forecast_posthoc"] = _forecast_posthoc(raw_new_ids, segment)
    result["supported_event_posthoc"] = _supported_posthoc(
        raw_new_ids, policy.decisions, env, segment
    )
    return result


def run_fixed(segment, *, checkpoint: Path, label: str,
              timeout: float, arm_filter: str | None = None,
              trace_axis: str | None = None,
              trace_window_s: tuple[float, float] | None = None) -> dict:
    model = frozen.load_model(checkpoint)
    rows = {}
    for name, _, _ in ARMS:
        if arm_filter is not None and name != arm_filter:
            continue
        print(f"{label} {name} starting", flush=True)
        try:
            value = simulate(segment, model, name, wall_timeout_s=timeout,
                             trace_axis=trace_axis, trace_window_s=trace_window_s)
        except Exception as exc:
            value = {"arm": name, "error": f"{type(exc).__name__}: {exc}"}
        rows[name] = value
        if "stats" in value:
            s = value["stats"]
            print(f"  H={s['hits']}/{s['targets']} Miss={s['misses']} "
                  f"TE={s['too_early']} OL={s['overloaded']} "
                  f"X={s['x_accuracy_percent']:.4f}%", flush=True)
            for miss in value.get("miss_diagnostics_posthoc", ()):
                print("  MISS", miss["target_ordinal"],
                      "due=", round(miss["target_due_s"], 4),
                      "category=", miss["category"],
                      "nearest=", miss["nearest_local_forecast"],
                      "decision=", miss["decision"], flush=True)
        else:
            print("  FAILED:", value["error"], flush=True)
    return {"label": label, "arms": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures", action="store_true",
                      help="only 120/240/480 BPM fixed ordinary synthetic charts")
    mode.add_argument("--special-fixtures", action="store_true",
                      help="deterministic SetSpeed/Twirl/Pause transition fixtures")
    mode.add_argument("--train-chart", type=Path,
                      help="explicit TRAIN chart path only; no Validation/Test")
    p.add_argument("--end-s", type=float, default=5.0,
                   help="only applied to --train-chart, relative chart segment end")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--arm", choices=tuple(name for name, _, _ in ARMS),
                   default=None, help="rerun only one failing comparison arm")
    p.add_argument("--case", default=None,
                   help="run only one named fixture/case from the selected mode")
    p.add_argument("--trace-axis", choices=PHYSICAL_AXES,
                   default=None, help="diagnose one physical axis only")
    p.add_argument("--trace-start-s", type=float, default=None)
    p.add_argument("--trace-end-s", type=float, default=None)
    p.add_argument("--wall-timeout-s", type=float, default=180.0)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("output directory exists; refusing overwrite")
    if args.wall_timeout_s <= 0:
        p.error("--wall-timeout-s must be positive")
    if args.trace_axis is None and (
        args.trace_start_s is not None or args.trace_end_s is not None
    ):
        p.error("--trace-axis is required with --trace-start-s and --trace-end-s")
    if args.trace_axis is not None and (
        args.trace_start_s is None or args.trace_end_s is None
    ):
        p.error("--trace-axis requires --trace-start-s and --trace-end-s")
    if args.train_chart is not None:
        if "train" not in (part.lower() for part in args.train_chart.parts):
            p.error("train-only integration gate; select an explicit Train chart")
        segment = build_playable_segment(
            compile_adofai(load_adofai(args.train_chart)), end_s=args.end_s
        )
        cases = [(args.train_chart.name, segment)]
    elif args.special_fixtures:
        from audit_hud_timing_tracker_transition_v14_matrix import generated
        cases = [
            (f"special-{variant}", generated((0., 90., 180., 270.),
                                              496., variant))
            for variant in ("speed_double", "twirl_speed", "pause_1")
        ]
    else:
        cases = [(f"ordinary-{int(bpm)}bpm", synthetic_segment(bpm))
                 for bpm in FIXTURE_BPMS]
    if args.case is not None:
        cases = [(name, segment) for name, segment in cases
                 if name == args.case]
        if len(cases) != 1:
            p.error("--case must exactly match a named case in this mode")
    report = {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "truth_policy_boundary": (
            "Oracle constructor only; HUD policy receives current visible "
            "Contact HUD, local TimingFrameUpdate and current physical Motor"
        ),
        "cases": [run_fixed(s, checkpoint=args.checkpoint,
                            label=name, timeout=args.wall_timeout_s,
                            arm_filter=args.arm, trace_axis=args.trace_axis,
                            trace_window_s=(
                                (args.trace_start_s, args.trace_end_s)
                                if args.trace_axis is not None else None
                            ))
                  for name, s in cases],
        "limitations": [
            "No training, no neural Timing v3; Timing v1.4 uses up to 80ms.",
            "Display HUD feedback is not exposed yet; v1.4 ignores feedback.",
            "Unsupported/ambiguous geometry abstains without an Oracle fallback.",
            "Timing source never supplies evaluator target ordinal to policy.",
            "This smoke run is not a Validation20 generalization claim.",
            "Candidate Network context remains frozen 65D.",
            "Supported-only scores are posthoc scorer ordinal associations, "
            "not proof that the selected finger produced that ordinal's Hit.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gates = {}
    for case in report["cases"]:
        arm_gates = {}
        for arm_name, record in case["arms"].items():
            if "error" in record:
                arm_gates[arm_name] = {
                    "passed": False,
                    "reason": record["error"],
                }
                continue
            stats = record["stats"]
            supported = record["supported_event_posthoc"]
            match = record["forecast_posthoc"]
            supported_safe = (
                supported["supported_truth_misses"] == 0
                and stats["too_early"] == 0
                and not stats["overloaded"]
            )
            no_false = (
                arm_name == "oracle_network"
                or match["matched_20ms"] == match["unique_forecasts"]
            )
            arm_gates[arm_name] = {
                "passed": supported_safe and no_false,
                "supported_truth_misses": supported["supported_truth_misses"],
                "too_early": stats["too_early"],
                "overloaded": stats["overloaded"],
                "false_forecast_ids_20ms": (
                    match["unique_forecasts"] - match["matched_20ms"]
                ),
                "uncovered_truth_targets": (
                    match["true_targets"] - match["matched_20ms"]
                ),
                "all_target_hit_fraction": (
                    stats["hits"] / stats["targets"] if stats["targets"] else None
                ),
                "supported_only_hit_fraction": supported[
                    "sufficient_lead_supported_hit_fraction"
                ],
            }
        gates[case["label"]] = arm_gates
    report["predeclared_gates"] = gates
    report["status"] = (
        "PASS" if all(x["passed"] for arms in gates.values()
                      for x in arms.values()) else "FAIL"
    )
    (args.output_dir / "integration-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved:", args.output_dir / "integration-summary.json", flush=True)
    print("gate:", report["status"], flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
