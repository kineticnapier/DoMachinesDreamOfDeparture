from __future__ import annotations

"""Read-only end-of-chart and endurance attribution for persistent Stage T Motor.

The baseline planner is unchanged. Every decision uses only the v2 263D HUD,
its local 10ms tick, previous commitments, and motor calibration. This module
reads truth after stepping solely to score and attribute outcomes.
"""

from collections import Counter
from dataclasses import dataclass

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.capacity import calibrate_n_key_press_lead

from . import prestige1 as p
from . import prestige1_autoplay_persistent_motor_audit as baseline
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-persistent-endurance-attribution-readonly-v1"
REPORT = "stage_timing_persistent_motor_endurance_attribution_readonly.json"
FIXTURES = ((1200., 18), (2000., 18), (4000., 18),
            (1200., 72), (2000., 72), (4000., 72),
            (1200., 144), (2000., 144), (4000., 144))
MATCH_TOL_S = .0004


@dataclass
class IntentTrace:
    due_s: float
    key: str
    discovered_s: float
    command_s: float | None = None
    physical_down_s: float | None = None
    evaluator_target_index: int | None = None
    judgement: str | None = None


def analyze_case(bpm: float, floors: int, calibration) -> dict:
    if (bpm, floors) not in FIXTURES:
        raise ValueError("Only fixed, predeclared endurance fixtures are supported")
    segment = baseline._fixture(bpm, floors=floors)
    planner = baseline.PersistentMotorPlanner(calibration)
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=p.KEY_COUNT,
        control_dt_s=baseline.DT, physics_dt_s=baseline.PHYSICS,
        capture_overload_trace=True,
    )
    observation = env.reset()
    traces: dict[int, IntentTrace] = {}
    predicted_statuses: Counter[str] = Counter()
    all_physical_events: list[dict] = []
    frames = 0
    max_steps = int((segment.duration_s + .5) / baseline.DT) + 3
    for tick in range(max_steps):
        # Causal learner-facing observation: only current 263D HUD.
        hud = tuple(encode_n_key_hud_real_chart_observation(observation))
        forecast = baseline.forecast_visible(hud)
        predicted_statuses[forecast.status] += 1
        before_tokens = set(planner.commitments)
        action = planner.action(hud)
        now_s = tick * baseline.DT
        for token in sorted(set(planner.commitments) - before_tokens):
            item = planner.commitments[token]
            if token in traces:
                raise RuntimeError("Repeated intent token in deterministic baseline")
            traces[token] = IntentTrace(item.predicted_s, item.key, now_s)
        for token, item in planner.commitments.items():
            old = traces.get(token)
            if old is None:
                raise RuntimeError("Persistent commitment missing from recorded history")
            index = planner.names.index(item.key)
            if action.values[index] > .5 and old.command_s is None:
                old.command_s = now_s
        start_event_count = len(env._trace_physical_keydowns)
        result = env.step(action)
        new_events = env._trace_physical_keydowns[start_event_count:]
        for event in new_events:
            key = event.key
            # Match to the first issued, unmatched intent on the *same key*.
            # This is a POSTHOC annotation and has no feedback into the planner.
            candidates = [row for row in traces.values()
                          if row.key == key and row.command_s is not None
                          and row.command_s <= event.time_s + 1e-9
                          and row.physical_down_s is None]
            record = min(candidates, key=lambda row: row.discovered_s) if candidates else None
            judgement = None
            if event.target_index is not None:
                target = segment.targets[event.target_index]
                judgement = env._classify(event.time_s - target.episode_time_s, target).value
            if record is not None:
                record.physical_down_s = event.time_s
                record.evaluator_target_index = event.target_index
                record.judgement = judgement
            all_physical_events.append({
                "time_ms": round(event.time_s * 1000., 6),
                "key": key,
                "evaluator_next_target_index": event.target_index,
                "judgement": judgement,
                "matched_plan_due_ms": (round(record.due_s * 1000., 6)
                                        if record is not None else None),
            })
        observation = result.observation
        frames += 1
        if result.done:
            break
    else:
        raise RuntimeError("Persistent Motor endurance probe did not terminate")

    misses: Counter[str] = Counter()
    outcomes: list[dict] = []
    for target_index, target in enumerate(segment.targets):
        due_s = target.episode_time_s
        token = planner.token(due_s)
        intent = traces.get(token)
        # A clock token collision cannot be silently treated as a match.
        if intent is not None and abs(intent.due_s-due_s) > MATCH_TOL_S:
            raise RuntimeError("Intent token does not correspond to evaluator floor time")
        hit = bool(env._used[target_index])
        miss = bool(env._missed[target_index])
        if not hit and not miss:
            raise RuntimeError("Target unresolved after terminated endurance episode")
        if hit:
            category = "hit"
        elif intent is None:
            category = "no_plan"
        elif intent.command_s is None:
            category = "planned_but_not_commanded"
        elif intent.physical_down_s is None:
            category = "commanded_but_no_matching_keydown"
        elif intent.evaluator_target_index != target_index:
            category = "keydown_evaluator_target_mismatch"
        else:
            category = "keydown_target_was_not_hit"
        if miss:
            misses[category] += 1
        outcomes.append({
            "ordinal": target_index + 1,
            "floor_index": target.floor_index,
            "target_ms": round(due_s * 1000., 6),
            "hit": hit, "miss": miss,
            "attribution": category,
            "planned_key": intent.key if intent else None,
            "plan_discovery_ms": (round(intent.discovered_s * 1000., 6) if intent else None),
            "first_command_ms": (round(intent.command_s * 1000., 6)
                                 if intent and intent.command_s is not None else None),
            "matched_physical_down_ms": (round(intent.physical_down_s * 1000., 6)
                                        if intent and intent.physical_down_s is not None else None),
            "physical_down_minus_intended_ms": (
                round((intent.physical_down_s-due_s)*1000., 6)
                if intent and intent.physical_down_s is not None else None),
            "physical_event_evaluator_target_index": (
                intent.evaluator_target_index if intent else None),
            "physical_event_judgement": intent.judgement if intent else None,
        })
    matched_intention_errors_ms = [
        (trace.physical_down_s - trace.due_s) * 1000.
        for trace in traces.values() if trace.physical_down_s is not None
    ]
    if sum(misses.values()) != env.stats.misses:
        raise RuntimeError("Attribution count does not reconcile with evaluator misses")
    if sum(x["hit"] for x in outcomes) != env.stats.hits:
        raise RuntimeError("Attribution count does not reconcile with evaluator hits")
    return {
        "bpm": bpm, "targets": floors, "duration_ms": segment.duration_s * 1000.,
        "frames": frames, "hits": env.stats.hits, "misses": env.stats.misses,
        "overloaded": env.stats.overloaded,
        "physical_keydowns": len(all_physical_events),
        "x_accuracy_percent": env.stats.x_accuracy_percent,
        "evaluator_mean_abs_error_ms": env.stats.mean_abs_error_ms,
        "matched_intentions_with_keydown": len(matched_intention_errors_ms),
        "intended_keydown_mae_ms": (
            sum(abs(x) for x in matched_intention_errors_ms) /
            len(matched_intention_errors_ms)
            if matched_intention_errors_ms else None
        ),
        "intended_keydown_max_abs_error_ms": (
            max(abs(x) for x in matched_intention_errors_ms)
            if matched_intention_errors_ms else None
        ),
        "planned_unique": planner.planned_unique,
        "hud_statuses": dict(sorted(predicted_statuses.items())),
        "miss_attribution": dict(sorted(misses.items())),
        "first_misses": [x for x in outcomes if x["miss"]][:24],
        "all_target_outcomes": outcomes,
        "physical_keydown_trace": all_physical_events,
        "motor_resets_after_start": 0,
    }


def run_audit(config: p.PrestigeConfig) -> dict:
    if p.KEY_COUNT != 8 or baseline.DT != .010 or baseline.PHYSICS != .001:
        raise RuntimeError("Frozen 8-key Motor and clock contract changed")
    destination = transfer.output_dir(config) / REPORT
    if destination.exists():
        raise FileExistsError("Refuse to overwrite previous endurance attribution report")
    calibration = calibrate_n_key_press_lead(
        8, control_dt_s=baseline.DT, physics_dt_s=baseline.PHYSICS)
    cases = [analyze_case(bpm, length, calibration) for bpm, length in FIXTURES]
    for row in cases:
        print(f"endurance {row['bpm']:>6.0f} BPM {row['targets']:>3} notes: "
              f"hits={row['hits']} misses={row['misses']} "
              f"planned={row['planned_unique']} "
              f"miss-causes={row['miss_attribution']}", flush=True)
    report = {
        "format": FORMAT, "cases": cases,
        "no_training": True, "no_student_checkpoint": True,
        "original_stage_a_fail_preserved": True,
        "geometry_hybrid_unchanged": True,
        "validation_evaluated": False, "final_evaluated": False,
        "fixed_preview_ms": baseline.PREVIEW*1000.,
        "policy_dt_ms": baseline.DT*1000.,
        "physics_dt_ms": baseline.PHYSICS*1000.,
        "notes": [
            "Unmodified persistent Stage T baseline; no hyperparameter tuning",
            "Only 263D HUD/current motor observation feeds forecast/planner",
            "Evaluator floor indices and exact timestamps only used after action",
            "Intent-to-keydown matching uses oldest issued unmatched plan per key",
            "Evaluator target index of a physical press may differ from intended target",
            "Attribution is diagnostic association, not a causal optimal-controller proof",
            "Hit count is not intended-action accuracy; evaluator pass windows may credit another floor",
            "Per-key oldest issued unmatched plan is diagnostic and may be ambiguous during interference",
            "No direct comparison to trained Student or Validation/FINAL",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(destination, report)
    return report
