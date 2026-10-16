from __future__ import annotations

"""Stage T read-only attribution: visible-prefix reservations vs physical KeyDown.

The exact *unchanged* PrefixPersistentMotorPlanner receives only the current
263D HUD and its own history. Privileged floor time, evaluator queues and
physical event traces are consulted *after* commands for diagnostic scoring.
No checkpoint, student training, Validation or FINAL is used here.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.keyboard import KeyEvent

from . import prestige1 as p
from . import prestige1_autoplay_persistent_motor_audit as baseline
from . import prestige1_autoplay_visible_prefix_audit as prefix
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-visible-prefix-physical-timing-attribution-v1"
REPORT = "stage_timing_visible_prefix_motor_timing_attribution_readonly.json"
FIXTURES = ((2000., 72), (2000., 144), (4000., 18), (4000., 72), (4000., 144))
MATCH_EPS_S = .0004
MAX_EXAMPLES = 24


@dataclass
class Intent:
    token: int
    due_s: float
    key: str
    discovered_s: float
    command_s: float | None = None
    first_command_position_mm: float | None = None
    first_command_velocity_mm_s: float | None = None
    first_command_activation: float | None = None
    first_command_pressed: bool | None = None
    keydown_s: float | None = None
    claimed_target_index: int | None = None
    evaluator_margin: str | None = None


def _r_ms(t: float | None) -> float | None:
    return round(t * 1000., 6) if t is not None else None


def analyze_case(bpm: float, notes: int, calibration) -> dict:
    if (bpm, notes) not in FIXTURES:
        raise ValueError("Only fixed diagnostic fixtures are allowed")
    if calibration.key_count != 8:
        raise ValueError("Expected unchanged 8-key calibration")
    seg = baseline._fixture(bpm, floors=notes)
    # Unmodified previous-stage planner: no learned policy or oracle teacher.
    planner = prefix.PrefixPersistentMotorPlanner(calibration)
    env = DiagnosticHudNKeyRealChartMotorEnv(
        seg, key_count=8, control_dt_s=.010, physics_dt_s=.001,
        capture_overload_trace=True,
    )
    obs = env.reset()
    intents: dict[int, Intent] = {}
    downs: list[dict] = []
    frame_records: list[dict] = []
    trace_statuses: Counter[str] = Counter()
    starts_while_key_already_pressed = 0
    runtime_bound = int((seg.duration_s+.5)/.010)+3
    for tick in range(runtime_bound):
        now = tick*.010  # Local control clock, not simulator privileged time.
        hud = tuple(encode_n_key_hud_real_chart_observation(obs))
        keys = obs.motor.key_names
        # The following forecast is for diagnosis only, not fed back to action.
        visible = prefix.forecast_visible_prefix(hud)
        trace_statuses[visible.status] += 1
        before = set(planner.commitments)
        action = planner.action(hud)
        for token in sorted(set(planner.commitments)-before):
            commitment = planner.commitments[token]
            if token in intents:
                raise RuntimeError("Duplicate commitment token in one episode")
            intents[token] = Intent(
                token=token, due_s=commitment.predicted_s,
                key=commitment.key, discovered_s=now,
            )
        # Record 10ms command onsets and *actual* motor state from visible HUD.
        # Do not read evaluator clock or target truth to determine any action.
        commanded = []
        for token, item in planner.commitments.items():
            rec = intents[token]
            index = keys.index(item.key)
            if action.values[index] > .5:
                commanded.append(item.key)
                if rec.command_s is None:
                    rec.command_s = now
                    rec.first_command_position_mm = obs.motor.positions_m[index]*1000.
                    rec.first_command_velocity_mm_s = obs.motor.velocities_m_s[index]*1000.
                    rec.first_command_pressed = bool(obs.motor.pressed_flags[index])
                    rec.first_command_activation = float(env.motor.body.states[index].activation)
                    if rec.first_command_pressed:
                        starts_while_key_already_pressed += 1
        frame_records.append({
            "local_time_ms": _r_ms(now),
            "action_down_keys": list(commanded),
            "active_reservations": len(planner.commitments),
            "forecast_status": visible.status,
        })
        physical_before = len(env._trace_physical_keydowns)
        result = env.step(action)
        for evt in env._trace_physical_keydowns[physical_before:]:
            # Evaluator annotations are used strictly after env.step(action).
            if evt.target_index is None:
                margin = "no_pending_target"
            else:
                target = seg.targets[evt.target_index]
                margin = env._classify(evt.time_s-target.episode_time_s,target).value
            matching = [rec for rec in intents.values() if (
                rec.key==evt.key and rec.command_s is not None and
                rec.command_s<=evt.time_s+1e-10 and rec.keydown_s is None
            )]
            record = min(matching,key=lambda rec:(rec.command_s,rec.discovered_s)) if matching else None
            if record is not None:
                record.keydown_s = evt.time_s
                record.claimed_target_index = evt.target_index
                record.evaluator_margin = margin
            downs.append({
                "time_ms": _r_ms(evt.time_s), "key": evt.key,
                "matched_intent_token": record.token if record else None,
                "matched_intended_due_ms": _r_ms(record.due_s) if record else None,
                "evaluator_next_target_index": evt.target_index,
                "margin": margin,
            })
        obs = result.observation
        if result.done:
            break
    else:
        raise RuntimeError("8K simulator did not terminate within time bound")

    # Offline truth joins: this entire block is only diagnostic/evaluator side.
    calib = dict(calibration.key_latencies_s)
    rows: list[dict] = []
    reasons: Counter[str] = Counter()
    late_command = 0
    intended_errors = []
    matched_offsets: Counter[str] = Counter()
    for ordinal, target in enumerate(seg.targets):
        token = planner.token(target.episode_time_s)
        intent = intents.get(token)
        if intent is not None and abs(intent.due_s-target.episode_time_s)>MATCH_EPS_S:
            raise RuntimeError("Token collision with evaluator target")
        hit, miss = bool(env._used[ordinal]),bool(env._missed[ordinal])
        if hit == miss:
            raise RuntimeError("Evaluator target neither/both used and missed")
        if intent is None:
            reason = "no_reservation"
        elif intent.command_s is None:
            reason = "reserved_not_commanded"
        elif intent.keydown_s is None:
            reason = "commanded_without_matched_keydown"
        elif intent.claimed_target_index != ordinal:
            reason = "keydown_claimed_different_target"
        elif miss:
            reason = "keydown_claimed_but_missed"
        else:
            reason = "correctly_claimed_target"
        if miss:
            reasons[reason] += 1
        if intent is not None and intent.command_s is not None:
            lead_s = intent.due_s-intent.command_s
            shortage_s = max(0.,calib[intent.key]-lead_s)
            late_command += shortage_s>1e-10
        else:
            lead_s = shortage_s = None
        if intent is not None and intent.keydown_s is not None:
            error_ms=(intent.keydown_s-target.episode_time_s)*1000.
            intended_errors.append(error_ms)
            if intent.claimed_target_index is not None:
                offset=intent.claimed_target_index-ordinal
                matched_offsets[str(offset)]+=1
        else:
            error_ms=None
            offset=None
        rows.append({
            "target_ordinal": ordinal+1, "due_ms": _r_ms(target.episode_time_s),
            "hit":hit,"miss":miss,"attribution":reason,
            "planned_key": intent.key if intent else None,
            "reservation_discovered_ms": _r_ms(intent.discovered_s) if intent else None,
            "command_start_ms": _r_ms(intent.command_s) if intent else None,
            "command_lead_ms":_r_ms(lead_s),
            "cold_calibration_latency_ms":_r_ms(calib[intent.key]) if intent else None,
            "cold_lead_shortfall_ms":_r_ms(shortage_s),
            "command_start_position_mm":intent.first_command_position_mm if intent else None,
            "command_start_velocity_mm_s":intent.first_command_velocity_mm_s if intent else None,
            "command_start_activation":intent.first_command_activation if intent else None,
            "actual_keydown_ms":_r_ms(intent.keydown_s) if intent else None,
            "keydown_minus_intended_ms":round(error_ms,6) if error_ms is not None else None,
            "physical_keydown_claimed_target_ordinal":intent.claimed_target_index+1
                if intent is not None and intent.claimed_target_index is not None else None,
            "claim_offset":offset,
            "physical_margin":intent.evaluator_margin if intent else None,
        })
    if len(rows)!=notes or sum(x['hit'] for x in rows)!=env.stats.hits or (
        sum(x['miss'] for x in rows)!=env.stats.misses or sum(reasons.values())!=env.stats.misses
    ):
        raise RuntimeError("Per-target accounting does not reconcile with evaluator")
    if env.stats.overloaded:
        raise RuntimeError("Unexpected overload in fixed nontraining diagnostic")
    mismatch_rows=[r for r in rows if r['attribution']=='keydown_claimed_different_target']
    returned = {
        "bpm":bpm,"targets":notes,"hits":env.stats.hits,"misses":env.stats.misses,
        "x_accuracy_percent":env.stats.x_accuracy_percent,
        "evaluator_mean_absolute_timing_error_ms":env.stats.mean_abs_error_ms,
        "reservations":len(intents),"physical_keydowns":len(downs),
        "matched_intents_with_down":len(intended_errors),
        "intended_keydown_mae_ms":(sum(abs(x) for x in intended_errors)/len(intended_errors)
            if intended_errors else None),
        "intended_keydown_mean_signed_error_ms":(sum(intended_errors)/len(intended_errors)
            if intended_errors else None),
        "intent_claim_offset_histogram":dict(sorted(matched_offsets.items(),key=lambda x:int(x[0]))),
        "command_started_with_insufficient_cold_lead":late_command,
        "first_commands_on_already_pressed_keys":starts_while_key_already_pressed,
        "miss_attribution":dict(sorted(reasons.items())),
        "all_target_outcomes":rows,
        "mismatch_examples":mismatch_rows[:MAX_EXAMPLES],
        "first_misses":[r for r in rows if r['miss']][:MAX_EXAMPLES],
        "physical_keydown_trace":downs,
        "frame_trace":frame_records,
        "forecast_statuses":dict(sorted(trace_statuses.items())),
        "no_motor_reset_between_notes":True,
        "no_oracle_inputs_to_planner":True,
    }
    return returned


def run_audit(config:p.PrestigeConfig)->dict:
    if p.KEY_COUNT!=8 or p.CONTROL_DT!=.010 or p.PHYSICS_DT!=.001:
        raise RuntimeError("Frozen motor clock contract changed")
    dest=transfer.output_dir(config)/REPORT
    if dest.exists():
        raise FileExistsError("Refuse to overwrite persistent-motor timing report")
    calibration=calibrate_n_key_press_lead(8,control_dt_s=.010,physics_dt_s=.001)
    cases=[analyze_case(bpm,length,calibration) for bpm,length in FIXTURES]
    for r in cases:
        print(f"timing {r['bpm']:>6.0f} BPM {r['targets']:>3} notes: "
              f"hits={r['hits']}/{r['targets']} X={r['x_accuracy_percent']:.2f}% "
              f"intended_MAE={r['intended_keydown_mae_ms']:.3f}ms "
              f"reasons={r['miss_attribution']}",flush=True)
    report={
        "format":FORMAT,"cases":cases,
        "no_training":True,"no_checkpoint":True,"no_validation":True,"no_final":True,
        "stage_a_fail_preserved":True,"frozen_geometry_hybrid_unchanged":True,
        "policy_control_dt_ms":10.,"physical_dt_ms":1.,
        "purpose":"Compare persistent HUD plans to actual KeyDown and evaluator target queues",
        "limitations":[
            "Exact target times and evaluator indices are used for after-the-fact joins only",
            "Matched keydown to earliest commanded unmatched intention on the same finger; heuristic not a causal proof",
            "Queue offset is a posthoc association, not evidence a player deliberately pressed that other target",
            "Cold lead deficit is not a complete dynamic compensation model; finger state may already be moving",
            "Physical keydown count and evaluator hit count are not interchangeable",
            "Only deterministic 2000/4000 BPM straight-floor fixtures; no general chart claims",
            "No planner tuning, Student training, checkpoints, Validation or FINAL",
        ],
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(dest,report)
    return report
