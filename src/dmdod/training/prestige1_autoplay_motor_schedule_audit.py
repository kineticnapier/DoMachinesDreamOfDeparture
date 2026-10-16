from __future__ import annotations

"""Read-only Stage T HUD future-event -> eight-key physical scheduling probe.

No Student, checkpoint, hidden chart clock, target schedule, or evaluator is
available to the planner. Inference only sees the 263D HUD, the previously
measured motor's *cold* actuation latencies, and a deterministic key priority.

A reference 1ms physical executor is simulated separately, and is NOT the
frozen Student's existing 10ms-held Motor control. Results are diagnostics,
not a trained policy, an optimal scheduler, or a feasibility certificate.
"""

from collections import Counter
from dataclasses import dataclass
from math import ceil, isfinite

from dmdod.motor.capacity import calibrate_n_key_press_lead, NKeyLeadCalibration
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv, n_key_names

from . import prestige1 as p
from . import prestige1_autoplay_future_events_audit as future
from . import prestige1_autoplay_high_bpm_simulator_audit as simulator
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-forecast-to-motor-readonly-v1"
REPORT = "stage_timing_forecast_motor_shadow_20k_readonly.json"
PHYSICS_DT_S = .001
CONTROL_DT_S = .010
SHADOW_DURATION_S = .080
SCORE_TOL_S = .0005


@dataclass(frozen=True)
class AssignedEvent:
    floor_offset: int
    target_offset_s: float
    key: str
    # Offset relative to this observation, 0 if the action should already
    # have begun. This is a command START, not a predicted physical KeyDown.
    command_start_s: float
    cold_keydown_latency_s: float
    cold_deadline_deficit_s: float


@dataclass(frozen=True)
class Plan:
    status: str
    assignments: tuple[AssignedEvent, ...]
    # A plan with late_cold_start is still a *provisional command sequence*.
    # Never report it as physically on-time, or pretend a 10ms forecast is
    # enough to prepare a 39ms latency motor from its rest position.

    @property
    def eligible(self) -> bool:
        return self.status in {"eligible", "late_cold_start"}


def plan_from_hud(hud: tuple[float, ...], calibration: NKeyLeadCalibration) -> Plan:
    """Assign each HUD-predicted target one distinct, currently released key.

    Calibrations come from the same frozen motor profile; these are not
    privileged chart signals. This baseline uses no memory of earlier plans,
    so *all fingers start at rest*. State-aware re-planning is future work.
    """
    if calibration.key_count != p.KEY_COUNT or (
        abs(calibration.physics_dt_s-PHYSICS_DT_S)>1e-12
    ):
        raise ValueError("Incompatible frozen eight-key physical calibration")
    prediction = future.forecast(hud)
    if not prediction.eligible:
        return Plan("forecast_"+prediction.status, ())
    # 8 position, 8 velocity, 8 press flags before orbit and floor encoding.
    names=n_key_names(p.KEY_COUNT)
    latencies=dict(calibration.key_latencies_s)
    available=[name for index,name in enumerate(names)
               if hud[2*p.KEY_COUNT+index]<.5]
    if len(prediction.times_s)>len(available):
        return Plan("insufficient_unpressed_keys", ())
    # Deterministic cold-latency-first allocation, not a learned fingering.
    available.sort(key=lambda key:(latencies[key],names.index(key)))
    assigned=[]
    for offset,target,key in zip(
        prediction.floor_offsets,prediction.times_s,available
    ):
        latency=latencies[key]
        assigned.append(AssignedEvent(
            floor_offset=offset,
            target_offset_s=float(target),
            key=key,
            command_start_s=max(0.,float(target)-latency),
            cold_keydown_latency_s=latency,
            cold_deadline_deficit_s=max(0.,latency-float(target)),
        ))
    late=any(item.cold_deadline_deficit_s>1e-8 for item in assigned)
    return Plan("late_cold_start" if late else "eligible",tuple(assigned))


def simulate_shadow_from_rest(plan: Plan, *, duration_s: float = SHADOW_DURATION_S) -> dict:
    """Actually step the unchanged 8-key 1ms motor with the planned commands.

    Reference microstep actions are diagnostic only. The Student has *not*
    gained the ability to issue 1ms commands. The baseline holds a finger's
    +1 command after its start until a physical KeyDown, then brakes it.
    """
    if not plan.eligible:
        raise ValueError("Cannot execute a rejected HUD plan")
    if duration_s<CONTROL_DT_S or not isfinite(duration_s):
        raise ValueError("Invalid motor reference duration")
    names=n_key_names(p.KEY_COUNT)
    env=NKeyMotorEnv(p.KEY_COUNT,control_dt_s=PHYSICS_DT_S,
                     physics_dt_s=PHYSICS_DT_S)
    env.reset()
    by_key={a.key:a for a in plan.assignments}
    pressed_once:set[str]=set()
    actual=[]
    microsteps=[]
    count=ceil(duration_s/PHYSICS_DT_S)
    for i in range(count):
        start=i*PHYSICS_DT_S
        cmd=tuple(
            -1. if name in pressed_once else
            1. if name in by_key and start+1e-10>=by_key[name].command_start_s
            else 0.
            for name in names
        )
        if i<int(round(CONTROL_DT_S/PHYSICS_DT_S)):
            microsteps.append(cmd)
        transition=env.step(NKeyAction(cmd))
        for event in transition.evaluator_events:
            if event.event is KeyEvent.DOWN and event.key in by_key:
                pressed_once.add(event.key)
                actual.append((event.key,float(event.time_s)))
    event_by_key=dict(actual)
    errors=[]
    missed=[]
    for planned in plan.assignments:
        if planned.key in event_by_key:
            errors.append((event_by_key[planned.key]-planned.target_offset_s)*1000.)
        else:
            missed.append(planned.key)
    first_ten=sum(t<=CONTROL_DT_S+1e-10 for _,t in actual)
    return {
        "actual_keydowns_ms":[{"key":key,"time_ms":t*1000.}
                              for key,t in actual],
        "key_downs_within_10ms":first_ten,
        "planned_events":len(plan.assignments),
        "actual_within_80ms":len(actual),
        "timing_errors_ms":errors,
        "not_actuated_within_80ms":missed,
        "reference_1ms_micro_actions_during_first_10ms":microsteps,
        "no_frozen_student_control_change":True,
    }


def run_audit(config:p.PrestigeConfig) -> dict:
    if abs(p.CONTROL_DT-CONTROL_DT_S)>1e-12 or abs(p.PHYSICS_DT-PHYSICS_DT_S)>1e-12:
        raise RuntimeError("Frozen Prestige I clock contract changed")
    root=transfer.output_dir(config)
    path=root/REPORT
    if path.exists():
        raise FileExistsError("Motor scheduling diagnostic exists; refuse replacement")
    calibration=calibrate_n_key_press_lead(
        p.KEY_COUNT, control_dt_s=PHYSICS_DT_S, physics_dt_s=PHYSICS_DT_S
    )
    results=[]
    for bpm in simulator.BPM_POINTS:
        for reverse in simulator.DIRECTIONS:
            segment,sha=simulator._fixture(bpm,reverse=reverse)
            samples,_=simulator._capture(segment)
            coarse=samples[::10]
            simulator._verify_direct_10ms(segment,coarse)
            plans=[plan_from_hud(frame.hud,calibration) for frame in coarse]
            statuses=Counter(plan.status for plan in plans)
            accepted=[plan for plan in plans if plan.eligible]
            events=[x for plan in accepted for x in plan.assignments]
            # One *actual physics* execution is made per fixture, if any
            # nonempty approved event plan exists. No evaluator ground truth
            # is used to construct the plan or the motor commands.
            first=next((plan for plan in accepted if plan.assignments),None)
            motor=simulate_shadow_from_rest(first) if first is not None else None
            floor_span_s=60./bpm  # Only valid for these straight 180-deg fixtures.
            row={
                "bpm":bpm,"direction":"ccw" if reverse else "cw",
                "fixture_sha256":sha,
                "control_frames":len(coarse),
                "status_counts":dict(sorted(statuses.items())),
                "planned_events":len(events),
                "planned_multi_event_frames":sum(len(x.assignments)>=2 for x in accepted),
                "cold_deadline_deficits":sum(x.cold_deadline_deficit_s>1e-8 for x in events),
                "min_cold_deadline_deficit_ms":min(
                    (x.cold_deadline_deficit_s*1000 for x in events),default=None),
                "max_cold_deadline_deficit_ms":max(
                    (x.cold_deadline_deficit_s*1000 for x in events),default=None),
                "straight_floor_visible_12_ahead_ms":12*floor_span_s*1000.,
                "earliest_cold_press_latency_ms":min(
                    calibration.key_latencies_s,key=lambda pair:pair[1])[1]*1000.,
                "first_nonempty_motor_reference":motor,
            }
            results.append(row)
            print(f"motor-shadow {bpm:>7.0f} {'ccw' if reverse else 'cw'} "
                  f"plans={len(events)} late-cold={row['cold_deadline_deficits']} "
                  f"multi-frames={row['planned_multi_event_frames']} "
                  f"motor10ms={motor['key_downs_within_10ms'] if motor else None}",
                  flush=True)
    report={
        "format":FORMAT,
        "results":results,
        "latency_by_key_ms":{k:v*1000. for k,v in calibration.key_latencies_s},
        "no_training":True,"no_model_checkpoint":True,
        "stage_a_fail_frozen":True,"geometry_hybrid_untouched":True,
        "validation_evaluated":False,"final_evaluated":False,
        "policy_control_dt_ms":CONTROL_DT_S*1000.,
        "reference_motor_dt_ms":PHYSICS_DT_S*1000.,
        "limitations":[
            "Planner uses only one causal 263D HUD frame and fixed motor calibration",
            "No plan persistence, prearming, prior practice, or optimized finger motion",
            "Cold-latency-first assignment is NOT optimal physical control",
            "The 1ms shadow executor is diagnostic; frozen Student still holds 10ms actions",
            "No privileged target timestamps or floor indices in the planner",
            "Physical preview is a fresh-from-rest motor; it cannot establish sustained playability",
            "Conditional HUD forecast has abstentions on ambiguous and special floors",
            "No training, validation, FINAL, or Student adoption",
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
