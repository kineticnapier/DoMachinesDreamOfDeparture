from __future__ import annotations

"""Stage T: read-only, causal 263D HUD -> future 10ms event-sequence probe.

The predictor does not access compiled charts, exact clock, target timestamps,
absolute floor number, or hidden direction. Only the posthoc scorer does.
This is a conservative ordinary-floor geometry baseline, not Student training.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from math import atan2, isfinite, pi
from pathlib import Path
import json

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_high_bpm_simulator_audit as simulator
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-hud-future-events-10ms-readonly-v1"
REPORT = "stage_timing_future_events_10ms_simulator_readonly.json"
HORIZON_S = 0.010
RAD_EPS = 1e-5
TIME_EPS_S = 1e-7
SCORE_TOL_S = 5e-5  # 0.05ms, for deterministic simulator numeric comparison
MAX_EXAMPLES = 8


@dataclass(frozen=True)
class Forecast:
    # Offsets RELATIVE to this frame's current visible floor, not chart indices.
    floor_offsets: tuple[int, ...]
    times_s: tuple[float, ...]
    status: str

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


def _direction_candidate(hud: tuple[float, ...], sign: int, bpm: float) -> Forecast:
    angular_rate = pi*bpm/60.
    orbit = atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES+1])
    event_times: list[float] = []
    event_offsets: list[int] = []
    elapsed = 0.
    for slot in range(cd.CURRENT_SLOT, cd.CURRENT_SLOT+cd.FLOORS.ahead_floors+1):
        floor = cd._visible_slot(hud, slot)
        if floor[-1] < .5:
            return Forecast((), (), "insufficient_lookahead")
        offset = slot-cd.CURRENT_SLOT
        if floor[6]>.5 or floor[7]>.5:
            return Forecast((), (), "midspin_or_multiplanet")
        # A Twirl on the *current* floor has already been applied. Its
        # direction must be inferred from the orbit, never read as truth.
        # SetSpeed/Pause at the current floor can also invalidate elapsed
        # phase, so they require separate models.
        active_markers = [j for j in range(cd.SKIP_MARKER_FROM,cd.SKIP_MARKER_TO)
                          if floor[j]>.5]
        if any(j!=cd.SKIP_MARKER_FROM for j in active_markers) or (
            offset>0 and active_markers
        ):
            return Forecast((), (), "special_event")
        entry = atan2(floor[2],floor[3])
        exit_angle = atan2(floor[4],floor[5])
        span=(sign*(exit_angle-entry))%(2*pi)
        if span<RAD_EPS or 2*pi-span<RAD_EPS:
            return Forecast((), (), "full_turn_or_zero_arc")
        if offset==0:
            progress=(sign*(orbit-entry))%(2*pi)
            # atan2 and the encoded sin/cos are rounded independently;
            # a physical exact entry may appear as ~2*pi for one sign.
            if progress<RAD_EPS or 2*pi-progress<RAD_EPS:
                progress=0.
            if progress>span+RAD_EPS:
                return Forecast((), (), "incompatible_direction")
            if span-progress < RAD_EPS:
                return Forecast((), (), "boundary_ambiguous")
            remaining=max(0.,span-progress)
        else:
            remaining=span
        until=remaining/angular_rate
        if elapsed+until > HORIZON_S+TIME_EPS_S:
            return Forecast(tuple(event_offsets),tuple(event_times),"eligible")
        # This floor's exit produces the *next* floor's playable note.
        # Require a visible, ordinary target, not a midspin or a special.
        next_floor=cd._visible_slot(hud,slot+1) if (
            slot+1<cd.FLOORS.floor_slots
        ) else ()
        if not next_floor or next_floor[-1]<.5:
            return Forecast((), (), "insufficient_lookahead")
        if any(next_floor[j]>.5 for j in (6,7)) or any(
            next_floor[j]>.5 for j in range(cd.SKIP_MARKER_FROM,
                                             cd.SKIP_MARKER_TO)
        ):
            return Forecast((), (), "special_future_floor")
        elapsed+=until
        if elapsed<=TIME_EPS_S:
            return Forecast((), (), "boundary_ambiguous")
        event_times.append(elapsed)
        event_offsets.append(offset+1)
        if elapsed>=HORIZON_S-TIME_EPS_S:
            return Forecast(tuple(event_offsets),tuple(event_times),"eligible")
    return Forecast((), (), "insufficient_lookahead")


def forecast(hud: tuple[float, ...]) -> Forecast:
    """Predict ordered future target times in (now, now+10ms] from one HUD.

    Both CW and CCW are evaluated; do NOT assume a privileged direction.
    If both histories are geometrically possible but give different
    future event times, abstain. Supports multiple targets per frame.
    """
    if len(hud)!=p.INPUT_DIM:
        return Forecast((),(),"invalid_hud_dimension")
    if not all(isfinite(x) for x in hud):
        return Forecast((),(),"nonfinite_hud")
    code=hud[cd.HUD_OFFSET]
    try:
        bpm=120.*(2.**code)
    except OverflowError:
        return Forecast((),(),"invalid_bpm")
    if not isfinite(bpm) or bpm<=0.:
        return Forecast((),(),"invalid_bpm")
    if max(abs(x) for x in hud[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2])<.5:
        return Forecast((),(),"invalid_orbit")
    candidates=[_direction_candidate(hud,sign,bpm) for sign in (1,-1)]
    viable=[c for c in candidates if c.eligible]
    if not viable:
        reasons=[x.status for x in candidates]
        # Never replace a special-event abstention with an assumed direction.
        for priority in ("special_future_floor","special_event",
                         "midspin_or_multiplanet","boundary_ambiguous",
                         "full_turn_or_zero_arc","insufficient_lookahead"):
            if priority in reasons:
                return Forecast((),(),priority)
        return Forecast((),(),"no_consistent_direction")
    if len(viable)==1:
        # Crucial: a direction must be eliminated by the CURRENT observed arc.
        # A future special event / missing lookahead / full-turn is *not*
        # evidence for the opposite direction. Otherwise a clock-phase
        # ambiguity can produce confidently wrong future event sequences.
        other=next(c for c in candidates if not c.eligible)
        if other.status != "incompatible_direction":
            return Forecast((),(),other.status)
        return viable[0]
    a,b=viable
    if a.floor_offsets!=b.floor_offsets or len(a.times_s)!=len(b.times_s) or any(
        abs(x-y)>TIME_EPS_S for x,y in zip(a.times_s,b.times_s)
    ):
        return Forecast((),(),"ambiguous_direction")
    return a


def evaluate_chart(segment, samples: tuple[simulator.Observed, ...]) -> dict:
    """All causal forecasts computed before any oracle timestamps are read."""
    predictions=tuple(forecast(sample.hud) for sample in samples)
    target_times=tuple(float(t.episode_time_s) for t in segment.targets)
    statuses=Counter()
    evaluated=correct=wrong=multi=correct_multi=0
    empty_correct=0
    abs_err: list[float]=[]
    mistakes=[]
    for sample,pred in zip(samples,predictions):
        statuses[pred.status]+=1
        if not pred.eligible:
            continue
        evaluated+=1
        # Evaluator only: these are NEVER fed into predictor.
        lo=bisect_right(target_times,sample.seconds+TIME_EPS_S)
        hi=bisect_right(target_times,sample.seconds+HORIZON_S+TIME_EPS_S)
        expected=tuple(x-sample.seconds for x in target_times[lo:hi])
        if len(expected)>=2:
            multi+=1
        success=len(expected)==len(pred.times_s) and all(
            abs(a-b)<=SCORE_TOL_S for a,b in zip(expected,pred.times_s)
        )
        if success:
            correct+=1
            if len(expected)>=2:correct_multi+=1
            if not expected:empty_correct+=1
            abs_err.extend(abs(a-b)*1000 for a,b in zip(expected,pred.times_s))
        else:
            wrong+=1
            if len(mistakes)<MAX_EXAMPLES:
                mistakes.append({
                    "time_ms":sample.seconds*1000.,
                    "expected_ms":[x*1000 for x in expected],
                    "predicted_ms":[x*1000 for x in pred.times_s],
                    "floor_offsets":pred.floor_offsets,
                })
    return {
        "total_frames":len(samples),
        "statuses":dict(sorted(statuses.items())),
        "eligible":evaluated,"exact_event_sequences":correct,"incorrect":wrong,
        "empty_correct":empty_correct,
        "eligible_multi_target":multi,"correct_multi_target":correct_multi,
        "event_time_mae_ms":sum(abs_err)/len(abs_err) if abs_err else None,
        "max_event_time_error_ms":max(abs_err) if abs_err else None,
        "examples":mistakes,
    }


def run_audit(config:p.PrestigeConfig)->dict:
    out=transfer.output_dir(config)
    path=out/REPORT
    if path.exists():
        raise FileExistsError("Read-only future-event report already exists; refuse overwrite")
    cases=[]
    for bpm in simulator.BPM_POINTS:
        for reverse in simulator.DIRECTIONS:
            segment,fixture_sha=simulator._fixture(bpm,reverse=reverse)
            samples,_=simulator._capture(segment)
            coarse=samples[::10]
            simulator._verify_direct_10ms(segment,coarse)
            # Shifted starts are observational phase probes from real 1ms
            # captures, NOT a claim that the frozen Student runs at 1ms.
            offsets={str(offset):evaluate_chart(segment,samples[offset::10])
                     for offset in (0,1,2,5)}
            primary=offsets["0"]
            cases.append({
                "bpm":bpm,"direction":"ccw" if reverse else "cw",
                "chart_sha256":fixture_sha,
                **primary,
                "phase_offset_10ms_probes":offsets,
            })
            eligible=sum(r['eligible'] for r in offsets.values())
            correct=sum(r['exact_event_sequences'] for r in offsets.values())
            multi=sum(r['eligible_multi_target'] for r in offsets.values())
            correct_multi=sum(r['correct_multi_target'] for r in offsets.values())
            wrong=sum(r['incorrect'] for r in offsets.values())
            print(f"future {bpm:>7.0f} {'ccw' if reverse else 'cw'}: "
                  f"{primary['exact_event_sequences']}/{primary['eligible']} "
                  f"at offset 0; phase-probes {correct}/{eligible} "
                  f"correct; multi={correct_multi}/{multi}, "
                  f"wrong={wrong}",flush=True)
    report={
        "format":FORMAT,"cases":cases,
        "horizon_ms":10.,"input_dim":p.INPUT_DIM,
        "fixed_26_physical_fixtures":True,
        "no_training":True,"no_student_checkpoint":True,
        "validation_evaluated":False,"final_evaluated":False,
        "stage_a_gate_overridden":False,
        "conditional_only":True,
        "limitations":[
            "Ordinary floor geometry and constant observed Tile BPM only",
            "Twirl on current floor may be inferred by phase; future Twirl abstains",
            "Other special events, full-turn and missing lookahead abstain",
            "Incorrect predictions are reported, never silently treated as no events",
            "Not an end-to-end Student, physical keydown or Motor execution test",
            "Simulated deterministic straight-floor fixture family only",
            "Shifted 1/2/5ms phases are 10ms decimations of a 1ms renderer trace, not frozen Student observation-clock modifications",
        ],
    }
    out.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
