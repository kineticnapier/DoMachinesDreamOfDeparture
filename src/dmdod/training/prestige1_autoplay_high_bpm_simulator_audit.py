from __future__ import annotations

"""Deterministic 8K simulator fixture audit through 20,000 Tile BPM.

This is a renderer/observability diagnostic, NOT training or an attempt to
simulate a human accurately hitting every note. The causal 263D observations
are produced by the real N-key chart HUD simulator (physics 1ms); 10ms
observations are sampled from that trace. Privileged floor indices and target
timestamps are read only AFTER HUD-only predictions have been computed.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite, pi
from pathlib import Path
import json

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.n_key import NKeyAction
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_projective_unwrap_audit as unwrap
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-physical-simulator-bpm-through-20000-readonly-v1"
REPORT = "stage_timing_simulator_20000bpm_readonly.json"
PHYSICS_DT_S = 0.001
CONTROL_DT_S = 0.010
MAX_BPM = 20000.
# Preset specification checks, NOT a hyperparameter search. Include clipping
# boundary, alias ties and both signs, plus the requested 20k ceiling.
BPM_POINTS = (600., 1920., 2380., 2999., 3000., 3001., 4000.,
              6000., 8000., 10000., 12000., 16000., 20000.)
DIRECTIONS = (False, True)  # normal, Twirl-CCW
FIXTURE_MIN_FLOORS = 24
FIXTURE_SECONDS = .18
MAX_EXAMPLES = 6


@dataclass(frozen=True)
class Observed:
    seconds: float
    hud: tuple[float, ...]


def _fixture(bpm:float,*,reverse:bool):
    if bpm not in BPM_POINTS:
        raise ValueError("Only fixed deterministic BPM fixtures are permitted")
    # 180-degree floor/target spacing, each nominally 60/BPM seconds.
    floors=max(FIXTURE_MIN_FLOORS,ceil(bpm*FIXTURE_SECONDS/60.)+12)
    chart_text=json.dumps({
        "angleData":[0.] * floors,
        "settings":{
            "bpm":bpm,"pitch":100.,
            "countdownTicks":0,"separateCountdownTime":False,
        },
        "actions":[{"floor":1,"eventType":"Twirl"}] if reverse else [],
    },sort_keys=True,separators=(",",":"))
    chart=compile_adofai(parse_adofai_text(chart_text))
    segment=build_playable_segment(chart,start_s=0.,end_s=chart.duration_s)
    if len(segment.targets)!=floors:
        raise RuntimeError("Synthetic chart target count drift")
    if not all(
        isfinite(floor.bpm) and abs(floor.bpm-bpm)<.03
        for floor in segment.chart.floors
    ):
        raise RuntimeError("Synthetic chart speed drift")
    return segment,sha256(chart_text.encode()).hexdigest()


def _capture(segment)->tuple[tuple[Observed,...],bool]:
    """Run actual 8K motor/HUD environment at physics 1ms with zero keys.

    At each physics tick a genuine simulator step is advanced; the 10ms
    observable trace is the explicitly subsampled 1ms trace. No synthetic
    orbit angle or hand-written HUD substitutes for the actual renderer.
    """
    env=DiagnosticHudNKeyRealChartMotorEnv(
        segment,key_count=p.KEY_COUNT,
        control_dt_s=PHYSICS_DT_S,physics_dt_s=PHYSICS_DT_S,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=.04,
    )
    obs=env.reset()
    no_press=NKeyAction((0.,)*p.KEY_COUNT)
    data=[]
    bound=ceil((segment.duration_s+.05)/PHYSICS_DT_S)+20
    done=False
    for index in range(bound):
        now=float(env.privileged_episode_time_s())
        visible=tuple(encode_n_key_hud_real_chart_observation(obs))
        if len(visible)!=p.INPUT_DIM or not all(isfinite(x) for x in visible):
            raise RuntimeError("Simulator produced invalid v2 263D HUD")
        if abs(now-index*PHYSICS_DT_S)>1e-7:
            raise RuntimeError("Simulator did not advance at physics tick")
        data.append(Observed(now,visible))
        state=env.step(no_press)
        obs=state.observation
        if state.done:
            done=True
            break
    if not done:
        raise RuntimeError("Simulator episode did not end within bounded steps")
    return tuple(data),done


def _verify_direct_10ms(segment,coarse:tuple[Observed,...])->None:
    """Require independent real 10ms physics/control replay to match 1ms decimation."""
    env=DiagnosticHudNKeyRealChartMotorEnv(
        segment,key_count=p.KEY_COUNT,
        control_dt_s=CONTROL_DT_S,physics_dt_s=PHYSICS_DT_S,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,tail_s=.04,
    )
    observation=env.reset()
    no_press=NKeyAction((0.,)*p.KEY_COUNT)
    for index,expected in enumerate(coarse):
        now=float(env.privileged_episode_time_s())
        hud=encode_n_key_hud_real_chart_observation(observation)
        if abs(now-expected.seconds)>1e-7:
            raise RuntimeError("Independent 10ms control clock differs from 1ms trace")
        if len(hud)!=len(expected.hud) or any(
            abs(a-b)>1e-5 for a,b in zip(hud,expected.hud)
        ):
            raise RuntimeError(
                f"Independent 10ms HUD differs from physical 1ms decimation at {index}")
        if index<len(coarse)-1:
            state=env.step(no_press)
            if state.done:
                raise RuntimeError("Independent 10ms environment ended early")
            observation=state.observation


def _phase_audit(samples:tuple[Observed,...],segment,*,dt_s:float)->dict:
    """Score only posthoc same-floor pairs against compiled renderer physics."""
    # Candidate inference uses ONLY adjacent HUDs, no floor index or truth.
    predictions=tuple(
        unwrap.unwrap_step(samples[i-1].hud,samples[i].hud,
                           sample_dt_s=dt_s)
        for i in range(1,len(samples))
    )
    codes=Counter()
    same=0
    tail_or_clamped=0
    good=0
    wrong=0
    max_err=0.
    examples=[]
    for i,guess in enumerate(predictions,1):
        # PRIVILEGED: only used now, after all predictions have been made.
        before=segment.chart_time_from_episode(samples[i-1].seconds)
        current=segment.chart_time_from_episode(samples[i].seconds)
        a=segment.chart.floors[
            max(0,_floor_index(segment,before))]
        b=segment.chart.floors[
            max(0,_floor_index(segment,current))]
        if a.index!=b.index or a.index==0:
            continue
        # The renderer clamps the final planet to its exit angle when the
        # last floor ends. Same floor index alone does not mean active orbit
        # motion: the episode has a post-chart tail. Exclude any interval
        # that reaches or exceeds exit_time_s from constant-speed scoring.
        if current>=b.exit_time_s-1e-10:
            tail_or_clamped+=1
            continue
        same+=1
        codes[guess.status]+=1
        if guess.radians is None:
            continue
        sign=-1. if b.is_ccw else 1.
        expected=sign*pi*b.bpm/60.*segment.pitch_ratio*(
            samples[i].seconds-samples[i-1].seconds)
        error=abs(guess.radians-expected)*180./pi
        if not isfinite(error):
            raise RuntimeError("Invalid angular error")
        max_err=max(max_err,error)
        if error<=.05:
            good+=1
        else:
            wrong+=1
            examples.append({
                "frame":i,
                "error_deg":error,
                "estimated_deg":guess.radians*180./pi,
                "expected_deg":expected*180./pi,
            })
    return {
        "sample_ms":dt_s*1000.,
        "total_pairs":max(0,len(samples)-1),
        "posthoc_same_floor_pairs":same,
        "posthoc_tail_or_clamped_pairs_excluded":tail_or_clamped,
        "eligible":good+wrong,
        "correct_within_0p05deg":good,
        "wrong_above_0p05deg":wrong,
        "max_error_deg":max_err if good+wrong else None,
        "statuses_same_floor":dict(sorted(codes.items())),
        "wrong_examples":examples[:MAX_EXAMPLES],
        "note":"Floor labels and true direction used only after causal HUD inference.",
    }


def _floor_index(segment,chart_time:float)->int:
    """Evaluator-only bisect; not visible to the RP1 predictor."""
    times=[f.target_time_s for f in segment.chart.floors]
    return max(0,min(len(times)-1,bisect_right(times,chart_time)-1))


def _timing_audit(samples:tuple[Observed,...],segment)->dict:
    """Separate 10ms countdown accuracy from angle recoverability."""
    frame_inputs=tuple(row.hud for row in samples)
    outputs,original=unwrap.causal_candidates(frame_inputs)
    target_times=[float(t.episode_time_s) for t in segment.targets]
    status={name:Counter() for name in unwrap.CANDIDATES}
    actual_error={name:[] for name in unwrap.CANDIDATES}
    unsafe={name:0 for name in unwrap.CANDIDATES}
    multiple=0
    observed=0
    max_skipped=0
    top=[]
    for i in range(1,len(samples)):
        t0,t1=samples[i-1].seconds,samples[i].seconds
        before=bisect_right(target_times,t0+1e-10)
        after=bisect_right(target_times,t1+1e-10)
        crossed=after-before
        max_skipped=max(max_skipped,crossed)
        if crossed>=2:
            multiple+=1
        # Oracle used ONLY after outputs computed above.
        next_index=bisect_right(target_times,t1+1e-10)
        truth=(target_times[next_index]-t1
               if next_index<len(target_times) else None)
        if truth is not None:
            observed+=1
        for name in unwrap.CANDIDATES:
            proposal=outputs[name][i]
            status[name][proposal.status]+=1
            if proposal.seconds is None:continue
            if crossed>=2:
                unsafe[name]+=1
            if truth is None:continue
            error=(proposal.seconds-truth)*1000.
            actual_error[name].append(error)
            if abs(error)>.05 and len(top)<MAX_EXAMPLES:
                top.append({
                    "frame":i,"bpm":unwrap._bpm(frame_inputs[i]),
                    "crossed_target_count":crossed,
                    "original_status":original[i],
                    "candidate":name,"signed_timing_error_ms":error,
                })
    candidates={}
    for name in unwrap.CANDIDATES:
        errors=actual_error[name]
        candidates[name]={
            "statuses":dict(sorted(status[name].items())),
            "eligible_with_next_target":len(errors),
            "mae_ms":(sum(abs(x) for x in errors)/len(errors)) if errors else None,
            "within_0p05ms":sum(abs(x)<=.05 for x in errors),
            "wrong_over_0p05ms":sum(abs(x)>.05 for x in errors),
            "eligible_on_multi_target_interval":unsafe[name],
        }
    return {
        "sample_ms":CONTROL_DT_S*1000.,
        "pairs":max(0,len(samples)-1),
        "pairs_with_next_target":observed,
        "pairs_crossing_two_or_more_targets":multiple,
        "max_targets_crossed_per_sample":max_skipped,
        "candidates":candidates,
        "incorrect_timing_examples":top,
        "note":"True target timestamps are evaluator-only; a correct RP1 step does not establish event identity.",
    }


def _one(bpm:float,reverse:bool)->dict:
    segment,chart_sha=_fixture(bpm,reverse=reverse)
    samples,done=_capture(segment)
    coarse=samples[::10]
    if not 5<len(coarse)<len(samples):
        raise RuntimeError("Insufficient 1ms/10ms HUD samples")
    _verify_direct_10ms(segment,coarse)
    # Output range must be finite and unclipped, including at 20k.
    expected=unwrap._bpm(samples[0].hud)
    if expected is None or abs(expected-bpm)>.03 or (
        bpm>1920. and samples[0].hud[cd.HUD_OFFSET]<=4.
    ):
        raise RuntimeError("Simulator BPM is clipped or corrupted")
    highres=_phase_audit(samples,segment,dt_s=PHYSICS_DT_S)
    regular=_phase_audit(coarse,segment,dt_s=CONTROL_DT_S)
    timing=_timing_audit(coarse,segment)
    return {
        "bpm":bpm,"direction":"ccw" if reverse else "cw",
        "chart_sha256":chart_sha,
        "target_count":len(segment.targets),
        "chart_duration_ms":segment.duration_s*1000.,
        "hud_encoded_tile_bpm":samples[0].hud[cd.HUD_OFFSET],
        "hud_decoded_tile_bpm":expected,
        "episode_done":done,
        "independent_10ms_replay_verified":True,
        "samples_1ms":len(samples),"samples_10ms":len(coarse),
        "phase_1ms":highres,
        "phase_10ms":regular,
        "next_target_timing_10ms":timing,
    }


def run_simulator_20k_audit(config:p.PrestigeConfig)->dict:
    # This does not call Stage A/T checkpoint loaders or evaluate any
    # user dataset. All charts are generated from public fixed constants.
    if p.CONTROL_DT!=CONTROL_DT_S or p.PHYSICS_DT!=PHYSICS_DT_S:
        raise RuntimeError("Prestige I physical clock contract changed")
    root=transfer.output_dir(config)
    path=root/REPORT
    if path.exists():
        raise FileExistsError("Refuse to replace prior deterministic simulator report")
    cases=[]
    for bpm in BPM_POINTS:
        for reverse in DIRECTIONS:
            result=_one(bpm,reverse)
            cases.append(result)
            phase=result["phase_1ms"]
            regular=result["phase_10ms"]
            timing=result["next_target_timing_10ms"]
            print(
                f"sim {bpm:>7.1f} BPM {result['direction']}: "
                f"v2={result['hud_decoded_tile_bpm']:.3f}, "
                f"1ms phase={phase['correct_within_0p05deg']}/{phase['posthoc_same_floor_pairs']} "
                f"(wrong={phase['wrong_above_0p05deg']}), "
                f"10ms phase={regular['correct_within_0p05deg']}/{regular['posthoc_same_floor_pairs']}, "
                f"10ms multi-target={timing['pairs_crossing_two_or_more_targets']}/"
                f"{timing['pairs']} max={timing['max_targets_crossed_per_sample']}",
                flush=True)
    if len(cases)!=len(BPM_POINTS)*len(DIRECTIONS):
        raise RuntimeError("Fixed simulator fixture set incomplete")
    failure=[r for r in cases if (
        r["phase_1ms"]["wrong_above_0p05deg"]>0 or
        r["hud_decoded_tile_bpm"]>20000.03
    )]
    report={
        "format":FORMAT,
        "purpose":"Deterministic real 8K physics/HUD renderer probe; not model adoption",
        "config_sha256":config.signature(),
        "sample_physics_ms":1.,
        "sample_control_ms":10.,
        "bpm_points":BPM_POINTS,
        "directions":("cw","ccw"),
        "max_bpm_requested":MAX_BPM,
        "no_bpm_clip":True,"no_training":True,
        "motor_student_training_permitted":False,
        "original_stage_a_gate_overridden":False,
        "validation_evaluated":False,"final_evaluated":False,
        "geometry_changed":False,"hybrid_changed":False,
        "checkpoint_loaded":False,
        "unsafe_for_adoption":bool(failure or any(
            r["next_target_timing_10ms"]["pairs_crossing_two_or_more_targets"]>0
            for r in cases
        )),
        "phase_1ms_wrong_cases":[{
            "bpm":r["bpm"],"direction":r["direction"],
            "wrong":r["phase_1ms"]["wrong_above_0p05deg"],
        } for r in failure],
        "cases":cases,
        "limitations":[
            "1ms HUD probe is diagnostic only; original Student sees 10ms HUD",
            "No key-down or human accuracy; no guarantee of playability at 20k",
            "10ms can cross several targets; event identity is not encoded by BPM",
            "No hidden times, chart floor ordinals or simulator clock in inference",
            "No training/chart holdout/Validation/FINAL",
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
