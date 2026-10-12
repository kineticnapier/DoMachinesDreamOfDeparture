from __future__ import annotations

"""Read-only 8-key motor sensitivity audit using the existing chart/motor physics.

This module NEVER changes the Student, motor coefficients, score rules or
checkpoint contents. Privileged target information is used only for scoring.
Open-loop replays stop at the last RECORDED control frame (no extrapolation).
Closed-loop re-infers the Student policy from newly observed physical states.
"""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.n_key import NKeyAction
from dmdod.training.fixed_state_action_audit import (
    EARLY_FAILURE_ANCHORS, assert_state_equal, file_sha256, select_controls,
)
from dmdod.training.timing_motor_probe import _unchanged_tensor
from dmdod.training.timing_motor_probe_v2 import onset_svg
from dmdod.training.real_chart import (
    clone_model_state, evaluate_role_continuous,
)

WINDOW_S = 0.200
CLOSED_LOOP_S = 0.400
AMPLITUDES = (0.02, 0.05, 0.10, 0.20)
ARMS = ("best", "balanced", "onset_focused")
EVENT_TOL_S = 1e-5


@dataclass(frozen=True)
class Intervention:
    kind: str
    key_indices: tuple[int, ...] = ()
    amount: float = 0.0
    shift_frames: int = 0
    oracle: bool = False

    @property
    def label(self) -> str:
        indices = ",".join(str(i) for i in self.key_indices)
        return f"{self.kind}:keys={indices}:delta={self.amount:+.2f}:shift={self.shift_frames:+d}"


def interventions(key_count: int, *, early_key_indices=()) -> tuple[Intervention, ...]:
    cases = [Intervention("baseline")]
    for key in range(key_count):
        for d in AMPLITUDES:
            for sign in (-1, 1):
                cases.append(Intervention("single_key", (key,), sign*d))
    for d in (0.05, 0.10):
        for sign in (-1, 1):
            cases.append(Intervention("coordinated_all", tuple(range(key_count)), sign*d))
    for shift in (-2, -1, 1, 2):
        cases.append(Intervention("time_shift", shift_frames=shift, oracle=shift<0))
    if early_key_indices:
        keys=tuple(sorted(set(early_key_indices)))
        for d in (0.05, 0.10):
            cases.append(Intervention("oracle_previous_early", keys, -d, oracle=True))
    return tuple(cases)


def alter_action(original: tuple[float, ...], case: Intervention) -> tuple[float, ...]:
    values=list(original)
    if case.kind in ("single_key","coordinated_all","oracle_previous_early"):
        for k in case.key_indices:
            values[k]=max(-1.,min(1.,values[k]+case.amount))
    return tuple(values)


def shifted_action(sequence: list[tuple[float, ...]], step: int,
                   case: Intervention, *, width: int) -> tuple[float, ...]:
    if case.kind != "time_shift":
        return alter_action(sequence[step],case)
    index = step - case.shift_frames
    if index < 0:
        # The delay has a physically neutral initial prefix, explicitly recorded.
        return (0.,)*width
    if index >= len(sequence):
        raise IndexError("Open-loop advance requires an unavailable future action")
    return sequence[index]


def _environment(prepared,named):
    return DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,key_count=prepared.key_count,
        capture_overload_trace=True,
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )


def _trace_partial(env,frame_data,physical_samples,*,mode,case,model):
    # Trace lists are diagnostic-only. A 200/400ms stop need not be terminal.
    keydowns=list(env._trace_physical_keydowns)
    early=list(env._trace_too_early)
    states=[]
    for tr in env._trace_action_frames:
        states.append({
            "command_time_s":float(tr.time_s),
            "action":list(tr.action_values),
            "position_mm_end":[x*1000. for x in tr.positions_m],
            "velocity_mm_s_end":[v*1000. for v in tr.velocities_m_s],
            "pressed_end":list(tr.pressed_flags),
        })
    key_rows=[
        {"time_s":float(k.time_s),"key":k.key,
         "target_index":k.target_index,
         "target_time_s":k.target_time_s,
         "error_ms":(1000*(k.time_s-k.target_time_s)
                     if k.target_time_s is not None else None)}
        for k in keydowns
    ]
    early_rows=[{
        "time_s":float(e.time_s),"key":e.key,"target_index":e.target_index,
        "error_ms":float(e.error_ms),
        "gauge_before":float(e.overload_before),
        "gauge_after":float(e.overload_after),
        "fail_overload":bool(e.fail_overload)
    } for e in early]
    failed=next((e["time_s"] for e in early_rows if e["fail_overload"]),None)
    return {
        "mode":mode,"case":case.label,"oracle":case.oracle,"model":model,
        "duration_s":float(env.privileged_episode_time_s()),
        "full_episode_ended":bool(env._done),
        "overload":bool(env._overload.overloaded),
        "overload_time_s":failed,
        "hits":int(env._hits),"misses":int(env._misses),
        "too_early":int(env._too_early),"keydowns":int(env.physical_keydowns),
        "gauge":float(env._overload.value),
        "key_events":key_rows,
        "too_early_events":early_rows,
        "action_frames":states,
        "contact_physics_samples":physical_samples,
        "mean_abs_action_change":(
            sum(sum(abs(x) for x in diff) for diff in frame_data)/
            (sum(len(diff) for diff in frame_data)) if frame_data else 0.
        ),
        "max_abs_action_change":max(
            (abs(v) for diff in frame_data for v in diff),default=0.
        ),
    }


def simulate(prepared,model,named,*,case:Intervention,
             mode:str,recorded_actions=None,horizon_s:float=CLOSED_LOOP_S,
             model_name:str="best",physics_trace:bool=False):
    """Mode-specific action source with identical physical environment/initial reset."""
    if mode not in ("open_loop","closed_loop"):
        raise ValueError("unknown replay mode")
    if mode=="open_loop":
        if not recorded_actions:
            raise ValueError("open-loop requires recorded reference actions")
        sequence=list(recorded_actions)
        if any(len(a)!=prepared.key_count for a in sequence):
            raise ValueError("reference action width mismatch")
        limit=len(sequence)
        if case.kind=="time_shift" and case.shift_frames<0:
            # Never fabricate baseline actions beyond recorded terminal.
            limit=max(0,limit+case.shift_frames)
        limit=min(limit,math.ceil(horizon_s/prepared.control_dt_s))
    else:
        if case.kind=="time_shift" and case.shift_frames<0:
            raise ValueError("noncausal action advance is open-loop oracle only")
        limit=math.ceil(horizon_s/prepared.control_dt_s)
    if limit<1:
        raise ValueError("no overlapping recorded frames for intervention")
    env=_environment(prepared,named)
    observation=env.reset()
    if physics_trace:
        env.motor.capture_physics_trace=True
    state=(model.initial_state(prepared.device) if mode=="closed_loop" else None)
    history=[]
    changes=[]
    contact=[]
    model.eval()
    with torch.no_grad():
        for step in range(limit):
            if mode=="open_loop":
                raw=sequence[step]
                modified=shifted_action(sequence,step,case,width=prepared.key_count)
            else:
                x=torch.tensor(
                    encode_n_key_hud_real_chart_observation(observation),
                    dtype=torch.float32,device=prepared.device,
                )
                if x.numel()!=263:
                    raise RuntimeError("Student visible 263D contract violated")
                mean,_,_,state=model.forward_step(x,state)
                raw=tuple(float(value.item()) for value in torch.tanh(mean))
                history.append(raw)
                if case.kind=="time_shift":
                    idx=step-case.shift_frames
                    modified=history[idx] if idx>=0 else (0.,)*prepared.key_count
                else:
                    modified=alter_action(raw,case)
            modified=tuple(max(-1.,min(1.,float(v))) for v in modified)
            changes.append(tuple(a-b for a,b in zip(modified,raw)))
            before_events=len(env._trace_physical_keydowns)
            preceding_positions=tuple(observation.motor.positions_m)
            outcome=env.step(NKeyAction(modified))
            observation=outcome.observation
            if physics_trace:
                substeps=env.motor.last_transition.physics_samples
                for k in env._trace_physical_keydowns[before_events:]:
                    idx=round(float(k.time_s)/prepared.physics_dt_s)-1
                    # Contact states are captured at precise physics substep;
                    # index into the CURRENT control frame only.
                    local_index=idx-step*env.motor.physics_substeps
                    if not 0<=local_index<len(substeps):
                        raise RuntimeError("contact physics sample index mismatch")
                    sample=substeps[local_index]
                    if abs(sample.time_s-k.time_s)>1e-8:
                        raise RuntimeError("contact physics time mismatch")
                    prior_positions=(
                        substeps[local_index-1].positions_m
                        if local_index>0 else preceding_positions
                    )
                    contact.append({
                        "time_s":float(k.time_s),"key":k.key,
                        "positions_mm":[x*1000. for x in sample.positions_m],
                        "velocities_mm_s":[
                            (position-old)*1000./prepared.physics_dt_s
                            for position,old in zip(sample.positions_m,prior_positions)
                        ],
                        "velocity_sampling":"1kHz_backward_position_difference",
                    })
            if outcome.done:
                break
    return _trace_partial(env,changes,contact,mode=mode,case=case,model=model_name)


def _key_pairs(reference:dict,altered:dict) -> tuple[list[dict],int,int]:
    """Match by physical key + occurrence, never assume target identity."""
    def group(result):
        found={}
        for item in result["key_events"]:
            found.setdefault(item["key"],[]).append(item)
        return found
    a,b=group(reference),group(altered)
    records=[]
    for key in sorted(set(a)|set(b)):
        left,right=a.get(key,[]),b.get(key,[])
        for i in range(max(len(left),len(right))):
            x=left[i] if i<len(left) else None
            y=right[i] if i<len(right) else None
            records.append({
                "key":key,"ordinal":i,
                "baseline_s":x["time_s"] if x else None,
                "modified_s":y["time_s"] if y else None,
                "delta_ms":((y["time_s"]-x["time_s"])*1000
                            if x and y else None),
                "matched":bool(x and y),
                "baseline_target_index":x["target_index"] if x else None,
                "modified_target_index":y["target_index"] if y else None,
            })
    return records,sum(e["matched"] for e in records),sum(not e["matched"] for e in records)


def _assert_key_events(expected,actual,*,where:str):
    if len(expected)!=len(actual):
        raise RuntimeError(f"baseline reproduction FAIL {where}: keydown count "
                           f"{len(actual)} != {len(expected)}")
    for i,(a,b) in enumerate(zip(expected,actual)):
        if a["key"]!=b["key"] or abs(a["time_s"]-b["time_s"])>EVENT_TOL_S:
            raise RuntimeError(
                f"baseline reproduction FAIL {where} keydown #{i}: "
                f"{a['key']}@{a['time_s']:.6f} != {b['key']}@{b['time_s']:.6f}"
            )


def verify_recorded_reference(trace,actual:dict,*,horizon:float,where:str):
    """Compare same-frame reference playback, NEVER beyond saved actions."""
    # Check only the physically covered interval; no extrapolation.
    reached=actual["duration_s"]+1e-7
    expected=[
        {"time_s":float(e.time_s),"key":e.key}
        for e in trace.physical_keydowns if e.time_s<=reached
    ]
    got=[{"time_s":e["time_s"],"key":e["key"]} for e in actual["key_events"]]
    _assert_key_events(expected,got,where=where)
    expected_early=[
        e for e in trace.too_early_events if e.time_s<=reached
    ]
    if len(expected_early)!=len(actual["too_early_events"]):
        raise RuntimeError(f"baseline reproduction FAIL {where}: TooEarly mismatch")
    for a,b in zip(expected_early,actual["too_early_events"]):
        if a.key!=b["key"] or abs(a.time_s-b["time_s"])>EVENT_TOL_S:
            raise RuntimeError(f"baseline reproduction FAIL {where}: TooEarly time mismatch")


def _full_baseline_reference(data:dict,results,traces,named,*,arm:str):
    """Validate all 20 per-anchor scores/timings before perturbing any actions."""
    refs=(data["baseline_anchors"] if arm=="best"
          else next(a["anchors"] for a in data["arm_comparison"]
                    if a["name"]==arm))
    if len(refs)!=20 or len(results)!=20 or len(traces)!=20:
        raise RuntimeError("baseline reproducibility requires all 20 fixed anchors")
    output=[]
    for i,(ref,(stats,downs),trace,segment) in enumerate(zip(refs,results,traces,named),1):
        expected={
            "hits":int(ref["hits"]),"misses":int(ref["misses"]),
            "too_early":int(ref["too_early"]),
            "keydowns":int(ref["keydowns"]),"overloaded":bool(ref["overloaded"]),
        }
        actual={
            "hits":int(stats.hits),"misses":int(stats.misses),
            "too_early":int(stats.too_early_presses),
            "keydowns":int(downs),"overloaded":bool(stats.overloaded),
        }
        if actual!=expected or bool(ref["early_overload"]) != (
            trace.termination.reason=="Overload" and trace.termination.time_s<=WINDOW_S+1e-9
        ):
            raise RuntimeError(f"baseline reproduction FAIL {arm} anchor #{i:02d}: "
                               f"{actual} != {expected}")
        times=ref["initial_keydown_times_s"]
        got=[e.time_s for e in trace.physical_keydowns if e.time_s<=WINDOW_S+1e-9]
        if len(times)!=len(got) or any(abs(a-b)>EVENT_TOL_S for a,b in zip(times,got)):
            raise RuntimeError(f"baseline reproduction FAIL {arm} anchor #{i:02d}: "
                               "physical KeyDown timestamps differ")
        if ref["overload_at_s"] is not None:
            if abs(ref["overload_at_s"]-trace.termination.time_s)>EVENT_TOL_S:
                raise RuntimeError(f"baseline reproduction FAIL {arm} anchor #{i:02d}: "
                                   "Overload timestamp differs")
        output.append({
            "arm":arm,"anchor":i,"chart":segment.chart_name,
            **actual,"initial_keydowns":len(got),"initial_keydown_times_s":times,
            "overload_time_s":ref["overload_at_s"],"PASS":True,
        })
    return output



def verify_prior_focus_events(prior_directory: Path, traces_by_arm: dict):
    """Fail if previously saved early KeyDown/TooEarly/Overload times disagree."""
    path=prior_directory/"motor_response_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"prior early motor timeline unavailable: {path}")
    groups={}
    with path.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            kind=row["kind"]
            if kind not in ("keydown","too_early","fail_overload"):
                continue
            key=(row["arm"],int(row["anchor"]))
            groups.setdefault(key,[]).append(
                (kind,row["key"],float(row["time_s"]))
            )
    if not groups:
        raise RuntimeError("prior focus CSV contains no scored motor events")
    validated=[]
    for (arm,index),expected in groups.items():
        if arm not in traces_by_arm or not 1<=index<=20:
            raise RuntimeError("prior event timeline references invalid arm/anchor")
        trace=traces_by_arm[arm][index-1]
        actual=[
            ("keydown",e.key,float(e.time_s))
            for e in trace.physical_keydowns if e.time_s<=WINDOW_S+1e-9
        ]+[
            ("fail_overload" if e.fail_overload else "too_early",
             e.key,float(e.time_s))
            for e in trace.too_early_events if e.time_s<=WINDOW_S+1e-9
        ]
        order=lambda entry:(entry[2],entry[0],entry[1])
        expected=sorted(expected,key=order)
        actual=sorted(actual,key=order)
        if len(expected)!=len(actual) or any(
            a[0]!=b[0] or a[1]!=b[1] or abs(a[2]-b[2])>EVENT_TOL_S
            for a,b in zip(expected,actual)
        ):
            raise RuntimeError(
                f"baseline reproduction FAIL {arm} anchor #{index:02d}: "
                "prior event timeline differs"
            )
        validated.append({"arm":arm,"anchor":index,"event_count":len(actual)})
    return validated


def _load_models(prepared,diagnostic_dir:Path,prior:dict):
    model=prepared.model
    source_state=clone_model_state(model)
    source_sha=file_sha256(prepared.source_checkpoint)
    if source_sha != prior.get("source_sha256"):
        raise RuntimeError("source Best checkpoint SHA256 differs from prior audit")
    result={"best":source_state}
    try:
        for arm in ("balanced","onset_focused"):
            f=diagnostic_dir/f"{arm}.diagnostic.pt"
            if not f.is_file():
                raise FileNotFoundError(f"saved experimental arm missing: {f}")
            ckpt=torch.load(f,map_location="cpu",weights_only=False)
            if ckpt.get("human_visible_checkpoint_role")!="diagnostic-candidate" or (
                ckpt.get("diagnostic_arm")!=arm
            ) or ckpt.get("diagnostic_source_sha256")!=source_sha:
                raise RuntimeError(f"diagnostic arm provenance mismatch: {arm}")
            candidate=ckpt["model_state"]
            if set(candidate)!=set(source_state):
                raise RuntimeError(f"diagnostic model keys differ: {arm}")
            fixed={name for name,p in model.named_parameters(remove_duplicate=False)
                   if not p.requires_grad}
            all_parameters={name for name,_ in model.named_parameters(remove_duplicate=False)}
            for key in candidate:
                if key in fixed or key not in all_parameters:
                    if not _unchanged_tensor(candidate[key],source_state[key]):
                        raise RuntimeError(f"diagnostic frozen/buffer changed: {arm}:{key}")
            model.load_state_dict(candidate,strict=True)
            model.prepare_recurrent_runtime()
            result[arm]=clone_model_state(model)
            assert_state_equal(result[arm],candidate,
                               context=f"candidate {arm} load mismatch")
        return result,source_sha
    finally:
        model.load_state_dict(source_state,strict=True)
        model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(model),source_state,
                           context="Best restored after loading diagnostic arms")

def write_csv(path:Path,fields:list[str],rows:list[dict]):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _category(result:dict,reference:dict) -> str:
    """Avoid labelling no-press suppression as an improvement."""
    if result["overload"] and not reference["overload"]:
        return "regression_overload"
    if not result["overload"] and reference["overload"]:
        if (result["keydowns"]==0 or
            (result["hits"]<reference["hits"] and
             result["keydowns"]<reference["keydowns"])):
            return "overload_avoided_by_suppression_or_underperformance"
        return "overload_avoided_check_miss_and_press_count"
    if result["keydowns"]==0 and reference["keydowns"]>0:
        return "no_press_not_improved"
    return "undetermined"


def _intervention_summary(base:dict,altered:dict,case:Intervention) -> dict:
    pairs,matched,unmatched=_key_pairs(base,altered)
    deltas=[x["delta_ms"] for x in pairs if x["delta_ms"] is not None]
    return {
        "case":case.label,"kind":case.kind,"oracle":case.oracle,
        "modified_key_indices":list(case.key_indices),
        "amount":case.amount,"shift_frames":case.shift_frames,
        "action_mean_abs_delta":altered["mean_abs_action_change"],
        "action_max_abs_delta":altered["max_abs_action_change"],
        "matched_keydown_pairs":matched,"unmatched_keydown_pairs":unmatched,
        "mean_keydown_delta_ms":sum(deltas)/len(deltas) if deltas else None,
        "max_abs_keydown_delta_ms":max((abs(x) for x in deltas),default=None),
        "keydowns":altered["keydowns"],
        "keydowns_delta":altered["keydowns"]-base["keydowns"],
        "too_early":altered["too_early"],
        "too_early_delta":altered["too_early"]-base["too_early"],
        "hits":altered["hits"],"misses":altered["misses"],
        "hits_delta":altered["hits"]-base["hits"],
        "overload":altered["overload"],
        "overload_time_s":altered["overload_time_s"],
        "baseline_overload_time_s":base["overload_time_s"],
        "baseline_keydowns":base["keydowns"],
        "baseline_hits":base["hits"],
        "gauge":altered["gauge"],
        "duration_s":altered["duration_s"],
        "classification":_category(altered,base),
    },pairs



def mechanism_sensitivity(rows:list[dict])->list[dict]:
    """Descriptive sensitivity only. NEVER ranks/selects an action for use."""
    grouped={}
    for row in rows:
        if row["kind"]=="baseline":
            continue
        key=(row["arm"],row["mode"],row["kind"])
        grouped.setdefault(key,[]).append(row)
    output=[]
    for (arm,mode,kind),subset in sorted(grouped.items()):
        changed=[x for x in subset if x["keydowns_delta"]!=0
                 or x["too_early_delta"]!=0 or x["hits_delta"]!=0
                 or (x["overload"]!= (x["baseline_overload_time_s"] is not None))]
        shifts=[abs(x["mean_keydown_delta_ms"]) for x in subset
                if x["mean_keydown_delta_ms"] is not None]
        output.append({
            "arm":arm,"mode":mode,"mechanism":kind,
            "n":len(subset),"changed_count_or_score_fraction":len(changed)/len(subset),
            "mean_abs_matched_keydown_delta_ms":(
                sum(shifts)/len(shifts) if shifts else None
            ),
            "overload_avoided_but_suppressed":sum(
                x["classification"]=="overload_avoided_by_suppression_or_underperformance"
                for x in subset
            ),
            "overload_avoided_not_yet_validated":sum(
                x["classification"]=="overload_avoided_check_miss_and_press_count"
                for x in subset
            ),
            "oracle_case_count":sum(x["oracle"] for x in subset),
        })
    return output


def run_motor_sensitivity_audit(prepared, *,
    prior_directory:Path, output_dir:Path, arm_state_dir:Path|None=None,
    closed_loop_s:float=CLOSED_LOOP_S) -> dict:
    if len(prepared.validation)!=20 or prepared.key_count!=8 or prepared.input_dim!=263:
        raise ValueError("requires fixed 20 Validation / 8 keys / 263D")
    if closed_loop_s < WINDOW_S or closed_loop_s>1.0:
        raise ValueError("closed_loop_s must be between 0.2 and 1.0")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {output_dir}")
    if output_dir.resolve()==prior_directory.resolve():
        raise ValueError("output cannot overwrite previous audit")
    if arm_state_dir is None:
        arm_state_dir=prior_directory
    expected_path=prior_directory/"pilot_reproduction.json"
    summary_path=prior_directory/"summary.json"
    if not expected_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("prior audit summary and pilot reproduction required")
    prior=json.loads(summary_path.read_text(encoding="utf-8"))
    pilot=json.loads(expected_path.read_text(encoding="utf-8"))
    if prior.get("kind")!="fixed_state_action_delta_audit":
        raise ValueError("not a recognized prior audit")
    model=prepared.model
    best=clone_model_state(model)
    states,source_sha=_load_models(prepared,arm_state_dir,prior)
    output_dir.mkdir(parents=True,exist_ok=True)
    baseline_rows=[]
    full_traces={}
    try:
        for arm in ARMS:
            model.load_state_dict(states[arm],strict=True)
            model.prepare_recurrent_runtime()
            traces=[]
            results=evaluate_role_continuous(
                model,prepared.validation,label=f"motor-sens-baseline-{arm}",
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s,device=prepared.device,
                verbose=False,trace_collector=traces,
            )
            baseline_rows.extend(_full_baseline_reference(
                pilot,results,traces,prepared.validation,arm=arm
            ))
            full_traces[arm]=tuple(traces)
        # All models and ALL fixed Validation anchors reproduced exactly.
        checked_prior_events=verify_prior_focus_events(prior_directory,full_traces)
        focus=tuple(sorted(set(EARLY_FAILURE_ANCHORS) |
                           set(select_controls(full_traces["best"]))))
        action_rows=[]
        physical_rows=[]
        closed_rows=[]
        json_results=[]
        for arm in ARMS:
            model.load_state_dict(states[arm],strict=True)
            model.prepare_recurrent_runtime()
            for index in focus:
                named=prepared.validation[index-1]
                original_trace=full_traces[arm][index-1]
                sequence=[tuple(f.action_values) for f in original_trace.action_frames]
                if not sequence:
                    raise RuntimeError(f"no original action sequence {arm} #{index}")
                early_keys={
                    e.key for e in original_trace.too_early_events
                    if e.time_s<=WINDOW_S+1e-9
                }
                from dmdod.motor.n_key import n_key_names
                names=n_key_names(prepared.key_count)
                oracle_indices=tuple(k for k,name in enumerate(names) if name in early_keys)
                cases=interventions(prepared.key_count,early_key_indices=oracle_indices)
                # Baseline open-loop EXACT saved action sequence, no inferred tail.
                reference_open=simulate(
                    prepared,model,named,case=Intervention("baseline"),
                    mode="open_loop",recorded_actions=sequence,horizon_s=closed_loop_s,
                    model_name=arm,physics_trace=True,
                )
                verify_recorded_reference(
                    original_trace,reference_open,horizon=closed_loop_s,
                    where=f"{arm} #{index:02d} open_loop"
                )
                # Baseline Closed-loop re-reads observations and newly simulated body.
                reference_closed=simulate(
                    prepared,model,named,case=Intervention("baseline"),
                    mode="closed_loop",horizon_s=closed_loop_s,
                    model_name=arm,physics_trace=True,
                )
                verify_recorded_reference(
                    original_trace,reference_closed,horizon=closed_loop_s,
                    where=f"{arm} #{index:02d} closed_loop"
                )
                for mode,reference in (("open_loop",reference_open),
                                       ("closed_loop",reference_closed)):
                    svg=output_dir/f"onset_{arm}_{index:02d}_{mode}.svg"
                    if svg.exists():
                        raise FileExistsError(f"onset SVG exists: {svg}")
                    onset_svg({
                        "index":index,
                        "chart":f"{named.chart_name} {arm} {mode}",
                        "frames":[{
                            "time_s":f["command_time_s"],
                            "action":f["action"],
                            "position_mm_end":f["position_mm_end"],
                            "velocity_mm_s_end":f["velocity_mm_s_end"],
                        } for f in reference["action_frames"]],
                        "events":[
                            {"time_s":e["time_s"],"key":e["key"],"kind":"keydown"}
                            for e in reference["key_events"]
                        ]+[
                            {"time_s":e["time_s"],"key":e["key"],
                             "kind":"fail_overload" if e["fail_overload"] else "too_early"}
                            for e in reference["too_early_events"]
                        ],
                    },svg)
                    for case in cases:
                        if case.kind=="baseline":
                            modified=reference
                        elif mode=="closed_loop" and case.shift_frames<0:
                            # A closed-loop action cannot be advanced without
                            # unseen future observations; never fake one.
                            continue
                        else:
                            modified=simulate(
                                prepared,model,named,case=case,mode=mode,
                                recorded_actions=sequence if mode=="open_loop" else None,
                                horizon_s=closed_loop_s,model_name=arm,
                                physics_trace=True,
                            )
                        comparator=reference
                        if mode=="open_loop" and case.shift_frames<0:
                            # Re-evaluate reference over the same shortened
                            # recorded horizon as this noncausal advance.
                            remaining=max(1,len(sequence)+case.shift_frames)
                            comparator=simulate(
                                prepared,model,named,case=Intervention("baseline"),
                                mode="open_loop",recorded_actions=sequence,
                                horizon_s=min(closed_loop_s,
                                              remaining*prepared.control_dt_s),
                                model_name=arm,
                            )
                        row,pairs=_intervention_summary(comparator,modified,case)
                        row.update({"arm":arm,"anchor":index,"chart":named.chart_name,
                                    "mode":mode,"reference_horizon_s":reference["duration_s"]})
                        destination=closed_rows if mode=="closed_loop" else action_rows
                        destination.append(row)
                        contact_by_key={}
                        for contact in modified["contact_physics_samples"]:
                            contact_by_key.setdefault(contact["key"],[]).append(contact)
                        for p in pairs:
                            recorded=contact_by_key.get(p["key"],[])
                            contact=(recorded[p["ordinal"]]
                                     if p["ordinal"]<len(recorded) else None)
                            physical_rows.append({
                                "arm":arm,"anchor":index,"chart":named.chart_name,
                                "mode":mode,"case":case.label,**p,
                                "contact_position_mm":(
                                    json.dumps(contact["positions_mm"]) if contact else None
                                ),
                                "contact_velocity_mm_s":(
                                    json.dumps(contact["velocities_mm_s"]) if contact else None
                                ),
                                "velocity_sampling":(
                                    contact["velocity_sampling"] if contact else None
                                ),
                            })
                        if case.kind=="baseline":
                            json_results.append({
                                "arm":arm,"anchor":index,"chart":named.chart_name,
                                "mode":mode,"baseline":modified
                            })
        write_csv(output_dir/"baseline_reproduction.csv",
                  ["arm","anchor","chart","hits","misses","too_early",
                   "keydowns","overloaded","initial_keydowns",
                   "initial_keydown_times_s","overload_time_s","PASS"],baseline_rows)
        cols=["arm","anchor","chart","mode","case","kind","oracle","modified_key_indices",
              "amount","shift_frames","action_mean_abs_delta","action_max_abs_delta",
              "matched_keydown_pairs","unmatched_keydown_pairs",
              "mean_keydown_delta_ms","max_abs_keydown_delta_ms",
              "keydowns","keydowns_delta","baseline_keydowns",
              "too_early","too_early_delta",
              "hits","misses","hits_delta","baseline_hits","overload","overload_time_s",
              "baseline_overload_time_s","gauge","duration_s",
              "reference_horizon_s","classification"]
        write_csv(output_dir/"action_sensitivity.csv",cols,action_rows)
        write_csv(output_dir/"closed_loop_interventions.csv",cols,closed_rows)
        write_csv(output_dir/"physical_keydown_deltas.csv",
                  ["arm","anchor","chart","mode","case","key","ordinal",
                   "baseline_s","modified_s","delta_ms","matched",
                   "baseline_target_index","modified_target_index",
                   "contact_position_mm","contact_velocity_mm_s",
                   "velocity_sampling"],physical_rows)
        (output_dir/"baseline_traces.json").write_text(
            json.dumps(json_results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )
        sensitivities=mechanism_sensitivity([*action_rows,*closed_rows])
        write_csv(
            output_dir/"mechanism_sensitivity.csv",
            ["arm","mode","mechanism","n","changed_count_or_score_fraction",
             "mean_abs_matched_keydown_delta_ms","overload_avoided_but_suppressed",
             "overload_avoided_not_yet_validated","oracle_case_count"],
            sensitivities,
        )
        result={
            "kind":"motor_sensitivity_v1",
            "source_sha256":source_sha,
            "prior_audit":str(prior_directory),
            "closed_loop_horizon_s":closed_loop_s,
            "open_loop_horizon_policy":"only original recorded frames; no padding",
            "focused_anchors":list(focus),
            "baseline_passed_anchors":len(baseline_rows),
            "prior_event_timelines_verified":checked_prior_events,
            "arms":list(ARMS),
            "mechanism_sensitivity":sensitivities,
            "interventions":{
                "single_key_perturbations":list(AMPLITUDES),
                "time_shift_frames":[-2,-1,1,2],
                "coordinated_deltas":[-.10,-.05,.05,.10],
                "negative_shift_is_open_loop_noncausal_oracle":True,
                "previous_too_early_key_cases_are_oracle":True,
            },
            "counts":{
                "open_loop_experiments":len(action_rows),
                "closed_loop_experiments":len(closed_rows),
                "keydown_pair_rows":len(physical_rows),
            },
            "limitations":[
                "Closed-loop observations change after physical perturbations.",
                "Open-loop tail is never guessed past original terminal.",
                "Short diagnostic horizons cannot establish final chart SAFE.",
                "No-press reduction in Overload is not automatically improvement.",
                "No Validation-based training/selection/hyperparameter search.",
                "No FINAL holdout or checkpoint modification.",
            ]
        }
        if file_sha256(prepared.source_checkpoint)!=source_sha:
            raise RuntimeError("original selected-best checkpoint changed during audit")
        (output_dir/"summary.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )
        return result
    finally:
        model.load_state_dict(best,strict=True)
        model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(model),best,
                           context="sensitivity audit restored Best")
