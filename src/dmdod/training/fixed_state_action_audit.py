from __future__ import annotations

"""Fixed-State Action Delta Audit: experimental artifacts, never promotion.

The existing onset pilot owns optimization and fixed Validation; hooks capture
the same train trajectories and Validation traces without extra collection.
Every model replays each fixed observation sequence from its OWN fresh RNN
state. Teacher labels are used exclusively for comparison, never as input.
"""

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from dmdod.training.human_visible_curriculum import (
    TEACHER_ACTIVE_THRESHOLD,
    n_key_actuation_loss_per_frame,
    validation_summary,
)
from dmdod.training.real_chart import clone_model_state, replay_policy_on_fixed_observations
from dmdod.training.timing_motor_probe import _unchanged_tensor
from dmdod.training.timing_motor_probe_v2 import onset_svg

THRESHOLD = 0.25
INITIAL_S = 0.200
EARLY_FAILURE_ANCHORS = (6, 10, 11, 15, 17)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_state_equal(actual: dict, expected: dict, *, context: str):
    if set(actual) != set(expected):
        raise RuntimeError(f"{context}: state keys changed")
    for key in expected:
        if not _unchanged_tensor(actual[key], expected[key]):
            raise RuntimeError(f"{context}: state mutated at {key}")


def _tensor_float_norm(tensor: torch.Tensor) -> float:
    # Trainable tensors are dense; frozen sparse connectome never gets densified.
    return float(torch.linalg.vector_norm(tensor.detach().double().reshape(-1)))


def parameter_deltas(model, original: dict, candidate: dict, *,
                     arm: str, gradient_norm: float | None = None) -> tuple[list[dict], dict]:
    # Include aliases so tied trainable parameters are not misclassified as frozen.
    trainable={name: param for name,param in model.named_parameters(remove_duplicate=False)
               if param.requires_grad}
    if set(original) != set(candidate):
        raise ValueError("parameter delta states have different keys")
    rows=[]
    changed_total=0
    trainable_total=0
    frozen_count=0
    nonfinite=[]
    layer_sums={}
    for key in sorted(original):
        a,b=original[key],candidate[key]
        if key not in trainable:
            if not _unchanged_tensor(a,b):
                raise RuntimeError(f"frozen parameter/buffer changed: {key}")
            frozen_count+=1
            continue
        if a.layout != torch.strided or b.layout != torch.strided:
            raise RuntimeError(f"trainable sparse state is unsupported: {key}")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"parameter shape/dtype changed: {key}")
        trainable_total+=a.numel()
        diff=b.detach().double()-a.detach().double()
        if not torch.isfinite(diff).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            nonfinite.append(key)
            raise RuntimeError(f"nonfinite trainable parameter delta: {key}")
        count=int(torch.count_nonzero(diff))
        changed_total+=count
        delta_norm=_tensor_float_norm(diff)
        start_norm=_tensor_float_norm(a)
        layer=key.rsplit(".",1)[0] if "." in key else key
        totals=layer_sums.setdefault(layer,[0.,0.,0,0])
        totals[0]+=delta_norm**2
        totals[1]+=start_norm**2
        totals[2]+=count
        totals[3]+=a.numel()
        grad=trainable[key].grad
        grad_last=None if grad is None else _tensor_float_norm(grad)
        rows.append({
            "arm":arm,"parameter":key,"layer":layer,
            "elements":a.numel(),"changed_elements":count,
            "start_l2":start_norm,"delta_l2":delta_norm,
            "relative_delta_l2":(delta_norm/start_norm if start_norm>0 else
                                 (0. if delta_norm==0 else None)),
            "last_gradient_l2":grad_last,
            "nonfinite":False,"small_update":(
                delta_norm/start_norm<1e-7 if start_norm>0 else delta_norm<1e-10
            ),
        })
    summary={
        "arm":arm,"trainable_parameter_elements":trainable_total,
        "changed_parameter_elements":changed_total,
        "frozen_and_buffer_state_entries_unchanged":frozen_count,
        "nonfinite_trainable_parameters":nonfinite,
        "training_gradient_mean_l2":gradient_norm,
        "layer_norms":[{
            "layer":name,"delta_l2":math.sqrt(x[0]),"original_l2":math.sqrt(x[1]),
            "relative_l2":(
                math.sqrt(x[0]/x[1]) if x[1] else (0. if x[0]==0 else None)
            ),
            "changed_elements":x[2],"elements":x[3]
        } for name,x in sorted(layer_sums.items())],
    }
    return rows,summary


def select_controls(best_traces, *, count=2) -> tuple[int,...]:
    selected=[]
    for index,trace in enumerate(best_traces,1):
        if index in EARLY_FAILURE_ANCHORS:
            continue
        if trace.termination.time_s <= INITIAL_S:
            continue
        if any(x.time_s <= INITIAL_S for x in trace.too_early_events):
            continue
        selected.append(index)
        if len(selected)>=count:
            break
    if not selected:
        raise RuntimeError("no normal-start Validation control anchor")
    return tuple(selected)


def trace_first_200ms(trace) -> dict:
    actions=[
        {"t_s":float(f.time_s),
         "actions":list(f.action_values),
         "positive_keys":sum(v>=THRESHOLD for v in f.action_values)}
        for f in trace.action_frames if f.time_s<INITIAL_S-1e-9
    ]
    keydowns=[
        {"t_s":float(e.time_s),"key":e.key,"target_index":e.target_index}
        for e in trace.physical_keydowns if e.time_s <= INITIAL_S+1e-9
    ]
    early=[
        {"t_s":float(e.time_s),"key":e.key,"fail_overload":bool(e.fail_overload)}
        for e in trace.too_early_events if e.time_s <= INITIAL_S+1e-9
    ]
    return {
        "actions":actions,"keydowns":keydowns,"too_early":early,
        "first_keydown_s":keydowns[0]["t_s"] if keydowns else None,
        "early_overload_s":(
            float(trace.termination.time_s)
            if trace.termination.reason=="Overload"
            and trace.termination.time_s<=INITIAL_S+1e-9 else None
        ),
        "initial_keydown_count":len(keydowns),
        "initial_max_positive_keys":max((f["positive_keys"] for f in actions),default=0),
    }


def _join_motor_timeline(arm: str, index: int, chart: str, trace) -> list[dict]:
    out=[]
    events=trace_first_200ms(trace)
    for frame in events["actions"]:
        out.append({
            "arm":arm,"anchor":index,"chart":chart,
            "kind":"action","time_s":frame["t_s"],
            "key":"","actions":json.dumps(frame["actions"],separators=(",",":")),
            "positive_keys":frame["positive_keys"],
            "early_overload_s":events["early_overload_s"]
        })
    for item in events["keydowns"]:
        out.append({
            "arm":arm,"anchor":index,"chart":chart,
            "kind":"keydown","time_s":item["t_s"],
            "key":item["key"],"actions":"","positive_keys":"",
            "early_overload_s":events["early_overload_s"]
        })
    for item in events["too_early"]:
        out.append({
            "arm":arm,"anchor":index,"chart":chart,
            "kind":"fail_overload" if item["fail_overload"] else "too_early",
            "time_s":item["t_s"],"key":item["key"],"actions":"",
            "positive_keys":"","early_overload_s":events["early_overload_s"]
        })
    return sorted(out,key=lambda r:(r["time_s"],r["kind"],r["key"]))




def save_motor_onset_svg(arm: str, index: int, chart: str, trace, output: Path):
    frames=[{
        "time_s":float(frame.time_s),
        "action":tuple(frame.action_values),
        "position_mm_end":tuple(float(x)*1000. for x in frame.positions_m),
        "velocity_mm_s_end":tuple(float(x)*1000. for x in frame.velocities_m_s),
    } for frame in trace.action_frames if frame.time_s<INITIAL_S-1e-9]
    events=[
        {"time_s":float(event.time_s),"kind":"keydown","key":event.key}
        for event in trace.physical_keydowns if event.time_s<=INITIAL_S+1e-9
    ]
    events.extend({
        "time_s":float(event.time_s),
        "kind":"fail_overload" if event.fail_overload else "too_early",
        "key":event.key,
    } for event in trace.too_early_events if event.time_s<=INITIAL_S+1e-9)
    report={"index":index,"chart":f"{chart} ({arm})",
            "frames":frames,"events":events}
    target=output/f"motor_onset_{arm}_{index:02d}.svg"
    if target.exists():
        raise FileExistsError(f"onset SVG exists: {target}")
    onset_svg(report,target)

def _write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("x",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _result_metrics(pred: torch.Tensor, target: torch.Tensor,
                    base: torch.Tensor, *, control_dt: float) -> dict:
    """Per-frame set-valued objective from the EXISTING training loss."""
    if pred.shape!=target.shape or pred.shape!=base.shape:
        raise ValueError("fixed-state replay tensor shapes disagree")
    pred,base,target=pred.float(),base.float(),target.float()
    n=pred.shape[0]
    if n==0:
        raise ValueError("empty fixed observation trajectory")
    delta=(pred-base).abs()
    set_loss=n_key_actuation_loss_per_frame(pred,target)
    base_loss=n_key_actuation_loss_per_frame(base,target)
    required=(target>TEACHER_ACTIVE_THRESHOLD).sum(1)
    commands=(pred>=THRESHOLD).sum(1)
    base_commands=(base>=THRESHOLD).sum(1)
    crosses=(pred>=THRESHOLD)!=(base>=THRESHOLD)
    # Use float64 for boundary times: float32 represents 200ms as slightly
    # below 200ms for common dt values (10ms/20ms/5ms), adding one frame.
    onset=torch.arange(n,dtype=torch.float64)*float(control_dt)<INITIAL_S-1e-9
    def numbers(selection):
        if not bool(selection.any()):
            return {"frames":0}
        d=delta[selection]
        c=commands[selection]
        t=required[selection]
        return {
            "frames":int(selection.sum()),"mean_abs_delta":float(d.mean()),
            "max_abs_delta":float(d.max()),
            "set_loss":float(set_loss[selection].mean()),
            "best_set_loss":float(base_loss[selection].mean()),
            "loss_change":float((set_loss-base_loss)[selection].mean()),
            "key_mse":float((pred[selection]-target[selection]).square().mean()),
            "required_press_count_mean":float(t.float().mean()),
            "commanded_press_count_mean":float(c.float().mean()),
            "best_commanded_press_count_mean":float(base_commands[selection].float().mean()),
            "missing_press_count_mean":float((t-c).clamp_min(0).float().mean()),
            "excess_press_count_mean":float((c-t).clamp_min(0).float().mean()),
            "threshold_crossings":int(crosses[selection].sum()),
            "frames_with_threshold_crossings":int(crosses[selection].any(1).sum()),
        }
    return {"all":numbers(torch.ones(n,dtype=torch.bool)),
            "onset":numbers(onset)}


def fixed_trajectory_rows(model, trajectories, states: dict, *, device, control_dt):
    """Re-run each model on the SAME Train observations (no environment step)."""
    per_frame=[]
    per_anchor=[]
    onset=[]
    for sequence in trajectories:
        # Recurrent initialization occurs independently per (model, trajectory).
        actions={}
        for arm,state in states.items():
            model.load_state_dict(state,strict=True)
            model.prepare_recurrent_runtime()
            actions[arm]=replay_policy_on_fixed_observations(
                model,sequence.observations,device=device
            )
        base=actions["best"]
        targets=sequence.teacher_actions.detach().cpu()
        for arm, pred in actions.items():
            metrics=_result_metrics(pred,targets,base,control_dt=control_dt)
            per_frame_loss=n_key_actuation_loss_per_frame(
                pred.float(),targets.float()
            ).detach().cpu()
            header={
                "arm":arm,"anchor":sequence.anchor_id+1,
                "source":sequence.source_kind,"frames":sequence.frames,
            }
            per_anchor.append({**header,**{
                ("onset_"+k):v for k,v in metrics["onset"].items()
            },**{
                ("all_"+k):v for k,v in metrics["all"].items()
            }})
            onset.append({**header,**metrics["onset"]})
            diff=(pred-base)
            teacher_positive=(targets>TEACHER_ACTIVE_THRESHOLD).sum(1)
            for i in range(sequence.frames):
                a,b,d=pred[i],base[i],diff[i]
                crossing=(a>=THRESHOLD)!=(b>=THRESHOLD)
                per_frame.append({
                    "arm":arm,"anchor":sequence.anchor_id+1,
                    "source":sequence.source_kind,"frame":i,
                    "time_s":i*control_dt,
                    "student_action":json.dumps(a.tolist(),separators=(",",":")),
                    "teacher_action":json.dumps(targets[i].tolist(),separators=(",",":")),
                    "best_action":json.dumps(b.tolist(),separators=(",",":")),
                    "delta_action":json.dumps(d.tolist(),separators=(",",":")),
                    "mean_abs_delta":float(d.abs().mean()),
                    "max_abs_delta":float(d.abs().max()),
                    "press_keys":int((a>=THRESHOLD).sum()),
                    "teacher_press_keys":int(teacher_positive[i]),
                    "threshold_crossing_keys":json.dumps(torch.nonzero(crossing).flatten().tolist()),
                    "threshold_crossing_count":int(crossing.sum()),
                    "set_valued_loss":float(per_frame_loss[i]),
                    "teacher_key_mse":float((a-targets[i]).square().mean()),
                })
    return per_frame,per_anchor,onset




def reference_reproduction(report: dict) -> dict:
    """Historical pilot checkpoints were not saved; report, never invent, agreement."""
    expected={
        "best":{"safe":12,"hits":1786,"early":278,"early_overload":5},
        "balanced":{"safe":10,"hits":1685,"early":297,"early_overload":5},
        "onset_focused":{"safe":7,"hits":1627,"early":235,"early_overload":5},
    }
    observed={"best":{
        **report["baseline_validation"],
        "early_overload":report["baseline_onset"]["early_overload"],
    }}
    for arm in report["arm_comparison"]:
        observed[arm["name"]]={
            **arm["validation"],"early_overload":arm["onset"]["early_overload"]
        }
    differences={}
    for arm,fields in expected.items():
        actual=observed.get(arm,{})
        mismatch={
            k:{"expected":v,"observed":actual.get(k)}
            for k,v in fields.items() if actual.get(k)!=v
        }
        differences[arm]=mismatch
    return {
        "expected":expected,
        "match_all":not any(differences.values()),
        "mismatches":differences,
        "disclaimer":"Reference mismatch means the pilot did not numerically reproduce; do not infer causality from historical comparisons.",
    }

class ActionDeltaAudit:
    def __init__(self,prepared, *, output_dir: Path, source_sha256: str,seed:int):
        self.prepared=prepared
        self.output_dir=output_dir
        self.source_sha256=source_sha256
        self.seed=seed
        self.original=clone_model_state(prepared.model)
        self.train_trajectories=None
        self.arm_states={"best":self.original}
        self.validation={}
        self.parameter_rows=[]
        self.parameter_summaries={}
        self.training_metrics={}
        self.checkpoint_paths={}

    def record_baseline(self, model, results, traces):
        assert_state_equal(clone_model_state(model),self.original,context="best baseline")
        self.validation["best"]=(list(results),tuple(traces))

    def record_train_trajectories(self, trajectories):
        self.train_trajectories=tuple(trajectories)

    def record_arm(self,name,model,optimizer,metrics,results,traces):
        if name not in ("balanced","onset_focused") or name in self.arm_states:
            raise ValueError(f"invalid or duplicate ablation arm: {name}")
        snapshot=clone_model_state(model)
        rows,summary=parameter_deltas(
            model,self.original,snapshot,arm=name,
            gradient_norm=float(metrics.mean_grad_norm)
        )
        self.parameter_rows.extend(rows)
        self.parameter_summaries[name]=summary
        self.training_metrics[name]=asdict(metrics)
        self.arm_states[name]=snapshot
        self.validation[name]=(list(results),tuple(traces))
        output=self.output_dir/f"{name}.diagnostic.pt"
        if output.exists():
            raise FileExistsError(f"diagnostic candidate exists: {output}")
        parent=dict(self.prepared.parent)
        parent.update(model.checkpoint_metadata())
        parent.update({
            "model_state":snapshot,
            "training_mode":"onset_fixed_state_action_audit",
            "human_visible_checkpoint_role":"diagnostic-candidate",
            "diagnostic_arm":name,
            "diagnostic_source_checkpoint":str(self.prepared.source_checkpoint),
            "diagnostic_source_sha256":self.source_sha256,
            "diagnostic_seed":self.seed,
            "diagnostic_training_metrics":asdict(metrics),
            "diagnostic_validation_summary":validation_summary(results),
            "diagnostic_no_automatic_promotion":True,
        })
        # "xb" refuses overwrite even when someone creates a file during training.
        with output.open("xb") as f:
            torch.save(parent,f)
        self.checkpoint_paths[name]=str(output)
        assert_state_equal(clone_model_state(model),snapshot,
                           context=f"{name} checkpoint save modified live state")
        assert_state_equal(self.original,
                           {k:v.detach().cpu().clone() for k,v in self.prepared.parent["model_state"].items()},
                           context="reference selected-best checkpoint state")

    def complete(self, *, pilot_report: dict) -> dict:
        if self.train_trajectories is None or set(self.arm_states)!={
            "best","balanced","onset_focused"
        } or set(self.validation)!=set(self.arm_states):
            raise RuntimeError("incomplete Action Delta Audit capture")
        assert_state_equal(clone_model_state(self.prepared.model),self.original,
                           context="post-training restored best model")
        model=self.prepared.model
        try:
            frames,anchors,onset=fixed_trajectory_rows(
                model,self.train_trajectories,self.arm_states,
                device=self.prepared.device,
                control_dt=self.prepared.control_dt_s,
            )
            selected=( *EARLY_FAILURE_ANCHORS,
                       *select_controls(self.validation["best"][1]))
            motor=[]
            motor_summary=[]
            for index in sorted(set(selected)):
                chart=self.prepared.validation[index-1].chart_name
                for arm,(results,traces) in self.validation.items():
                    trace=traces[index-1]
                    save_motor_onset_svg(arm,index,chart,trace,self.output_dir)
                    motor.extend(_join_motor_timeline(arm,index,chart,trace))
                    detail=trace_first_200ms(trace)
                    motor_summary.append({
                        "arm":arm,"anchor":index,"chart":chart,
                        "keydowns_0_200ms":detail["initial_keydown_count"],
                        "first_keydown_s":detail["first_keydown_s"],
                        "early_overload_s":detail["early_overload_s"],
                        "max_positive_action_keys":detail["initial_max_positive_keys"],
                        "keydown_times":detail["keydowns"],
                    })
            _write_csv(
                self.output_dir/"fixed_state_action_deltas.csv",
                ["arm","anchor","source","frame","time_s","student_action",
                 "teacher_action","best_action","delta_action","mean_abs_delta",
                 "max_abs_delta","press_keys","teacher_press_keys",
                 "threshold_crossing_keys","threshold_crossing_count",
                 "set_valued_loss","teacher_key_mse"],
                frames,
            )
            _write_csv(
                self.output_dir/"fixed_state_per_anchor.csv",
                sorted({key for row in anchors for key in row}),anchors
            )
            _write_csv(
                self.output_dir/"onset_action_comparison.csv",
                sorted({key for row in onset for key in row}),onset
            )
            _write_csv(
                self.output_dir/"motor_response_comparison.csv",
                ["arm","anchor","chart","kind","time_s","key","actions",
                 "positive_keys","early_overload_s"],
                motor
            )
            _write_csv(
                self.output_dir/"parameter_deltas.csv",
                ["arm","parameter","layer","elements","changed_elements",
                 "start_l2","delta_l2","relative_delta_l2",
                 "last_gradient_l2","nonfinite","small_update"],
                self.parameter_rows,
            )
            source_now=file_sha256(self.prepared.source_checkpoint)
            if source_now!=self.source_sha256:
                raise RuntimeError("source best checkpoint changed on disk during audit")
            audit={
                "kind":"fixed_state_action_delta_audit",
                "source_checkpoint":str(self.prepared.source_checkpoint),
                "source_sha256":self.source_sha256,
                "seed":self.seed,
                "arms":["best","balanced","onset_focused"],
                "candidate_checkpoints":self.checkpoint_paths,
                "parameter_summaries":self.parameter_summaries,
                "training_metrics":self.training_metrics,
                "fixed_state_train_trajectories":len(self.train_trajectories),
                "fixed_state_frames":len(frames),
                "fixed_state_per_anchor":anchors,
                "onset_action_comparison":onset,
                "motor_focus_anchors":list(sorted(set(selected))),
                "motor_summary":motor_summary,
                "pilot_validation":pilot_report["baseline_validation"],
                "reference_reproduction":reference_reproduction(pilot_report),
                "pilot_arm_validation":[
                    {"arm":a["name"],**a["validation"],"onset":a["onset"]}
                    for a in pilot_report["arm_comparison"]
                ],
                "limitations":[
                    "Fixed-state Train trajectories and closed-loop Validation are distinct datasets.",
                    "Threshold-key count alone does not describe continuous Action deltas.",
                    "Per-frame set-valued loss uses existing teacher action targets.",
                    "Closed-loop motor timings are not fixed-state causal attributions.",
                    "No FINAL holdout, no promotion, no Student structural/loss modifications.",
                    "Arm checkpoint files are diagnostic-candidate, not selected-best.",
                ],
            }
            (self.output_dir/"best_reference.json").write_text(
                json.dumps({
                    "checkpoint_path":str(self.prepared.source_checkpoint),
                    "sha256":self.source_sha256,
                    "source_mode":self.prepared.parent.get("training_mode"),
                    "source_checkpoint_role":self.prepared.parent.get("human_visible_checkpoint_role"),
                    "stored_best_not_copied":True,
                    "original_state_verified_unchanged":True,
                    "model_training_start_seed":self.seed,
                },indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
            )
            (self.output_dir/"summary.json").write_text(
                json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                encoding="utf-8"
            )
            return audit
        finally:
            model.load_state_dict(self.original,strict=True)
            model.prepare_recurrent_runtime()
            assert_state_equal(clone_model_state(model),self.original,
                               context="post-audit restored best model")
