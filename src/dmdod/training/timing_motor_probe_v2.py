from __future__ import annotations

"""Offline timing readout audit and bounded initial-motor replay (no Student fitting).

Readouts train from existing Train JSONL. Validation logs are evaluation-only.
The optional replay stops by 200 ms and never writes a checkpoint.
"""

import csv
import gzip
import html
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.n_key import NKeyAction
from dmdod.training.timing_motor_probe import (
    LinearTimeReadout, TrainedReadout, _check_student_unchanged,
    predict_readout,
)

EARLY_FAILURE_ANCHORS = (6, 10, 11, 15, 17)
FOCUS_READOUT_ANCHORS = (9, 12, 14, 16, 18)
INITIAL_WINDOW_S = 0.200
EXTREME_Z = 20.0


def load_frame_log(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"existing frame log missing: {path}")
    obs, states, actions, labels, times, indices = [], [], [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            obs.append(data["visible_263d"])
            states.append(data["student_rnn_state_after_step"])
            actions.append(data["continuous_action"])
            labels.append(float("nan") if data["target_remaining_s"] is None
                          else float(data["target_remaining_s"]))
            times.append(float(data["t_s"]))
            indices.append(data["next_target_index"])
    if not obs:
        raise ValueError(f"empty log: {path}")
    result = {
        "visible": torch.tensor(obs, dtype=torch.float32),
        "hidden": torch.tensor(states, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "times": times, "indices": indices,
    }
    if result["visible"].shape[1] != 263 or result["actions"].shape[1] != 8:
        raise ValueError(f"unexpected Student input/action dimensions: {path}")
    return result


def _usable(log: dict) -> torch.Tensor:
    return torch.isfinite(log["labels"])


def timing_error(pred: torch.Tensor, labels: torch.Tensor, *, prehit=False) -> dict:
    mask = torch.isfinite(labels) & torch.isfinite(pred)
    if prehit:
        mask &= (labels >= 0) & (labels <= .150)
    bad = (~torch.isfinite(pred)) & torch.isfinite(labels)
    if bad.any():
        raise RuntimeError("Readout generated nonfinite predictions; refusing to clip")
    abs_ms = (pred[mask] - labels[mask]).abs().double() * 1000.
    if not abs_ms.numel():
        return {"n":0,"mae_ms":None,"p50_ms":None,"p90_ms":None,
                "p99_ms":None,"max_abs_ms":None}
    return {"n":int(abs_ms.numel()),"mae_ms":float(abs_ms.mean()),
            "p50_ms":float(torch.quantile(abs_ms,.5)),
            "p90_ms":float(torch.quantile(abs_ms,.9)),
            "p99_ms":float(torch.quantile(abs_ms,.99)),
            "max_abs_ms":float(abs_ms.max())}


def metrics(pred: torch.Tensor, y: torch.Tensor) -> dict:
    return {"all": timing_error(pred,y),
            "prehit_0_150ms":timing_error(pred,y,prehit=True)}


def constant_baselines(labels: torch.Tensor, median: float) -> dict:
    return {
        "train_median": metrics(torch.full_like(labels,median),labels),
        "zero_ms": {"prehit_0_150ms":timing_error(torch.zeros_like(labels),labels,prehit=True)},
        "75_ms": {"prehit_0_150ms":timing_error(torch.full_like(labels,.075),labels,prehit=True)},
    }


def fit_logged_readout(x: torch.Tensor, y: torch.Tensor, *,
                       device: torch.device, seed: int, steps: int,
                       batch_size: int = 256) -> tuple[TrainedReadout, list[dict]]:
    valid = torch.isfinite(y) & torch.isfinite(x).all(1)
    x, y = x[valid].float(), y[valid].float()
    if x.shape[0] < 2 or steps < 1:
        raise ValueError("insufficient Train frames or invalid readout steps")
    x_mean = x.mean(0)
    x_std = x.std(0,unbiased=False).clamp_min(1e-4)
    y_mean = y.mean()
    y_std = y.std(unbiased=False).clamp_min(1e-4)
    generator=torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head=LinearTimeReadout(x.shape[1]).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-2,weight_decay=1e-3)
    curve=[]
    normalized=(x-x_mean)/x_std
    targets=(y-y_mean)/y_std
    for step in range(1,steps+1):
        ids=torch.randint(x.shape[0],(min(batch_size,len(y)),),generator=generator)
        xb=normalized[ids].to(device)
        yb=targets[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.smooth_l1_loss(head(xb),yb)
        loss.backward()
        optimizer.step()
        if step==1 or step%10==0 or step==steps:
            with torch.no_grad():
                total=0.
                count=0
                for batch in range(0,len(y),512):
                    z=normalized[batch:batch+512].to(device)
                    prediction=(head(z).cpu()*y_std+y_mean)
                    actual=y[batch:batch+512]
                    total+=float((prediction-actual).abs().double().sum())
                    count+=len(actual)
            curve.append({"step":step,"batch_loss":float(loss.detach()),
                          "train_mae_ms":1000.*total/count})
    head.eval()
    return TrainedReadout(head,x_mean,x_std,y_mean,y_std),curve


def fixed_ridge_visible(x: torch.Tensor, y: torch.Tensor, *, penalty=10.0):
    """Predeclared one-shot visible-only ridge, no Validation hyperparameter search."""
    mask=torch.isfinite(y)&torch.isfinite(x).all(1)
    data=x[mask].double()
    truth=y[mask].double()
    center=data.mean(0)
    scale=data.std(0,unbiased=False).clamp_min(1e-4)
    z=(data-center)/scale
    ty=truth.mean()
    xc=z.T@z
    xc.diagonal().add_(penalty)
    weights=torch.linalg.solve(xc,z.T@(truth-ty))
    def predict(values):
        return (((values.double()-center)/scale)@weights+ty).float()
    return predict


def feature_diagnostics(train_x: torch.Tensor, val_logs: list[tuple[int,dict]],
                        readout: TrainedReadout, kind: str,
                        output_dir: Path) -> dict:
    train_mean=train_x.mean(0).float()
    train_raw_std=train_x.std(0,unbiased=False).float()
    normalize_std=readout.input_std.float()
    max_raw=torch.zeros_like(train_mean)
    max_z=torch.zeros_like(train_mean)
    extreme=[]
    anomalies=[]
    w=readout.model.linear.weight.detach().float().cpu().flatten()
    scale=float(readout.target_std)
    for anchor,record in val_logs:
        x=record[kind].float()
        if not torch.isfinite(x).all():
            raise RuntimeError(f"nonfinite Validation {kind} feature at #{anchor}")
        z=(x-readout.input_mean)/normalize_std
        max_raw=torch.maximum(max_raw,x.abs().amax(0))
        max_z=torch.maximum(max_z,z.abs().amax(0))
        where=torch.where(z.abs().amax(dim=1) > EXTREME_Z)[0]
        for idx in where[:50].tolist():
            extreme.append({"anchor":anchor,"frame_index":idx,
                            "t_s":record["times"][idx],
                            "max_abs_z":float(z[idx].abs().max())})
        pred=predict_readout(readout,x,torch.device(next(readout.model.parameters()).device))
        dif=(pred-record["labels"]).abs()
        dif[~torch.isfinite(record["labels"])]=-1
        top=torch.argsort(dif,descending=True)[:3]
        for i in top.tolist():
            if not torch.isfinite(record["labels"][i]):
                continue
            contrib=z[i]*w*scale*1000.
            best=torch.argsort(contrib.abs(),descending=True)[:10]
            anomalies.append({
                "anchor":anchor,"frame_index":i,"t_s":record["times"][i],
                "truth_s":float(record["labels"][i]),"raw_prediction_s":float(pred[i]),
                "abs_error_ms":float(dif[i]*1000.),
                "max_abs_z":float(z[i].abs().max()),
                "feature_contributions":[
                    {"feature_index":j,"raw_value":float(x[i,j]),
                     "z":float(z[i,j]),"linear_contribution_ms":float(contrib[j]),
                     "train_mean":float(train_mean[j]),
                     "train_std_raw":float(train_raw_std[j])}
                    for j in best.tolist()
                ],
            })
    with (output_dir/f"feature_stats_{kind}.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["feature_index","train_mean","train_raw_std","train_normalization_std",
                         "validation_max_abs_raw","validation_max_abs_z"])
        for i in range(train_x.shape[1]):
            writer.writerow([i,float(train_mean[i]),float(train_raw_std[i]),
                             float(normalize_std[i]),float(max_raw[i]),float(max_z[i])])
    anomalies=sorted(anomalies,key=lambda e:-e["abs_error_ms"])[:30]
    audit={"kind":kind,"frames_over_abs_z_20_sample":extreme[:100],
           "total_recorded_extreme_sample":len(extreme),
           "global_max_abs_z":float(max_z.max()),
           "largest_raw_prediction_errors":anomalies}
    (output_dir/f"feature_outliers_{kind}.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return audit


def readout_diagnostics(existing: Path, output_dir: Path, *,
                        steps: int, device: torch.device) -> dict:
    logs=existing/"frames"
    train=[load_frame_log(logs/f"train_{i:02d}.jsonl.gz") for i in range(1,21)]
    train_y=torch.cat([r["labels"][_usable(r)] for r in train])
    median=float(train_y.median())
    train_x={
        key:torch.cat([r[key][_usable(r)] for r in train])
        for key in ("visible","hidden")
    }
    result={"train_median_s":median,"models":{},"validation":{},"focus":list(FOCUS_READOUT_ANCHORS)}
    fitted={}
    for kind,seed in (("visible",2402),("hidden",2401)):
        readout,curve=fit_logged_readout(train_x[kind],train_y,device=device,
                                         seed=seed,steps=steps)
        fitted[kind]=readout
        pred=predict_readout(readout,train_x[kind],device)
        result["models"][kind]={
            "train":metrics(pred,train_y),
            "readout":"AdamW linear, fixed steps/hyperparameters",
        }
        with (output_dir/f"convergence_{kind}.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=("step","batch_loss","train_mae_ms"))
            writer.writeheader()
            writer.writerows(curve)
        _line_svg(curve,output_dir/f"convergence_{kind}.svg",f"{kind} Train MAE (ms)")
    result["models"]["train_median"]={"train":constant_baselines(train_y,median)}
    visible_ridge=fixed_ridge_visible(train_x["visible"],train_y)
    result["models"]["visible_ridge"]={
        "train":metrics(visible_ridge(train_x["visible"]),train_y),
        "penalty":10.0,"fit_only":"Train","selected_by_validation":False,
    }
    validation_logs=[(i,load_frame_log(logs/f"validation_{i:02d}.jsonl.gz"))
                     for i in range(1,21)]
    for index,record in validation_logs:
        y=record["labels"]
        report={"baselines":constant_baselines(y,median)}
        for kind in ("hidden","visible"):
            prediction=predict_readout(fitted[kind],record[kind],device)
            report[kind]=metrics(prediction,y)
            # Raw predictions are never clipped; file includes every Validation frame.
            with (output_dir/f"predictions_{kind}_{index:02d}.csv").open(
                "w",newline="",encoding="utf-8-sig"
            ) as f:
                writer=csv.writer(f)
                writer.writerow(["frame","time_s","target_remaining_s",
                                 "raw_prediction_s","signed_error_ms"])
                for j,(t,label,p) in enumerate(zip(record["times"],y,prediction)):
                    finite=math.isfinite(float(label))
                    writer.writerow([j,t,float(label) if finite else "",
                                     float(p),float((p-label)*1000) if finite else ""])
        report["visible_ridge"]=metrics(visible_ridge(record["visible"]),y)
        result["validation"][str(index)]=report
    with (output_dir/"timing_per_anchor_v2.csv").open(
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        writer=csv.writer(f)
        writer.writerow(["anchor","hidden_mae_ms","visible_mae_ms",
                         "ridge_visible_mae_ms","median_mae_ms",
                         "hidden_prehit_mae_ms","visible_prehit_mae_ms",
                         "median_prehit_mae_ms","zero_prehit_mae_ms",
                         "75ms_prehit_mae_ms"])
        def val(anchor,key,part="all"):
            return result["validation"][str(anchor)][key][part]["mae_ms"]
        for index in range(1,21):
            base=result["validation"][str(index)]["baselines"]
            writer.writerow([
                index,val(index,"hidden"),val(index,"visible"),
                val(index,"visible_ridge"),
                base["train_median"]["all"]["mae_ms"],
                val(index,"hidden","prehit_0_150ms"),
                val(index,"visible","prehit_0_150ms"),
                base["train_median"]["prehit_0_150ms"]["mae_ms"],
                base["zero_ms"]["prehit_0_150ms"]["mae_ms"],
                base["75_ms"]["prehit_0_150ms"]["mae_ms"],
            ])
    for kind in ("hidden","visible"):
        result["models"][kind]["validation_mae_ms"]=weighted_metric(
            [result["validation"][str(i)][kind]["all"] for i in range(1,21)])
        result["models"][kind]["validation_prehit_mae_ms"]=weighted_metric(
            [result["validation"][str(i)][kind]["prehit_0_150ms"] for i in range(1,21)])
        result["models"][kind]["feature_audit"]=feature_diagnostics(
            train_x[kind],validation_logs,fitted[kind],kind,output_dir)
    for kind in ("visible_ridge",):
        result["models"][kind]["validation_mae_ms"]=weighted_metric(
            [result["validation"][str(i)][kind]["all"] for i in range(1,21)])
    result["baselines"]={
        name:{
            "validation_all_mae_ms":weighted_metric([
                result["validation"][str(i)]["baselines"][name]["all"] for i in range(1,21)])
            if name=="train_median" else None,
            "validation_prehit_mae_ms":weighted_metric([
                result["validation"][str(i)]["baselines"][name]["prehit_0_150ms"]
                for i in range(1,21)])
        } for name in ("train_median","zero_ms","75_ms")
    }
    return result


def weighted_metric(parts: list[dict]) -> float | None:
    n=sum(p["n"] for p in parts)
    return None if not n else sum(p["n"]*p["mae_ms"] for p in parts if p["n"])/n


def _line_svg(curve: list[dict], path:Path, title:str):
    values=[p["train_mae_ms"] for p in curve]
    ymax=max(values) if values else 1.0
    ymax=max(ymax,1.)
    points=" ".join(
        f"{50+800*i/max(1,len(values)-1):.1f},{350-280*v/ymax:.1f}"
        for i,v in enumerate(values)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420">'
        f'<rect width="900" height="420" fill="white"/>'
        f'<text x="50" y="35" font-size="19">{html.escape(title)}</text>'
        f'<line x1="50" y1="350" x2="850" y2="350" stroke="gray"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<text x="50" y="385">Training iterations →</text>'
        f'<text x="50" y="60">max={ymax:.1f} ms</text></svg>',
        encoding="utf-8",
    )


def _saved_motor_events(existing: Path, index: int, *, until: float) -> list[tuple]:
    path=existing/f"motor_validation_{index:02d}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"existing motor log missing: {path}")
    with path.open(encoding="utf-8-sig",newline="") as f:
        return sorted(
            (float(r["time_s"]),r["key"],r["kind"])
            for r in csv.DictReader(f)
            if r["kind"] in ("keydown","keyup")
            and float(r["time_s"]) <= until + 1e-9
        )


def select_control_anchors(existing: Path, *, count: int = 2) -> tuple[int,...]:
    """Select non-early-overload and zero-initial-TooEarly controls from old logs."""
    summary_path=existing/"summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"existing summary missing: {summary_path}")
    summaries=json.loads(summary_path.read_text(encoding="utf-8"))["validation"]
    chosen=[]
    for i,item in enumerate(summaries,1):
        if i in EARLY_FAILURE_ANCHORS:
            continue
        motor=item["motor"]
        if float(motor["terminal_s"]) < INITIAL_WINDOW_S:
            continue
        motor_path=existing/f"motor_validation_{i:02d}.csv"
        with motor_path.open(encoding="utf-8-sig",newline="") as f:
            early=sum(str(r.get("too_early","")).strip().lower() in ("true","1")
                      for r in csv.DictReader(f) if r["kind"]=="keydown"
                      and float(r["time_s"]) <= INITIAL_WINDOW_S+1e-9)
        if early==0:
            chosen.append(i)
        if len(chosen)>=count:
            break
    if not chosen:
        raise RuntimeError("no suitable normal-start control in existing logs")
    return tuple(chosen)


def replay_first_200ms(model, prepared, named, index: int, existing: Path) -> dict:
    """Replay only initial motor motion, fail closed on action/event mismatch.

    Unlike collect_student_rollout this does NOT finish the whole chart.
    """
    old=load_frame_log(existing/"frames"/f"validation_{index:02d}.jsonl.gz")
    env=DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,key_count=model.key_count,
        capture_overload_trace=True,
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation=env.reset()
    state=model.initial_state(prepared.device)
    timeline=[]
    events=[]
    model.eval()
    with torch.no_grad():
        for tick in range(math.ceil(INITIAL_WINDOW_S/prepared.control_dt_s)):
            t=float(env.privileged_episode_time_s())
            if t>=INITIAL_WINDOW_S-1e-9:
                break
            encoded=encode_n_key_hud_real_chart_observation(observation)
            if len(encoded)!=263:
                raise RuntimeError("replay visible input changed")
            input_tensor=torch.tensor(encoded,dtype=torch.float32,device=prepared.device)
            mean,_,_,state=model.forward_step(input_tensor,state)
            action=tuple(float(x.item()) for x in torch.tanh(mean))
            if tick>=len(old["times"]) or abs(t-old["times"][tick])>1e-6:
                raise RuntimeError(f"#{index:02d} short replay frame indexing differs from old logs")
            action_delta=max(abs(a-b) for a,b in zip(action,old["actions"][tick].tolist()))
            if action_delta>1e-4:
                raise RuntimeError(
                    f"#{index:02d} short replay differs from existing actions: "
                    f"t={t:.3f}s max_abs={action_delta:g}; refusing physical inference"
                )
            next_target=env.privileged_next_target()
            target_remaining=(
                None if next_target is None else float(next_target.episode_time_s)-t
            )
            result=env.step(NKeyAction(action))
            observation=result.observation
            body=result.observation.motor
            timeline.append({
                "time_s":t,"action":action,
                "position_mm_end":tuple(float(v)*1000 for v in body.positions_m),
                "velocity_mm_s_end":tuple(float(v)*1000 for v in body.velocities_m_s),
                "pressed_end":tuple(bool(v) for v in body.pressed_flags),
                "target_remaining_s":target_remaining,
                "next_target_index":env._trace_action_frames[-1].next_target_index,
                "commanded_keys_ge_025":sum(v>=.25 for v in action),
            })
            tr=env.motor.last_transition
            if tr is None:
                raise RuntimeError("no physical transition during replay")
            for event in tr.evaluator_events:
                if event.event.value in ("down","up"):
                    events.append({
                        "time_s":float(event.time_s), "key":str(event.key),
                        "kind":"keydown" if event.event.value=="down" else "keyup",
                    })
            if result.done:
                break
    observed=sorted((e["time_s"],e["key"],e["kind"]) for e in events)
    prior=_saved_motor_events(
        existing,index,until=min(INITIAL_WINDOW_S,
                                 timeline[-1]["time_s"]+prepared.control_dt_s)
    )
    if len(observed)!=len(prior) or any(
        a[1:]!=b[1:] or abs(a[0]-b[0])>1e-5
        for a,b in zip(observed,prior)
    ):
        raise RuntimeError(
            f"#{index:02d} replay physical events differ from old logs "
            f"({len(observed)} vs {len(prior)}); refusing physical inference"
        )
    for e in env._trace_too_early:
        if e.time_s <= INITIAL_WINDOW_S+1e-9:
            events.append({
                "time_s":float(e.time_s),"key":e.key,
                "kind":"too_early","target_index":e.target_index,
                "timing_error_ms":float(e.error_ms),
                "gauge_before":float(e.overload_before),
                "gauge_after":float(e.overload_after),
            })
            if e.fail_overload:
                events.append({
                    "time_s":float(e.time_s),"key":e.key,
                    "kind":"fail_overload","target_index":e.target_index,
                })
    events.sort(key=lambda e:(e["time_s"],e["kind"],e["key"]))
    return {
        "index":index,"chart":named.chart_name,
        "window_s":INITIAL_WINDOW_S,
        "frames":timeline,"events":events,
        "replay_matches_existing_actions_and_events":True,
        "first_commanded_keys_ge_025":timeline[0]["commanded_keys_ge_025"] if timeline else None,
        "first_keydown_s":next(
            (e["time_s"] for e in events if e["kind"]=="keydown"),None
        ),
        "overload_s":next(
            (e["time_s"] for e in events if e["kind"]=="fail_overload"),None
        ),
        "too_early_count":sum(e["kind"]=="too_early" for e in events),
        "max_commanded_keys_ge_025":max(
            (f["commanded_keys_ge_025"] for f in timeline), default=0
        ),
        "first_multi_command_s":next(
            (f["time_s"] for f in timeline if f["commanded_keys_ge_025"]>=2),
            None
        ),
        "max_physical_pressed_keys":max(
            (sum(f["pressed_end"]) for f in timeline),default=0
        ),
    }


def save_onset_csv(report: dict, output: Path) -> None:
    with (output/f"onset_{report['index']:02d}_frames.csv").open(
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        writer=csv.writer(f)
        writer.writerow(["time_s","key_index","continuous_action","position_mm_end",
                         "velocity_mm_s_end","pressed_end",
                         "next_target_index","target_remaining_s",
                         "commanded_keys_ge_025"])
        for frame in report["frames"]:
            for key in range(len(frame["action"])):
                writer.writerow([
                    frame["time_s"],key,frame["action"][key],
                    frame["position_mm_end"][key],frame["velocity_mm_s_end"][key],
                    int(frame["pressed_end"][key]),frame["next_target_index"],
                    frame["target_remaining_s"],frame["commanded_keys_ge_025"],
                ])
    with (output/f"onset_{report['index']:02d}_events.csv").open(
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=[
            "time_s","key","kind","target_index","timing_error_ms",
            "gauge_before","gauge_after"
        ])
        writer.writeheader()
        for event in report["events"]:
            writer.writerow(event)
    (output/f"onset_{report['index']:02d}.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )


def onset_svg(report: dict, path: Path) -> None:
    """Self-contained interactive-independent SVG: 8 action/position/velocity curves."""
    width,height=1000,760
    colors=("#2563eb","#dc2626","#059669","#d97706",
            "#7c3aed","#db2777","#0d9488","#64748b")
    frames=report["frames"]
    pieces=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="55" y="30" font-family="sans-serif" font-size="18">'
        f'#{report["index"]:02d} {html.escape(report["chart"])} — first 200ms</text>',
    ]
    panels=[
        ("Continuous Action", "action", 70, -1., 1.),
        ("Finger position (mm)", "position_mm_end", 285, None, None),
        ("Finger velocity (mm/s)", "velocity_mm_s_end", 500, None, None),
    ]
    left,right=55,970
    panel_height=165
    for label,field,top,minimum,maximum in panels:
        if minimum is None:
            values=[v for row in frames for v in row[field]]
            minimum=min(values,default=0.)
            maximum=max(values,default=1.)
            if maximum-minimum<1e-9:
                maximum=minimum+1.
        pieces.append(
            f'<text x="{left}" y="{top-8}" font-family="sans-serif" font-size="14">'
            f'{label} [{minimum:.3g},{maximum:.3g}]</text>'
        )
        pieces.append(f'<rect x="{left}" y="{top}" width="{right-left}" '
                      f'height="{panel_height}" fill="none" stroke="#bbb"/>')
        for key in range(8):
            points=[]
            for row in frames:
                x=left+(right-left)*max(0,min(1,row["time_s"]/INITIAL_WINDOW_S))
                val=row[field][key]
                y=top+panel_height-(val-minimum)/(maximum-minimum)*panel_height
                points.append(f"{x:.2f},{y:.2f}")
            if points:
                pieces.append(
                    f'<polyline points="{" ".join(points)}" fill="none" '
                    f'stroke="{colors[key]}" stroke-width="1.6"/>'
                )
        for event in report["events"]:
            if event["kind"] not in ("keydown","keyup","too_early","fail_overload"):
                continue
            x=left+(right-left)*event["time_s"]/INITIAL_WINDOW_S
            color={"keydown":"#2563eb","keyup":"#999",
                   "too_early":"#f59e0b","fail_overload":"#dc2626"}[event["kind"]]
            pieces.append(
                f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" '
                f'y2="{top+panel_height}" stroke="{color}" '
                f'stroke-dasharray="3 3" opacity="0.55"/>'
            )
    pieces.append('<text x="55" y="710" font-size="13" font-family="sans-serif">'
                  'Blue=KeyDown, grey=KeyUp, amber=TooEarly, red=FailOverload; '
                  'line colors identify 8 virtual keys</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces),encoding="utf-8")


def run_probe_v2(existing: Path, output: Path, *, prepared=None,
                 steps: int=250, skip_replay: bool=False) -> dict:
    if not (existing/"summary.json").is_file():
        raise FileNotFoundError(f"v1 report not found: {existing/'summary.json'}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"v2 output directory not empty: {output}")
    output.mkdir(parents=True,exist_ok=True)
    device=prepared.device if prepared is not None else torch.device("cpu")
    print("=== Timing/Motor v2: offline readout reliability ===",flush=True)
    readout=readout_diagnostics(existing,output,steps=steps,device=device)
    output_data={"readout":readout,"initial_motor":{},
                 "source":str(existing),"replay_indices":[],
                 "notes":[
                     "Readouts re-fit only on cached Train frames; no Student training.",
                     "Validation never used for readout training, normalization or ridge tuning.",
                     "Raw predictions are never clipped.",
                     "Command threshold crossing is not proof of Student intent.",
                     "Physical short replays are accepted only if matching existing action/event logs.",
                     "No FINAL holdout and no checkpoint writes.",
                 ]}
    if not skip_replay:
        if prepared is None:
            raise ValueError("prepared Student model required for bounded replay")
        controls=select_control_anchors(existing)
        selected=tuple(sorted(set(EARLY_FAILURE_ANCHORS)|set(controls)))
        original={k:v.detach().cpu().clone() for k,v in prepared.model.state_dict().items()}
        try:
            for index in selected:
                result=replay_first_200ms(
                    prepared.model,prepared,prepared.validation[index-1],
                    index,existing
                )
                result["group"]=("early_overload" if index in EARLY_FAILURE_ANCHORS
                                  else "normal_start_control")
                save_onset_csv(result,output)
                onset_svg(result,output/f"onset_{index:02d}.svg")
                output_data["initial_motor"][str(index)]={
                    k:v for k,v in result.items() if k not in ("frames","events")
                }
                output_data["replay_indices"].append(index)
                print(
                    f"onset #{index:02d} {result['chart']} group={result['group']} "
                    f"first_commands={result['first_commanded_keys_ge_025']} "
                    f"TooEarly={result['too_early_count']} "
                    f"Overload={result['overload_s']}",flush=True
                )
        finally:
            _check_student_unchanged(prepared.model,original)
        with (output/"onset_comparison.csv").open(
            "w",newline="",encoding="utf-8-sig"
        ) as f:
            writer=csv.writer(f)
            writer.writerow(["anchor","chart","group","max_commanded_keys_ge_025",
                             "first_multi_command_s","max_physical_pressed_keys",
                             "first_keydown_s","too_early_count","overload_s",
                             "replay_verified"])
            for index in selected:
                row=output_data["initial_motor"][str(index)]
                writer.writerow([
                    index,row["chart"],row["group"],
                    row["max_commanded_keys_ge_025"],row["first_multi_command_s"],
                    row["max_physical_pressed_keys"],row["first_keydown_s"],
                    row["too_early_count"],row["overload_s"],
                    int(row["replay_matches_existing_actions_and_events"])
                ])
    (output/"summary_v2.json").write_text(
        json.dumps(output_data,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(f"v2 output: {output/'summary_v2.json'}",flush=True)
    return output_data
