from __future__ import annotations

"""Train-only zero-variance mask for an offline visible-time linear readout.

Raw linear, fixed-penalty ridge, and constant baselines remain comparable.
No Student is instantiated or updated here.
"""

import csv
import json
from pathlib import Path

import torch

from dmdod.training.timing_motor_probe import predict_readout
from dmdod.training.timing_motor_probe_v2 import (
    constant_baselines,
    fit_logged_readout,
    fixed_ridge_visible,
    load_frame_log,
    metrics,
    weighted_metric,
)


def train_variance_mask(train: torch.Tensor) -> torch.Tensor:
    if train.ndim != 2 or not torch.isfinite(train).all():
        raise ValueError("Train features must be finite [frames, features]")
    # Exact zero variance is detected on Train only; no Validation leakage.
    std=train.std(dim=0,unbiased=False)
    mask=std>0
    if not bool(mask.any()):
        raise ValueError("All Train features have zero variance")
    return mask


def run_masked_readout_audit(
    log_root: Path,
    report_path: Path,
    *,
    device: torch.device,
    steps:int=250,
) -> dict:
    csv_path=report_path.with_suffix(".csv")
    if report_path.exists() or csv_path.exists():
        raise FileExistsError("masked readout output already exists")
    frames=log_root/"frames"
    train=[
        load_frame_log(frames/f"train_{i:02d}.jsonl.gz")
        for i in range(1,21)
    ]
    x=torch.cat([
        row["visible"][torch.isfinite(row["labels"])] for row in train
    ])
    y=torch.cat([
        row["labels"][torch.isfinite(row["labels"])] for row in train
    ])
    if x.shape[1]!=263:
        raise RuntimeError("visible readout input width changed")
    mask=train_variance_mask(x)
    selected=torch.nonzero(mask).flatten()
    masked,masked_curve=fit_logged_readout(
        x[:,mask],y,device=device,seed=2402,steps=steps
    )
    raw,_=fit_logged_readout(
        x,y,device=device,seed=2402,steps=steps
    )
    ridge=fixed_ridge_visible(x,y,penalty=10.0)
    median=float(y.median())
    compare={
        "train":{},
        "validation":{},
        "mask":{"input_dim":int(x.shape[1]),"kept":int(mask.sum()),
                "removed":int((~mask).sum()),
                "removed_indices":torch.nonzero(~mask).flatten().tolist()},
        "mask_fitted_on":"Train only",
        "ridge_penalty":10.0,
        "validation_used_for_model_selection":False,
    }
    compare["train"]={
        "masked":metrics(predict_readout(masked,x[:,mask],device),y),
        "raw":metrics(predict_readout(raw,x,device),y),
        "ridge":metrics(ridge(x),y),
        "constants":constant_baselines(y,median),
    }
    for i in range(1,21):
        row=load_frame_log(frames/f"validation_{i:02d}.jsonl.gz")
        xv=row["visible"]
        yy=row["labels"]
        compare["validation"][str(i)]={
            "masked":metrics(predict_readout(masked,xv[:,mask],device),yy),
            "raw":metrics(predict_readout(raw,xv,device),yy),
            "ridge":metrics(ridge(xv),yy),
            "constants":constant_baselines(yy,median),
        }
    for key in ("masked","raw","ridge"):
        compare["validation_aggregate_"+key]={
            section:weighted_metric([
                compare["validation"][str(i)][key][section]
                for i in range(1,21)
            ])
            for section in ("all","prehit_0_150ms")
        }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x",encoding="utf-8") as f:
        json.dump(compare,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    with csv_path.open("x",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow([
            "anchor","raw_mae_ms","masked_mae_ms","ridge_mae_ms",
            "median_mae_ms","raw_prehit_mae_ms","masked_prehit_mae_ms",
            "ridge_prehit_mae_ms","median_prehit_mae_ms",
            "zero_prehit_mae_ms","75ms_prehit_mae_ms",
        ])
        for i in range(1,21):
            r=compare["validation"][str(i)]
            writer.writerow([
                i,r["raw"]["all"]["mae_ms"],
                r["masked"]["all"]["mae_ms"],r["ridge"]["all"]["mae_ms"],
                r["constants"]["train_median"]["all"]["mae_ms"],
                r["raw"]["prehit_0_150ms"]["mae_ms"],
                r["masked"]["prehit_0_150ms"]["mae_ms"],
                r["ridge"]["prehit_0_150ms"]["mae_ms"],
                r["constants"]["train_median"]["prehit_0_150ms"]["mae_ms"],
                r["constants"]["zero_ms"]["prehit_0_150ms"]["mae_ms"],
                r["constants"]["75_ms"]["prehit_0_150ms"]["mae_ms"],
            ])
    return compare
