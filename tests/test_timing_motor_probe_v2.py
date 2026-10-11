from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training.timing_motor_probe import (
    _write_event_csv, collect_student_rollout, save_frame_data,
)
from dmdod.training.timing_motor_probe_v2 import (
    constant_baselines, feature_diagnostics, fit_logged_readout,
    load_frame_log, metrics, onset_svg, replay_first_200ms,
    save_onset_csv, select_control_anchors, timing_error, weighted_metric,
)


def test_constant_baselines_and_prehit_use_train_median_without_validation_fit():
    labels=torch.tensor([0.,.075,.15,2.])
    report=constant_baselines(labels,.2)
    assert report["train_median"]["all"]["n"]==4
    assert report["zero_ms"]["prehit_0_150ms"]["mae_ms"]==pytest.approx(75.)
    assert report["75_ms"]["prehit_0_150ms"]["mae_ms"]==pytest.approx(50.)
    assert report["train_median"]["prehit_0_150ms"]["n"]==3
    assert weighted_metric([{"n":1,"mae_ms":1.},{"n":3,"mae_ms":3.}])==2.5


def test_bad_raw_predictions_are_not_clipped_and_are_reported():
    truth=torch.tensor([0.01,0.0,.10])
    prediction=torch.tensor([100.,-50.,.12])
    report=metrics(prediction,truth)
    assert report["all"]["max_abs_ms"]>90_000
    assert report["prehit_0_150ms"]["n"]==3
    with pytest.raises(RuntimeError,match="nonfinite"):
        timing_error(torch.tensor([float("nan")]),torch.tensor([1.]))


def test_convergence_curve_and_train_only_linear_fit():
    x=torch.linspace(-1,1,256).unsqueeze(1)
    train=torch.cat([x,x**2],dim=1)
    y=(.5+.2*x[:,0]).clone()
    stored=train.clone()
    fitted,history=fit_logged_readout(
        train,y,device=torch.device("cpu"),seed=2,steps=50,
    )
    assert len(history)>=5
    assert history[0]["step"]==1 and history[-1]["step"]==50
    assert all("train_mae_ms" in row for row in history)
    assert torch.equal(train,stored)
    assert torch.equal(fitted.input_mean,stored.mean(0))


def test_load_existing_gzipped_frames_and_263d_invariant(tmp_path):
    p=tmp_path/"f.jsonl.gz"
    data={"visible_263d":[0.0]*263,
          "student_rnn_state_after_step":[.5]*4225,
          "continuous_action":[0.0]*8,"t_s":0.0,
          "next_target_index":0,"target_remaining_s":0.3}
    with gzip.open(p,"wt",encoding="utf-8") as f:
        f.write(json.dumps(data)+"\n")
    d=load_frame_log(p)
    assert d["visible"].shape==(1,263)
    assert d["hidden"].shape==(1,4225)
    assert d["labels"][0]==pytest.approx(.3)
    data["visible_263d"]=[0.]
    with gzip.open(p,"wt",encoding="utf-8") as f:
        f.write(json.dumps(data)+"\n")
    with pytest.raises(ValueError,match="dimensions"):
        load_frame_log(p)


def test_feature_outliers_top_ten_and_unclipped_attribution(tmp_path):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(3,1)
            with torch.no_grad():
                self.linear.weight.fill_(1.)
                self.linear.bias.zero_()
        def forward(self,x):
            return self.linear(x).flatten()
    from dmdod.training.timing_motor_probe import TrainedReadout
    train=torch.tensor([[0.,0.,0.],[1.,1.,1.],[-1.,-1.,-1.]])
    model=Head()
    fitted=TrainedReadout(
        model,train.mean(0),train.std(0,unbiased=False).clamp_min(1e-4),
        torch.tensor(0.),torch.tensor(1.)
    )
    record={"visible":torch.tensor([[100.,0.,0.]]),
            "labels":torch.tensor([.0]),"times":[0.1]}
    audit=feature_diagnostics(train,[(9,record)],fitted,"visible",tmp_path)
    assert audit["global_max_abs_z"]>20
    f=audit["largest_raw_prediction_errors"][0]
    assert f["anchor"]==9
    assert f["feature_contributions"][0]["feature_index"]==0
    assert abs(f["raw_prediction_s"])>10
    assert len(f["feature_contributions"])==3
    assert (tmp_path/"feature_stats_visible.csv").exists()


def test_control_selection_uses_existing_events_without_replaying(tmp_path):
    rows=[{"motor":{"terminal_s":1.0}} for _ in range(20)]
    (tmp_path/"summary.json").write_text(json.dumps({"validation":rows}),encoding="utf-8")
    for index in range(1,21):
        with (tmp_path/f"motor_validation_{index:02d}.csv").open(
            "w",newline="",encoding="utf-8"
        ) as f:
            writer=csv.DictWriter(f,fieldnames=["kind","time_s","too_early"])
            writer.writeheader()
            writer.writerow({"kind":"keydown","time_s":.05,
                             "too_early":index==1})
    assert select_control_anchors(tmp_path,count=2)==(2,3)


def _fixture_short_replay(tmp_path):
    from dmdod.adofai_chart import parse_adofai_text
    from dmdod.adofai_timing import compile_adofai
    from dmdod.adofai_playable import build_playable_segment
    chart=parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,"pitch":100,'
        '"countdownTicks":0,"separateCountdownTime":false},"actions":[]}'
    )
    compiled=compile_adofai(chart)
    segment=build_playable_segment(compiled,start_s=0.,end_s=compiled.duration_s)
    class Student(torch.nn.Module):
        key_count=8
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.tensor([0.]))
        def initial_state(self,device):
            return torch.zeros(4,device=device)
        def forward_step(self,x,state):
            assert x.shape==(263,)
            return self.bias*torch.ones(8,device=x.device),None,None,state+1.
    prepared=SimpleNamespace(
        key_count=8,input_dim=263,device=torch.device("cpu"),
        control_dt_s=.01,physics_dt_s=.001
    )
    named=SimpleNamespace(chart_name="tiny",segment=segment)
    student=Student()
    rollout=collect_student_rollout(student,prepared,named,role="validation",index=1)
    (tmp_path/"frames").mkdir()
    save_frame_data(rollout,tmp_path/"frames"/"validation_01.jsonl.gz")
    _write_event_csv(tmp_path/"motor_validation_01.csv",rollout)
    return student,prepared,named


def test_bounded_replay_matches_cached_student_actions_and_motor_events(tmp_path):
    student,prepared,named=_fixture_short_replay(tmp_path)
    before={k:v.detach().clone() for k,v in student.state_dict().items()}
    report=replay_first_200ms(student,prepared,named,1,tmp_path)
    assert report["replay_matches_existing_actions_and_events"]
    assert len(report["frames"])<=21
    assert report["frames"][0]["time_s"]==0
    assert len(report["frames"][0]["action"])==8
    assert len(report["frames"][0]["position_mm_end"])==8
    assert len(report["frames"][0]["velocity_mm_s_end"])==8
    assert report["frames"][0]["next_target_index"]==0
    save_onset_csv(report,tmp_path)
    onset_svg(report,tmp_path/"onset_01.svg")
    assert (tmp_path/"onset_01_frames.csv").exists()
    assert (tmp_path/"onset_01_events.csv").exists()
    assert (tmp_path/"onset_01.svg").read_text(encoding="utf-8").startswith("<svg ")
    for k,v in before.items():
        assert torch.equal(v,student.state_dict()[k])


def test_bounded_replay_refuses_actions_that_diverge_from_cached_logs(tmp_path):
    student,prepared,named=_fixture_short_replay(tmp_path)
    with torch.no_grad():
        student.bias.fill_(1.)
    with pytest.raises(RuntimeError,match="differs from existing actions"):
        replay_first_200ms(student,prepared,named,1,tmp_path)


def test_cannot_overwrite_existing_probe_results(tmp_path):
    from dmdod.training.timing_motor_probe_v2 import run_probe_v2
    original=tmp_path/"run2"
    original.mkdir()
    (original/"summary.json").write_text("{}",encoding="utf-8")
    output=tmp_path/"v2"
    output.mkdir()
    (output/"old.txt").write_text("preserve",encoding="utf-8")
    with pytest.raises(FileExistsError,match="not empty"):
        run_probe_v2(original,output,skip_replay=True)
    assert (output/"old.txt").read_text()=="preserve"
