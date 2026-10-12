from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.envs.n_key import (
    EpisodeTerminationTrace,
    NKeyActionFrameTrace,
    NKeyOverloadTrace,
    NKeyPhysicalKeyDownTrace,
    TooEarlyKeyDownTrace,
)
from dmdod.training.fixed_state_action_audit import (
    ActionDeltaAudit,
    _result_metrics,
    _write_csv,
    assert_state_equal,
    file_sha256,
    fixed_trajectory_rows,
    parameter_deltas,
    reference_reproduction,
    select_controls,
    trace_first_200ms,
)
from dmdod.training.human_visible_curriculum import (
    CurriculumEpochMetrics, HumanVisibleTrajectory,
)
from dmdod.training.real_chart import clone_model_state


class ToyPolicy(torch.nn.Module):
    key_count=8

    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([0.1]))
        self.frozen=torch.nn.Parameter(torch.tensor([7.]),requires_grad=False)
        self.register_buffer("step_buffer",torch.tensor(4,dtype=torch.int64))
        self.initialized=0

    def initial_state(self,device):
        self.initialized+=1
        return torch.zeros(2,device=device)

    def forward_step(self,x,state):
        return (
            (state[0]+self.weight)*torch.ones(8,device=x.device),
            None,None,state+1.,
        )

    def prepare_recurrent_runtime(self):
        pass

    def checkpoint_metadata(self):
        return {"key_count":8,"input_dim":263}


def trajectory(anchor=0,source="expert",length=30):
    obs=torch.zeros((length,263))
    labels=torch.zeros((length,8))
    labels[5:10,:2]=1.
    return HumanVisibleTrajectory(
        anchor_id=anchor,source_kind=source,
        observations=obs,
        teacher_actions=labels,
        student_actions=torch.zeros_like(labels),
    )


def test_replay_same_visible_frames_three_models_with_fresh_rnn_each():
    model=ToyPolicy()
    best=clone_model_state(model)
    a={k:v.clone() for k,v in best.items()}
    b={k:v.clone() for k,v in best.items()}
    a["weight"]=torch.tensor([0.4])
    b["weight"]=torch.tensor([0.7])
    frames,anchors,onset=fixed_trajectory_rows(
        model,[trajectory(length=25)],
        {"best":best,"balanced":a,"onset_focused":b},
        device=torch.device("cpu"),control_dt=.01,
    )
    assert model.initialized==3
    assert len(frames)==25*3
    assert len(anchors)==3
    assert len(onset)==3
    baseline=[r for r in frames if r["arm"]=="best"]
    candidate=[r for r in frames if r["arm"]=="balanced"]
    assert baseline[0]["mean_abs_delta"]==0.
    assert candidate[0]["mean_abs_delta"]>0.
    assert candidate[1]["mean_abs_delta"]>0.
    assert candidate[0]["teacher_press_keys"]==0
    assert candidate[5]["teacher_press_keys"]==2
    assert all(row["time_s"]==index*.01 for index,row in enumerate(candidate))
    assert all("set_valued_loss" in r and "threshold_crossing_keys" in r
               for r in frames)
    assert next(r for r in anchors if r["arm"]=="balanced")["onset_frames"]==20


@pytest.mark.parametrize(("control_dt","expected_frames"),[
    (0.005,40),
    (0.01,20),
    (0.02,10),
    (0.03,7),
    (0.2,1),
])
def test_onset_200ms_boundary_uses_full_precision(control_dt,expected_frames):
    # Frames exactly at 200ms must not be counted in the <200ms onset.
    target=torch.zeros((50,8))
    pred=torch.zeros_like(target)
    summary=_result_metrics(pred,target,pred,control_dt=control_dt)
    assert summary["onset"]["frames"]==expected_frames

def test_set_valued_loss_changes_even_without_threshold_crossing():
    t=torch.zeros((5,8))
    t[:,0]=1.
    base=torch.full((5,8),.3)
    candidate=torch.full((5,8),.4)
    result=_result_metrics(candidate,t,base,control_dt=.01)
    assert result["onset"]["threshold_crossings"]==0
    assert result["onset"]["mean_abs_delta"]==pytest.approx(.1)
    assert result["onset"]["loss_change"]!=0


def test_parameter_deltas_counts_and_preserves_frozen_parameters():
    model=ToyPolicy()
    old=clone_model_state(model)
    new={k:v.clone() for k,v in old.items()}
    new["weight"]=torch.tensor([.15])
    with torch.no_grad():
        model.weight.copy_(new["weight"])
    model.weight.grad=torch.tensor([.12])
    rows,summary=parameter_deltas(model,old,new,arm="balanced",gradient_norm=.4)
    assert len(rows)==1
    assert rows[0]["parameter"]=="weight"
    assert rows[0]["changed_elements"]==1
    assert rows[0]["last_gradient_l2"]==pytest.approx(.12)
    assert summary["changed_parameter_elements"]==1
    assert summary["trainable_parameter_elements"]==1
    assert summary["frozen_and_buffer_state_entries_unchanged"]==2
    new["frozen"]=torch.tensor([8.])
    with pytest.raises(RuntimeError,match="frozen parameter/buffer"):
        parameter_deltas(model,old,new,arm="balanced")


def test_parameter_deltas_sparse_coo_csr_never_densifies():
    model=ToyPolicy()
    coo=torch.sparse_coo_tensor(
        torch.tensor([[0,1],[0,1]]),
        torch.tensor([2.,3.]),(2,2)
    ).coalesce()
    csr=coo.to_sparse_csr()
    model.register_buffer("fixed_coo",coo)
    model.register_buffer("fixed_csr",csr)
    old=clone_model_state(model)
    now=clone_model_state(model)
    assert_state_equal(now,old,context="sparse")
    parameter_deltas(model,old,now,arm="balanced")
    changed={k:v.clone() for k,v in now.items()}
    csr_values=changed["fixed_csr"].values()
    csr_values[0]+=1
    with pytest.raises(RuntimeError,match="fixed_csr"):
        assert_state_equal(changed,old,context="sparse mutated")
    with pytest.raises(RuntimeError,match="frozen parameter/buffer"):
        parameter_deltas(model,old,changed,arm="balanced")


def _trace(*,time=.7,reason="Time limit",downs=(),early=()):
    actions=tuple(NKeyActionFrameTrace(
        time_s=i*.01,action_values=(.4,)*8,
        positions_m=(0.,)*8,velocities_m_s=(0.,)*8,
        pressed_flags=(False,)*8,next_target_index=0,
        next_target_time_s=.5,
    ) for i in range(20))
    physical=tuple(
        NKeyPhysicalKeyDownTrace(t,"left_1",0,.5) for t in downs
    )
    early_events=tuple(
        TooEarlyKeyDownTrace(t,"left_1",0,-400.,0.,2.,False)
        for t in early
    )
    return NKeyOverloadTrace(
        termination=EpisodeTerminationTrace(
            time_s=time,reason=reason,hits=0,too_early=len(early),
            keydowns=len(downs),next_target_index=0,overload_value=0.,
        ),
        keydown_times_s=tuple(downs),
        too_early_events=early_events,
        action_frames=actions,physical_keydowns=physical,
    )


def test_initial_motor_trace_counts_actions_and_physical_keydowns():
    trace=_trace(time=.09,reason="Overload",downs=(.04,.07),early=(.04,.07))
    info=trace_first_200ms(trace)
    assert info["early_overload_s"]==pytest.approx(.09)
    assert info["initial_keydown_count"]==2
    assert info["first_keydown_s"]==pytest.approx(.04)
    assert info["initial_max_positive_keys"]==8
    assert len(info["actions"])==20


def test_normal_start_controls_are_selected_from_baseline_only():
    rows=[_trace() for _ in range(20)]
    rows[0]=_trace(time=.05,reason="Overload")
    rows[1]=_trace(early=(.03,))
    selected=select_controls(rows,count=2)
    assert 1 not in selected and 2 not in selected
    assert selected==(3,4)


def test_source_sha256_exact_and_no_existing_file_replacement(tmp_path):
    checkpoint=tmp_path/"best.pt"
    checkpoint.write_bytes(b"fixed-selected-best")
    assert file_sha256(checkpoint)==hashlib.sha256(b"fixed-selected-best").hexdigest()
    file=tmp_path/"fixed.csv"
    _write_csv(file,["alpha","beta"],[{"alpha":1,"beta":2}])
    assert next(csv.DictReader(file.open(encoding="utf-8-sig"))) == {
        "alpha":"1","beta":"2"
    }
    with pytest.raises(FileExistsError):
        _write_csv(file,["alpha"],[{"alpha":1}])


def test_reproduction_is_explicitly_pass_or_mismatch():
    report={
        "baseline_validation":{"safe":12,"hits":1786,"early":278},
        "baseline_onset":{"early_overload":5},
        "arm_comparison":[
            {"name":"balanced","validation":{"safe":10,"hits":1685,"early":297},
             "onset":{"early_overload":5}},
            {"name":"onset_focused","validation":{"safe":7,"hits":1627,"early":235},
             "onset":{"early_overload":5}},
        ],
    }
    assert reference_reproduction(report)["match_all"]
    report["arm_comparison"][0]["validation"]["hits"]=1700
    mismatch=reference_reproduction(report)
    assert not mismatch["match_all"]
    assert mismatch["mismatches"]["balanced"]["hits"]["observed"]==1700


def test_checkpoint_audit_saves_only_diagnostic_candidates(tmp_path, monkeypatch):
    import dmdod.training.fixed_state_action_audit as audit_module
    monkeypatch.setattr(
        audit_module, "validation_summary",
        lambda results: {"safe": 12, "hits": 1786, "early": 278},
    )
    model=ToyPolicy()
    start=clone_model_state(model)
    best_file=tmp_path/"best.pt"
    best_file.write_bytes(b"immutable source")
    parent={
        "model_state":start,
        "human_visible_checkpoint_role":"selected-best",
        "training_mode":"human_visible_curriculum_dagger",
    }
    prepared=SimpleNamespace(
        model=model,parent=parent,source_checkpoint=best_file,
        validation=[SimpleNamespace(chart_name=f"v{i}") for i in range(20)],
        device=torch.device("cpu"),control_dt_s=.01,
    )
    audit=ActionDeltaAudit(
        prepared,output_dir=tmp_path,
        source_sha256=file_sha256(best_file),seed=2401,
    )
    with torch.no_grad():
        model.weight.add_(.2)
    audit.record_arm(
        "balanced",model,object(),
        CurriculumEpochMetrics(10,80,.2,.15,.1,.2,{}),
        [object()]*20,[_trace()]*20,
    )
    candidate=tmp_path/"balanced.diagnostic.pt"
    assert candidate.is_file()
    record=torch.load(candidate,map_location="cpu",weights_only=False)
    assert record["human_visible_checkpoint_role"]=="diagnostic-candidate"
    assert record["diagnostic_arm"]=="balanced"
    assert parent["human_visible_checkpoint_role"]=="selected-best"
    assert record["diagnostic_source_sha256"]==hashlib.sha256(
        b"immutable source"
    ).hexdigest()
    with pytest.raises(ValueError,match="duplicate"):
        audit.record_arm(
            "balanced",model,object(),
            CurriculumEpochMetrics(10,80,.2,.15,.1,.2,{}),
            [object()]*20,[_trace()]*20,
        )
    assert best_file.read_bytes()==b"immutable source"
