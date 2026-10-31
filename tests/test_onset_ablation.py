from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training.human_visible_curriculum import (
    CurriculumEpochMetrics, HumanVisibleReplayWindow, HumanVisibleTrajectory,
    build_balanced_epoch_plan,
)
from dmdod.training.onset_ablation import (
    ARMS, _early_metrics, onset_split_plan, plan_coverage,
    run_onset_learning_ablation,
)
from dmdod.training.onset_readout_audit import train_variance_mask


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(3.0))

    def initial_state(self,device):
        return torch.zeros(1,device=device)

    def prepare_recurrent_runtime(self):
        pass


def trajectories(length=250):
    items=[]
    for anchor in range(20):
        for source in ("expert","dagger"):
            x=torch.arange(length,dtype=torch.float32).reshape(-1,1)
            action=torch.ones((length,8))
            items.append(HumanVisibleTrajectory(
                anchor_id=anchor,source_kind=source,
                observations=x,teacher_actions=action,
                student_actions=action*0,
            ))
    return items


def windows_and_plan(model):
    all_trajectories=trajectories()
    base=[]
    for t in all_trajectories:
        base.append(HumanVisibleReplayWindow(
            anchor_id=t.anchor_id,source_kind=t.source_kind,
            observations=t.observations[:192],
            teacher_actions=t.teacher_actions[:192],
            student_actions=t.student_actions[:192],
            initial_state=model.initial_state(torch.device("cpu")),
            burn_in=0,start=0,
        ))
    plan=build_balanced_epoch_plan(
        base,anchor_ids=list(range(20)),updates_per_epoch=10,
        anchors_per_batch=4,seed=2401,
    )
    return all_trajectories,base,plan


def test_equal_supervised_frame_budget_and_real_twenty_frame_onset():
    model=TinyModel()
    all_trajectories,base,plan=windows_and_plan(model)
    focused,focused_plan=onset_split_plan(
        base,plan,all_trajectories,model=model
    )
    for old,new in zip(plan,focused_plan):
        assert len(new)==2*len(old)
        assert sum(base[i].supervised_frames for i in old)==sum(
            focused[i].supervised_frames for i in new
        )
        for i in new[::2]:
            w=focused[i]
            assert w.supervised_frames==20
            assert w.burn_in==0
            assert w.start==0
            assert torch.equal(w.observations[:,0],torch.arange(20).float())
        for i in new[1::2]:
            w=focused[i]
            assert w.supervised_frames==172
            assert w.burn_in==20
    assert sum(base[i].supervised_frames for batch in plan for i in batch)==(
        sum(focused[i].supervised_frames for batch in focused_plan for i in batch)
    )


def test_balanced_10_updates_cover_all_twenty_anchors_both_sources():
    model=TinyModel()
    _,base,plan=windows_and_plan(model)
    count=plan_coverage(base,plan,20)
    assert len(plan)==10
    assert count["unique_anchors"]==20
    assert all(
        row["expert"]==row["dagger"]==2
        for row in count["per_anchor_source_windows"].values()
    )


def test_onset_split_handles_short_tail_without_extra_labels():
    model=TinyModel()
    data=trajectories(length=4)
    window=HumanVisibleReplayWindow(
        anchor_id=0,source_kind="expert",
        observations=data[0].observations[:4],
        teacher_actions=data[0].teacher_actions[:4],
        student_actions=data[0].student_actions[:4],
        initial_state=model.initial_state(torch.device("cpu")),
        burn_in=0,start=0
    )
    out,plan=onset_split_plan([window],((0,),),data,model=model)
    assert len(plan[0])==2
    assert [out[i].supervised_frames for i in plan[0]]==[3,1]


def test_onset_split_refuses_duplicate_train_trajectory_keys():
    model=TinyModel()
    t=trajectories()
    with pytest.raises(ValueError,match="duplicate"):
        onset_split_plan([],(),[t[0],t[0]],model=model)


def test_early_overload_and_action_counts_are_reported_without_press_suppression():
    def trace(index):
        overload=index==1
        return SimpleNamespace(
            termination=SimpleNamespace(
                reason="Overload" if overload else "Time limit",
                time_s=.08 if overload else 2.
            ),
            action_frames=(
                SimpleNamespace(time_s=0.,action_values=(.8,.9,.75,0,0,0,0,0)),
                SimpleNamespace(time_s=.01,action_values=(.01,)*8),
            ),
            physical_keydowns=(
                SimpleNamespace(time_s=.005),
                SimpleNamespace(time_s=.05),
            ),
        )
    results=[
        (SimpleNamespace(hits=10,misses=2,too_early_presses=3,overloaded=True),2),
        (SimpleNamespace(hits=30,misses=1,too_early_presses=0,overloaded=False),2),
    ]
    agg,rows=_early_metrics(
        results,[trace(1),trace(2)],
        [SimpleNamespace(chart_name="x"),SimpleNamespace(chart_name="y")]
    )
    assert agg["early_overload"]==1
    assert agg["hits"]==40
    assert agg["misses"]==3
    assert agg["keydowns"]==4
    assert rows[0]["initial_max_positive_action_keys"]==3
    assert rows[0]["initial_keydown_times_s"]==[.005,.05]
    assert rows[0]["overload_at_s"]==pytest.approx(.08)


def test_train_only_zero_variance_mask_does_not_drop_new_validation_values():
    train=torch.tensor([[0.,1.,2.],[0.,2.,2.],[0.,3.,2.]])
    mask=train_variance_mask(train)
    assert mask.tolist()==[False,True,False]
    validation=torch.tensor([[100.,5.,200.]])
    assert validation[:,mask].shape==(1,1)
    with pytest.raises(ValueError,match="zero variance"):
        train_variance_mask(torch.ones((3,4)))


def test_full_pilot_restores_best_and_optimizer_for_each_arm(monkeypatch,tmp_path):
    import dmdod.training.onset_ablation as ablation

    model=TinyModel()
    all_trajectories,base,plan=windows_and_plan(model)
    names=[]
    optimizer_markers=[]
    labels=[]
    updates=[]
    traces_seen=[]

    class DummyOptimizer:
        marker=0

    class Stat:
        overloaded=False
        hits=100
        misses=20
        too_early_presses=10

    def rollout(*a,trace_collector,**k):
        traces_seen.append(float(model.weight.detach()))
        trace=SimpleNamespace(
            termination=SimpleNamespace(reason="Time limit",time_s=2.0),
            action_frames=(),physical_keydowns=(),
        )
        trace_collector.extend([trace]*20)
        return [(Stat(),100)]*20

    def train(model,win,scheduled,*,optimizer,**kwargs):
        names.append(float(model.weight.detach()))
        optimizer_markers.append(optimizer.marker)
        labels.append(sum(win[i].supervised_frames for batch in scheduled for i in batch))
        updates.append(len(scheduled))
        with torch.no_grad():
            model.weight.add_(.1 if len(win)==len(base) else .2)
        return CurriculumEpochMetrics(10,len(scheduled),.2,.1,.1,.2,{})

    prepared=SimpleNamespace(
        model=model,anchors=[SimpleNamespace(chart_name=f"a{i}") for i in range(20)],
        validation=[SimpleNamespace(chart_name=f"v{i}") for i in range(20)],
        parent={
            "human_visible_optimizer_state":{},
            "human_visible_best_epoch":2,
            "human_visible_epoch":2,
            "human_visible_best_validation_summary":{"safe":20},
        },
        source_checkpoint=tmp_path/"best.pt",
        control_dt_s=.01,physics_dt_s=.001,
        device=torch.device("cpu")
    )
    config=SimpleNamespace(
        mode="human_visible_curriculum_dagger",
        human_visible=SimpleNamespace(
            alpha_sweep_enabled=False,
            updates_per_epoch=10,
            burn_in_steps=96,supervised_steps=192,
            anchors_per_batch=4,grad_clip=1.,seed=2401,
        ),
    )
    monkeypatch.setattr(ablation,"curriculum_checkpoint_role",lambda p:"selected-best")
    monkeypatch.setattr(ablation,"_check_validation_context",lambda *args:None)
    monkeypatch.setattr(ablation,"_check_validation_summary",lambda *args:None)
    monkeypatch.setattr(ablation,"_collect_expert_trajectories",
                        lambda p:[t for t in all_trajectories if t.source_kind=="expert"])
    monkeypatch.setattr(ablation,"_collect_dagger_trajectories",
                        lambda p,m,**kw:([t for t in all_trajectories if t.source_kind=="dagger"],5000,0))
    monkeypatch.setattr(ablation,"build_replay_windows",lambda *args,**kw:base)
    monkeypatch.setattr(ablation,"build_balanced_epoch_plan",lambda *args,**kw:plan)
    monkeypatch.setattr(ablation,"build_human_visible_optimizer",lambda *args:DummyOptimizer())
    monkeypatch.setattr(ablation,"_load_checked_optimizer",
                        lambda opt,parent,model:setattr(opt,"marker",27))
    monkeypatch.setattr(ablation,"train_human_visible_epoch",train)
    monkeypatch.setattr(ablation,"evaluate_role_continuous",rollout)
    monkeypatch.setattr(ablation,"validation_summary",lambda results:{
        "safe":20,"hits":2000,"early":200,"keydowns":2000,"targets":4000,
    })
    report=tmp_path/"pilot.json"
    result=run_onset_learning_ablation(prepared,config,report_path=report)
    assert report.is_file() and report.with_suffix(".csv").is_file()
    assert names==[3.0,3.0]
    assert optimizer_markers==[27,27]
    assert updates==[10,10]
    assert labels[0]==labels[1]
    assert len(traces_seen)==3
    assert model.weight.detach().item() == 3.0
    assert all(r["selected_as_best"] is False for r in result["arm_comparison"])
    with pytest.raises(FileExistsError):
        run_onset_learning_ablation(prepared,config,report_path=report)
