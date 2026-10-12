from __future__ import annotations

"""Two-arm, read-only-checkpoint motor onset pilot.

Both arms use the same original selected-best model/AdamW, pre-collected
Train-only expert and student-driven DAgger, and the same anchor/window plan.
Only the distribution of supervised frames within each selected window differs.
No curriculum advancement, checkpoint writes, or best selection occurs.
"""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch

from dmdod.training.human_visible_curriculum import (
    HumanVisibleReplayWindow,
    HumanVisibleTrajectory,
    _check_validation_context,
    _check_validation_summary,
    _collect_expert_trajectories,
    _collect_dagger_trajectories,
    _load_checked_optimizer,
    build_balanced_epoch_plan,
    build_human_visible_optimizer,
    build_replay_windows,
    curriculum_checkpoint_role,
    stable_seed,
    train_human_visible_epoch,
    validation_summary,
)
from dmdod.training.real_chart import clone_model_state, evaluate_role_continuous

PILOT_UPDATES = 10
ONSET_FRAMES = 20
EARLY_OVERLOAD_S = 0.200
ARMS = ("balanced", "onset_focused")


def onset_split_plan(
    windows: list[HumanVisibleReplayWindow],
    plan: tuple[tuple[int, ...], ...],
    trajectories: list[HumanVisibleTrajectory],
    *,
    model,
    onset_frames: int = ONSET_FRAMES,
) -> tuple[list[HumanVisibleReplayWindow], tuple[tuple[int, ...], ...]]:
    """Replace each selected base window by onset + remaining-context windows.

    Every selection retains exactly the same NUMBER of supervised frames.
    Onset is always a real first-N-frame sequence, never a 192-frame window
    that merely happens to contain the onset.
    """
    if onset_frames < 1:
        raise ValueError("onset_frames must be positive")
    trajectory_by_key={(t.anchor_id,t.source_kind):t for t in trajectories}
    if len(trajectory_by_key)!=len(trajectories):
        raise ValueError("duplicate trajectory/source key")
    output=list(windows)
    selected: dict[int,tuple[int,...]]={}
    for base_index in sorted({index for batch in plan for index in batch}):
        w=windows[base_index]
        length=w.supervised_frames
        if length <= 0:
            raise RuntimeError("replay window has no supervised frames")
        traj=trajectory_by_key[(w.anchor_id,w.source_kind)]
        count=min(onset_frames,length-1) if length>1 else 1
        if traj.frames<count:
            raise RuntimeError("Train trajectory does not contain requested onset")
        onset=HumanVisibleReplayWindow(
            anchor_id=w.anchor_id,source_kind=w.source_kind,
            observations=traj.observations[:count],
            teacher_actions=traj.teacher_actions[:count],
            student_actions=traj.student_actions[:count],
            initial_state=model.initial_state(traj.observations.device).detach().clone(),
            burn_in=0,start=0,
        )
        out=[len(output)]
        output.append(onset)
        rest=length-count
        if rest:
            # Same selected trajectory, end-aligned supervised context.
            # Extending no-grad burn-in avoids pretending a discontinuity
            # between the retained observations and the chosen background.
            new_burn=w.burn_in+count
            background=HumanVisibleReplayWindow(
                anchor_id=w.anchor_id,source_kind=w.source_kind,
                observations=w.observations,
                teacher_actions=w.teacher_actions,
                student_actions=w.student_actions,
                initial_state=w.initial_state,
                burn_in=new_burn,
                start=w.start+count,
            )
            if background.supervised_frames!=rest:
                raise RuntimeError("onset background lost supervised frames")
            out.append(len(output))
            output.append(background)
        if sum(output[i].supervised_frames for i in out)!=length:
            raise RuntimeError("onset split changed teacher-label budget")
        selected[base_index]=tuple(out)
    new_plan=tuple(tuple(i for base in batch for i in selected[base]) for batch in plan)
    for old,new in zip(plan,new_plan):
        old_labels=sum(windows[i].supervised_frames for i in old)
        new_labels=sum(output[i].supervised_frames for i in new)
        if old_labels!=new_labels:
            raise RuntimeError("optimizer batch supervised-frame counts diverged")
    return output,new_plan


def plan_coverage(windows,plan,expected_anchors:int)->dict:
    selected=[windows[i] for batch in plan for i in batch]
    counts={str(anchor):{"expert":0,"dagger":0} for anchor in range(expected_anchors)}
    for w in selected:
        counts[str(w.anchor_id)][w.source_kind]+=1
    missing=[key for key,entry in counts.items()
             if entry["expert"]==0 or entry["dagger"]==0]
    return {
        "per_anchor_source_windows":counts,
        "unique_anchors":expected_anchors-len(missing),
        "missing_anchors":missing,
        "supervised_label_frames":sum(w.supervised_frames for w in selected),
        "windows":len(selected),
        "optimizer_updates":len(plan),
    }


def _early_metrics(results,traces,segments)->tuple[dict,list[dict]]:
    if len(results)!=len(traces) or len(results)!=len(segments):
        raise RuntimeError("fixed Validation count mismatch")
    rows=[]
    for index,((stats,downs),trace,named) in enumerate(
        zip(results,traces,segments),1
    ):
        early=trace.termination.reason=="Overload" and (
            0.0 <= trace.termination.time_s <= EARLY_OVERLOAD_S+1e-9
        )
        frames=[f for f in trace.action_frames if f.time_s<EARLY_OVERLOAD_S-1e-9]
        physical=[
            e for e in trace.physical_keydowns if e.time_s<=EARLY_OVERLOAD_S+1e-9
        ]
        rows.append({
            "anchor":index,"chart":named.chart_name,
            "safe":not stats.overloaded,"hits":int(stats.hits),
            "misses":int(stats.misses),
            "too_early":int(stats.too_early_presses),
            "keydowns":int(downs),"overloaded":bool(stats.overloaded),
            "early_overload":early,
            "overload_at_s":float(trace.termination.time_s) if early else None,
            "termination_reason":trace.termination.reason,
            "initial_keydowns":len(physical),
            "initial_keydown_times_s":[e.time_s for e in physical],
            "initial_max_positive_action_keys":max(
                (sum(x>=0.25 for x in f.action_values) for f in frames),default=0
            ),
            "initial_action_keycounts":[
                {"t_s":f.time_s,"positive_keys":sum(x>=0.25 for x in f.action_values)}
                for f in frames
            ],
        })
    return {
        "early_overload":sum(x["early_overload"] for x in rows),
        "safe":sum(x["safe"] for x in rows),
        "hits":sum(x["hits"] for x in rows),
        "misses":sum(x["misses"] for x in rows),
        "too_early":sum(x["too_early"] for x in rows),
        "keydowns":sum(x["keydowns"] for x in rows),
        "initial_keydowns":sum(x["initial_keydowns"] for x in rows),
        "anchors":len(rows),
    },rows


def _source_preflight(prepared,config,report_path,updates):
    if config.mode!="human_visible_curriculum_dagger":
        raise ValueError("requires human_visible_curriculum_dagger")
    if len(prepared.anchors)!=20 or len(prepared.validation)!=20:
        raise ValueError("requires fixed 20 Train and 20 Validation anchors")
    if updates!=PILOT_UPDATES:
        raise ValueError(f"pilot requires exactly {PILOT_UPDATES} optimizer updates per arm")
    if config.human_visible.alpha_sweep_enabled:
        raise ValueError("alpha sweep must be disabled")
    if curriculum_checkpoint_role(prepared.parent)!="selected-best":
        raise ValueError("pilot requires selected-best source")
    if "human_visible_optimizer_state" not in prepared.parent:
        raise ValueError("source checkpoint has no optimizer state")
    if (
        report_path.resolve()==prepared.source_checkpoint.resolve()
        or report_path.exists() or report_path.with_suffix(".csv").exists()
    ):
        raise FileExistsError(f"ablation report path occupied: {report_path}")
    if config.human_visible.updates_per_epoch!=updates:
        raise ValueError("config updates_per_epoch must equal requested pilot updates")
    _check_validation_context(prepared.parent,config,prepared)
    return int(prepared.parent.get("human_visible_best_epoch",
                prepared.parent["human_visible_epoch"]))+1


def _seed_arm(seed:int,epoch:int):
    random.seed(stable_seed(seed,"onset-random",epoch))
    torch.manual_seed(stable_seed(seed,"onset-torch",epoch) % (2**63))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stable_seed(seed,"onset-cuda",epoch)%(2**63))


def run_onset_learning_ablation(prepared,config,*,report_path:Path,
                               updates:int=PILOT_UPDATES,audit=None)->dict:
    epoch=_source_preflight(prepared,config,report_path,updates)
    model=prepared.model
    visible=config.human_visible
    seed=int(visible.seed)
    best_state=clone_model_state(model)
    baseline_traces=[]
    baseline_results=evaluate_role_continuous(
        model,prepared.validation,label="onset-best",
        control_dt_s=prepared.control_dt_s,physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,verbose=False,trace_collector=baseline_traces,
    )
    baseline=validation_summary(baseline_results)
    _check_validation_summary(
        baseline,prepared.parent["human_visible_best_validation_summary"]
    )
    baseline_onset,baseline_anchor=_early_metrics(
        baseline_results,baseline_traces,prepared.validation
    )
    if audit is not None:
        audit.record_baseline(model, baseline_results, baseline_traces)
    expert=_collect_expert_trajectories(prepared)
    # Beta=0: physical Student-driven Train states with Teacher-labelled
    # counterfactual actions. Same exact trajectories for BOTH training arms.
    dagger,dagger_total,dagger_intervened=_collect_dagger_trajectories(
        prepared,model,beta=0.0,epoch=epoch,base_seed=seed
    )
    trajectories=[*expert,*dagger]
    if audit is not None:
        audit.record_train_trajectories(trajectories)
    if {(t.anchor_id,t.source_kind) for t in trajectories}!={
        (i,source) for i in range(20) for source in ("expert","dagger")
    }:
        raise RuntimeError("missing Train anchor/source trajectory")
    windows=build_replay_windows(
        model,trajectories,burn_in_steps=visible.burn_in_steps,
        supervised_steps=visible.supervised_steps,
    )
    plan=build_balanced_epoch_plan(
        windows,anchor_ids=list(range(20)),updates_per_epoch=updates,
        anchors_per_batch=visible.anchors_per_batch,
        seed=stable_seed(seed,"plan",epoch),
    )
    base_coverage=plan_coverage(windows,plan,20)
    if base_coverage["unique_anchors"]!=20:
        raise RuntimeError("not all 20 Train anchors used in pilot updates")
    if not all(
        row["expert"]==row["dagger"]
        for row in base_coverage["per_anchor_source_windows"].values()
    ):
        raise RuntimeError("Expert/DAgger sampling imbalance")
    focused_windows,focused_plan=onset_split_plan(
        windows,plan,trajectories,model=model
    )
    focused_coverage=plan_coverage(focused_windows,focused_plan,20)
    if (focused_coverage["supervised_label_frames"] !=
            base_coverage["supervised_label_frames"]):
        raise RuntimeError("Arm budgets unequal")
    output_arms=[]
    try:
        for name in ARMS:
            model.load_state_dict(best_state,strict=True)
            model.prepare_recurrent_runtime()
            optimizer=build_human_visible_optimizer(model,visible)
            _load_checked_optimizer(optimizer,prepared.parent,model)
            _seed_arm(seed,epoch)
            selected_windows=windows if name=="balanced" else focused_windows
            selected_plan=plan if name=="balanced" else focused_plan
            stats=train_human_visible_epoch(
                model,selected_windows,selected_plan,
                optimizer=optimizer,grad_clip=visible.grad_clip,
                control_dt_s=prepared.control_dt_s,
            )
            if stats.updates!=updates:
                raise RuntimeError("Arm optimizer updates diverged")
            traces=[]
            results=evaluate_role_continuous(
                model,prepared.validation,label=f"onset-{name}",
                control_dt_s=prepared.control_dt_s,physics_dt_s=prepared.physics_dt_s,
                device=prepared.device,verbose=False,trace_collector=traces,
            )
            summary=validation_summary(results)
            onset,anchors=_early_metrics(results,traces,prepared.validation)
            if audit is not None:
                audit.record_arm(name, model, optimizer, stats, results, traces)
            print(f"onset {name}: SAFE={summary['safe']}/20 "
                  f"H={summary['hits']} Early={summary['early']} "
                  f"Keydowns={summary['keydowns']} "
                  f"Misses={onset['misses']} first200msOverload={onset['early_overload']}")
            output_arms.append({
                "name":name,"updates":stats.updates,"training":asdict(stats),
                "selection_plan":plan_coverage(selected_windows,selected_plan,20),
                "validation":summary,"onset":onset,"anchors":anchors,
                "selected_as_best":False,"checkpoint_written":False,
            })
            del optimizer
    finally:
        model.load_state_dict(best_state,strict=True)
        model.prepare_recurrent_runtime()
    report={
        "kind":"human_visible_onset_learning_pilot",
        "source_checkpoint":str(prepared.source_checkpoint),
        "source_epoch":epoch-1,"training_epoch":epoch,
        "updates_per_arm":updates,"beta_train_collection":0.0,
        "seed":seed,"train_anchors":20,"validation_anchors":20,
        "dagger_frames":dagger_total,"dagger_intervened_frames":dagger_intervened,
        "baseline_validation":baseline,"baseline_onset":baseline_onset,
        "baseline_anchors":baseline_anchor,
        "arm_comparison":output_arms,
        "notes":[
            "Both arms start from identical selected-best model and AdamW states.",
            "Expert and Student-driven beta=0 DAgger Train trajectories shared.",
            "Balanced anchor/source selections and supervised LABEL COUNTS match.",
            "Arm B oversamples genuine 0-200ms windows (short 20-frame slices).",
            "Arm B uses separate onset/background windows with unchanged loss.",
            "Window/backprop counts differ (Arm B uses more shorter windows).",
            "SAFE alone does not imply better playing: inspect Hits, Misses, Keydowns.",
            "Validation was never used to train or select a checkpoint.",
            "FINAL holdout and original checkpoint are untouched.",
        ],
    }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    csv_path=report_path.with_suffix(".csv")
    if csv_path.exists():
        raise FileExistsError(f"CSV report already exists: {csv_path}")
    with csv_path.open("x",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["arm","anchor","chart","safe","hits","misses",
                         "too_early","keydowns","overloaded","early_overload",
                         "overload_at_s","initial_keydowns",
                         "initial_max_positive_action_keys"])
        for arm_name,rows in [
            ("best",baseline_anchor),
            *[(a["name"],a["anchors"]) for a in output_arms],
        ]:
            for row in rows:
                writer.writerow([
                    arm_name,row["anchor"],row["chart"],int(row["safe"]),
                    row["hits"],row["misses"],row["too_early"],row["keydowns"],
                    int(row["overloaded"]),int(row["early_overload"]),
                    row["overload_at_s"],row["initial_keydowns"],
                    row["initial_max_positive_action_keys"],
                ])
    return report
