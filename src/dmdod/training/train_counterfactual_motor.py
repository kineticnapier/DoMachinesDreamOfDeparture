from __future__ import annotations

"""Train-only, causal motor counterfactuals and controlled 10-update A/B pilot.

All privileged timings are evaluator-side Train diagnostics. The Student always
receives the original 263D HUD. No existing motor/loss/checkpoint code changes.
"""

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import torch

from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.motor.n_key import NKeyAction, n_key_names
from dmdod.training.fixed_state_action_audit import assert_state_equal, file_sha256
from dmdod.training.human_visible import freeze_human_visible_controller
from dmdod.training.human_visible_curriculum import (
    _collect_dagger_trajectories,
    _collect_expert_trajectories,
    _load_checked_optimizer,
    build_balanced_epoch_plan,
    build_human_visible_optimizer,
    build_replay_windows,
    context_balanced_actuation_loss,
    stable_seed,
    validation_summary,
)
from dmdod.training.motor_sensitivity_audit import _environment
from dmdod.training.onset_ablation import _early_metrics, _seed_arm, _source_preflight
from dmdod.training.real_chart import clone_model_state, evaluate_role_continuous

DELTA_GRID = (-0.02, -0.05, -0.10, -0.20, 0.02, 0.05, 0.10, 0.20)
DURATION_GRID = (1, 2, 4, 8)
PRECONTACT_GRID = (8, 4, 2, 1)
ONSET_S = 0.200
HORIZON_S = 0.400
MAX_CASES_PER_ANCHOR = 128
UPDATES = 10
AUX_WEIGHT = 0.20


@dataclass(frozen=True)
class LocalIntervention:
    key: int
    start: int
    duration: int
    delta: float

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass
class MotorRollout:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    executed_actions: torch.Tensor
    used: tuple[int, ...]
    missed: tuple[int, ...]
    keydowns: tuple[tuple[str, float], ...]
    early: tuple[tuple[str, float], ...]
    hits: int
    misses: int
    too_early: int
    overloaded: bool
    duration_s: float
    terminated: bool


@dataclass
class MotorCorrection:
    anchor_id: int
    intervention: LocalIntervention
    observations: torch.Tensor
    corrected_actions: torch.Tensor
    base_hits: int
    hits: int
    teacher_hits: int


def patched_action(action: tuple[float, ...], intervention: LocalIntervention | None,
                   step: int) -> tuple[float, ...]:
    if intervention is None or not intervention.start <= step < intervention.end:
        return action
    values = list(action)
    k = intervention.key
    values[k] = min(1.0, max(-1.0, values[k] + intervention.delta))
    return tuple(values)


def rollout(prepared, model, named, intervention: LocalIntervention | None = None,
            *, horizon_s: float = HORIZON_S) -> MotorRollout:
    """Always recompute the policy using post-perturbation motor observations."""
    env = _environment(prepared, named)
    observation = env.reset()
    state = model.initial_state(prepared.device)
    obs, raw, executed = [], [], []
    terminated = False
    limit = math.ceil(horizon_s / prepared.control_dt_s)
    model.eval()
    with torch.no_grad():
        for t in range(limit):
            vector = encode_n_key_hud_real_chart_observation(observation)
            if len(vector) != 263:
                raise RuntimeError("Student 263D observation contract violated")
            x = torch.tensor(vector, dtype=torch.float32, device=prepared.device)
            means, _, _, state = model.forward_step(x, state)
            action = tuple(float(y.item()) for y in torch.tanh(means))
            altered = patched_action(action, intervention, t)
            obs.append(vector)
            raw.append(action)
            executed.append(altered)
            next_step = env.step(NKeyAction(altered))
            observation = next_step.observation
            if next_step.done:
                terminated = True
                break
    return MotorRollout(
        observations=torch.tensor(obs, dtype=torch.float32, device=prepared.device),
        raw_actions=torch.tensor(raw, dtype=torch.float32, device=prepared.device),
        executed_actions=torch.tensor(executed, dtype=torch.float32, device=prepared.device),
        used=tuple(i for i, flag in enumerate(env._used) if flag),
        missed=tuple(i for i, flag in enumerate(env._missed) if flag),
        keydowns=tuple((e.key, float(e.time_s)) for e in env._trace_physical_keydowns),
        early=tuple((e.key, float(e.time_s)) for e in env._trace_too_early),
        hits=int(env._hits), misses=int(env._misses), too_early=int(env._too_early),
        overloaded=bool(env._overload.overloaded),
        duration_s=float(env.privileged_episode_time_s()),
        terminated=terminated,
    )


def assert_train_rollout_matches_dagger(base: MotorRollout, dagger, *, atol=1e-5) -> None:
    """Fail closed if counterfactual origin differs from collected beta=0 states."""
    length = int(base.raw_actions.shape[0])
    if dagger.frames < length:
        raise RuntimeError("Train DAgger rollout ended before motor baseline")
    if not torch.allclose(base.raw_actions, dagger.student_actions[:length], atol=atol, rtol=0):
        raise RuntimeError("Train beta=0 Student actions differ from motor baseline")
    if not torch.allclose(base.observations, dagger.observations[:length], atol=atol, rtol=0):
        raise RuntimeError("Train beta=0 observations differ from motor baseline")


def candidate_cases(base: MotorRollout, *, dt: float, key_count: int,
                    max_cases: int = MAX_CASES_PER_ANCHOR) -> list[LocalIntervention]:
    """Bounded diagnostic-only search around *Train* physical TooEarly events."""
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    key_indices = {name: i for i, name in enumerate(n_key_names(key_count))}
    sources = []
    for key, event_time in base.early:
        if key in key_indices and event_time <= ONSET_S + 1e-9:
            if all(existing_key != key_indices[key] for existing_key, _ in sources):
                sources.append((key_indices[key], event_time))
            if len(sources) >= 2:
                break
    result, seen = [], set()
    # Spread the budget over keys, amplitudes, durations and precontact leads.
    # A sorted-by-absolute-delta prefix would silently exclude ±0.10/0.20.
    for lead in PRECONTACT_GRID:
        for duration in DURATION_GRID:
            for delta in DELTA_GRID:
                for key, event_time in sources:
                    event_step = max(0, math.ceil(event_time / dt) - 1)
                    start = max(0, event_step - lead)
                    case = LocalIntervention(key, start, duration, delta)
                    if case.end > len(base.raw_actions):
                        continue
                    if case not in seen:
                        seen.add(case)
                        result.append(case)
    return result[:max_cases]


def accept_counterfactual(base: MotorRollout, alternative: MotorRollout,
                          teacher: MotorRollout) -> tuple[bool, str]:
    """Guard against learning no-press solutions or sacrificing due hits."""
    if alternative.overloaded:
        return False, "overload"
    if not alternative.keydowns:
        return False, "no_keydowns"
    if teacher.overloaded or teacher.terminated and teacher.duration_s < ONSET_S:
        return False, "expert_reference_invalid"
    if not set(base.used).issubset(alternative.used):
        return False, "lost_baseline_hit"
    if alternative.hits < base.hits:
        return False, "lost_hits"
    if alternative.too_early >= base.too_early:
        return False, "not_fewer_too_early"
    # The candidate must demonstrate some meaningful scoring, measured
    # against a separate Train expert, not just survive with no presses.
    min_hits = max(base.hits, math.ceil(0.75 * teacher.hits))
    if alternative.hits < min_hits:
        return False, "insufficient_teacher_relative_hits"
    if not base.overloaded:
        if alternative.misses > base.misses:
            return False, "more_misses"
    elif alternative.misses > teacher.misses + 1:
        return False, "too_many_misses_vs_expert"
    if not base.overloaded and alternative.hits == base.hits:
        return False, "no_hit_gain_on_safe_baseline"
    return True, "physical_improvement_train_only"


def choose_correction(prepared, model, named, anchor_id: int,
                      dagger, *, max_cases: int = MAX_CASES_PER_ANCHOR):
    """A causal counterfactual: labels and observations from the SAME rollout."""
    base = rollout(prepared, model, named)
    assert_train_rollout_matches_dagger(base, dagger)
    # The original expert is run by a separate call for comparison, not used
    # as a Student observation. Its full-rollout labels are already collected.
    from dmdod.n_key_capacity import CenterFirstNKeyTeacher
    from dmdod.training.n_key import _teacher_action
    env = _environment(prepared, named)
    observation = env.reset()
    teacher_policy = CenterFirstNKeyTeacher(prepared.key_count)
    for _ in range(math.ceil(HORIZON_S / prepared.control_dt_s)):
        action = _teacher_action(env, teacher_policy, observation, lead_s=prepared.lead_s)
        result = env.step(action)
        observation = result.observation
        if result.done:
            break
    expert = MotorRollout(
        observations=base.observations[:0], raw_actions=base.raw_actions[:0],
        executed_actions=base.executed_actions[:0],
        used=tuple(i for i, f in enumerate(env._used) if f),
        missed=tuple(i for i, f in enumerate(env._missed) if f),
        keydowns=tuple((e.key, float(e.time_s)) for e in env._trace_physical_keydowns),
        early=tuple((e.key, float(e.time_s)) for e in env._trace_too_early),
        hits=int(env._hits), misses=int(env._misses), too_early=int(env._too_early),
        overloaded=bool(env._overload.overloaded),
        duration_s=float(env.privileged_episode_time_s()), terminated=bool(env._done),
    )
    candidates = candidate_cases(base, dt=prepared.control_dt_s,
                                 key_count=prepared.key_count, max_cases=max_cases)
    rows, good = [], []
    for case in candidates:
        altered = rollout(prepared, model, named, case)
        accepted, why = accept_counterfactual(base, altered, expert)
        row = {
            "anchor":anchor_id+1,"chart":named.chart_name,
            "key":n_key_names(prepared.key_count)[case.key],
            "start_frame":case.start,"duration_frames":case.duration,
            "delta":case.delta,"accepted":accepted,"reason":why,
            "base_hits":base.hits,"base_early":base.too_early,
            "base_overload":base.overloaded,"expert_hits":expert.hits,
            "hits":altered.hits,"misses":altered.misses,
            "too_early":altered.too_early,"overload":altered.overloaded,
            "keydowns":len(altered.keydowns),
        }
        rows.append(row)
        if accepted:
            # Prefer small perturbations and short duration after hard guards.
            good.append((abs(case.delta)*case.duration, -altered.hits,
                         altered.too_early, case.start, case, altered))
    if not good:
        return None, rows, base
    good.sort(key=lambda x: x[:4])
    _, _, _, _, winner, trajectory = good[0]
    if winner.end > trajectory.observations.shape[0]:
        raise RuntimeError("selected intervention ran past trajectory termination")
    correction = MotorCorrection(
        anchor_id=anchor_id, intervention=winner,
        observations=trajectory.observations,
        corrected_actions=trajectory.executed_actions,
        base_hits=base.hits, hits=trajectory.hits, teacher_hits=expert.hits,
    )
    return correction, rows, base


def auxiliary_loss(model, correction: MotorCorrection) -> torch.Tensor:
    """Only the causally patched key/frames are supervised on altered states."""
    item = correction.intervention
    state = model.initial_state(correction.observations.device)
    if item.start:
        with torch.no_grad():
            _, _, state = model.forward_sequence(correction.observations[:item.start], state)
        state = state.detach()
    means, _, _ = model.forward_sequence(
        correction.observations[item.start:item.end], state
    )
    predicted = torch.tanh(means)[:, item.key]
    target = correction.corrected_actions[item.start:item.end, item.key]
    if predicted.numel() != item.duration:
        raise RuntimeError("corrected frames were lost")
    return (predicted - target).square().mean()


def train_arm(model, windows, plan, optimizer, *, corrections, use_aux,
              grad_clip, control_dt_s):
    """Same base windows and optimizer steps. Optional Train-only auxiliary."""
    trainable = freeze_human_visible_controller(model)
    model.train()
    metrics = {"updates":0,"base_losses":[],"aux_losses":[],"grad_norms":[],
               "aux_updates":0}
    for batch in plan:
        optimizer.zero_grad(set_to_none=True)
        base_loss = None
        for index in batch:
            w = windows[index]
            state = w.initial_state.detach()
            if w.burn_in:
                with torch.no_grad():
                    _, _, state = model.forward_sequence(w.observations[:w.burn_in], state)
                state = state.detach()
            means, _, _ = model.forward_sequence(w.observations[w.burn_in:], state)
            predicted = torch.tanh(means)
            loss, _ = context_balanced_actuation_loss(
                predicted, w.teacher_actions[w.burn_in:],
                w.student_actions[w.burn_in:], control_dt_s=control_dt_s,
            )
            term = loss / len(batch)
            base_loss = term if base_loss is None else base_loss + term
        if base_loss is None:
            raise RuntimeError("empty optimizer batch")
        aux = None
        if use_aux:
            selected = {windows[i].anchor_id for i in batch}
            active = [corrections[k] for k in sorted(selected) if k in corrections]
            if active:
                aux = torch.stack([auxiliary_loss(model, c) for c in active]).mean()
        total = base_loss + (AUX_WEIGHT * aux if aux is not None else 0.0)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("nonfinite Train loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        metrics["updates"] += 1
        metrics["base_losses"].append(float(base_loss.detach()))
        metrics["aux_losses"].append(float(aux.detach()) if aux is not None else None)
        metrics["grad_norms"].append(float(norm.detach()))
        metrics["aux_updates"] += int(aux is not None)
    return metrics


def run_counterfactual_motor_pilot(prepared, config, *, output_dir: Path,
                                    max_cases_per_anchor: int = MAX_CASES_PER_ANCHOR):
    """Train-only discovery, fixed-plan A/B, then untouched Validation metrics."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory occupied: {output_dir}")
    if max_cases_per_anchor < 1 or max_cases_per_anchor > 512:
        raise ValueError("max_cases_per_anchor must be between 1 and 512")
    # Reuse exactly the existing onset pilot's selected-best / Validation context checks.
    epoch = _source_preflight(prepared, config, output_dir / "pilot.json", UPDATES)
    if prepared.key_count != 8 or prepared.input_dim != 263:
        raise ValueError("requires 8 keys and Student 263D HUD")
    model = prepared.model
    best = clone_model_state(model)
    sha_before = file_sha256(prepared.source_checkpoint)
    report = {"kind":"train_counterfactual_motor_pilot_v1",
              "source_sha256":sha_before,"train_anchors":20,
              "validation_anchors":20,"updates_per_arm":UPDATES,
              "auxiliary_loss_weight":AUX_WEIGHT,
              "max_cases_per_anchor":max_cases_per_anchor,
              "delta_grid":list(DELTA_GRID),
              "duration_grid":list(DURATION_GRID),
              "uses_validation_for_teacher":False,
              "used_final":False,"selected_best":False}
    try:
        baseline_traces = []
        baseline_results = evaluate_role_continuous(
            model, prepared.validation, label="cf-motor-best",
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device, verbose=False, trace_collector=baseline_traces,
        )
        from dmdod.training.human_visible_curriculum import _check_validation_summary
        baseline = validation_summary(baseline_results)
        _check_validation_summary(
            baseline, prepared.parent["human_visible_best_validation_summary"]
        )
        expert_data = _collect_expert_trajectories(prepared)
        dagger_data, frames, intervened = _collect_dagger_trajectories(
            prepared, model, beta=0.0, epoch=epoch,
            base_seed=int(config.human_visible.seed),
        )
        if intervened != 0 or len(dagger_data) != 20 or len(expert_data) != 20:
            raise RuntimeError("Train-only beta=0 DAgger collection mismatch")
        corrections = {}
        exploration = []
        for i, named in enumerate(prepared.anchors):
            winner, cases, origin = choose_correction(
                prepared, model, named, i, dagger_data[i],
                max_cases=max_cases_per_anchor,
            )
            exploration.extend(cases)
            if winner is not None:
                corrections[i] = winner
            print(f"Train counterfactual {i+1:02d}/20: "
                  f"cases={len(cases)} accepted={sum(x['accepted'] for x in cases)} "
                  f"teacher={'yes' if winner is not None else 'none'}", flush=True)
        if not corrections:
            # No optimizer steps on an unqualified teacher. Preserve the
            # Train-only diagnostics so the failure can be investigated.
            report.update({
                "status":"no_train_corrections_no_training",
                "train_dagger_frames":frames,
                "train_candidate_evaluations":len(exploration),
                "train_candidates_accepted":0,
                "train_anchors_with_correction":0,
                "comparison":[],
            })
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir/"train_candidates.csv").open(
                "x",encoding="utf-8-sig",newline=""
            ) as f:
                if exploration:
                    writer=csv.DictWriter(f,fieldnames=list(exploration[0]))
                    writer.writeheader()
                    writer.writerows(exploration)
            with (output_dir/"pilot.json").open("x",encoding="utf-8") as f:
                json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
                f.write("\n")
            return report
        trajectories = [*expert_data, *dagger_data]
        windows = build_replay_windows(
            model, trajectories, burn_in_steps=config.human_visible.burn_in_steps,
            supervised_steps=config.human_visible.supervised_steps,
        )
        plan = build_balanced_epoch_plan(
            windows, anchor_ids=list(range(20)), updates_per_epoch=UPDATES,
            anchors_per_batch=config.human_visible.anchors_per_batch,
            seed=stable_seed(int(config.human_visible.seed), "plan", epoch),
        )
        comparison = []
        for arm, use_aux in (("baseline",False),("physics_aware",True)):
            model.load_state_dict(best, strict=True)
            model.prepare_recurrent_runtime()
            optimizer = build_human_visible_optimizer(model, config.human_visible)
            _load_checked_optimizer(optimizer, prepared.parent, model)
            _seed_arm(int(config.human_visible.seed), epoch)
            metrics = train_arm(
                model, windows, plan, optimizer, corrections=corrections,
                use_aux=use_aux, grad_clip=config.human_visible.grad_clip,
                control_dt_s=prepared.control_dt_s,
            )
            if metrics["updates"] != UPDATES:
                raise RuntimeError("A/B optimizer update budgets diverged")
            traces = []
            results = evaluate_role_continuous(
                model, prepared.validation, label=f"cf-motor-{arm}",
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s, device=prepared.device,
                verbose=False, trace_collector=traces,
            )
            totals = validation_summary(results)
            early, anchors = _early_metrics(results,traces,prepared.validation)
            comparison.append({"arm":arm,"training":metrics,
                               "validation":totals,"onset":early,
                               "anchors":anchors})
            print(f"{arm}: SAFE={totals['safe']}/20 "
                  f"H={totals['hits']} Early={totals['early']} "
                  f"initialOverload={early['early_overload']}/20",flush=True)
            del optimizer
        report.update({
            "train_dagger_frames":frames,
            "train_intervened_frames":intervened,
            "train_candidate_evaluations":len(exploration),
            "train_candidates_accepted":sum(r["accepted"] for r in exploration),
            "train_anchors_with_correction":len(corrections),
            "train_corrections":[{
                "anchor":i+1,"key":n_key_names(8)[c.intervention.key],
                "start_frame":c.intervention.start,
                "duration_frames":c.intervention.duration,
                "delta":c.intervention.delta,
                "base_hits":c.base_hits,"hits":c.hits,
                "teacher_hits":c.teacher_hits,
            } for i,c in sorted(corrections.items())],
            "source_validation":baseline,
            "comparison":comparison,
        })
        if file_sha256(prepared.source_checkpoint) != sha_before:
            raise RuntimeError("source checkpoint changed during Train pilot")
        output_dir.mkdir(parents=True,exist_ok=True)
        # All artifacts are append-only; never touch checkpoints or FINAL.
        with (output_dir/"train_candidates.csv").open("x",encoding="utf-8-sig",newline="") as f:
            if exploration:
                writer=csv.DictWriter(f,fieldnames=list(exploration[0]))
                writer.writeheader()
                writer.writerows(exploration)
        with (output_dir/"pilot.json").open("x",encoding="utf-8") as f:
            json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
            f.write("\n")
        return report
    finally:
        model.load_state_dict(best,strict=True)
        model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(model),best,
                           context="Train counterfactual pilot restored Best")
