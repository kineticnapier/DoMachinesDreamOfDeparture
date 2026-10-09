from __future__ import annotations

"""Curriculum DAgger training helpers for the human-visible controller."""

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
import random
import time

import torch

from dmdod.training.dagger_continuation import (
    collect_n_key_intervention_dagger_sequence_with_continuation,
)
from dmdod.training.human_visible import freeze_human_visible_controller
from dmdod.training.n_key import (
    MSE_COEF,
    NEUTRAL_PUSH_COEF,
    NEUTRAL_PUSH_LIMIT,
    PRESS_MARGIN,
    PRESS_MARGIN_COEF,
    RELEASE_MARGIN,
    RELEASE_MARGIN_COEF,
    TEACHER_ACTIVE_THRESHOLD,
    collect_n_key_expert_sequence,
)
from dmdod.training.real_chart import (
    aggregate,
    clone_model_state,
    evaluate_role_continuous,
    safe_anchor_count,
    save_checkpoint,
    summarize,
)


BETA_SCHEDULE = (1.0, 0.75, 0.50, 0.25, 0.10, 0.0)
TRAINING_MODE = "human_visible_curriculum_dagger"
TRAINER_VERSION = "3.1.0-human-visible-curriculum-dagger"
CHECKPOINT_FORMAT_VERSION = 35

_CONTEXT_WEIGHTS = {
    "press_due": 2.0,
    "release_due": 2.0,
    "multi_press": 2.5,
    "high_speed": 2.0,
    "recovery": 3.0,
    "idle": 1.0,
}


@dataclass(frozen=True, slots=True)
class HumanVisibleTrajectory:
    anchor_id: int
    source_kind: str
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    student_actions: torch.Tensor

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(slots=True)
class HumanVisibleReplayWindow:
    anchor_id: int
    source_kind: str
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    student_actions: torch.Tensor
    initial_state: torch.Tensor
    burn_in: int
    start: int

    @property
    def supervised_frames(self) -> int:
        return int(self.observations.shape[0]) - int(self.burn_in)


@dataclass(frozen=True, slots=True)
class CurriculumEpochMetrics:
    updates: int
    windows: int
    mean_loss: float
    final_loss: float
    mean_grad_norm: float
    max_grad_norm: float
    context_loss: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def n_key_actuation_loss_per_frame(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-frame form of the existing set-valued N-key actuation objective."""

    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have matching shape [T, K]")
    key_count = int(predicted.shape[1])
    if key_count < 2 or key_count % 2 != 0:
        raise ValueError("N-key action width must be an even integer >= 2")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    available = ~release

    press_count = press.sum(dim=1, keepdim=True)
    available_count = available.sum(dim=1, keepdim=True)
    if bool((press_count > available_count).any()):
        raise ValueError("teacher requests more presses than non-held keys")

    ranked, _ = torch.sort(
        predicted.masked_fill(release, -2.0),
        dim=1,
        descending=True,
    )
    ranks = torch.arange(key_count, device=predicted.device).reshape(1, key_count)
    ranked_press = ranks < press_count
    ranked_available = ranks < available_count
    ranked_neutral = ranked_available & ~ranked_press

    desired_ranked = ranked_press.to(dtype=predicted.dtype)
    available_mse = (
        (ranked - desired_ranked).square() * ranked_available
    ).sum(dim=1)
    release_mse = ((predicted + 1.0).square() * release).sum(dim=1)
    mse = (available_mse + release_mse) / float(key_count)

    press_gap = torch.relu(PRESS_MARGIN - ranked).square()
    press_den = ranked_press.sum(dim=1).clamp_min(1)
    press_loss = (press_gap * ranked_press).sum(dim=1) / press_den

    release_gap = torch.relu(predicted - RELEASE_MARGIN).square()
    release_den = release.sum(dim=1).clamp_min(1)
    release_loss = (release_gap * release).sum(dim=1) / release_den

    unsafe_neutral_push = torch.relu(ranked - NEUTRAL_PUSH_LIMIT).square()
    neutral_den = ranked_neutral.sum(dim=1).clamp_min(1)
    neutral_loss = (
        unsafe_neutral_push * ranked_neutral
    ).sum(dim=1) / neutral_den

    return (
        MSE_COEF * mse
        + PRESS_MARGIN_COEF * press_loss
        + RELEASE_MARGIN_COEF * release_loss
        + NEUTRAL_PUSH_COEF * neutral_loss
    )


def decision_context_masks(
    teacher_actions: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    control_dt_s: float,
) -> dict[str, torch.Tensor]:
    if teacher_actions.shape != student_actions.shape:
        raise ValueError("teacher/student actions must have matching shapes")
    if teacher_actions.ndim != 2:
        raise ValueError("teacher/student actions must have shape [T, K]")

    press = teacher_actions > TEACHER_ACTIVE_THRESHOLD
    release = teacher_actions < -TEACHER_ACTIVE_THRESHOLD
    press_due = press.any(dim=1)
    release_due = release.any(dim=1)
    multi_press = press.sum(dim=1) >= 2

    high_speed = torch.zeros_like(press_due)
    # A teacher press can persist for several control frames while the motor is
    # moving. Treat only press onsets as distinct rhythm events, otherwise a
    # single sustained command would be mislabeled as a high-speed pattern.
    previous_press = torch.zeros_like(press_due)
    if press_due.numel() > 1:
        previous_press[1:] = press_due[:-1]
    press_onset = press_due & ~previous_press
    event_indices = torch.nonzero(press_onset, as_tuple=False).flatten()
    max_gap_steps = max(1, int(round(0.100 / float(control_dt_s))))
    if event_indices.numel() >= 2:
        gaps = event_indices[1:] - event_indices[:-1]
        close = gaps <= max_gap_steps
        if bool(close.any()):
            left = event_indices[:-1][close]
            right = event_indices[1:][close]
            high_speed[left] = True
            high_speed[right] = True

    teacher_press_count = press.sum(dim=1)
    student_press_count = (
        student_actions > TEACHER_ACTIVE_THRESHOLD
    ).sum(dim=1)
    release_bad = (
        release & (student_actions > RELEASE_MARGIN)
    ).any(dim=1)
    mismatch = (teacher_press_count != student_press_count) | release_bad

    recovery = torch.zeros_like(mismatch)
    recovery_steps = max(1, int(round(0.250 / float(control_dt_s))))
    for offset in range(recovery_steps + 1):
        if offset == 0:
            recovery |= mismatch
        elif offset < mismatch.shape[0]:
            recovery[offset:] |= mismatch[:-offset]

    idle = ~(press_due | release_due | recovery)
    return {
        "press_due": press_due,
        "release_due": release_due,
        "multi_press": multi_press,
        "high_speed": high_speed,
        "recovery": recovery,
        "idle": idle,
    }


def context_balanced_actuation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    control_dt_s: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_frame = n_key_actuation_loss_per_frame(predicted, target)
    masks = decision_context_masks(
        target,
        student_actions,
        control_dt_s=control_dt_s,
    )

    terms: list[torch.Tensor] = []
    weights: list[float] = []
    context_means: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            continue
        mean = per_frame[mask].mean()
        context_means[name] = mean
        terms.append(mean * float(_CONTEXT_WEIGHTS[name]))
        weights.append(float(_CONTEXT_WEIGHTS[name]))

    if not terms:
        loss = per_frame.mean()
    else:
        loss = torch.stack(terms).sum() / sum(weights)
    return loss, context_means


def build_human_visible_optimizer(model, visible_config) -> torch.optim.AdamW:
    freeze_human_visible_controller(model)
    head_params: list[torch.nn.Parameter] = []
    controller_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("controller_delta."):
            head_params.append(parameter)
        else:
            controller_params.append(parameter)

    groups = []
    if controller_params:
        groups.append(
            {
                "params": controller_params,
                "lr": float(visible_config.controller_lr),
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(visible_config.lr),
            }
        )
    if not groups:
        raise RuntimeError("human-visible optimizer has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        weight_decay=float(visible_config.weight_decay),
    )


@torch.no_grad()
def build_replay_windows(
    model,
    trajectories: list[HumanVisibleTrajectory],
    *,
    burn_in_steps: int,
    supervised_steps: int,
) -> list[HumanVisibleReplayWindow]:
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be non-negative")
    if supervised_steps <= 0:
        raise ValueError("supervised_steps must be positive")

    model.eval()
    model.prepare_recurrent_runtime()
    windows: list[HumanVisibleReplayWindow] = []

    for trajectory in trajectories:
        starts = list(range(0, trajectory.frames, supervised_steps))
        burn_starts = sorted(
            {
                max(0, int(start) - int(burn_in_steps))
                for start in starts
            }
        )
        state = model.initial_state(trajectory.observations.device)
        state_at: dict[int, torch.Tensor] = {}
        cursor = 0
        for burn_start in burn_starts:
            if burn_start > cursor:
                _, _, state = model.forward_sequence(
                    trajectory.observations[cursor:burn_start],
                    state,
                )
                state = state.detach()
                cursor = burn_start
            state_at[burn_start] = state.detach().clone()

        for start in starts:
            burn_start = max(0, start - burn_in_steps)
            end = min(trajectory.frames, start + supervised_steps)
            if end <= start:
                continue
            windows.append(
                HumanVisibleReplayWindow(
                    anchor_id=int(trajectory.anchor_id),
                    source_kind=str(trajectory.source_kind),
                    observations=trajectory.observations[burn_start:end],
                    teacher_actions=trajectory.teacher_actions[burn_start:end],
                    student_actions=trajectory.student_actions[burn_start:end],
                    initial_state=state_at[burn_start],
                    burn_in=int(start - burn_start),
                    start=int(start),
                )
            )

    if not windows:
        raise ValueError("no replay windows were built")
    return windows


def build_balanced_epoch_plan(
    windows: list[HumanVisibleReplayWindow],
    *,
    anchor_ids: list[int],
    updates_per_epoch: int,
    anchors_per_batch: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be positive")
    if anchors_per_batch <= 0:
        raise ValueError("anchors_per_batch must be positive")
    if not anchor_ids:
        raise ValueError("anchor_ids must not be empty")

    pools: dict[tuple[int, str], list[int]] = {}
    for index, window in enumerate(windows):
        pools.setdefault(
            (int(window.anchor_id), str(window.source_kind)),
            [],
        ).append(index)

    for anchor_id in anchor_ids:
        for source_kind in ("expert", "dagger"):
            if not pools.get((int(anchor_id), source_kind)):
                raise ValueError(
                    f"missing {source_kind} replay windows for anchor {anchor_id}"
                )

    rng = random.Random(int(seed))
    order = [int(anchor_id) for anchor_id in anchor_ids]
    rng.shuffle(order)
    cursor = 0
    chosen_per_update = min(int(anchors_per_batch), len(order))
    plan: list[tuple[int, ...]] = []

    for _ in range(int(updates_per_epoch)):
        chosen: list[int] = []
        for _ in range(chosen_per_update):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            chosen.append(order[cursor])
            cursor += 1

        batch: list[int] = []
        for anchor_id in chosen:
            batch.append(rng.choice(pools[(anchor_id, "expert")]))
            batch.append(rng.choice(pools[(anchor_id, "dagger")]))
        plan.append(tuple(batch))
    return tuple(plan)


def train_human_visible_epoch(
    model,
    windows: list[HumanVisibleReplayWindow],
    plan: tuple[tuple[int, ...], ...],
    *,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    control_dt_s: float,
) -> CurriculumEpochMetrics:
    if grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if not plan:
        raise ValueError("epoch plan must not be empty")

    trainable = freeze_human_visible_controller(model)
    model.train()

    losses: list[float] = []
    grad_norms: list[float] = []
    context_values: dict[str, list[float]] = {
        name: [] for name in _CONTEXT_WEIGHTS
    }
    total_windows = 0

    for batch in plan:
        if not batch:
            raise ValueError("empty batch in epoch plan")
        optimizer.zero_grad(set_to_none=True)
        batch_loss_value = 0.0

        for window_index in batch:
            window = windows[int(window_index)]
            state = window.initial_state.detach()
            burn = int(window.burn_in)

            if burn:
                with torch.no_grad():
                    _, _, state = model.forward_sequence(
                        window.observations[:burn],
                        state,
                    )
                state = state.detach()

            means, _, _ = model.forward_sequence(
                window.observations[burn:],
                state,
            )
            predicted = torch.tanh(means)
            target = window.teacher_actions[burn:]
            student = window.student_actions[burn:]
            loss, contexts = context_balanced_actuation_loss(
                predicted,
                target,
                student,
                control_dt_s=control_dt_s,
            )
            (loss / float(len(batch))).backward()
            batch_loss_value += float(loss.detach().item()) / float(len(batch))
            total_windows += 1
            for name, value in contexts.items():
                context_values[name].append(float(value.detach().item()))

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(grad_clip),
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(batch_loss_value)
        grad_norms.append(float(grad_norm.detach().item()))

    context_summary = {
        name: sum(values) / len(values)
        for name, values in context_values.items()
        if values
    }
    return CurriculumEpochMetrics(
        updates=len(plan),
        windows=total_windows,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
        mean_grad_norm=sum(grad_norms) / len(grad_norms),
        max_grad_norm=max(grad_norms),
        context_loss=context_summary,
    )


def validation_summary(results) -> dict[str, float | int | bool]:
    basic = summarize(results)
    targets = max(1, int(basic.targets))
    pp_numerator = 0.0
    mae_numerator = 0.0
    mae_weight = 0
    for stats, _ in results:
        count = int(getattr(stats, "targets", 0))
        pp_numerator += float(getattr(stats, "perfect_rate", 0.0)) * count
        mae = getattr(stats, "mean_abs_error_ms", None)
        hits = int(getattr(stats, "hits", 0))
        if mae is not None and math.isfinite(float(mae)) and hits > 0:
            mae_numerator += float(mae) * hits
            mae_weight += hits

    return {
        "safe": int(safe_anchor_count(results)),
        "anchors": int(len(results)),
        "hits": int(basic.hits),
        "targets": int(basic.targets),
        "x": float(basic.x_accuracy_percent),
        "pp": 100.0 * pp_numerator / targets,
        "mae_ms": (
            mae_numerator / mae_weight if mae_weight > 0 else float("inf")
        ),
        "early": int(basic.early),
        "overloaded": bool(basic.overloaded),
        "keydowns": int(basic.keydowns),
    }


def validation_rank(summary: dict) -> tuple[float, ...]:
    mae = float(summary["mae_ms"])
    if not math.isfinite(mae):
        mae = float("inf")
    return (
        float(summary["safe"]),
        float(summary["hits"]),
        float(summary["x"]),
        float(summary["pp"]),
        -mae,
        -float(summary["early"]),
    )


def progression_passes(current: dict, baseline: dict) -> bool:
    return (
        int(current["safe"]) >= int(baseline["safe"])
        and float(current["hits"]) >= 0.98 * float(baseline["hits"])
        and float(current["x"]) >= float(baseline["x"]) - 1.0
    )


def catastrophic_regression(
    current: dict,
    best: dict,
) -> bool:
    anchors = max(1, int(current["anchors"]))
    safe_drop = int(best["safe"]) - int(current["safe"])
    safe_threshold = max(2, int(math.ceil(0.15 * anchors)))
    if safe_drop >= safe_threshold:
        return True
    if (
        int(current["safe"]) <= int(best["safe"])
        and float(current["hits"]) < 0.70 * float(best["hits"])
    ):
        return True
    return False


def _format_validation(summary: dict) -> str:
    mae = float(summary["mae_ms"])
    mae_text = "inf" if not math.isfinite(mae) else f"{mae:.2f}ms"
    return (
        f"SAFE={summary['safe']}/{summary['anchors']} "
        f"H={summary['hits']}/{summary['targets']} "
        f"X={summary['x']:.2f}% PP={summary['pp']:.1f}% "
        f"MAE={mae_text} early={summary['early']} "
        f"over={summary['overloaded']} keydowns={summary['keydowns']}"
    )


def _checkpoint_payload(
    prepared,
    config,
    model,
    optimizer,
    *,
    epoch: int,
    level: int,
    validation_streak: int,
    catastrophic_streak: int,
    baseline_summary: dict,
    current_summary: dict,
    best_summary: dict,
    best_checkpoint_path: Path,
    history: list[dict],
    stopped_reason: str,
) -> dict:
    payload = dict(prepared.parent)
    payload.update(model.checkpoint_metadata())
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": clone_model_state(model),
            "training_mode": TRAINING_MODE,
            "training_config": config.as_dict(),
            "human_visible_training_semantics": (
                "fixed-parent+curriculum-dagger+balanced-context-v2"
            ),
            "human_visible_epoch": int(epoch),
            "human_visible_dagger_level": int(level),
            "human_visible_beta": float(BETA_SCHEDULE[level]),
            "human_visible_validation_streak": int(validation_streak),
            "human_visible_catastrophic_streak": int(catastrophic_streak),
            "human_visible_optimizer_state": copy.deepcopy(
                optimizer.state_dict()
            ),
            "human_visible_parent_validation_summary": dict(
                baseline_summary
            ),
            "human_visible_current_validation_summary": dict(
                current_summary
            ),
            "human_visible_best_validation_summary": dict(best_summary),
            "human_visible_best_checkpoint_path": str(best_checkpoint_path),
            "human_visible_history": list(history),
            "human_visible_stopped_reason": str(stopped_reason),
            "dagger_selection_uses_validation": True,
            "survival_guard_enabled": False,
            "survival_guard_semantics": None,
        }
    )
    return payload


def _collect_expert_trajectories(prepared) -> list[HumanVisibleTrajectory]:
    trajectories: list[HumanVisibleTrajectory] = []
    for anchor_id, named in enumerate(prepared.anchors):
        rollout = collect_n_key_expert_sequence(
            named.segment,
            key_count=prepared.key_count,
            lead_s=prepared.lead_s,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            source=f"curriculum-expert-{anchor_id}-{named.chart_name}",
        )
        trajectories.append(
            HumanVisibleTrajectory(
                anchor_id=anchor_id,
                source_kind="expert",
                observations=rollout.sequence.observations,
                teacher_actions=rollout.sequence.teacher_actions,
                student_actions=rollout.sequence.teacher_actions,
            )
        )
    return trajectories


def _collect_dagger_trajectories(
    prepared,
    model,
    *,
    beta: float,
    epoch: int,
    base_seed: int,
) -> tuple[list[HumanVisibleTrajectory], int, int]:
    trajectories: list[HumanVisibleTrajectory] = []
    total_frames = 0
    intervened_frames = 0
    for anchor_id, named in enumerate(prepared.anchors):
        rollout = collect_n_key_intervention_dagger_sequence_with_continuation(
            model,
            named.segment,
            lead_s=prepared.lead_s,
            beta=beta,
            seed=stable_seed(base_seed, "beta", epoch, anchor_id),
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            source=f"curriculum-dagger-e{epoch}-{anchor_id}-{named.chart_name}",
            action_mode="continuous",
            continue_after_failure=True,
        )
        trajectories.append(
            HumanVisibleTrajectory(
                anchor_id=anchor_id,
                source_kind="dagger",
                observations=rollout.sequence.observations,
                teacher_actions=rollout.sequence.teacher_actions,
                student_actions=rollout.student_actions,
            )
        )
        total_frames += int(rollout.sequence.frames)
        intervened_frames += int(rollout.interventions.sum().item())
    return trajectories, total_frames, intervened_frames


def run_human_visible_curriculum(
    prepared,
    config,
    *,
    format_duration,
) -> None:
    if not prepared.validation:
        raise SystemExit(
            "human_visible_curriculum_dagger requires at least one Validation segment"
        )

    model = prepared.model
    visible = config.human_visible
    budget = config.budget
    freeze_human_visible_controller(model)

    total_budget_s = float(budget.hours) * 3600.0
    reserve_s = float(budget.reserve_minutes) * 60.0
    train_budget_s = total_budget_s - reserve_s
    start_time = time.monotonic()

    progress_path = prepared.output_checkpoint.with_name(
        prepared.output_checkpoint.stem
        + ".progress"
        + prepared.output_checkpoint.suffix
    )

    print("=== DMDOD human-visible curriculum DAgger ===")
    print(
        f"source={prepared.source_checkpoint} "
        f"output={prepared.output_checkpoint} "
        f"keys={prepared.key_count} input={prepared.input_dim}D "
        f"device={prepared.device}"
    )
    print(
        f"anchors={len(prepared.anchors)} "
        f"validation={len(prepared.validation)} "
        f"updates/epoch={visible.updates_per_epoch} "
        f"batch={min(visible.anchors_per_batch, len(prepared.anchors))} anchors x 2 sources "
        f"burn-in={visible.burn_in_steps} supervised={visible.supervised_steps}"
    )
    print(
        f"lr controller={visible.controller_lr:g} residual={visible.lr:g} "
        f"weight-decay={visible.weight_decay:g} grad-clip={visible.grad_clip:g}"
    )
    print(
        "Safety: no single-anchor training veto; fixed Validation controls "
        "progression, checkpoint ranking, and catastrophic rollback."
    )

    optimizer = build_human_visible_optimizer(model, visible)
    resuming = str(
        prepared.parent.get("training_mode", "")
    ) == TRAINING_MODE
    if resuming and "human_visible_optimizer_state" in prepared.parent:
        optimizer.load_state_dict(
            prepared.parent["human_visible_optimizer_state"]
        )

    print("=== fixed Validation baseline ===")
    baseline_results = evaluate_role_continuous(
        model,
        prepared.validation,
        label="baseline-val",
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        device=prepared.device,
        verbose=False,
    )
    current_summary = validation_summary(baseline_results)
    stored_baseline = prepared.parent.get(
        "human_visible_parent_validation_summary"
    )
    baseline_summary = (
        dict(stored_baseline)
        if resuming and isinstance(stored_baseline, dict)
        else dict(current_summary)
    )

    epoch = int(
        prepared.parent.get("human_visible_epoch", 0)
        if resuming
        else 0
    )
    level = int(
        prepared.parent.get("human_visible_dagger_level", 0)
        if resuming
        else 0
    )
    level = min(max(level, 0), len(BETA_SCHEDULE) - 1)
    validation_streak = int(
        prepared.parent.get("human_visible_validation_streak", 0)
        if resuming
        else 0
    )
    catastrophic_streak = int(
        prepared.parent.get("human_visible_catastrophic_streak", 0)
        if resuming
        else 0
    )
    history = list(
        prepared.parent.get("human_visible_history", [])
        if resuming
        else []
    )

    best_state = clone_model_state(model)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_summary = dict(current_summary)

    best_path_value = prepared.parent.get(
        "human_visible_best_checkpoint_path"
    )
    if resuming and best_path_value:
        candidate_path = Path(str(best_path_value))
        if candidate_path.exists():
            best_checkpoint = torch.load(
                candidate_path,
                map_location=prepared.device,
                weights_only=False,
            )
            if (
                str(best_checkpoint.get("training_mode", ""))
                == TRAINING_MODE
                and "model_state" in best_checkpoint
            ):
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in best_checkpoint["model_state"].items()
                }
                if "human_visible_optimizer_state" in best_checkpoint:
                    best_optimizer_state = copy.deepcopy(
                        best_checkpoint["human_visible_optimizer_state"]
                    )
                stored_best = best_checkpoint.get(
                    "human_visible_best_validation_summary"
                )
                if isinstance(stored_best, dict):
                    best_summary = dict(stored_best)

    if not resuming:
        save_checkpoint(
            prepared.output_checkpoint,
            _checkpoint_payload(
                prepared,
                config,
                model,
                optimizer,
                epoch=epoch,
                level=level,
                validation_streak=validation_streak,
                catastrophic_streak=catastrophic_streak,
                baseline_summary=baseline_summary,
                current_summary=current_summary,
                best_summary=best_summary,
                best_checkpoint_path=prepared.output_checkpoint,
                history=history,
                stopped_reason="baseline-validated",
            ),
        )

    print("=== collecting static expert trajectories ===")
    expert_trajectories = _collect_expert_trajectories(prepared)
    expert_frames = sum(item.frames for item in expert_trajectories)
    print(f"expert-data: {expert_frames} frames")

    stopped_reason = "max-epochs"
    max_epoch_number = epoch + int(visible.max_epochs)

    while epoch < max_epoch_number:
        elapsed = time.monotonic() - start_time
        if elapsed >= train_budget_s:
            stopped_reason = "time-budget"
            break

        epoch += 1
        beta = float(BETA_SCHEDULE[level])
        print(
            f"=== epoch {epoch} beta={beta:.2f} "
            f"elapsed={format_duration(elapsed)} "
            f"remaining={format_duration(train_budget_s - elapsed)} ==="
        )

        dagger_trajectories, dagger_frames, intervention_frames = (
            _collect_dagger_trajectories(
                prepared,
                model,
                beta=beta,
                epoch=epoch,
                base_seed=visible.seed,
            )
        )
        intervention_rate = (
            intervention_frames / dagger_frames
            if dagger_frames
            else 0.0
        )
        print(
            f"collect: expert={expert_frames} dagger={dagger_frames} "
            f"intervention={intervention_rate:.3f}"
        )

        trajectories = [
            *expert_trajectories,
            *dagger_trajectories,
        ]
        windows = build_replay_windows(
            model,
            trajectories,
            burn_in_steps=visible.burn_in_steps,
            supervised_steps=visible.supervised_steps,
        )
        plan = build_balanced_epoch_plan(
            windows,
            anchor_ids=list(range(len(prepared.anchors))),
            updates_per_epoch=visible.updates_per_epoch,
            anchors_per_batch=visible.anchors_per_batch,
            seed=stable_seed(visible.seed, "plan", epoch),
        )
        metrics = train_human_visible_epoch(
            model,
            windows,
            plan,
            optimizer=optimizer,
            grad_clip=visible.grad_clip,
            control_dt_s=prepared.control_dt_s,
        )
        context_text = " ".join(
            f"{name}={value:.4f}"
            for name, value in sorted(metrics.context_loss.items())
        )
        print(
            f"train: updates={metrics.updates} windows={metrics.windows} "
            f"loss={metrics.mean_loss:.6f}->{metrics.final_loss:.6f} "
            f"grad={metrics.mean_grad_norm:.4g}/{metrics.max_grad_norm:.4g} "
            f"ctx[{context_text}]"
        )

        validation_results = evaluate_role_continuous(
            model,
            prepared.validation,
            label=f"val-e{epoch}",
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            verbose=False,
        )
        current_summary = validation_summary(validation_results)
        current_rank = validation_rank(current_summary)
        best_rank = validation_rank(best_summary)
        selected_best = current_rank > best_rank

        passed = progression_passes(
            current_summary,
            baseline_summary,
        )
        validation_streak = (
            validation_streak + 1 if passed else 0
        )

        catastrophic = (
            False
            if selected_best
            else catastrophic_regression(
                current_summary,
                best_summary,
            )
        )
        catastrophic_streak = (
            catastrophic_streak + 1 if catastrophic else 0
        )

        if selected_best:
            best_state = clone_model_state(model)
            best_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )
            best_summary = dict(current_summary)
            save_checkpoint(
                prepared.output_checkpoint,
                _checkpoint_payload(
                    prepared,
                    config,
                    model,
                    optimizer,
                    epoch=epoch,
                    level=level,
                    validation_streak=validation_streak,
                    catastrophic_streak=catastrophic_streak,
                    baseline_summary=baseline_summary,
                    current_summary=current_summary,
                    best_summary=best_summary,
                    best_checkpoint_path=prepared.output_checkpoint,
                    history=history,
                    stopped_reason="running-best",
                ),
            )

        promoted = False
        if (
            validation_streak >= int(visible.progression_streak)
            and level < len(BETA_SCHEDULE) - 1
        ):
            level += 1
            validation_streak = 0
            promoted = True

        rolled_back = False
        if catastrophic_streak >= int(
            visible.catastrophic_patience
        ):
            model.load_state_dict(best_state)
            model.prepare_recurrent_runtime()
            freeze_human_visible_controller(model)
            optimizer.load_state_dict(best_optimizer_state)
            catastrophic_streak = 0
            rolled_back = True

        history.append(
            {
                "epoch": int(epoch),
                "beta": float(beta),
                "level": int(level),
                "intervention_rate": float(intervention_rate),
                **metrics.as_dict(),
                "validation": dict(current_summary),
                "selected_best": bool(selected_best),
                "progression_pass": bool(passed),
                "promoted": bool(promoted),
                "catastrophic": bool(catastrophic),
                "rolled_back": bool(rolled_back),
            }
        )

        print(
            f"val: {_format_validation(current_summary)} "
            f"rank={'BEST' if selected_best else 'keep'} "
            f"pass={validation_streak}/{visible.progression_streak} "
            f"promote={int(promoted)} catastrophic={int(catastrophic)} "
            f"rollback={int(rolled_back)}"
        )

        continuation_summary = (
            dict(best_summary) if rolled_back else dict(current_summary)
        )
        save_checkpoint(
            progress_path,
            _checkpoint_payload(
                prepared,
                config,
                model,
                optimizer,
                epoch=epoch,
                level=level,
                validation_streak=validation_streak,
                catastrophic_streak=catastrophic_streak,
                baseline_summary=baseline_summary,
                current_summary=continuation_summary,
                best_summary=best_summary,
                best_checkpoint_path=prepared.output_checkpoint,
                history=history,
                stopped_reason="running-progress",
            ),
        )
        print(f"autosave progress: {progress_path}")

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    freeze_human_visible_controller(model)
    optimizer.load_state_dict(best_optimizer_state)
    save_checkpoint(
        prepared.output_checkpoint,
        _checkpoint_payload(
            prepared,
            config,
            model,
            optimizer,
            epoch=epoch,
            level=level,
            validation_streak=validation_streak,
            catastrophic_streak=catastrophic_streak,
            baseline_summary=baseline_summary,
            current_summary=best_summary,
            best_summary=best_summary,
            best_checkpoint_path=prepared.output_checkpoint,
            history=history,
            stopped_reason=stopped_reason,
        ),
    )

    if progress_path.exists():
        progress_path.unlink()
        print(f"removed completed progress checkpoint: {progress_path}")

    print("=== selected fixed-Validation best checkpoint ===")
    print(
        f"stop={stopped_reason} epoch={epoch} beta={BETA_SCHEDULE[level]:.2f} "
        f"{_format_validation(best_summary)}"
    )
    print(
        f"human-visible curriculum final elapsed="
        f"{format_duration(time.monotonic() - start_time)} "
        f"checkpoint={prepared.output_checkpoint}"
    )
