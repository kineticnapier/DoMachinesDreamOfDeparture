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
from dmdod.n_key_real_chart import NKeyOverloadTrace
from dmdod.training.real_chart import (
    aggregate,
    clone_model_state,
    evaluate_role_continuous,
    replay_policy_on_fixed_observations,
    safe_anchor_count,
    save_checkpoint,
    summarize,
)


BETA_SCHEDULE = (1.0, 0.75, 0.50, 0.25, 0.10, 0.0)
TRAINING_MODE = "human_visible_curriculum_dagger"
TRAINER_VERSION = "3.3.0-human-visible-curriculum-resume"
CHECKPOINT_FORMAT_VERSION = 37

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


@dataclass(frozen=True, slots=True)
class ValidationAnchorSnapshot:
    """Immutable values from one already-computed fixed-Validation rollout."""

    index: int
    chart_name: str
    chart_sha256: str
    start_s: float
    end_s: float
    hits: int
    early: int
    overloaded: bool
    keydowns: int

    @property
    def safe(self) -> bool:
        return not self.overloaded


def snapshot_validation_anchors(validation, results) -> tuple[ValidationAnchorSnapshot, ...]:
    """Pair existing Validation results with their ordered chart segments."""

    if len(validation) != len(results):
        raise ValueError("Validation segments/results length mismatch")
    return tuple(
        ValidationAnchorSnapshot(
            index=index,
            chart_name=str(named.chart_name),
            chart_sha256=str(named.chart_sha256),
            start_s=float(named.start_s),
            end_s=float(named.end_s),
            hits=int(stats.hits),
            early=int(stats.too_early_presses),
            overloaded=bool(stats.overloaded),
            keydowns=int(keydowns),
        )
        for index, (named, (stats, keydowns)) in enumerate(
            zip(validation, results), 1
        )
    )


def _format_anchor_delta(
    previous: ValidationAnchorSnapshot,
    current: ValidationAnchorSnapshot,
) -> str:
    return (
        f"#{previous.index:02d} {previous.chart_name!r} "
        f"[{previous.start_s:.3f}-{previous.end_s:.3f}s "
        f"sha={previous.chart_sha256[:8]}] "
        f"SAFE {int(previous.safe)}->{int(current.safe)} "
        f"Hits {previous.hits}->{current.hits} "
        f"({current.hits - previous.hits:+d}) "
        f"Early {previous.early}->{current.early} "
        f"({current.early - previous.early:+d}) "
        f"Overload {int(previous.overloaded)}->{int(current.overloaded)} "
        f"Keydowns {previous.keydowns}->{current.keydowns} "
        f"({current.keydowns - previous.keydowns:+d})"
    )


def format_catastrophic_anchor_diagnostics(
    previous_best: tuple[ValidationAnchorSnapshot, ...],
    current: tuple[ValidationAnchorSnapshot, ...],
    *,
    max_hit_drops: int = 5,
) -> tuple[str, ...]:
    """Report chart-level deterioration without performing another rollout."""

    if max_hit_drops < 0:
        raise ValueError("max_hit_drops must be non-negative")
    if len(previous_best) != len(current):
        raise ValueError("Validation anchor count changed")
    for before, after in zip(previous_best, current):
        if (
            before.index,
            before.chart_sha256,
            before.start_s,
            before.end_s,
        ) != (
            after.index,
            after.chart_sha256,
            after.start_s,
            after.end_s,
        ):
            raise ValueError("Validation anchor identity/order changed")

    lost = [
        (before, after)
        for before, after in zip(previous_best, current)
        if before.safe and not after.safe
    ]
    gained = [
        (before, after)
        for before, after in zip(previous_best, current)
        if not before.safe and after.safe
    ]
    drops = sorted(
        (
            (before, after)
            for before, after in zip(previous_best, current)
            if after.hits < before.hits
        ),
        key=lambda pair: (pair[1].hits - pair[0].hits, pair[0].index),
    )[:max_hit_drops]

    lines = ["catastrophic Validation anchors (against previous selected best):"]
    for title, pairs in (
        ("SAFE lost", lost),
        ("SAFE gained", gained),
        (f"largest Hits drops (top {max_hit_drops})", drops),
    ):
        lines.append(f"  {title}: {len(pairs)}" if pairs else f"  {title}: none")
        lines.extend(f"    {_format_anchor_delta(a, b)}" for a, b in pairs)
    return tuple(lines)



def first_keydown_count_divergence(
    best: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    window_s: float = 0.25,
    step_s: float = 0.05,
    min_count_delta: int = 3,
) -> tuple[float, int, int] | None:
    """First locally significant press-count difference; physical keys are interchangeable."""
    if window_s <= 0.0 or step_s <= 0.0 or min_count_delta <= 0:
        raise ValueError("KeyDown divergence parameters must be positive")
    if not best and not candidate:
        return None
    end = max((*best, *candidate), default=0.0)
    for step in range(int(math.ceil(end / step_s)) + 1):
        start = step * step_s
        a = sum(start <= t < start + window_s for t in best)
        b = sum(start <= t < start + window_s for t in candidate)
        if abs(a - b) >= min_count_delta:
            return start, a, b
    return None


def format_overload_trace_comparison(
    best: NKeyOverloadTrace,
    candidate: NKeyOverloadTrace,
    *,
    anchor_label: str,
    max_too_early: int = 10,
) -> tuple[str, ...]:
    """Compare termination and physical KeyDown history without another rollout."""
    if max_too_early < 0:
        raise ValueError("max_too_early must be non-negative")
    a, b = best.termination, candidate.termination
    lines = [
        f"  overload trace {anchor_label}:",
        f"    termination: best t={a.time_s:.3f}s {a.reason} -> "
        f"candidate t={b.time_s:.3f}s {b.reason} "
        f"(dt={b.time_s - a.time_s:+.3f}s)",
        f"    best: Hits={a.hits} Early={a.too_early} Keydowns={a.keydowns} "
        f"next_target={a.next_target_index} gauge={a.overload_value:.3f}",
        f"    candidate: Hits={b.hits} Early={b.too_early} "
        f"Keydowns={b.keydowns} next_target={b.next_target_index} "
        f"gauge={b.overload_value:.3f}",
    ]
    divergence = first_keydown_count_divergence(
        best.keydown_times_s, candidate.keydown_times_s
    )
    if divergence is None:
        lines.append("    key-independent KeyDown divergence: none (250ms window, delta>=3)")
    else:
        t, n_best, n_cand = divergence
        lines.append(
            f"    first key-independent KeyDown divergence: t={t:.3f}s "
            f"best={n_best} candidate={n_cand} (250ms window, delta>=3)"
        )
    for title, trace in (("best", best), ("candidate", candidate)):
        events = trace.too_early_events[-max_too_early:] if max_too_early else ()
        lines.append(
            f"    {title} TooEarly: total={len(trace.too_early_events)} "
            f"last={len(events)} times_s="
            f"[{', '.join(f'{event.time_s:.3f}' for event in events)}]"
        )
        for event in events:
            lines.append(
                f"      t={event.time_s:.3f}s key={event.key} "
                f"target_index={event.target_index} error={event.error_ms:+.2f}ms "
                f"gauge={event.overload_before:.3f}->{event.overload_after:.3f}"
                f"{' FAIL_OVERLOAD' if event.fail_overload else ''}"
            )
    if best.action_frames and candidate.action_frames:
        lines.extend(format_action_output_diagnostics(
            best, candidate, divergence=divergence
        ))
    return tuple(lines)


def format_action_output_diagnostics(
    best: NKeyOverloadTrace,
    candidate: NKeyOverloadTrace,
    *,
    divergence: tuple[float, int, int] | None,
    max_frames_per_window: int = 5,
) -> tuple[str, ...]:
    """Compare policy commands and resulting motor states, with no extra rollouts."""
    if max_frames_per_window <= 0:
        raise ValueError("max_frames_per_window must be positive")
    # Legacy/minimal traces do not carry per-control-step action frames.
    # An empty trace is not a failed alignment; there is nothing to compare.
    if not best.action_frames or not candidate.action_frames:
        return ()
    windows: list[tuple[str, float, float]] = []
    if divergence is not None:
        start = divergence[0]
        windows.append(("first KeyDown-count divergence", start, start + 0.25))
    if candidate.termination.reason == "Overload":
        end = candidate.termination.time_s
        windows.append(("candidate FailOverload lead-up", max(0.0, end - 0.10), end))
    if not windows:
        return ()

    from dmdod.n_key_motor import n_key_names

    lines: list[str] = []
    for label, start, end in windows:
        # Control time and input frequency must match; compare identical steps
        # rather than retiming one trajectory to an unrelated motor state.
        pairs = [
            (i, best.action_frames[i], b)
            for i, b in enumerate(candidate.action_frames)
            if i < len(best.action_frames)
            and start - 1e-9 <= b.time_s <= end + 1e-9
            and abs(best.action_frames[i].time_s - b.time_s) <= 0.001
            and len(best.action_frames[i].action_values) == len(b.action_values)
        ]
        if not pairs:
            lines.append(
                f"    action window {label} [{start:.3f}, {end:.3f}]s: no aligned frames"
            )
            continue
        priority = sorted(
            pairs,
            key=lambda p: (
                -max(abs(x - y) for x, y in zip(p[1].action_values, p[2].action_values)),
                p[0],
            ),
        )
        selected = set()
        # Keep the beginning and ending frames as well as the largest changes.
        if max_frames_per_window > 1:
            selected.update((pairs[0][0], pairs[-1][0]))
        selected.update(p[0] for p in priority[:max(0, max_frames_per_window - len(selected))])
        sampled = [p for p in pairs if p[0] in selected]
        lines.append(
            f"    action window {label} [{start:.3f}, {end:.3f}]s: "
            f"{len(sampled)}/{len(pairs)} frames (command at start, motor at end)"
        )
        for _, a, b in sampled:
            keys = n_key_names(len(a.action_values))
            diffs = [abs(x - y) for x, y in zip(a.action_values, b.action_values)]
            active = set(sorted(range(len(keys)), key=lambda i: (-diffs[i], i))[:2])
            active.update(i for i, d in enumerate(diffs) if d >= 0.20)
            active.update(i for i in range(len(keys)) if a.pressed_flags[i] != b.pressed_flags[i])
            for tr in (best, candidate):
                for e in tr.physical_keydowns:
                    if a.time_s - 1e-9 <= e.time_s < a.time_s + 0.010001 and e.key in keys:
                        active.add(keys.index(e.key))
            def target(f):
                if f.next_target_index is None:
                    return "none"
                return f"#{f.next_target_index}@{f.next_target_time_s:.3f}s"
            lines.append(
                f"      t={a.time_s:.3f}s next-target best={target(a)} "
                f"candidate={target(b)} max-abs-action-delta={max(diffs):.3f}"
            )
            for i in sorted(active):
                lines.append(
                    f"        {keys[i]} tanh(mu)={a.action_values[i]:+.3f}/{b.action_values[i]:+.3f} "
                    f"position={a.positions_m[i]*1000:.2f}/{b.positions_m[i]*1000:.2f}mm "
                    f"velocity={a.velocities_m_s[i]*1000:.1f}/{b.velocities_m_s[i]*1000:.1f}mm/s "
                    f"pressed={int(a.pressed_flags[i])}/{int(b.pressed_flags[i])}"
                )
        for name, tr in (("best", best), ("candidate", candidate)):
            downs = [
                e for e in tr.physical_keydowns
                if start - 1e-9 <= e.time_s <= end + 1e-9
            ]
            if downs:
                lines.append(f"      {name} physical KeyDowns in window: {len(downs)}")
                for e in downs[-12:]:
                    next_target = (
                        f"#{e.target_index}@{e.target_time_s:.3f}s"
                        if e.target_index is not None else "none"
                    )
                    lines.append(
                        f"        t={e.time_s:.3f}s key={e.key} next-target={next_target}"
                    )
    return tuple(lines)



def format_open_loop_action_comparison(
    best_trace: NKeyOverloadTrace,
    candidate_open_loop: torch.Tensor,
    *,
    anchor_label: str,
    candidate_termination_s: float,
    divergence: tuple[float, int, int] | None = None,
    max_frames_per_window: int = 5,
) -> tuple[str, ...]:
    """Quantify parameter/RNN differences under the best policy's fixed observations.

    The best action values are from the original best rollout, and the
    candidate values are from a new inference pass on those exact observations.
    No motor or game event is simulated in this counterfactual.
    """
    if max_frames_per_window <= 0:
        raise ValueError("max_frames_per_window must be positive")
    if candidate_open_loop.ndim != 2:
        raise ValueError("candidate open-loop actions must be 2D")
    n = int(candidate_open_loop.shape[0])
    if not best_trace.action_frames or n == 0:
        return (f"  open-loop {anchor_label}: no fixed-observation frames",)
    if n > len(best_trace.action_frames):
        raise ValueError("open-loop actions longer than fixed observations")
    width = len(best_trace.action_frames[0].action_values)
    if int(candidate_open_loop.shape[1]) != width:
        raise ValueError("open-loop key count mismatch")

    reference = torch.tensor(
        [frame.action_values for frame in best_trace.action_frames[:n]],
        dtype=torch.float32,
    )
    deltas = (reference - candidate_open_loop.cpu()).abs()
    per_step = deltas.max(dim=1).values
    all_values = deltas.reshape(-1)
    threshold_count = int((per_step >= 0.25).sum().item())
    first_large = torch.nonzero(per_step >= 0.25).flatten()
    first_t = (
        f"{best_trace.action_frames[int(first_large[0])].time_s:.3f}s"
        if first_large.numel() else "none"
    )
    lines = [
        f"  open-loop {anchor_label}: fixed best observations, candidate model "
        f"with independent RNN state; frames={n}",
        f"    mean|action delta|={all_values.mean().item():.4f} "
        f"p95={torch.quantile(all_values, 0.95).item():.4f} "
        f"max={all_values.max().item():.4f} "
        f"frames(max delta>=0.25)={threshold_count}/{n} first={first_t}",
    ]
    windows: list[tuple[str, float, float]] = []
    if divergence is not None:
        windows.append(("first KeyDown-count divergence", divergence[0], divergence[0] + 0.25))
    windows.append((
        "candidate termination lead-up",
        max(0.0, candidate_termination_s - 0.10),
        candidate_termination_s,
    ))
    from dmdod.n_key_motor import n_key_names
    keys = n_key_names(width)
    for label, start, end in windows:
        indices = [
            i for i, frame in enumerate(best_trace.action_frames[:n])
            if start - 1e-9 <= frame.time_s <= end + 1e-9
        ]
        if not indices:
            lines.append(f"    fixed-input window {label}: no frames")
            continue
        highest = sorted(indices, key=lambda i: (-float(per_step[i]), i))
        selected: set[int] = set()
        if max_frames_per_window > 1:
            selected.update((indices[0], indices[-1]))
        selected.update(highest[:max(0, max_frames_per_window - len(selected))])
        lines.append(
            f"    fixed-input window {label} [{start:.3f}, {end:.3f}]s "
            f"frames={len(selected)}/{len(indices)}"
        )
        for i in sorted(selected):
            pairs = sorted(
                ((j, float(deltas[i, j])) for j in range(width)),
                key=lambda pair: (-pair[1], pair[0]),
            )[:3]
            detail = " ".join(
                f"{keys[j]}={float(reference[i, j]):+.3f}/{float(candidate_open_loop[i, j]):+.3f}"
                f"(d={delta:.3f})"
                for j, delta in pairs
            )
            lines.append(
                f"      t={best_trace.action_frames[i].time_s:.3f}s "
                f"max_delta={float(per_step[i]):.3f} {detail}"
            )
    lines.append(
        "    NOTE: fixed-observation replay has no physics/score/Overload outcome; "
        "it isolates policy inference from closed-loop observation divergence."
    )
    return tuple(lines)

def catastrophic_trace_anchor_indices(
    previous: tuple[ValidationAnchorSnapshot, ...],
    current: tuple[ValidationAnchorSnapshot, ...],
) -> tuple[int, ...]:
    """Detail every SAFE loss and the five largest Hits drops (0-based offsets)."""
    if len(previous) != len(current):
        raise ValueError("Validation trace anchor count changed")
    lost = {
        i for i, (a, b) in enumerate(zip(previous, current))
        if a.safe and not b.safe
    }
    drops = sorted(
        (i for i, (a, b) in enumerate(zip(previous, current)) if b.hits < a.hits),
        key=lambda i: (current[i].hits - previous[i].hits, i),
    )[:5]
    return tuple(sorted(lost | set(drops)))


def select_best_trace_reference(
    previous: tuple[NKeyOverloadTrace, ...],
    candidate: tuple[NKeyOverloadTrace, ...],
    *,
    selected_best: bool,
) -> tuple[NKeyOverloadTrace, ...]:
    return candidate if selected_best else previous


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


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    level: int
    validation_streak: int
    catastrophic_streak: int
    progression_pass: bool
    promoted: bool
    rolled_back: bool


def update_curriculum_state(
    *,
    level: int,
    passed: bool,
    catastrophic: bool,
    validation_streak: int,
    catastrophic_streak: int,
    progression_streak: int,
    catastrophic_patience: int,
) -> CurriculumDecision:
    """Advance curriculum counters without letting failures earn promotion."""

    if progression_streak <= 0:
        raise ValueError("progression_streak must be positive")
    if catastrophic_patience <= 0:
        raise ValueError("catastrophic_patience must be positive")

    progression_pass = bool(passed and not catastrophic)
    validation_streak = (
        int(validation_streak) + 1 if progression_pass else 0
    )
    catastrophic_streak = (
        int(catastrophic_streak) + 1 if catastrophic else 0
    )

    rolled_back = catastrophic_streak >= int(catastrophic_patience)
    if rolled_back:
        # A rollback restores a previous policy/optimizer state. Any progression
        # evidence accumulated by the discarded continuation is invalid too.
        validation_streak = 0
        catastrophic_streak = 0

    promoted = False
    if (
        not rolled_back
        and validation_streak >= int(progression_streak)
        and int(level) < len(BETA_SCHEDULE) - 1
    ):
        level = int(level) + 1
        validation_streak = 0
        promoted = True

    return CurriculumDecision(
        level=int(level),
        validation_streak=int(validation_streak),
        catastrophic_streak=int(catastrophic_streak),
        progression_pass=progression_pass,
        promoted=promoted,
        rolled_back=rolled_back,
    )


def effective_updates_per_epoch(
    *,
    level: int,
    transition_epochs_remaining: int,
    updates_per_epoch: int,
    transition_updates_per_epoch: int,
) -> int:
    """Use a smaller optimizer budget immediately after a beta transition."""

    if int(level) > 0 and int(transition_epochs_remaining) > 0:
        return int(transition_updates_per_epoch)
    return int(updates_per_epoch)


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



# Resume checks are deliberately restricted to this trainer. They do not
# reinterpret generic/legacy action-trust checkpoint roles.
def curriculum_checkpoint_role(checkpoint: dict) -> str | None:
    if str(checkpoint.get("training_mode", "")) != TRAINING_MODE:
        return None
    legacy = {
        "running-best": "selected-best",
        "baseline-validated": "selected-best",
        "running-progress": "continuation-progress",
    }.get(str(checkpoint.get("human_visible_stopped_reason", "")))
    explicit = checkpoint.get("human_visible_checkpoint_role")
    if explicit is None:
        if legacy is None:
            raise SystemExit("curriculum checkpoint role cannot be determined safely")
        return legacy
    if explicit not in {"selected-best", "continuation-progress"}:
        raise SystemExit(f"unsupported curriculum checkpoint role: {explicit!r}")
    if legacy is not None and explicit != legacy:
        raise SystemExit("curriculum checkpoint role conflicts with stopped_reason")
    return str(explicit)


def _checkpoint_best_identity(checkpoint: dict) -> tuple[int, float, int]:
    if curriculum_checkpoint_role(checkpoint) != "selected-best":
        raise SystemExit("expected a selected-best curriculum checkpoint")
    epoch = int(checkpoint.get("human_visible_best_epoch", checkpoint.get("human_visible_epoch", -1)))
    if epoch < 0:
        raise SystemExit("selected-best checkpoint has no valid best epoch")
    beta = checkpoint.get("human_visible_best_beta")
    if beta is None:
        matches = [
            item for item in checkpoint.get("human_visible_history", [])
            if isinstance(item, dict)
            and int(item.get("epoch", -1)) == epoch
            and bool(item.get("selected_best", False))
            and "beta" in item
        ]
        if len(matches) != 1:
            raise SystemExit("cannot infer old selected-best beta from unique best history entry")
        beta = matches[0]["beta"]
    beta = float(beta)
    indices = [i for i, scheduled in enumerate(BETA_SCHEDULE)
               if math.isclose(beta, scheduled, rel_tol=0, abs_tol=1e-8)]
    if len(indices) != 1:
        raise SystemExit(f"best beta is not in the curriculum schedule: {beta}")
    if int(checkpoint.get("human_visible_epoch", epoch)) != epoch:
        raise SystemExit("selected-best checkpoint epoch does not match its best epoch")
    return epoch, float(BETA_SCHEDULE[indices[0]]), indices[0]


def _validation_identity(prepared) -> list[dict]:
    return [
        {
            "chart_sha256": str(item.chart_sha256),
            "start_s": float(item.start_s),
            "end_s": float(item.end_s),
        }
        for item in prepared.validation
    ]


def _check_validation_context(checkpoint: dict, config, prepared) -> None:
    previous = checkpoint.get("training_config")
    if not isinstance(previous, dict):
        raise SystemExit("resume requires saved training_config")
    prior_run = previous.get("run", {})
    prior_data = previous.get("data", {})
    if not isinstance(prior_run, dict) or not isinstance(prior_data, dict):
        raise SystemExit("resume training_config has invalid run/data sections")
    if Path(str(prior_run.get("dataset", ""))).resolve() != Path(config.run.dataset).resolve():
        raise SystemExit("resume dataset path differs from saved training_config")
    for name in ("anchor_limit", "validation_limit"):
        old = prior_data.get(name)
        new = getattr(config.data, name)
        if old != new:
            raise SystemExit(f"resume {name} differs from saved training_config: {old} != {new}")
    if "human_visible_validation_identity" in checkpoint:
        previous_items = checkpoint["human_visible_validation_identity"]
        current_items = _validation_identity(prepared)
        if len(previous_items) != len(current_items):
            raise SystemExit("fixed Validation segment count changed")
        for old, new in zip(previous_items, current_items):
            if old["chart_sha256"] != new["chart_sha256"]:
                raise SystemExit("fixed Validation chart SHA changed")
            for name in ("start_s", "end_s"):
                if not math.isclose(float(old[name]), new[name], abs_tol=1e-8, rel_tol=0):
                    raise SystemExit(f"fixed Validation {name} changed")


def _check_validation_summary(actual: dict, recorded: dict) -> None:
    if not isinstance(recorded, dict):
        raise SystemExit("selected-best checkpoint has no saved Validation summary")
    for name in ("anchors", "targets", "safe", "hits"):
        if int(actual[name]) != int(recorded[name]):
            raise SystemExit(f"fixed Validation mismatch: {name}: {actual[name]} != {recorded[name]}")
    # Stored checkpoint stats are already rounded by the simulator pipeline.
    for name, tolerance in (("x", 0.05), ("pp", 0.05), ("mae_ms", 0.25)):
        a, b = float(actual[name]), float(recorded[name])
        if not (a == b or math.isclose(a, b, rel_tol=0, abs_tol=tolerance)):
            raise SystemExit(f"fixed Validation mismatch: {name}: {a} != {b}")


def _check_optimizer_state(optimizer: torch.optim.Optimizer, checkpoint: dict, model) -> dict:
    saved = checkpoint.get("human_visible_optimizer_state")
    if not isinstance(saved, dict) or not isinstance(saved.get("param_groups"), list):
        raise SystemExit("resume checkpoint is missing valid AdamW optimizer state")
    groups = saved["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise SystemExit("resume AdamW parameter-group count mismatch")
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    current_names = [
        [parameter_names[id(p)] for p in group["params"]]
        for group in optimizer.param_groups
    ]
    previous_names = checkpoint.get("human_visible_optimizer_parameter_names")
    if previous_names is not None and previous_names != current_names:
        raise SystemExit("resume AdamW parameter names/order mismatch")
    saved_state = saved.get("state")
    if not isinstance(saved_state, dict) or not saved_state:
        raise SystemExit("resume AdamW has no optimizer moments")
    expected_ids = {identifier for group in groups for identifier in group.get("params", [])}
    if not set(saved_state).issubset(expected_ids):
        raise SystemExit("resume AdamW state refers to an unknown parameter")
    for i, (old, current) in enumerate(zip(groups, optimizer.param_groups)):
        old_ids, current_params = old.get("params", []), current["params"]
        if len(old_ids) != len(current_params) or not old_ids:
            raise SystemExit(f"resume AdamW parameter-group {i} shape mismatch")
        if len(set(old_ids)) != len(old_ids):
            raise SystemExit(f"resume AdamW parameter-group {i} has duplicate parameter IDs")
        for field in ("lr", "weight_decay", "betas", "eps", "amsgrad"):
            if field in old and old[field] != current[field]:
                raise SystemExit(f"resume AdamW group {i} {field} mismatch")
        for identifier, parameter in zip(old_ids, current_params):
            state = saved_state.get(identifier, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state and tuple(state[name].shape) != tuple(parameter.shape):
                    raise SystemExit(f"resume AdamW {name} tensor shape mismatch")
    return saved


def _load_checked_optimizer(optimizer, checkpoint: dict, model) -> None:
    state = _check_optimizer_state(optimizer, checkpoint, model)
    try:
        optimizer.load_state_dict(copy.deepcopy(state))
    except (ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"resume AdamW state load failed: {exc}") from exc


@dataclass(slots=True)
class CurriculumStart:
    role: str | None
    epoch: int
    level: int
    validation_streak: int
    catastrophic_streak: int
    transition_epochs_remaining: int
    history: list[dict]
    baseline_summary: dict | None
    best_summary: dict | None
    best_state: dict[str, torch.Tensor]
    best_optimizer_state: dict
    best_epoch: int
    best_beta: float
    seed_output: bool


def prepare_curriculum_start(prepared, config, model, optimizer) -> CurriculumStart:
    """Validate source/paths and restore optimizer before writing anything."""
    source = prepared.source_checkpoint.resolve()
    output = prepared.output_checkpoint.resolve()
    progress_path = prepared.output_checkpoint.with_name(
        prepared.output_checkpoint.stem + ".progress" + prepared.output_checkpoint.suffix
    )
    if source == output:
        raise SystemExit("source checkpoint and output must differ")
    parent = prepared.parent
    role = curriculum_checkpoint_role(parent)
    best_checkpoint = parent
    seed_output = False
    if role == "selected-best":
        if prepared.output_checkpoint.exists():
            raise SystemExit("resume output already exists; refusing to overwrite initial best")
        if progress_path.exists():
            raise SystemExit(f"resume progress exists: {progress_path}; resume from that progress instead")
        _check_validation_context(parent, config, prepared)
        best_epoch, best_beta, level = _checkpoint_best_identity(parent)
        epoch = best_epoch
        _load_checked_optimizer(optimizer, parent, model)
        streak, catastrophic_streak = 0, 0
        transition_remaining = int(config.human_visible.transition_epochs) if level > 0 else 0
        seed_output = True
    elif role == "continuation-progress":
        if source != progress_path.resolve():
            raise SystemExit(f"progress source must be the matching progress file: {progress_path}")
        if not prepared.output_checkpoint.is_file():
            raise SystemExit(f"referenced selected-best output does not exist: {prepared.output_checkpoint}")
        path_value = parent.get("human_visible_best_checkpoint_path")
        if not path_value or Path(str(path_value)).resolve() != output:
            raise SystemExit("progress best_checkpoint_path does not match output")
        best_checkpoint = torch.load(prepared.output_checkpoint, map_location="cpu", weights_only=False)
        if curriculum_checkpoint_role(best_checkpoint) != "selected-best":
            raise SystemExit("progress referenced file is not selected-best")
        _check_validation_context(parent, config, prepared)
        _check_validation_context(best_checkpoint, config, prepared)
        if parent.get("human_visible_parent_validation_summary") != best_checkpoint.get("human_visible_parent_validation_summary"):
            raise SystemExit("progress and best have different fixed parent Validation baselines")
        best_epoch, best_beta, _ = _checkpoint_best_identity(best_checkpoint)
        if parent.get("human_visible_best_validation_summary") != best_checkpoint.get("human_visible_best_validation_summary"):
            raise SystemExit("progress and selected-best Validation rankings disagree")
        _check_optimizer_state(optimizer, best_checkpoint, model)
        _load_checked_optimizer(optimizer, parent, model)
        epoch = int(parent["human_visible_epoch"])
        level = int(parent["human_visible_dagger_level"])
        if not 0 <= level < len(BETA_SCHEDULE):
            raise SystemExit("progress curriculum level is invalid")
        streak = int(parent["human_visible_validation_streak"])
        catastrophic_streak = int(parent["human_visible_catastrophic_streak"])
        transition_remaining = int(parent.get("human_visible_transition_epochs_remaining", 0))
        if min(epoch, streak, catastrophic_streak, transition_remaining) < 0:
            raise SystemExit("progress has negative curriculum state")
        if parent.get("human_visible_best_epoch") is not None and int(parent["human_visible_best_epoch"]) != best_epoch:
            raise SystemExit("progress and selected-best epochs disagree")
    else:
        epoch, level, streak, catastrophic_streak, transition_remaining = 0, 0, 0, 0, 0
        best_epoch, best_beta = 0, float(BETA_SCHEDULE[0])
        if progress_path.exists():
            raise SystemExit(f"progress exists: {progress_path}; refusing to overwrite it")

    best_summary = (
        best_checkpoint.get("human_visible_best_validation_summary") if role else None
    )
    baseline_summary = (
        parent.get("human_visible_parent_validation_summary") if role else None
    )
    if role and (not isinstance(best_summary, dict) or not isinstance(baseline_summary, dict)):
        raise SystemExit("resume checkpoint is missing baseline/best Validation summaries")
    return CurriculumStart(
        role=role,
        epoch=epoch,
        level=level,
        validation_streak=streak,
        catastrophic_streak=catastrophic_streak,
        transition_epochs_remaining=transition_remaining,
        history=list(parent.get("human_visible_history", []) if role else []),
        baseline_summary=dict(baseline_summary) if baseline_summary is not None else None,
        best_summary=dict(best_summary) if best_summary is not None else None,
        best_state={name: tensor.detach().cpu().clone() for name, tensor in best_checkpoint["model_state"].items()} if role else clone_model_state(model),
        best_optimizer_state=copy.deepcopy(best_checkpoint["human_visible_optimizer_state"] if role else optimizer.state_dict()),
        best_epoch=best_epoch,
        best_beta=best_beta,
        seed_output=seed_output,
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
    best_epoch: int,
    best_beta: float,
    transition_epochs_remaining: int,
    history: list[dict],
    stopped_reason: str,
    checkpoint_role: str,
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
            "human_visible_checkpoint_role": str(checkpoint_role),
            "human_visible_validation_identity": _validation_identity(prepared),
            "human_visible_optimizer_parameter_names": [
                [next(name for name, p in model.named_parameters() if p is param)
                 for param in group["params"]]
                for group in optimizer.param_groups
            ],
            "human_visible_training_semantics": (
                "fixed-parent+curriculum-dagger+balanced-context-v3"
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
            "human_visible_best_epoch": int(best_epoch),
            "human_visible_best_beta": float(best_beta),
            "human_visible_transition_epochs_remaining": int(
                transition_epochs_remaining
            ),
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
        f"transition-updates={visible.transition_updates_per_epoch}"
        f"x{visible.transition_epochs} "
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
    start = prepare_curriculum_start(prepared, config, model, optimizer)
    epoch = start.epoch
    level = start.level
    validation_streak = start.validation_streak
    catastrophic_streak = start.catastrophic_streak
    transition_epochs_remaining = start.transition_epochs_remaining
    history = start.history
    best_state = start.best_state
    best_optimizer_state = start.best_optimizer_state
    best_epoch = start.best_epoch
    best_beta = start.best_beta

    print("=== fixed Validation baseline / best verification ===")
    best_overload_traces: list[NKeyOverloadTrace] = []
    best_fixed_observations: list[torch.Tensor] = []
    # A progress model may not be the best model. Evaluate the referenced best
    # separately, then restore the exact continuation model and optimizer.
    if start.role == "continuation-progress":
        continuation_state = clone_model_state(model)
        model.load_state_dict(best_state)
        model.prepare_recurrent_runtime()
    try:
        baseline_results = evaluate_role_continuous(
            model,
            prepared.validation,
            label="resume-best-val" if start.role else "baseline-val",
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            verbose=False,
            trace_collector=best_overload_traces,
            observation_collector=best_fixed_observations,
        )
        evaluated_best_summary = validation_summary(baseline_results)
    finally:
        if start.role == "continuation-progress":
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            freeze_human_visible_controller(model)

    best_validation_anchors = snapshot_validation_anchors(
        prepared.validation, baseline_results
    )

    if start.role:
        _check_validation_summary(evaluated_best_summary, start.best_summary)
        best_summary = dict(start.best_summary)
        baseline_summary = dict(start.baseline_summary)
        print(f"verified stored best: {_format_validation(best_summary)}")
    else:
        baseline_summary = dict(evaluated_best_summary)
        best_summary = dict(evaluated_best_summary)
    current_summary = (
        dict(prepared.parent["human_visible_current_validation_summary"])
        if start.role == "continuation-progress"
        else dict(evaluated_best_summary)
    )

    # Saving a selected-best source to a new name must precede training, so a
    # run with zero subsequent improvements still has a usable best checkpoint.
    if start.role != "continuation-progress":
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
                best_epoch=best_epoch,
                best_beta=best_beta,
                transition_epochs_remaining=transition_epochs_remaining,
                history=history,
                checkpoint_role="selected-best",
                stopped_reason="baseline-validated",
            ),
        )
        print(f"autosave initial best: {prepared.output_checkpoint}")

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
        updates_this_epoch = effective_updates_per_epoch(
            level=level,
            transition_epochs_remaining=transition_epochs_remaining,
            updates_per_epoch=visible.updates_per_epoch,
            transition_updates_per_epoch=visible.transition_updates_per_epoch,
        )
        plan = build_balanced_epoch_plan(
            windows,
            anchor_ids=list(range(len(prepared.anchors))),
            updates_per_epoch=updates_this_epoch,
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

        candidate_overload_traces: list[NKeyOverloadTrace] = []
        candidate_fixed_observations: list[torch.Tensor] = []
        validation_results = evaluate_role_continuous(
            model,
            prepared.validation,
            label=f"val-e{epoch}",
            control_dt_s=prepared.control_dt_s,
            physics_dt_s=prepared.physics_dt_s,
            device=prepared.device,
            verbose=False,
            trace_collector=candidate_overload_traces,
            observation_collector=candidate_fixed_observations,
        )
        current_summary = validation_summary(validation_results)
        current_rank = validation_rank(current_summary)
        best_rank = validation_rank(best_summary)
        selected_best = current_rank > best_rank

        passed = progression_passes(
            current_summary,
            baseline_summary,
        )

        catastrophic = (
            False
            if selected_best
            else catastrophic_regression(
                current_summary,
                best_summary,
            )
        )

        if catastrophic:
            current_anchors = snapshot_validation_anchors(
                prepared.validation, validation_results
            )
            print(f"=== val-e{epoch} catastrophic vs best-e{best_epoch} ===")
            for line in format_catastrophic_anchor_diagnostics(
                best_validation_anchors, current_anchors
            ):
                print(line)
            if (
                len(best_overload_traces) != len(prepared.validation)
                or len(candidate_overload_traces) != len(prepared.validation)
            ):
                raise RuntimeError("Validation trace collection count mismatch")
            for index in catastrophic_trace_anchor_indices(
                best_validation_anchors, current_anchors
            ):
                label = f"#{index + 1:02d} {prepared.validation[index].chart_name}"
                for line in format_overload_trace_comparison(
                    best_overload_traces[index],
                    candidate_overload_traces[index],
                    anchor_label=label,
                ):
                    print(line)
                # The candidate is replayed on the already-recorded BEST
                # observations. No additional environment rollout occurs.
                reference = best_overload_traces[index]
                candidate_end = candidate_overload_traces[index].termination.time_s
                stop_steps = sum(
                    frame.time_s <= candidate_end + 1e-9
                    for frame in reference.action_frames
                )
                if len(best_fixed_observations) != len(prepared.validation):
                    raise RuntimeError("best fixed observation count mismatch")
                fixed_actions = replay_policy_on_fixed_observations(
                    model,
                    best_fixed_observations[index],
                    device=prepared.device,
                    max_steps=stop_steps,
                )
                divergence = first_keydown_count_divergence(
                    reference.keydown_times_s,
                    candidate_overload_traces[index].keydown_times_s,
                )
                for line in format_open_loop_action_comparison(
                    reference,
                    fixed_actions,
                    anchor_label=label,
                    candidate_termination_s=candidate_end,
                    divergence=divergence,
                ):
                    print(line)

        if selected_best:
            best_fixed_observations = candidate_fixed_observations
            best_overload_traces = list(select_best_trace_reference(
                tuple(best_overload_traces),
                tuple(candidate_overload_traces),
                selected_best=True,
            ))
            best_validation_anchors = snapshot_validation_anchors(
                prepared.validation, validation_results
            )
            best_state = clone_model_state(model)
            best_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )
            best_summary = dict(current_summary)
            best_epoch = int(epoch)
            best_beta = float(beta)

        decision = update_curriculum_state(
            level=level,
            passed=passed,
            catastrophic=catastrophic,
            validation_streak=validation_streak,
            catastrophic_streak=catastrophic_streak,
            progression_streak=visible.progression_streak,
            catastrophic_patience=visible.catastrophic_patience,
        )
        level = decision.level
        validation_streak = decision.validation_streak
        catastrophic_streak = decision.catastrophic_streak
        passed = decision.progression_pass
        promoted = decision.promoted
        rolled_back = decision.rolled_back

        if rolled_back:
            model.load_state_dict(best_state)
            model.prepare_recurrent_runtime()
            freeze_human_visible_controller(model)
            optimizer.load_state_dict(best_optimizer_state)

        if promoted or rolled_back:
            transition_epochs_remaining = (
                int(visible.transition_epochs) if level > 0 else 0
            )
        elif transition_epochs_remaining > 0:
            transition_epochs_remaining -= 1

        history.append(
            {
                "epoch": int(epoch),
                "beta": float(beta),
                "level": int(level),
                "intervention_rate": float(intervention_rate),
                "updates_this_epoch": int(updates_this_epoch),
                "transition_epochs_remaining": int(
                    transition_epochs_remaining
                ),
                **metrics.as_dict(),
                "validation": dict(current_summary),
                "selected_best": bool(selected_best),
                "progression_pass": bool(passed),
                "promoted": bool(promoted),
                "catastrophic": bool(catastrophic),
                "rolled_back": bool(rolled_back),
            }
        )

        if selected_best:
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
                    best_epoch=best_epoch,
                    best_beta=best_beta,
                    transition_epochs_remaining=transition_epochs_remaining,
                    history=history,
                    checkpoint_role="selected-best",
                    stopped_reason="running-best",
                ),
            )
            print(f"autosave best: {prepared.output_checkpoint}")

        print(
            f"val: {_format_validation(current_summary)} "
            f"rank={'BEST' if selected_best else 'keep'} "
            f"pass={validation_streak}/{visible.progression_streak} "
            f"promote={int(promoted)} catastrophic={int(catastrophic)} "
            f"rollback={int(rolled_back)} "
            f"next-updates={effective_updates_per_epoch(level=level, transition_epochs_remaining=transition_epochs_remaining, updates_per_epoch=visible.updates_per_epoch, transition_updates_per_epoch=visible.transition_updates_per_epoch)}"
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
                best_epoch=best_epoch,
                best_beta=best_beta,
                transition_epochs_remaining=transition_epochs_remaining,
                history=history,
                checkpoint_role="continuation-progress",
                stopped_reason="running-progress",
            ),
        )
        print(f"autosave progress: {progress_path}")

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    freeze_human_visible_controller(model)
    optimizer.load_state_dict(best_optimizer_state)
    if progress_path.exists():
        progress_path.unlink()
        print(f"removed completed progress checkpoint: {progress_path}")

    print("=== selected fixed-Validation best checkpoint ===")
    print(
        f"stop={stopped_reason} final-epoch={epoch} "
        f"best-epoch={best_epoch} best-beta={best_beta:.2f} "
        f"{_format_validation(best_summary)}"
    )
    print(
        f"human-visible curriculum final elapsed="
        f"{format_duration(time.monotonic() - start_time)} "
        f"checkpoint={prepared.output_checkpoint}"
    )
