from __future__ import annotations

"""Read-only, same-policy beta=1/.5/0 DAgger *collection* ablation.

Collected trajectories are discarded after measurement. There are no optimizer
updates, checkpoint writes, or fixed-Validation selection calls. Initial scores
refer to executed Train-anchor mixtures, not policy-only Validation.
"""

from dataclasses import dataclass
import math
from typing import Callable

import torch

from dmdod.training.dagger_continuation import (
    NKeyInterventionRollout,
    collect_n_key_intervention_dagger_sequence_with_continuation,
)


BETAS = (1.0, 0.5, 0.0)


@dataclass(frozen=True, slots=True)
class BetaAnchorStats:
    beta: float
    anchor_index: int
    chart_name: str
    targets: int
    hits: int
    too_early: int
    keydowns: int
    overloaded: bool
    terminal_control_time_s: float
    frames: int
    pre_terminal_frames: int
    post_terminal_frames: int
    interventions: int
    # Sums use explicit denominators so differing episode lengths are comparable.
    abs_student_teacher_pre: float
    abs_student_teacher_post: float
    action_elements_pre: int
    action_elements_post: int
    pressed_key_frames_pre: float
    pressed_key_frames_post: float
    abs_velocity_pre: float
    abs_velocity_post: float

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(1, self.frames)


def summarize_rollout(
    rollout: NKeyInterventionRollout,
    *,
    beta: float,
    anchor_index: int,
    chart_name: str,
) -> BetaAnchorStats:
    actions = rollout.student_actions
    teacher = rollout.sequence.teacher_actions
    executed = rollout.executed_actions
    interventions = rollout.interventions
    observations = rollout.sequence.observations
    frames, keys = actions.shape
    first_terminal = int(rollout.terminal_frame_count)
    if (
        not 1 <= first_terminal <= frames
        or teacher.shape != actions.shape or executed.shape != actions.shape
        or interventions.shape != (frames,) or observations.shape[0] != frames
        or observations.shape[1] < keys * 3
        or rollout.terminal_control_time_s is None
        or not math.isfinite(rollout.terminal_control_time_s)
    ):
        raise RuntimeError("beta collection probe: invalid rollout or first-terminal boundary")
    if not torch.isfinite(actions).all() or not torch.isfinite(teacher).all():
        raise RuntimeError("beta collection probe: non-finite student/teacher action")

    # Check that intervention recording exactly matches the executed vectors.
    selected = torch.where(interventions[:, None], teacher, actions)
    if not torch.equal(selected, executed):
        raise RuntimeError("beta collection probe: executed actions mismatch recorded interventions")

    delta = (actions - teacher).abs()
    # Motor prefix: [positions K, velocities K, pressed flags K, ...].
    velocities = observations[:, keys:2 * keys].abs()
    pressed = observations[:, 2 * keys:3 * keys]
    if not torch.logical_or(pressed == 0.0, pressed == 1.0).all():
        raise RuntimeError("beta collection probe: invalid pressed-flag features")
    with torch.no_grad():
        def total(matrix: torch.Tensor, start: int, end: int) -> float:
            return float(matrix[start:end].sum().item())

        return BetaAnchorStats(
            beta=float(beta),
            anchor_index=int(anchor_index),
            chart_name=str(chart_name),
            targets=int(rollout.stats.targets),
            hits=int(rollout.stats.hits),
            too_early=int(rollout.stats.too_early_presses),
            keydowns=int(rollout.physical_keydowns),
            overloaded=bool(rollout.stats.overloaded),
            terminal_control_time_s=float(rollout.terminal_control_time_s),
            frames=frames,
            pre_terminal_frames=first_terminal,
            post_terminal_frames=frames - first_terminal,
            interventions=int(interventions.sum().item()),
            abs_student_teacher_pre=total(delta, 0, first_terminal),
            abs_student_teacher_post=total(delta, first_terminal, frames),
            action_elements_pre=first_terminal * keys,
            action_elements_post=(frames - first_terminal) * keys,
            pressed_key_frames_pre=total(pressed, 0, first_terminal),
            pressed_key_frames_post=total(pressed, first_terminal, frames),
            abs_velocity_pre=total(velocities, 0, first_terminal),
            abs_velocity_post=total(velocities, first_terminal, frames),
        )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def format_beta_summary(rows: tuple[BetaAnchorStats, ...]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("beta collection probe requires collected anchors")
    first = rows[0]
    if any(r.beta != first.beta for r in rows):
        raise ValueError("beta summary mixes different beta values")
    frames = sum(r.frames for r in rows)
    pre = sum(r.pre_terminal_frames for r in rows)
    post = sum(r.post_terminal_frames for r in rows)
    interventions = sum(r.interventions for r in rows)
    elements_pre = sum(r.action_elements_pre for r in rows)
    elements_post = sum(r.action_elements_post for r in rows)
    keys = first.action_elements_pre // first.pre_terminal_frames
    if any(r.action_elements_pre != r.pre_terminal_frames * keys for r in rows):
        raise ValueError("beta summary has inconsistent key counts")
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    return (
        f"=== beta={first.beta:.2f} Train-anchor collection summary ===",
        f"anchors={len(rows)} not-overloaded={sum(not r.overloaded for r in rows)}/{len(rows)} "
        f"overloads={sum(r.overloaded for r in rows)} "
        f"Hits={sum(r.hits for r in rows)}/{sum(r.targets for r in rows)} "
        f"TooEarly={sum(r.too_early for r in rows)} "
        f"Keydowns={sum(r.keydowns for r in rows)}",
        f"frames={frames} pre-terminal={pre} post-terminal={post} "
        f"teacher-executed={interventions} ({interventions / frames:.3f})",
        f"student-teacher |action|/element: "
        f"pre={fmt(_ratio(sum(r.abs_student_teacher_pre for r in rows), elements_pre))} "
        f"post={fmt(_ratio(sum(r.abs_student_teacher_post for r in rows), elements_post))} "
        f"(post reflects diagnostic continuation)",
        f"pressed keys/frame: "
        f"pre={fmt(_ratio(sum(r.pressed_key_frames_pre for r in rows), pre))} "
        f"post={fmt(_ratio(sum(r.pressed_key_frames_post for r in rows), post))} "
        f"|velocity|/key: "
        f"pre={fmt(_ratio(sum(r.abs_velocity_pre for r in rows), elements_pre))} "
        f"post={fmt(_ratio(sum(r.abs_velocity_post for r in rows), elements_post))}",
    )


def run_beta_collection_probe(
    prepared,
    model,
    *,
    seed: int,
    collect: Callable = collect_n_key_intervention_dagger_sequence_with_continuation,
) -> tuple[BetaAnchorStats, ...]:
    """Run beta variants with the same checkpoint, Train anchors, and RNG seed.

    Intervention coin flips have the same RNG seed across betas. Student RNN
    states and teacher reservation states reset independently per episode.
    """
    from dmdod.training.human_visible_curriculum import stable_seed

    rows: list[BetaAnchorStats] = []
    print("=== DMDOD read-only beta DAgger collection probe ===")
    print(
        f"Train anchors={len(prepared.anchors)} betas={BETAS} "
        f"keys={prepared.key_count} device={prepared.device} "
        f"failure-continuation=ON (not a Validation rollout)"
    )
    model.eval()
    for beta in BETAS:
        group: list[BetaAnchorStats] = []
        for index, named in enumerate(prepared.anchors, 1):
            rollout = collect(
                model, named.segment,
                lead_s=prepared.lead_s,
                beta=beta,
                seed=stable_seed(seed, "beta-collection-probe", index - 1),
                press_threshold=0.25,
                release_threshold=-0.45,
                control_dt_s=prepared.control_dt_s,
                physics_dt_s=prepared.physics_dt_s,
                device=prepared.device,
                source=f"beta-probe-{beta:.2f}-{index:02d}",
                action_mode="continuous",
                continue_after_failure=True,
            )
            row = summarize_rollout(
                rollout, beta=beta, anchor_index=index,
                chart_name=named.chart_name,
            )
            group.append(row)
            rows.append(row)
            print(
                f"beta={beta:.2f} Train #{index:02d}/{len(prepared.anchors)} "
                f"{named.chart_name}: end~{row.terminal_control_time_s:.3f}s "
                f"H={row.hits}/{row.targets} Early={row.too_early} "
                f"Overload={int(row.overloaded)} Keydowns={row.keydowns} "
                f"post-failure-frames={row.post_terminal_frames} "
                f"intervention={row.intervention_rate:.3f}"
            )
            del rollout
        for line in format_beta_summary(tuple(group)):
            print(line)
    print(
        "NOTE: summaries are first-terminal outcomes of executed Train-anchor "
        "teacher/student mixtures. They are NOT student-only Validation scores."
    )
    print(
        "No optimizer updates, training replay, checkpoint writes, "
        "or selection/rollback performed."
    )
    return tuple(rows)
