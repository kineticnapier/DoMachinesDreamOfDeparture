from __future__ import annotations

import torch

from dmdod.training.human_visible_curriculum import (
    HumanVisibleReplayWindow,
    build_balanced_epoch_plan,
    catastrophic_regression,
    context_balanced_actuation_loss,
    effective_updates_per_epoch,
    n_key_actuation_loss_per_frame,
    progression_passes,
    update_curriculum_state,
)


def _window(anchor_id: int, source_kind: str, start: int) -> HumanVisibleReplayWindow:
    observations = torch.zeros(8, 10)
    actions = torch.zeros(8, 4)
    return HumanVisibleReplayWindow(
        anchor_id=anchor_id,
        source_kind=source_kind,
        observations=observations,
        teacher_actions=actions,
        student_actions=actions,
        initial_state=torch.zeros(3),
        burn_in=0,
        start=start,
    )


def test_per_frame_loss_keeps_free_press_permutation_invariant() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    predicted_a = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
    predicted_b = torch.tensor([[0.0, 0.9, 0.0, 0.0]])

    loss_a = n_key_actuation_loss_per_frame(predicted_a, target)
    loss_b = n_key_actuation_loss_per_frame(predicted_b, target)

    assert torch.allclose(loss_a, loss_b)


def test_context_balancing_is_invariant_to_duplicate_identical_idle_frames() -> None:
    target_short = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    predicted_short = torch.tensor(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
        ]
    )
    student_short = predicted_short.clone()

    target_long = torch.cat(
        [target_short[:1], target_short[1:].repeat(20, 1)],
        dim=0,
    )
    predicted_long = torch.cat(
        [predicted_short[:1], predicted_short[1:].repeat(20, 1)],
        dim=0,
    )
    student_long = predicted_long.clone()

    short, _ = context_balanced_actuation_loss(
        predicted_short,
        target_short,
        student_short,
        control_dt_s=0.01,
    )
    long, _ = context_balanced_actuation_loss(
        predicted_long,
        target_long,
        student_long,
        control_dt_s=0.01,
    )

    assert torch.allclose(short, long, atol=1e-7, rtol=1e-6)


def test_balanced_epoch_plan_is_deterministic_and_source_balanced() -> None:
    windows = []
    for anchor_id in range(3):
        for source_kind in ("expert", "dagger"):
            windows.append(_window(anchor_id, source_kind, 0))
            windows.append(_window(anchor_id, source_kind, 8))

    first = build_balanced_epoch_plan(
        windows,
        anchor_ids=[0, 1, 2],
        updates_per_epoch=5,
        anchors_per_batch=2,
        seed=123,
    )
    second = build_balanced_epoch_plan(
        windows,
        anchor_ids=[0, 1, 2],
        updates_per_epoch=5,
        anchors_per_batch=2,
        seed=123,
    )

    assert first == second
    for batch in first:
        assert len(batch) == 4
        selected = [windows[index] for index in batch]
        assert sum(item.source_kind == "expert" for item in selected) == 2
        assert sum(item.source_kind == "dagger" for item in selected) == 2
        for offset in range(0, len(selected), 2):
            assert selected[offset].anchor_id == selected[offset + 1].anchor_id


def test_single_safe_anchor_drop_is_not_catastrophic_for_six_validation_anchors() -> None:
    best = {
        "safe": 6,
        "anchors": 6,
        "hits": 100,
    }
    one_drop = {
        "safe": 5,
        "anchors": 6,
        "hits": 99,
    }
    two_drop = {
        "safe": 4,
        "anchors": 6,
        "hits": 99,
    }

    assert catastrophic_regression(one_drop, best) is False
    assert catastrophic_regression(two_drop, best) is True


def test_progression_gate_is_relative_to_fixed_baseline() -> None:
    baseline = {"safe": 3, "hits": 100, "x": 50.0}
    assert progression_passes(
        {"safe": 3, "hits": 98, "x": 49.0},
        baseline,
    )
    assert not progression_passes(
        {"safe": 2, "hits": 110, "x": 60.0},
        baseline,
    )


def test_catastrophic_epoch_cannot_advance_progression() -> None:
    decision = update_curriculum_state(
        level=1,
        passed=True,
        catastrophic=True,
        validation_streak=1,
        catastrophic_streak=0,
        progression_streak=3,
        catastrophic_patience=2,
    )

    assert decision.progression_pass is False
    assert decision.validation_streak == 0
    assert decision.promoted is False
    assert decision.rolled_back is False


def test_rollback_resets_progression_streak() -> None:
    decision = update_curriculum_state(
        level=1,
        passed=True,
        catastrophic=True,
        validation_streak=2,
        catastrophic_streak=1,
        progression_streak=3,
        catastrophic_patience=2,
    )

    assert decision.rolled_back is True
    assert decision.validation_streak == 0
    assert decision.catastrophic_streak == 0
    assert decision.promoted is False


def test_clean_streak_promotes_level() -> None:
    decision = update_curriculum_state(
        level=0,
        passed=True,
        catastrophic=False,
        validation_streak=2,
        catastrophic_streak=0,
        progression_streak=3,
        catastrophic_patience=2,
    )

    assert decision.progression_pass is True
    assert decision.promoted is True
    assert decision.level == 1
    assert decision.validation_streak == 0


def test_transition_epochs_use_reduced_update_budget() -> None:
    assert effective_updates_per_epoch(
        level=1,
        transition_epochs_remaining=3,
        updates_per_epoch=32,
        transition_updates_per_epoch=16,
    ) == 16
    assert effective_updates_per_epoch(
        level=1,
        transition_epochs_remaining=0,
        updates_per_epoch=32,
        transition_updates_per_epoch=16,
    ) == 32
    assert effective_updates_per_epoch(
        level=0,
        transition_epochs_remaining=3,
        updates_per_epoch=32,
        transition_updates_per_epoch=16,
    ) == 32
