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


# Resume-specific CPU checks. No chart dataset or CUDA device is needed.
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest

from dmdod.training.human_visible_curriculum import (
    BETA_SCHEDULE,
    TRAINING_MODE,
    _check_validation_context,
    _check_validation_summary,
    _checkpoint_best_identity,
    _check_optimizer_state,
    _load_checked_optimizer,
    curriculum_checkpoint_role,
    prepare_curriculum_start,
)


def _small_resume_model_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(
        [{"params": list(model[0].parameters()), "lr": 3e-5},
         {"params": list(model[1].parameters()), "lr": 5e-5}],
        weight_decay=1e-4,
    )
    optimizer.zero_grad()
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    return model, optimizer


def _tiny_resume_context(tmp_path, model, optimizer, *, role="selected-best"):
    source = tmp_path / ("source.progress.pt" if role == "continuation-progress" else "source.pt")
    output = tmp_path / ("source.pt" if role == "continuation-progress" else "copy.pt")
    summary = {"safe": 1, "anchors": 1, "hits": 2, "targets": 5,
               "x": 50.0, "pp": 40.0, "mae_ms": 22.0}
    old = {
        "training_mode": TRAINING_MODE,
        "human_visible_stopped_reason": "running-progress" if role == "continuation-progress" else "running-best",
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "human_visible_optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "training_config": {"run": {"dataset": str(tmp_path / "dataset")},
                            "data": {"anchor_limit": 20, "validation_limit": 20}},
        "human_visible_epoch": 2,
        "human_visible_dagger_level": 1,
        "human_visible_history": [{"epoch": 2, "beta": 1.0, "selected_best": True}],
        "human_visible_parent_validation_summary": dict(summary),
        "human_visible_best_validation_summary": dict(summary),
        "human_visible_current_validation_summary": dict(summary),
        "human_visible_validation_streak": 1,
        "human_visible_catastrophic_streak": 1,
    }
    data = SimpleNamespace(anchor_limit=20, validation_limit=20)
    run = SimpleNamespace(dataset=str(tmp_path / "dataset"))
    config = SimpleNamespace(run=run, data=data, human_visible=SimpleNamespace(transition_epochs=3))
    prepared = SimpleNamespace(parent=old, source_checkpoint=source, output_checkpoint=output,
                               validation=[SimpleNamespace(chart_sha256="abc", start_s=0, end_s=10)])
    return prepared, config, summary


def test_old_selected_best_beta_is_read_from_matching_history(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, summary = _tiny_resume_context(tmp_path, model, optim)
    assert _checkpoint_best_identity(prepared.parent) == (2, 1.0, 0)
    prepared.parent["human_visible_history"] = []
    with pytest.raises(SystemExit, match="cannot infer"):
        _checkpoint_best_identity(prepared.parent)


def test_new_best_resume_resets_streak_and_retains_optimizer(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    start = prepare_curriculum_start(prepared, config, model, optim)
    assert (start.epoch, start.level, start.best_epoch, start.best_beta) == (2, 0, 2, 1.0)
    assert (start.validation_streak, start.catastrophic_streak) == (0, 0)
    assert start.seed_output is True
    assert start.best_optimizer_state["state"]


def test_new_best_resume_rejects_existing_output_or_progress(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    prepared.output_checkpoint.touch()
    with pytest.raises(SystemExit, match="already exists"):
        prepare_curriculum_start(prepared, config, model, optim)
    prepared.output_checkpoint.unlink()
    progress = prepared.output_checkpoint.with_name("copy.progress.pt")
    progress.touch()
    with pytest.raises(SystemExit, match="progress exists"):
        prepare_curriculum_start(prepared, config, model, optim)


def test_new_best_resume_rejects_source_equal_output(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    prepared.output_checkpoint = prepared.source_checkpoint
    with pytest.raises(SystemExit, match="must differ"):
        prepare_curriculum_start(prepared, config, model, optim)


def test_missing_or_wrong_optimizer_state_is_rejected(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    del prepared.parent["human_visible_optimizer_state"]
    with pytest.raises(SystemExit, match="optimizer state"):
        prepare_curriculum_start(prepared, config, model, optim)
    prepared.parent["human_visible_optimizer_state"] = {"state": {}, "param_groups": []}
    with pytest.raises(SystemExit, match="group count"):
        prepare_curriculum_start(prepared, config, model, optim)


def test_progress_resume_requires_existing_matching_best(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim, role="continuation-progress")
    prepared.parent["human_visible_best_checkpoint_path"] = str(prepared.output_checkpoint)
    with pytest.raises(SystemExit, match="does not exist"):
        prepare_curriculum_start(prepared, config, model, optim)
    torch.save({"training_mode": TRAINING_MODE, "human_visible_stopped_reason": "running-progress"},
               prepared.output_checkpoint)
    with pytest.raises(SystemExit, match="not selected-best"):
        prepare_curriculum_start(prepared, config, model, optim)


def test_progress_resume_restores_counters_and_best_reference(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim, role="continuation-progress")
    best = dict(prepared.parent)
    best["human_visible_stopped_reason"] = "running-best"
    torch.save(best, prepared.output_checkpoint)
    prepared.parent["human_visible_best_checkpoint_path"] = str(prepared.output_checkpoint)
    prepared.parent["human_visible_epoch"] = 8
    prepared.parent["human_visible_dagger_level"] = 2
    prepared.parent["human_visible_transition_epochs_remaining"] = 2
    start = prepare_curriculum_start(prepared, config, model, optim)
    assert (start.epoch, start.level, start.validation_streak, start.catastrophic_streak) == (8, 2, 1, 1)
    assert start.transition_epochs_remaining == 2
    assert start.seed_output is False
    assert start.best_epoch == 2


def test_resume_validation_rejects_modified_limits_or_results(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, summary = _tiny_resume_context(tmp_path, model, optim)
    config.data.validation_limit = 19
    with pytest.raises(SystemExit, match="validation_limit"):
        _check_validation_context(prepared.parent, config, prepared)
    with pytest.raises(SystemExit, match="fixed Validation mismatch"):
        _check_validation_summary(dict(summary, hits=3), summary)


def test_resume_optimizer_moments_are_loaded_and_can_be_restored(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    original = copy.deepcopy(optim.state_dict())
    for item in optim.state.values():
        item["exp_avg"].zero_()
    _load_checked_optimizer(optim, prepared.parent, model)
    assert any(torch.count_nonzero(s["exp_avg"]) for s in optim.state.values())
    assert len(original["state"]) == len(optim.state)


def test_resume_checkpoint_roles_fail_closed():
    assert curriculum_checkpoint_role({"training_mode": TRAINING_MODE,
        "human_visible_stopped_reason": "running-best"}) == "selected-best"
    assert curriculum_checkpoint_role({"training_mode": TRAINING_MODE,
        "human_visible_stopped_reason": "running-progress"}) == "continuation-progress"
    with pytest.raises(SystemExit, match="cannot be determined"):
        curriculum_checkpoint_role({"training_mode": TRAINING_MODE})


def test_validation_identity_mismatch_is_rejected(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    prepared.parent["human_visible_validation_identity"] = [
        {"chart_sha256": "different", "start_s": 0, "end_s": 10}
    ]
    with pytest.raises(SystemExit, match="SHA changed"):
        _check_validation_context(prepared.parent, config, prepared)


def test_optimizer_shape_mismatch_is_rejected(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim)
    data = prepared.parent["human_visible_optimizer_state"]
    key = next(iter(data["state"]))
    data["state"][key]["exp_avg"] = torch.ones(20)
    with pytest.raises(SystemExit, match="tensor shape mismatch"):
        _check_optimizer_state(optim, prepared.parent, model)


def test_progress_rejects_different_saved_best_summary(tmp_path):
    model, optim = _small_resume_model_and_optimizer()
    prepared, config, _ = _tiny_resume_context(tmp_path, model, optim, role="continuation-progress")
    best = copy.deepcopy(prepared.parent)
    best["human_visible_stopped_reason"] = "running-best"
    best["human_visible_best_validation_summary"]["hits"] = 99
    torch.save(best, prepared.output_checkpoint)
    prepared.parent["human_visible_best_checkpoint_path"] = str(prepared.output_checkpoint)
    with pytest.raises(SystemExit, match="rankings disagree"):
        prepare_curriculum_start(prepared, config, model, optim)
