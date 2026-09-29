from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v056 as v056  # noqa: E402
import train_real_chart_v070 as trainer  # noqa: E402
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM  # noqa: E402
from dmdod.real_chart_hud_features import HUD_FEATURE_DIM, HUD_REAL_CHART_INPUT_DIM  # noqa: E402
from dmdod.recurrent_policy import RecurrentActorCritic  # noqa: E402


def test_v070_uses_separate_hud_checkpoint_format_and_dimension():
    assert trainer.TRAINER_VERSION == "0.7.0-human-visible-hud"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 14
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v070_hud.pt")
    assert HUD_REAL_CHART_INPUT_DIM == REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM
    assert HUD_REAL_CHART_INPUT_DIM == 245


def test_v070_import_does_not_mutate_legacy_encoder_before_main():
    # v0.6.x checkpoints/evaluator remain usable unless the v0.7.0 trainer is
    # explicitly installed inside its own training process.
    assert REAL_CHART_INPUT_DIM == 233


def test_v070_installs_hud_width_for_legacy_dagger_validator(monkeypatch):
    # v0.6.0 creates v0.5.5 DAggerSequence objects. Their __post_init__ reads
    # the v0.5.5 module-global width at runtime, so v0.7 must update it to 245D.
    monkeypatch.setattr(trainer.v055, "REAL_CHART_INPUT_DIM", REAL_CHART_INPUT_DIM)
    trainer._install_dagger_input_dimension()
    assert trainer.v055.REAL_CHART_INPUT_DIM == HUD_REAL_CHART_INPUT_DIM

    observations = torch.zeros((3, HUD_REAL_CHART_INPUT_DIM))
    actions = torch.zeros((3, 2))
    sequence = trainer.v055.DAggerSequence(observations, actions, "hud-test")
    assert sequence.frames == 3


def test_exact_train_proposal_cache_reuses_model_state(monkeypatch):
    trainer._install_dagger_input_dimension()
    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=8,
        initial_log_std=-1.20,
    )
    observations = torch.zeros((4, HUD_REAL_CHART_INPUT_DIM))
    actions = torch.zeros((4, 2))
    sequence = trainer.v055.DAggerSequence(observations, actions, "cache-test")
    stable = v056._make_stable_sequence(
        sequence,
        press_recovery_cap=8,
        expert=True,
    )

    calls = {"count": 0}

    def fake_train(model, sequences, *, optimizer, chunk_steps, reverse_order):
        del sequences, optimizer, chunk_steps, reverse_order
        calls["count"] += 1
        with torch.no_grad():
            model.actor_mean.bias.add_(0.125)
        return 1.2345

    monkeypatch.setattr(trainer, "_ORIGINAL_TRAIN_ONE_EPOCH", fake_train)
    monkeypatch.setattr(trainer, "_TRAIN_PROPOSAL_CACHE", {})
    monkeypatch.setattr(trainer, "_SEQUENCE_SIGNATURE_CACHE", {})
    monkeypatch.setattr(trainer, "_TRAIN_PROPOSAL_CACHE_HITS", 0)
    monkeypatch.setattr(trainer, "_TRAIN_PROPOSAL_CACHE_MISSES", 0)

    base_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(trainer.v057._policy_parameters(model), lr=1e-4)
    first_loss = trainer._cached_train_one_epoch(
        model,
        [stable],
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=False,
    )
    first_state = copy.deepcopy(model.state_dict())
    assert calls["count"] == 1
    assert trainer._TRAIN_PROPOSAL_CACHE_MISSES == 1

    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(trainer.v057._policy_parameters(model), lr=1e-4)
    second_loss = trainer._cached_train_one_epoch(
        model,
        [stable],
        optimizer=optimizer,
        chunk_steps=192,
        reverse_order=False,
    )

    assert calls["count"] == 1
    assert trainer._TRAIN_PROPOSAL_CACHE_HITS == 1
    assert second_loss == pytest.approx(first_loss)
    for name, value in model.state_dict().items():
        assert torch.equal(value, first_state[name])


def test_train_proposal_cache_distinguishes_sequence_order(monkeypatch):
    trainer._install_dagger_input_dimension()
    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=8,
        initial_log_std=-1.20,
    )
    observations = torch.zeros((2, HUD_REAL_CHART_INPUT_DIM))
    actions = torch.zeros((2, 2))
    sequence = trainer.v055.DAggerSequence(observations, actions, "cache-order-test")
    stable = v056._make_stable_sequence(sequence, press_recovery_cap=8, expert=True)

    calls = {"count": 0}

    def fake_train(model, sequences, *, optimizer, chunk_steps, reverse_order):
        del model, sequences, optimizer, chunk_steps, reverse_order
        calls["count"] += 1
        return 0.5

    monkeypatch.setattr(trainer, "_ORIGINAL_TRAIN_ONE_EPOCH", fake_train)
    monkeypatch.setattr(trainer, "_TRAIN_PROPOSAL_CACHE", {})
    monkeypatch.setattr(trainer, "_SEQUENCE_SIGNATURE_CACHE", {})

    base_state = copy.deepcopy(model.state_dict())
    for reverse in (False, True):
        model.load_state_dict(base_state)
        optimizer = torch.optim.Adam(trainer.v057._policy_parameters(model), lr=1e-4)
        trainer._cached_train_one_epoch(
            model,
            [stable],
            optimizer=optimizer,
            chunk_steps=192,
            reverse_order=reverse,
        )

    assert calls["count"] == 2


def test_parallel_eval_defaults_to_physical_core_guess_and_allows_override(monkeypatch):
    monkeypatch.delenv("DMDOD_EVAL_WORKERS", raising=False)
    monkeypatch.setattr(trainer.os, "cpu_count", lambda: 12)
    assert trainer._configured_eval_workers() == 6

    monkeypatch.setenv("DMDOD_EVAL_WORKERS", "3")
    assert trainer._configured_eval_workers() == 3


def test_parallel_eval_worker_count_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("DMDOD_EVAL_WORKERS", "wat")
    with pytest.raises(SystemExit, match="DMDOD_EVAL_WORKERS"):
        trainer._configured_eval_workers()
