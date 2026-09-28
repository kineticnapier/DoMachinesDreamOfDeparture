from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v057 as trainer  # noqa: E402
from dmdod.recurrent_policy import RecurrentActorCritic  # noqa: E402


def _result(
    *,
    hits: int = 87,
    misses: int = 20,
    xacc: float = 63.22,
    pp: float = 0.43,
    early: int = 14,
    overloaded: bool = False,
    targets: int = 107,
):
    stats = SimpleNamespace(
        hits=hits,
        misses=misses,
        x_accuracy_percent=xacc,
        perfect_rate=pp,
        too_early_presses=early,
        overloaded=overloaded,
        targets=targets,
    )
    return trainer.v054.StudentEvalResult(stats, hits + early)


def _stable_sequence(frames: int, source: str):
    observations = torch.zeros(
        (frames, trainer.REAL_CHART_INPUT_DIM),
        dtype=torch.float32,
    )
    actions = torch.zeros((frames, 2), dtype=torch.float32)
    if frames >= 2:
        actions[1, 0] = 1.0
    sequence = trainer.v055.DAggerSequence(observations, actions, source)
    return trainer.v056._make_stable_sequence(
        sequence,
        press_recovery_cap=trainer.v056.DEFAULT_PRESS_RECOVERY_CAP,
        expert=source == "expert",
    )


def test_conservative_guard_rejects_observed_v056_accuracy_early_regression():
    best = _result()
    # v0.5.6 accepted a similar H=90 / X=61.41 / early=24 candidate.
    candidate = _result(hits=90, misses=17, xacc=61.41, pp=0.42, early=24)
    decision = trainer._conservative_guard(best, candidate)
    assert not decision.accepted
    assert "early regression" in decision.reason or "XAcc regression" in decision.reason


def test_conservative_guard_rejects_safe_to_overload_even_with_more_hits():
    best = _result()
    candidate = _result(hits=100, misses=7, xacc=80.0, pp=0.70, early=8, overloaded=True)
    decision = trainer._conservative_guard(best, candidate)
    assert not decision.accepted
    assert decision.reason == "safe->overload"


def test_conservative_guard_accepts_accuracy_gain_without_material_regression():
    best = _result()
    candidate = _result(hits=88, misses=19, xacc=66.0, pp=0.46, early=15)
    decision = trainer._conservative_guard(best, candidate)
    assert decision.accepted
    assert decision.reason == "accuracy-first better"


def test_one_epoch_resets_gru_state_for_each_trajectory():
    class CountingPolicy(RecurrentActorCritic):
        def __init__(self) -> None:
            super().__init__(input_dim=trainer.REAL_CHART_INPUT_DIM, hidden_dim=8)
            self.initial_state_calls = 0

        def initial_state(self, device: torch.device) -> torch.Tensor:
            self.initial_state_calls += 1
            return super().initial_state(device)

    model = CountingPolicy()
    sequences = [_stable_sequence(4, "expert"), _stable_sequence(3, "student")]
    optimizer = trainer._new_optimizer(model, 1e-4)
    loss = trainer._train_one_epoch(
        model,
        sequences,
        optimizer=optimizer,
        chunk_steps=2,
        reverse_order=False,
    )
    assert loss >= 0.0
    assert model.initial_state_calls == len(sequences)


def test_accuracy_key_prefers_xacc_before_extra_hits_after_guards():
    accurate = _result(hits=87, xacc=65.0, pp=0.45, early=14)
    more_hits = _result(hits=90, misses=17, xacc=64.0, pp=0.45, early=14)
    assert trainer._accuracy_key(accurate) > trainer._accuracy_key(more_hits)
