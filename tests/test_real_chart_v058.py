from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v058 as trainer  # noqa: E402


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


def test_state_interpolation_scales_full_bc_proposal_without_touching_integer_state():
    base = {
        "weight": torch.tensor([0.0, 2.0], dtype=torch.float32),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    proposal = {
        "weight": torch.tensor([4.0, -2.0], dtype=torch.float32),
        "counter": torch.tensor([99], dtype=torch.int64),
    }

    result = trainer._interpolate_state(base, proposal, 0.25)

    assert torch.allclose(result["weight"], torch.tensor([1.0, 1.0]))
    assert torch.equal(result["counter"], base["counter"])
    assert result["counter"].data_ptr() != base["counter"].data_ptr()


def test_state_interpolation_endpoints_are_exact_for_float_parameters():
    base = {"weight": torch.tensor([1.0, -3.0])}
    proposal = {"weight": torch.tensor([5.0, 9.0])}

    assert torch.equal(trainer._interpolate_state(base, proposal, 0.0)["weight"], base["weight"])
    assert torch.equal(trainer._interpolate_state(base, proposal, 1.0)["weight"], proposal["weight"])


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_state_interpolation_rejects_invalid_alpha(alpha: float):
    state = {"weight": torch.tensor([0.0])}
    with pytest.raises(ValueError):
        trainer._interpolate_state(state, state, alpha)


def test_trust_choice_prefers_best_accuracy_survivor_not_largest_alpha():
    best = _result()
    full_step = trainer.TrustCandidate(
        1.0,
        _result(hits=88, misses=19, xacc=64.0, pp=0.44, early=14),
        trainer.v057.ConservativeDecision(True, "accuracy-first better"),
    )
    half_step = trainer.TrustCandidate(
        0.5,
        _result(hits=88, misses=19, xacc=66.0, pp=0.46, early=13),
        trainer.v057.ConservativeDecision(True, "accuracy-first better"),
    )
    quarter_step_rejected = trainer.TrustCandidate(
        0.25,
        _result(hits=100, misses=7, xacc=80.0, pp=0.80, early=2, overloaded=True),
        trainer.v057.ConservativeDecision(False, "safe->overload"),
    )

    choice = trainer._choose_trust_candidate(
        best,
        [full_step, half_step, quarter_step_rejected],
    )

    assert choice.accepted
    assert choice.alpha == 0.5
    assert choice.evaluation is half_step.evaluation


def test_trust_choice_rolls_back_when_every_alpha_fails_guard():
    best = _result()
    candidates = [
        trainer.TrustCandidate(
            alpha,
            _result(hits=0, misses=0, xacc=15.0, pp=0.0, early=4, overloaded=True),
            trainer.v057.ConservativeDecision(False, "safe->overload"),
        )
        for alpha in trainer.DEFAULT_TRUST_ALPHAS
    ]

    choice = trainer._choose_trust_candidate(best, candidates)

    assert not choice.accepted
    assert choice.alpha is None
    assert choice.evaluation is None


def test_default_trust_schedule_reaches_one_thirty_second():
    assert trainer.DEFAULT_TRUST_ALPHAS == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
