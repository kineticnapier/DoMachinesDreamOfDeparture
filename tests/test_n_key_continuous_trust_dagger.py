from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_real_chart_v163_n_key_continuous_trust_dagger as trainer


def _stats(
    *,
    hits: int,
    targets: int = 100,
    xacc: float = 50.0,
    early: int = 0,
    overloaded: bool = False,
):
    return SimpleNamespace(
        hits=hits,
        targets=targets,
        x_accuracy_percent=xacc,
        x_accuracy_points=xacc * targets / 100.0,
        x_accuracy_denominator=float(targets),
        too_early_presses=early,
        overloaded=overloaded,
    )


def _candidate(
    *,
    alpha: float,
    hits: int,
    safe: bool,
    xacc: float = 50.0,
) -> trainer.TrustCandidate:
    return trainer.TrustCandidate(
        alpha=alpha,
        state={"weight": torch.tensor([alpha])},
        results=[(_stats(hits=hits, xacc=xacc, overloaded=not safe), hits)],
        guard_accepted=safe,
        guard_reasons=() if safe else ("unsafe",),
    )


def test_interpolate_state_scales_proposal_from_accepted_base() -> None:
    base = {"weight": torch.tensor([0.0, 4.0])}
    proposal = {"weight": torch.tensor([8.0, 0.0])}

    quarter = trainer._interpolate_state(base, proposal, 0.25)

    assert torch.allclose(quarter["weight"], torch.tensor([2.0, 3.0]))


def test_interpolate_state_rejects_changed_nonfloating_buffer() -> None:
    base = {"count": torch.tensor([1], dtype=torch.int64)}
    proposal = {"count": torch.tensor([2], dtype=torch.int64)}

    with pytest.raises(ValueError, match="non-floating"):
        trainer._interpolate_state(base, proposal, 0.5)


def test_clone_optimizer_state_isolated_from_later_proposal_updates() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()

    accepted = trainer._clone_optimizer_state(optimizer)
    live_state = optimizer.state_dict()
    live_exp_avg = next(iter(live_state["state"].values()))["exp_avg"]
    live_exp_avg.add_(10.0)

    accepted_exp_avg = next(iter(accepted["state"].values()))["exp_avg"]
    assert not torch.equal(live_exp_avg, accepted_exp_avg)


def test_clone_optimizer_state_accepts_saved_state_dict() -> None:
    saved = {
        "state": {0: {"exp_avg": torch.tensor([1.0])}},
        "param_groups": [{"params": [0], "lr": 0.1}],
    }

    cloned = trainer._clone_optimizer_state(saved)
    saved["state"][0]["exp_avg"].add_(5.0)

    assert torch.equal(cloned["state"][0]["exp_avg"], torch.tensor([1.0]))


def test_restore_optimizer_state_rolls_back_rejected_proposal_momentum() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    accepted = trainer._clone_optimizer_state(optimizer)

    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    assert optimizer.state

    trainer._restore_optimizer_state(optimizer, accepted)

    assert optimizer.state == {}


def test_select_improving_candidate_ignores_unsafe_and_nonimproving() -> None:
    reference = [(_stats(hits=100, xacc=30.0), 100)]
    candidates = [
        _candidate(alpha=1.0, hits=150, safe=False),
        _candidate(alpha=0.5, hits=95, safe=True, xacc=99.0),
        _candidate(alpha=0.25, hits=120, safe=True, xacc=20.0),
        _candidate(alpha=0.125, hits=110, safe=True, xacc=90.0),
    ]

    chosen = trainer._select_improving_candidate(reference, candidates)

    assert chosen is candidates[2]
    assert chosen.alpha == 0.25


def test_select_improving_candidate_can_keep_current_state() -> None:
    reference = [(_stats(hits=100), 100)]
    candidates = [
        _candidate(alpha=1.0, hits=130, safe=False),
        _candidate(alpha=0.5, hits=90, safe=True),
    ]

    assert trainer._select_improving_candidate(reference, candidates) is None


def test_v163_checkpoint_records_trust_region_optimizer_selection() -> None:
    state = {"weight": torch.tensor([1.0])}
    optimizer_state = {
        "state": {0: {"step": torch.tensor(1.0)}},
        "param_groups": [{"params": [0], "lr": 3e-4}],
    }
    parent = {
        "format_version": 18,
        "trainer_version": "1.6.1-n-key-dagger",
        "dagger_round": 1,
        "final_used_for_selection": False,
        "finalized": False,
    }
    history = [
        {
            "epoch": 1,
            "alpha": 0.25,
            "guard_accepted": True,
            "accepted_for_continuation": True,
        }
    ]

    payload = trainer._checkpoint_payload(
        parent,
        model_state=state,
        optimizer_state=optimizer_state,
        source_checkpoint=Path("dagger1.pt"),
        output_checkpoint=Path("dagger2-trust.pt"),
        round_index=2,
        completed_epoch=4,
        requested_epochs=4,
        selected_epoch=3,
        selected_alpha=0.125,
        lr=3e-4,
        expert_frames=100,
        dagger_frames=90,
        losses=[1.0, 0.8, 0.7, 0.6],
        trust_alphas=(1.0, 0.5, 0.25, 0.125),
        trust_history=history,
    )

    assert trainer.TRAINER_VERSION == "1.6.3-n-key-continuous-trust-dagger-opt-rollback"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 21
    assert payload["dagger_round"] == 2
    assert payload["dagger_action_mode"] == "continuous"
    assert payload["dagger_selected_epoch"] == 3
    assert payload["dagger_selected_alpha"] == pytest.approx(0.125)
    assert payload["dagger_trust_continuation"] == "accepted-model+optimizer-state"
    assert payload["dagger_trust_optimizer_continuation"] == "accepted-proposal-or-rollback"
    assert payload["dagger_optimizer_state"] == optimizer_state
    assert payload["dagger_trust_history"] == history
    assert payload["dagger_selection_uses_validation"] is False
    assert payload["final_used_for_selection"] is False
    assert payload["finalized"] is False
