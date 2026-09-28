from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v056 as trainer  # noqa: E402


def _sequence(frames: int, source: str, *, press_run: int = 0) -> trainer.StableSequence:
    observations = torch.zeros((frames, trainer.REAL_CHART_INPUT_DIM), dtype=torch.float32)
    actions = torch.zeros((frames, 2), dtype=torch.float32)
    if press_run:
        actions[: min(frames, press_run), 0] = 1.0
    base = trainer.v055.DAggerSequence(observations, actions, source)
    return trainer._make_stable_sequence(
        base,
        press_recovery_cap=trainer.DEFAULT_PRESS_RECOVERY_CAP,
        expert=False,
    )


def _eval(*, hits: int, early: int, overloaded: bool, targets: int = 107, xacc: float = 60.0):
    stats = SimpleNamespace(
        overloaded=overloaded,
        hits=hits,
        too_early_presses=early,
        x_accuracy_percent=xacc,
        perfect_rate=0.4,
        misses=targets - hits,
        targets=targets,
    )
    return trainer.v054.StudentEvalResult(stats=stats, physical_keydowns=hits + early)


def test_teacher_mixture_schedule_decays_to_student_only():
    assert trainer._mixture_beta(1) == 0.50
    assert trainer._mixture_beta(2) == 0.25
    assert trainer._mixture_beta(3) == 0.10
    assert trainer._mixture_beta(4) == 0.00
    assert trainer._mixture_beta(8) == 0.00


def test_press_recovery_cap_masks_excess_positive_labels_without_dropping_context():
    actions = torch.zeros((12, 2), dtype=torch.float32)
    actions[:10, 0] = 1.0
    actions[10, 0] = -1.0
    actions[11, 0] = 1.0

    weights = trainer._press_recovery_weights(actions, cap=3)

    assert weights.shape == actions.shape
    assert torch.equal(weights[:3, 0], torch.ones(3))
    assert torch.equal(weights[3:10, 0], torch.zeros(7))
    assert weights[10, 0].item() == 1.0
    assert weights[11, 0].item() == 1.0
    assert torch.equal(weights[:, 1], torch.ones(12))


def test_student_replay_never_exceeds_one_expert_length_and_prefers_newer_trajectories():
    old = _sequence(70, "old")
    middle = _sequence(20, "middle")
    newest = _sequence(25, "newest")

    kept = trainer._trim_student_replay([old, middle, newest], max_frames=100)

    assert trainer._student_replay_frames(kept) <= 100
    assert [item.sequence.source for item in kept] == ["middle", "newest"]


def test_guard_rejects_safe_to_overload_and_large_hit_regression():
    best = _eval(hits=87, early=14, overloaded=False)

    overload = trainer._guard_decision(best, _eval(hits=95, early=5, overloaded=True))
    assert not overload.accepted
    assert overload.reason == "safe->overload"

    regressed = trainer._guard_decision(best, _eval(hits=70, early=2, overloaded=False))
    assert not regressed.accepted
    assert regressed.reason.startswith("hit regression>")


def test_guard_accepts_strictly_better_safe_candidate():
    best = _eval(hits=87, early=14, overloaded=False, xacc=63.0)
    candidate = _eval(hits=90, early=10, overloaded=False, xacc=65.0)

    decision = trainer._guard_decision(best, candidate)

    assert decision.accepted
    assert decision.reason == "better"


def test_weighted_loss_ignores_masked_recovery_press_elements():
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    predicted = torch.zeros_like(target)
    full = torch.ones_like(target)
    masked = torch.ones_like(target)
    masked[1, 0] = 0.0

    full_loss = trainer._weighted_actuation_loss(predicted, target, full)
    masked_loss = trainer._weighted_actuation_loss(predicted, target, masked)

    assert torch.isfinite(full_loss)
    assert torch.isfinite(masked_loss)
    # The masked element contributes no press-recovery error; the remaining
    # first press still keeps a non-zero training signal.
    assert masked_loss > 0.0
    assert masked_loss <= full_loss
